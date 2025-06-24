'''
server.py

1) listen for incoming connections
2) receive video frames
3) process frames and send results back

'''

import cv2
import numpy as np
import socket
import time
import av
import os
from multiprocessing import Process, Queue


from typing import List, Tuple

from networking import (
    HEADER_NORMAL, HEADER_TERMINATE,
    connect_dual_tcp, transmit_data, receive_data,
    measure_timelag,
    bytes_to_int32, int32_to_bytes,
    bytes_to_bool, bool_to_bytes
)
from preprocessing import (
    shift_anchor_features,
    apply_affine_and_pad, get_padded_image,
    estimate_affine_in_padded_anchor_fast,
    create_dirtiness_map,
    ndarray_to_bytes, bytes_to_ndarray
)

PAD_FILLING_COLOR = [0, 0, 0]  # RGB for padding

def rigid_from_mvs(mvs: np.ndarray):
    """
    모션벡터(N, 5) → 현재 프레임 ➜ 참조(이전) 프레임으로 가는 2×3 rigid 변환 추정.
    실패 시 None 반환.
    """
    if mvs is None or mvs.shape[0] < 3 or mvs.shape[1] != 5:
        return None
    # mvs: [dst_x, dst_y, motion_x, motion_y, motion_scale]
    dst = mvs[:, 0:2].astype(np.float32)
    src = dst + (mvs[:, 2:4] / mvs[:, 4:5]).astype(np.float32)
    if dst.shape[0] < 3 or src.shape[0] < 3:
        return None
    M, _ = cv2.estimateAffinePartial2D(dst, src,
                                       method=cv2.RANSAC,
                                       ransacReprojThreshold=2.0,
                                       confidence=0.995,
                                       refineIters=10)
    return M                                                   # shape (2,3) or None


def thread_receive_video(
    socket_rx: socket.socket, 
    frame_queue: Queue,
    queue_recv_timestamp: Queue,
    compress: str,
) -> float:
    decoder = None
    if compress == "h264":
        decoder = av.codec.CodecContext.create("h264", "r")

    while True:
        data = receive_data(socket_rx)
        if data is None:
            break

        fidx     = bytes_to_int32(data)
        refresh  = bytes_to_bool(receive_data(socket_rx))
        dmap     = bytes_to_ndarray(receive_data(socket_rx))
        shift_x  = bytes_to_int32(receive_data(socket_rx))
        shift_y  = bytes_to_int32(receive_data(socket_rx))
        blk_size = bytes_to_int32(receive_data(socket_rx))

        # ── anchor_image_padded 복원 ─────────────────────────────
        if compress == "none":
            anchor_img = bytes_to_ndarray(receive_data(socket_rx))

        elif compress == "jpeg":
            arr = np.frombuffer(receive_data(socket_rx), dtype=np.uint8)
            anchor_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        elif compress == "h264":
            # *** 순서 교정: 먼저 pkt_count 읽기 ***
            pkt_cnt = bytes_to_int32(receive_data(socket_rx))
            frames  = []
            for _ in range(pkt_cnt):
                pkt_bytes = receive_data(socket_rx)
                frames.extend(decoder.decode(av.Packet(pkt_bytes)))

            if not frames:                           # 혹시 버퍼링 남았다면 flush
                frames.extend(decoder.decode(None))
            anchor_img = frames[0].to_ndarray(format="bgr24")

        # -------------------------------------------------------
        queue_recv_timestamp.put(time.time())
        frame_queue.put((fidx, refresh, anchor_img, dmap,
                         (shift_x, shift_y), blk_size))

        
        if compress == "h264":
            pkt_cnt = bytes_to_int32(receive_data(socket_rx))
            frames  = []
            for _ in range(pkt_cnt):
                pkt_bytes = receive_data(socket_rx)
                frames.extend(decoder.decode(av.Packet(pkt_bytes)))

            if not frames:                           # 혹시 버퍼링 남았다면 flush
                frames.extend(decoder.decode(None))

    frame_queue.put((-1, None, None, None, (None, None), None))


    

def thread_process_video(
    socket_tx: socket.socket, 
    frame_queue: Queue, 
    result_queue: Queue,
    queue_proc_timestamp: Queue,
    device: str='cuda',
) -> float:
    """
    Thread function to process video frames received from the client and send results back.

    Args:
        socket_rx (socket.socket): The socket object to receive data from.
        socket_tx (socket.socket): The socket object to send data to.
    """
    import torch
    from models import MaskedRCNN_ViT_B_FPN_Contexted
    from models import DINO_4Scale_Swin_Contexted

    model = MaskedRCNN_ViT_B_FPN_Contexted(device=device)
    # model = DINO_4Scale_Swin_Contexted(device=device)
    model.load_weight("models/model_final_61ccd1.pkl")
    model.eval()
    print('Model loaded and ready to process frames.')

    num_warmup = 10
    for _ in range(num_warmup):
        dummy_input = np.zeros((1024, 1024, 3), dtype=np.uint8)
        model.forward_contexted(dummy_input)

    transmit_data(socket_tx, b"start!")

    start_timestamp = time.time()

    # Sizes: H, W
    input_size = (1024, 1024)

    anchor_features = None
    compute_rates = []

    num_frames = 0

    while True:
        # Fetch frame
        fidx, refresh, target_ndarray, dirtiness_map, (shift_x, shift_y), block_size = frame_queue.get()
        if fidx == -1:
            break

        dirtiness_map = torch.from_numpy(dirtiness_map).to(device)

        if anchor_features is not None and shift_x == 0 and shift_y == 0:
            # Shift anchor features if needed
            anchor_features = shift_anchor_features(
                anchor_features, shift_x, shift_y
            )
        
        # Process the frame
        if refresh:
            target_padded_ndarray = get_padded_image(
                target_ndarray, input_size, filling_color=PAD_FILLING_COLOR
            )
            (boxes, labels, scores), cached_features_dict = model.forward_contexted(
                target_padded_ndarray
            )
            dirtiness_map = torch.ones(1, 64, 64, 1)
        else:
            (boxes, labels, scores), cached_features_dict = model.forward_contexted(
                target_ndarray,
                anchor_features=anchor_features,
                dirtiness_map=dirtiness_map,
            )
        
        anchor_features = cached_features_dict

        # Threshold the results
        result_thres = 0.5
        boxes = boxes[scores > result_thres, :]
        labels = labels[scores > result_thres]
        scores = scores[scores > result_thres]

        # Statistics
        compute_rates.append(dirtiness_map.mean().item())

        timestamp = time.time()
        print(f"Processed {fidx} | boxes: {boxes.shape}, scores: {scores.shape}, labels: {labels.shape} | timestamp: {timestamp}")
        queue_proc_timestamp.put(timestamp)

        # Send results back to the client
        result_queue.put((
            fidx, target_ndarray, dirtiness_map, boxes, scores, labels, 
            (shift_x, shift_y), block_size
        ))

        num_frames += 1

    end_timestamp = time.time()

    print(f"Frame rate: {num_frames / (end_timestamp - start_timestamp)} FPS")
    print(f"Compute rate: {np.mean(compute_rates)}")

    # transmit_data(socket_tx, b"", HEADER_TERMINATE)  # Send termination signal
    result_queue.put((-1, None, None, None, None, None, (None, None), None))  # Send termination signal

    time.sleep(1)  # Give some time for the send thread to finish

    return end_timestamp - start_timestamp


def thread_send_result(
    socket_rx: socket.socket, socket_tx: socket.socket, result_queue: Queue
) -> float:
    """
    """

    from models.constants import COCO_LABELS_LIST, COCO_COLORS_ARRAY

    cum_shift_x, cum_shift_y = 0, 0

    while True:
        
        # Receive results from the processing thread
        fidx, target_ndarray, dirtiness_map, boxes, \
            scores, labels, (shift_x, shift_y), block_size = result_queue.get()
        
        if fidx == -1:
            break

        # Send results back to the client
        transmit_data(socket_tx, fidx.to_bytes(4, 'big'))
        transmit_data(socket_tx, ndarray_to_bytes(boxes))
        transmit_data(socket_tx, ndarray_to_bytes(scores))
        transmit_data(socket_tx, ndarray_to_bytes(labels))

        # Store the results in a file
        # for the dirty area, boost a green channel
        dmap_resized = cv2.resize(dirtiness_map[0, :, :, 0].cpu().numpy(), (target_ndarray.shape[1], target_ndarray.shape[0]), interpolation=cv2.INTER_NEAREST)
        dmap_resized = dmap_resized.astype(np.uint8)
        target_ndarray = target_ndarray.astype(np.uint16)
        target_ndarray[:, :, 1] += dmap_resized * 30
        target_ndarray = np.clip(target_ndarray, 0, 255).astype(np.uint8)

        # make boxes at the frame
        target_ndarray = target_ndarray.astype(np.uint8)
        for box, score, label in zip(boxes, scores, labels):
            box = box.astype(np.int32)
            cv2.rectangle(target_ndarray, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(target_ndarray, f"{COCO_LABELS_LIST[label]}: {score:.2f}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # make full frame yellow rect
        cv2.rectangle(target_ndarray, (0, 0), (target_ndarray.shape[1], target_ndarray.shape[0]), (0, 255, 255), 2)

        # shift the target_ndarray by the shift_x, shift_y
        if shift_x is not None and shift_y is not None:
            shift_x *= block_size
            shift_y *= block_size
            cum_shift_x += shift_x
            cum_shift_y += shift_y

        if cum_shift_x != 0 or cum_shift_y != 0:
            target_ndarray = np.roll(target_ndarray, shift=(cum_shift_y, cum_shift_x), axis=(0, 1))

        cv2.imwrite(f"inferenced/{fidx:05d}.jpg", target_ndarray)

    transmit_data(socket_tx, b"", HEADER_TERMINATE)  # Send termination signal



def main(args):
    server_ip = args.server_ip
    server_port1 = args.server_port
    server_port2 = server_port1 + 1
    device = args.device

    os.makedirs("inferenced", exist_ok=True)

    # Connect to the client
    socket_rx, socket_tx = connect_dual_tcp(
        server_ip, (server_port1, server_port2), node_type="server"
    )

    timelag = measure_timelag(socket_rx, socket_tx, "server")
    print(f"Timelag: {timelag} seconds")

    # Receive metadata
    metadata = receive_data(socket_rx)
    metadata = metadata.decode('utf-8')
    metadata = eval(metadata)  # Convert string back to dictionary
    
    gop = metadata.get("gop", 30)
    compress = metadata.get("compress", "none")
    frame_shape = metadata.get("frame_shape", (854, 480))

    print(f"Received metadata: {metadata}")

    # Prepare processes
    frame_queue = Queue(maxsize=100)
    result_queue = Queue(maxsize=100)

    queue_recv_timestamp = Queue(maxsize=100)
    queue_proc_timestamp = Queue(maxsize=100)

    thread_recv = Process(
        target=thread_receive_video, 
        args=(
            socket_rx, 
            frame_queue,
            queue_recv_timestamp, 
            compress,
        )
    )
    thread_proc = Process(
        target=thread_process_video, 
        args=(
            socket_tx, 
            frame_queue, 
            result_queue, 
            queue_proc_timestamp, 
            device
        )
    )
    thread_send = Process(
        target=thread_send_result, 
        args=(
            socket_rx, 
            socket_tx, 
            result_queue
        )
    )

    thread_recv.start()
    thread_proc.start()
    thread_send.start()
    
    # Wait for the processing thread to finish
    thread_recv.join()
    thread_proc.join()
    thread_send.join()

    # Receive statistics
    transmit_times = bytes_to_ndarray(receive_data(socket_rx)) + timelag
    preproc_times = bytes_to_ndarray(receive_data(socket_rx)) + timelag
    result_times = bytes_to_ndarray(receive_data(socket_rx)) + timelag

    # Process statistics
    receive_times = []
    while not queue_recv_timestamp.empty():
        receive_times.append(queue_recv_timestamp.get())
    receive_times = np.array(receive_times)

    proc_times = []
    while not queue_proc_timestamp.empty():
        proc_times.append(queue_proc_timestamp.get())
    proc_times = np.array(proc_times)

    ### avg total latency
    latencies = [receive - transmit for transmit, receive in zip(transmit_times, result_times)]
    print(f"Average latency: {np.mean(latencies):.4f} seconds")

    ### avg preproc latency
    preproc_latencies = [
        preproc - transmit for transmit, preproc in zip(transmit_times, preproc_times)
    ]
    print(f" > Average preproc latency: {np.mean(preproc_latencies):.4f} seconds")

    ### avg transfer latency
    transfer_latencies = [
        receive - preproc for preproc, receive in zip(preproc_times, receive_times)
    ]
    print(f" > Average transfer latency: {np.mean(transfer_latencies):.4f} seconds")

    ### avg proc latency
    proc_latencies = [
        proc - receive for proc, receive in zip(proc_times, receive_times)
    ]
    print(f" > Average proc latency: {np.mean(proc_latencies):.4f} seconds")
    
    ### avg return latency
    return_latencies = [
        receive - proc for receive, proc in zip(result_times, proc_times)
    ]
    print(f" > Average return latency: {np.mean(return_latencies):.4f} seconds")

    
    # Save logs as CSV
    ### Columns: frame_idx, send_time, receive_time, preproc_time, proc_time, return_time
    log_text = "frame_idx, send_time, receive_time, preproc_time, proc_time, return_time\n"
    for i in range(len(transmit_times)):
        log_text += f"{i}, {transmit_times[i]:.6f}, {receive_times[i]:.6f}, {preproc_times[i]:.6f}, {proc_times[i]:.6f}, {result_times[i]:.6f}\n"

    with open("logs/server_log.csv", "w") as f:
        f.write(log_text)
    print("Logs saved to server_log.csv")


    # Close sockets
    socket_rx.close()
    socket_tx.close()

    # Create inferenced video using ffmpeg
    os.system("ffmpeg -framerate 30 -i inferenced/%05d.jpg -c:v libx264 -pix_fmt yuv420p -an -y inferenced.mp4 2> /dev/null")
    os.system("rm -rf inferenced")  # Clean up images


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Server for processing video frames.")
    parser.add_argument("--server-ip", type=str, default="localhost", help="Server IP address.")
    parser.add_argument("--server-port", type=int, default=65432, help="Server port.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (cpu or cuda).")

    args = parser.parse_args()

    main(args)