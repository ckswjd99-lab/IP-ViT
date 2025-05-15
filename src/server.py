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
from multiprocessing import Process, Queue


from typing import List, Tuple

from networking import (
    HEADER_NORMAL, HEADER_TERMINATE,
    connect_dual_tcp, transmit_data, receive_data,
    measure_timelag,
)
from preprocessing import (
    estimate_affine_in_padded_anchor, apply_affine_and_pad, get_padded_image,
    estimate_affine_in_padded_anchor_fast,
    create_dirtiness_map,
    ndarray_to_bytes, bytes_to_ndarray
)


def thread_receive_video(
    socket_rx: socket.socket, 
    frame_queue: Queue,
    frame_shape: Tuple[int, int],
    queue_recv_timestamp: Queue,
    queue_preproc_timestamp: Queue,
    compress: str,
    gop: int,
) -> float:
    
    """
    Thread function to receive video frames from the client and put them into a queue.

    Args:
        socket_rx (socket.socket): The socket object to receive data from.
        frame_queue (Queue): Queue to store received frames.
        queue_recv_timestamp (Queue): Queue to store receive timestamps.
        queue_preproc_timestamp (Queue): Queue to store preprocessing timestamps.
        compress (str): Compression method to use ('jpeg', 'h264', or 'none').
        gop (int): Group of pictures size for video encoding.
    """
    import torch

    input_size = (1024, 1024)

    anchor_image_padded = None

    if compress == "h264":
        codec = av.codec.CodecContext.create('h264', 'r')
    
    while True:
        # Receive frame idx
        data = receive_data(socket_rx)
        if data is None:
            break

        fidx = int.from_bytes(data[:4], 'big')
        print(f"Received {fidx} | timestamp: {time.time()}")

        # Optionally decode the frame
        if compress == "jpeg":
            data = bytes_to_ndarray(receive_data(socket_rx))
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        elif compress == "h264":
            data = receive_data(socket_rx)
            packet = av.Packet(data)
            frames = codec.decode(packet)
            frame = frames[0].to_ndarray(format='bgr24')
            
        else:
            frame = bytes_to_ndarray(receive_data(socket_rx))
        queue_recv_timestamp.put(time.time())

        target_ndarray = cv2.resize(frame, frame_shape)

        # Preprocess the frame
        refresh = False
        if fidx % gop == 0 or anchor_image_padded is None:
            refresh = True
            dirtiness_map = torch.ones(1, 64, 64, 1)
            target_padded_ndarray = get_padded_image(target_ndarray, input_size)
        else:
            # try image refinement
            # affine_matrix = estimate_affine_in_padded_anchor(anchor_image_padded, frame)
            affine_matrix = estimate_affine_in_padded_anchor_fast(
                anchor_image_padded, target_ndarray
            )
            target_padded_ndarray = apply_affine_and_pad(target_ndarray, affine_matrix)
            
            scaling_factor = np.linalg.norm(affine_matrix[:2, :2])
            dirtiness_map = create_dirtiness_map(anchor_image_padded, target_padded_ndarray)
            
            refresh |= (target_padded_ndarray is None)
            refresh |= (scaling_factor < 0.95)
        
        anchor_image_padded = target_padded_ndarray

        queue_preproc_timestamp.put(time.time())

        # Put the frame into the queue
        frame_queue.put((fidx, refresh, target_padded_ndarray, dirtiness_map.cpu().numpy()))

        cv2.imwrite(f"recv/{fidx:05d}.jpg", target_padded_ndarray)
        
    frame_queue.put((-1, None, None, None))  # Send termination signal
    

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

    model = MaskedRCNN_ViT_B_FPN_Contexted(device=device)
    model.load_weight("models/model_final_61ccd1.pkl")
    model.eval()
    print('Model loaded and ready to process frames.')

    transmit_data(socket_tx, b"start!")

    start_timestamp = time.time()

    # Sizes: H, W
    input_size = (1024, 1024)

    anchor_features = None
    compute_rates = []

    num_frames = 0

    while True:
        # Fetch frame
        fidx, refresh, target_ndarray, dirtiness_map = frame_queue.get()
        if fidx == -1:
            break

        dirtiness_map = torch.from_numpy(dirtiness_map).to(device)
        
        # Process the frame
        if refresh:
            target_padded_ndarray = get_padded_image(target_ndarray, input_size)
            (boxes, labels, scores), cached_features_dict = model.forward_contexted(
                target_padded_ndarray
            )
            dirtiness_map = torch.ones(1, 64, 64, 1)
        else:
            (boxes, labels, scores), cached_features_dict = model.forward_contexted(
                target_padded_ndarray,
                anchor_features=anchor_features,
                dirtiness_map=dirtiness_map
            )
        
        anchor_features = cached_features_dict

        cv2.imwrite(f"recv/{fidx:05d}.jpg", target_padded_ndarray)
        
        # Statistics
        compute_rates.append(dirtiness_map.mean().item())

        timestamp = time.time()
        print(f"Processed {fidx} | boxes: {boxes.shape}, scores: {scores.shape}, labels: {labels.shape} | timestamp: {timestamp}")
        queue_proc_timestamp.put(timestamp)

        # Send results back to the client
        result_queue.put((fidx, boxes, scores, labels))

        num_frames += 1

    end_timestamp = time.time()

    print(f"Frame rate: {num_frames / (end_timestamp - start_timestamp)} FPS")
    print(f"Compute rate: {np.mean(compute_rates)}")

    # transmit_data(socket_tx, b"", HEADER_TERMINATE)  # Send termination signal
    result_queue.put((-1, None, None, None))  # Send termination signal

    return end_timestamp - start_timestamp


def thread_send_result(
    socket_rx: socket.socket, socket_tx: socket.socket, result_queue: Queue
) -> float:
    """
    """

    while True:
        
        # Receive results from the processing thread
        fidx, boxes, scores, labels = result_queue.get()
        if fidx == -1:
            break

        # Send results back to the client
        transmit_data(socket_tx, fidx.to_bytes(4, 'big'))
        transmit_data(socket_tx, ndarray_to_bytes(boxes))
        transmit_data(socket_tx, ndarray_to_bytes(scores))
        transmit_data(socket_tx, ndarray_to_bytes(labels))

    transmit_data(socket_tx, b"", HEADER_TERMINATE)  # Send termination signal



def main(args):
    server_ip = args.server_ip
    server_port1 = args.server_port
    server_port2 = server_port1 + 1
    device = args.device

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
    queue_preproc_timestamp = Queue(maxsize=100)
    queue_proc_timestamp = Queue(maxsize=100)

    thread_recv = Process(
        target=thread_receive_video, 
        args=(
            socket_rx, 
            frame_queue,
            frame_shape,
            queue_recv_timestamp, 
            queue_preproc_timestamp,
            compress,
            gop,
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
    result_times = bytes_to_ndarray(receive_data(socket_rx)) + timelag

    # Process statistics
    receive_times = []
    while not queue_recv_timestamp.empty():
        receive_times.append(queue_recv_timestamp.get())
    receive_times = np.array(receive_times)

    preproc_times = []
    while not queue_preproc_timestamp.empty():
        preproc_times.append(queue_preproc_timestamp.get())
    preproc_times = np.array(preproc_times)

    proc_times = []
    while not queue_proc_timestamp.empty():
        proc_times.append(queue_proc_timestamp.get())
    proc_times = np.array(proc_times)

    ### avg total latency
    latencies = [receive - transmit for transmit, receive in zip(transmit_times, result_times)]
    print(f"Average latency: {np.mean(latencies):.4f} seconds")

    ### avg transfer latency
    transfer_latencies = [
        receive - transmit for transmit, receive in zip(transmit_times, receive_times)
    ]
    print(f" > Average transfer latency: {np.mean(transfer_latencies):.4f} seconds")

    ### avg preproc latency
    preproc_latencies = [
        preproc - receive for preproc, receive in zip(preproc_times, receive_times)
    ]
    print(f" > Average preproc latency: {np.mean(preproc_latencies):.4f} seconds")

    ### avg proc latency
    proc_latencies = [
        proc - preproc for proc, preproc in zip(proc_times, preproc_times)
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



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Server for processing video frames.")
    parser.add_argument("--server-ip", type=str, default="localhost", help="Server IP address.")
    parser.add_argument("--server-port", type=int, default=65432, help="Server port.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (cpu or cuda).")

    args = parser.parse_args()

    main(args)