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


def thread_receive_video(socket_rx: socket.socket, socket_tx: socket.socket, frame_queue: Queue) -> float:
    
    """
    Thread function to receive video frames from the client and put them into a queue.

    Args:
        socket_rx (socket.socket): The socket object to receive data from.
        socket_tx (socket.socket): The socket object to send data to.
        frame_queue (Queue): Queue to store received frames.
    """
    import torch

    input_size = (1024, 1024)
    frame_size = (854, 480)

    gop = 30
    anchor_image_padded = None

    while True:
        # Receive frame
        data = receive_data(socket_rx)
        if data is None:
            break

        fidx = int.from_bytes(data[:4], 'big')
        frame = bytes_to_ndarray(receive_data(socket_rx))
        # print(f"Received {fidx} | frame: {frame.shape} | timestamp: {time.time()}")

        target_ndarray = cv2.resize(frame, frame_size)

        # Preprocess the frame
        refresh = False
        if fidx % gop == 0 or anchor_image_padded is None:
            refresh = True
            dirtiness_map = torch.ones(1, 64, 64, 1)
            target_padded_ndarray = get_padded_image(target_ndarray, input_size)
        else:
            # try image refinement
            # affine_matrix = estimate_affine_in_padded_anchor(anchor_image_padded, frame)
            affine_matrix = estimate_affine_in_padded_anchor_fast(anchor_image_padded, frame)
            target_padded_ndarray = apply_affine_and_pad(target_ndarray, affine_matrix)
            
            scaling_factor = np.linalg.norm(affine_matrix[:2, :2])
            dirtiness_map = create_dirtiness_map(anchor_image_padded, target_padded_ndarray)
            
            refresh |= (target_padded_ndarray is None)
            refresh |= (scaling_factor < 0.95)
        
        anchor_image_padded = target_padded_ndarray

        # Put the frame into the queue
        frame_queue.put((fidx, refresh, target_ndarray, dirtiness_map.cpu().numpy()))
        
    frame_queue.put((-1, None, None, None))  # Send termination signal
    

def thread_process_video(
    socket_rx: socket.socket, 
    socket_tx: socket.socket, 
    frame_queue: Queue, 
    result_queue: Queue,
    device: str='cuda'
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
            (boxes, labels, scores), cached_features_dict = model.forward_contexted(target_padded_ndarray)
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

        print(f"Processed {fidx} | boxes: {boxes.shape}, scores: {scores.shape}, labels: {labels.shape} | timestamp: {time.time()}")

        # Convert results to bytes
        # transmit_data(socket_tx, fidx.to_bytes(4, 'big'))
        # transmit_data(socket_tx, ndarray_to_bytes(boxes))
        # transmit_data(socket_tx, ndarray_to_bytes(scores))
        # transmit_data(socket_tx, ndarray_to_bytes(labels))
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
    socket_rx, socket_tx = connect_dual_tcp(server_ip, (server_port1, server_port2), node_type="server")

    timelag = measure_timelag(socket_rx, socket_tx, "server")
    print(f"Timelag: {timelag} seconds")

    # Prepare processes
    frame_queue = Queue(maxsize=100)
    result_queue = Queue(maxsize=100)

    thread_recv = Process(target=thread_receive_video, args=(socket_rx, socket_tx, frame_queue))
    thread_proc = Process(target=thread_process_video, args=(socket_rx, socket_tx, frame_queue, result_queue, device))
    thread_send = Process(target=thread_send_result, args=(socket_rx, socket_tx, result_queue))
    

    # Receive metadata
    metadata = receive_data(socket_rx)
    metadata = metadata.decode('utf-8')
    metadata = eval(metadata)  # Convert string back to dictionary
    frame_rate = metadata.get("frame_rate", 30)

    print(f"Received metadata: {metadata}")
    

    thread_recv.start()
    thread_proc.start()
    thread_send.start()
    
    # Wait for the processing thread to finish
    thread_recv.join()
    thread_proc.join()
    thread_send.join()

    # Close sockets
    socket_rx.close()
    socket_tx.close()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Server for processing video frames.")
    parser.add_argument("--server_ip", type=str, default="localhost", help="Server IP address.")
    parser.add_argument("--server_port", type=int, default=65432, help="Server port.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (cpu or cuda).")

    args = parser.parse_args()

    main(args)