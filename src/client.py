'''
device.py

1) connect to server
2) load video and send it to the model
3) receive the result

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
from preprocessing import ndarray_to_bytes, bytes_to_ndarray, load_video


def thread_send_video(socket: socket.socket, video_path: str, frame_rate: float=30) -> float:
    """
    Thread function to send video frames to the server.

    Args:
        socket (socket.socket): The socket object to send data over.
        video_path (str): Path to the video file.
        frame_rate (float): Frame rate for sending video frames.
    """
    frames = load_video(video_path)

    start_timestamp = time.time()
    for fidx, frame in enumerate(frames):
        # Transmit frame idx
        fidx_bytes = fidx.to_bytes(4, 'big')
        transmit_data(socket, fidx_bytes)

        # Encode the frame
        transmit_data(socket, ndarray_to_bytes(frame))

        print(f"Transferred {fidx} | frame: {frame.shape} | timestamp: {time.time()}")

    transmit_data(socket, b"", HEADER_TERMINATE)  # Send termination signal
    end_timestamp = time.time()

    return end_timestamp - start_timestamp


def thread_receive_results(socket: socket.socket) -> float:
    """
    Thread function to receive results from the server.

    Args:
        socket (socket.socket): The socket object to receive data from.
    """
    while True:
        data = receive_data(socket)
        if data is None:
            break
        
        fidx = int.from_bytes(data[:4], 'big')
        boxes = bytes_to_ndarray(receive_data(socket))
        scores = bytes_to_ndarray(receive_data(socket))
        labels = bytes_to_ndarray(receive_data(socket))

        print(f"Received {fidx} | boxes: {boxes.shape}, scores: {scores.shape}, labels: {labels.shape} | timestamp: {time.time()}")
        

def main(args):
    server_ip = args.server_ip
    server_port1 = args.server_port
    server_port2 = server_port1 + 1
    video_path = args.video_path
    frame_rate = args.frame_rate

    # Connect to the server
    socket_rx, socket_tx = connect_dual_tcp(server_ip, (server_port1, server_port2), node_type="client")

    timelag = measure_timelag(socket_rx, socket_tx, "client")
    print(f"Timelag: {timelag} seconds")

    # Prepare processes
    thread_recv = Process(target=thread_receive_results, args=(socket_rx,))
    thread_send = Process(target=thread_send_video, args=(socket_tx, video_path, frame_rate))


    # Send metadata
    metadata = {
        "frame_rate": frame_rate,
    }
    metadata_bytes = str(metadata).encode('utf-8')
    transmit_data(socket_tx, metadata_bytes)

    print("Sent metadata:", metadata)

    receive_data(socket_rx)  # Wait for server to be ready

    # Start sending video
    thread_recv.start()
    thread_send.start()

    # Wait for the threads to finish
    thread_send.join()
    thread_recv.join()

    # Close sockets
    socket_rx.close()
    socket_tx.close()





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Client for sending video to server.")
    parser.add_argument("--server_ip", type=str, default="localhost", help="Server IP address.")
    parser.add_argument("--server_port", type=int, default=65432, help="Server port.")
    parser.add_argument("--video_path", type=str, default="input.mp4", help="Path to the video file.")
    parser.add_argument("--frame_rate", type=float, default=30.0, help="Frame rate for sending video.")

    args = parser.parse_args()

    main(args)