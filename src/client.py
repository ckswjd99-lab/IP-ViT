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
from multiprocessing import Process, Queue, Semaphore

from typing import List, Tuple

from networking import (
    HEADER_NORMAL, HEADER_TERMINATE,
    connect_dual_tcp, transmit_data, receive_data,
    measure_timelag,
)
from preprocessing import ndarray_to_bytes, bytes_to_ndarray, load_video


def thread_send_video(
    socket: socket.socket, 
    video_path: str, 
    frame_rate: float, 
    sem_offload, 
    queue_timestamp: Queue, 
    compress: bool
) -> float:
    """
    Thread function to send video frames to the server.

    Args:
        socket (socket.socket): The socket object to send data over.
        video_path (str): Path to the video file.
        frame_rate (float): Frame rate for sending video frames.
    """
    frames = load_video(video_path)

    for fidx, frame in enumerate(frames):
        if sem_offload is not None:
            sem_offload.acquire()  # Wait for server to process the frame

        # Transmit frame idx
        fidx_bytes = fidx.to_bytes(4, 'big')
        transmit_data(socket, fidx_bytes)

        # Optionally encode the frame
        if compress:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            _, frame = cv2.imencode('.jpg', frame, encode_param)
            frame = np.array(frame)

        # Encode the frame
        transmit_data(socket, ndarray_to_bytes(frame))

        timestamp = time.time()
        print(f"Transferred {fidx} | frame: {frame.shape} | timestamp: {timestamp}")
        queue_timestamp.put(timestamp)

    transmit_data(socket, b"", HEADER_TERMINATE)  # Send termination signal


def thread_receive_results(
    socket: socket.socket, 
    sem_offload, 
    queue_timestamp: Queue
) -> float:
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

        timestamp = time.time()
        print(f"Received {fidx} | boxes: {boxes.shape}, scores: {scores.shape}, labels: {labels.shape} | timestamp: {timestamp}")
        queue_timestamp.put(timestamp)

        if sem_offload is not None:
            sem_offload.release()  # Release semaphore to indicate processing is done
        

def main(args):
    server_ip = args.server_ip
    server_port1 = args.server_port
    server_port2 = server_port1 + 1
    video_path = args.video_path
    gop = args.frame_rate
    sequential = args.sequential
    compress = args.compress

    # Connect to the server
    socket_rx, socket_tx = connect_dual_tcp(
        server_ip, (server_port1, server_port2), node_type="client"
    )

    timelag = measure_timelag(socket_rx, socket_tx, "client")
    print(f"Timelag: {timelag} seconds")

    # Create queues for recording
    queue_transmit_timestamp = Queue(maxsize=100)
    queue_receive_timestamp = Queue(maxsize=100)

    # Create a semaphore for offloading
    sem_offload = Semaphore(1) if sequential else None

    # Prepare processes
    thread_recv = Process(
        target=thread_receive_results, 
        args=(
            socket_rx, 
            sem_offload, 
            queue_receive_timestamp
        )
    )
    thread_send = Process(
        target=thread_send_video, 
        args=(
            socket_tx, 
            video_path, 
            gop, 
            sem_offload, 
            queue_transmit_timestamp,
            compress
        )
    )

    # Send metadata
    metadata = {
        "gop": gop,
        "compress": compress,
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

    # Send statistics
    transmit_times = []
    receive_times = []
    while not queue_transmit_timestamp.empty():
        transmit_times.append(queue_transmit_timestamp.get())
    while not queue_receive_timestamp.empty():
        receive_times.append(queue_receive_timestamp.get())
    transmit_times = np.array(transmit_times)
    receive_times = np.array(receive_times)

    transmit_data(socket_tx, ndarray_to_bytes(transmit_times))
    transmit_data(socket_tx, ndarray_to_bytes(receive_times))

    # Close sockets
    socket_rx.close()
    socket_tx.close()

    # Print the stats

    latencies = [receive - transmit for transmit, receive in zip(transmit_times, receive_times)]
    print(f"Average latency: {np.mean(latencies):.4f} seconds")
    print(f"Max latency: {np.max(latencies):.4f} seconds")
    print(f"Min latency: {np.min(latencies):.4f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Client for sending video to server.")
    parser.add_argument("--server-ip", type=str, default="localhost", help="Server IP address.")
    parser.add_argument("--server-port", type=int, default=65432, help="Server port.")
    parser.add_argument("--video-path", type=str, default="input.mp4", help="Path to the video file.")
    parser.add_argument("--frame-rate", type=float, default=30, help="Frame rate for sending video.")
    parser.add_argument("--sequential", type=bool, default=True, help="Sender waits until the result of the previous frame is received.")
    parser.add_argument("--compress", action="store_true", help="Compress video frames before sending.")

    args = parser.parse_args()

    main(args)