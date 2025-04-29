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
    connect_dual_tcp, transmit_data, receive_data
)
from preprocessing import ndarray_to_bytes, bytes_to_ndarray


def thread_process_video(socket_rx: socket.socket, socket_tx: socket.socket, device: str='cuda') -> float:
    """
    Thread function to process video frames received from the client and send results back.

    Args:
        socket_rx (socket.socket): The socket object to receive data from.
        socket_tx (socket.socket): The socket object to send data to.
    """
    import torch
    from models import FasterRCNN_ResNet50_FPN_Contexted, MaskedRCNN_ViT_B_FPN_Contexted

    # model = FasterRCNN_ResNet50_FPN_Contexted(device=device)
    model = MaskedRCNN_ViT_B_FPN_Contexted(device=device)
    model.load_weight("models/model_final_61ccd1.pkl")
    model.eval()

    start_timestamp = time.time()

    num_frames = 0
    feature_cache = None
    while True:
        data = receive_data(socket_rx)
        if data is None:
            transmit_data(socket_tx, b"", HEADER_TERMINATE)
            break

        fidx = int.from_bytes(data[:4], 'big')

        frame = bytes_to_ndarray(receive_data(socket_rx))
        print(f"Received frame {fidx} of shape: {frame.shape}")

        # Process the frame with the model
        (boxes, scores, labels), _ = model.forward_contexted(frame)

        # Convert results to bytes
        transmit_data(socket_tx, fidx.to_bytes(4, 'big'))
        transmit_data(socket_tx, ndarray_to_bytes(boxes))
        transmit_data(socket_tx, ndarray_to_bytes(scores))
        transmit_data(socket_tx, ndarray_to_bytes(labels))

        num_frames += 1

    end_timestamp = time.time()

    print(f"Frame rate: {num_frames / (end_timestamp - start_timestamp)} FPS")

    return end_timestamp - start_timestamp

def main(args):
    server_ip = args.server_ip
    server_port1 = args.server_port
    server_port2 = server_port1 + 1
    device = args.device

    # Connect to the client
    socket_rx, socket_tx = connect_dual_tcp(server_ip, (server_port1, server_port2), type="server")

    # Prepare processes
    thread_proc = Process(target=thread_process_video, args=(socket_rx, socket_tx, device))
    thread_proc.start()

    # Receive metadata
    metadata = receive_data(socket_rx)
    metadata = metadata.decode('utf-8')
    metadata = eval(metadata)  # Convert string back to dictionary
    frame_rate = metadata.get("frame_rate", 30)

    print(f"Received metadata: {metadata}")

    # Wait for the processing thread to finish
    thread_proc.join()

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