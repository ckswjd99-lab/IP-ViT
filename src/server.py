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
    print('Model loaded and ready to process frames.')

    transmit_data(socket_tx, b"start!")

    start_timestamp = time.time()

    # Sizes: H, W
    input_size = (1024, 1024)
    frame_size = (854, 480)

    gop = 30
    num_frames = 0
    anchor_image_padded = None
    anchor_features = None
    compute_rates = []

    while True:
        # Receive frame
        data = receive_data(socket_rx)
        if data is None:
            transmit_data(socket_tx, b"", HEADER_TERMINATE)
            break

        fidx = int.from_bytes(data[:4], 'big')

        frame = bytes_to_ndarray(receive_data(socket_rx))
        print(f"Received {fidx} | frame: {frame.shape} | timestamp: {time.time()}")

        target_ndarray = cv2.resize(frame, frame_size)

        # Preprocess the frame
        refresh = False
        if fidx % gop == 0 or anchor_image_padded is None or anchor_features is None:
            refresh = True
        else:
            # try image refinement
            # affine_matrix = estimate_affine_in_padded_anchor(anchor_image_padded, frame)
            affine_matrix = estimate_affine_in_padded_anchor_fast(anchor_image_padded, frame)
            target_padded_ndarray = apply_affine_and_pad(target_ndarray, affine_matrix)
            
            scaling_factor = np.linalg.norm(affine_matrix[:2, :2])
            dirtiness_map = create_dirtiness_map(anchor_image_padded, target_padded_ndarray)
            
            refresh |= (target_padded_ndarray is None)
            refresh |= (scaling_factor < 0.95)
        
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
        
        anchor_image_padded = target_padded_ndarray
        anchor_features = cached_features_dict

        cv2.imwrite(f"recv/{fidx:05d}.jpg", target_padded_ndarray)
        
        # Statistics
        compute_rates.append(dirtiness_map.mean().item())

        # Convert results to bytes
        transmit_data(socket_tx, fidx.to_bytes(4, 'big'))
        transmit_data(socket_tx, ndarray_to_bytes(boxes))
        transmit_data(socket_tx, ndarray_to_bytes(scores))
        transmit_data(socket_tx, ndarray_to_bytes(labels))

        num_frames += 1

    end_timestamp = time.time()

    print(f"Frame rate: {num_frames / (end_timestamp - start_timestamp)} FPS")
    print(f"Compute rate: {np.mean(compute_rates)}")

    return end_timestamp - start_timestamp

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