#!/usr/bin/env python

import yarp
import numpy as np
import cv2
import sys
import time
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Video file path
    # VIDEO_PATH = "input.mp4"
    VIDEO_PATH = ""
    
    # Initialize YARP network
    yarp.Network.init()
    
    # Create output port to send images
    output_port = yarp.BufferedPortImageRgb()
    output_port.open("/test_yarp/image:o")
    logging.info(f"Output port opened: {output_port.getName()}")
    
    # # Open video file
    # cap = cv2.VideoCapture(VIDEO_PATH)
    # if not cap.isOpened():
    #     logging.error(f"Failed to open video file: {VIDEO_PATH}")
    #     output_port.close()
    #     yarp.Network.fini()
    #     return

    if VIDEO_PATH is not "":
        if not os.path.exists(VIDEO_PATH):
            raise FileNotFoundError(f"Video path not found: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)
        # try to set desired properties for consistency (may be ignored for some file formats)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        # Connect to realsense / probe available cameras
        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if cap.read()[0]:
                cameras.append(i)
                cap.release()
        print(f"Available cameras: {cameras}")
        if len(cameras) == 0:
            raise Exception("No cameras available")
        cap = cv2.VideoCapture(cameras[-1])
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = 640
    height = 480

    logging.info(f"Video: {width}x{height}, {fps} FPS, {frame_count} frames")
    
    # Setup YARP image buffer
    yarp_image = yarp.ImageRgb()
    yarp_image.resize(width, height)
    
    frame_delay = 1.0 / fps if fps > 0 else 0.033  # Default to ~30fps if fps is unknown
    
    logging.info("Starting video stream... Press Ctrl+C to stop")
    frame_idx = 0
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()

            if not ret:
                logging.info("End of video reached, looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back to start
                continue
            
            frame_idx += 1
            
            # OpenCV reads in BGR, convert to RGB for YARP
            frame = cv2.resize(frame, (width, height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Copy frame to YARP image
            yarp_image.setExternal(frame_rgb.data, width, height)
            
            # Send image through YARP port
            output_image = output_port.prepare()
            output_image.copy(yarp_image)
            output_port.write()
            
            if frame_idx % 30 == 0:
                logging.info(f"Sent frame {frame_idx}/{frame_count}")
            
            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_delay - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logging.info("Stopping video stream")
    
    finally:
        cap.release()
        output_port.close()
        yarp.Network.fini()
        logging.info("Test module closed")


if __name__ == "__main__":
    main()
