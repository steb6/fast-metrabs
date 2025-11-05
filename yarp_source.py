#!/usr/bin/env python

import yarp
import numpy as np
import cv2
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Video file path
    VIDEO_PATH = "input.mp4"
    
    # Initialize YARP network
    yarp.Network.init()
    
    # Create output port to send images
    output_port = yarp.BufferedPortImageRgb()
    output_port.open("/test_yarp/image:o")
    logging.info(f"Output port opened: {output_port.getName()}")
    
    # Try to connect to the HPE module input port
    logging.info("Attempting to connect to /hpe/image:i ...")
    if not yarp.Network.connect("/test_yarp/image:o", "/hpe/image:i"):
        logging.warning("Could not auto-connect to /hpe/image:i. You may need to connect manually.")
    else:
        logging.info("Connected to /hpe/image:i")
    
    # Open video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logging.error(f"Failed to open video file: {VIDEO_PATH}")
        output_port.close()
        yarp.Network.fini()
        return
    
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
