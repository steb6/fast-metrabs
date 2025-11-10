#!/usr/bin/env python

import yarp
import numpy as np
import logging
import sys
import cv2
import pycuda.autoinit
from dataclasses import asdict
from utils.modules import HumanDetector
from inference import HumanPoseEstimator
from configs import HPE, HD
from utils.visualizer import MPLPosePrinter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
hpe_namespace = "hpe"


class HumanPoseEstimationModule(yarp.RFModule):
    def configure(self, rf):
        # Load configuration for image dimensions
        self.image_height = 480
        self.image_width = 640
        
        # Create input port for receiving images
        self.input_port = yarp.BufferedPortImageRgb()
        self.input_port.open(f"/{hpe_namespace}/image:i")
        logging.info(f"Input port opened: {self.input_port.getName()}")

        # Create output port for sending annotated images
        self.output_port = yarp.BufferedPortImageRgb()
        self.output_port.open(f"/{hpe_namespace}/image:o")
        logging.info(f"Output port opened: {self.output_port.getName()}")

        # Create output port for sending pose data (as Bottle)
        self.pose_output_port = yarp.Port()
        self.pose_output_port.open(f"/{hpe_namespace}/pose:o")
        logging.info(f"Pose output port opened: {self.pose_output_port.getName()}")

        # Setup input buffer for reading images
        self.input_buffer_array = np.ones((self.image_height, self.image_width, 3), dtype=np.uint8)
        self.input_buffer_image = yarp.ImageRgb()
        self.input_buffer_image.resize(self.image_width, self.image_height)
        self.input_buffer_image.setExternal(self.input_buffer_array.data, self.image_width, self.image_height)
        logging.info(f"Input buffer created with shape: {self.input_buffer_array.shape}")

        # Setup output buffer for writing processed images
        self.output_buffer_array = np.ones((self.image_height, self.image_width, 3), dtype=np.uint8)
        self.output_buffer_image = yarp.ImageRgb()
        self.output_buffer_image.resize(self.image_width, self.image_height)
        self.output_buffer_image.setExternal(self.output_buffer_array.data, self.image_width, self.image_height)
        logging.info(f"Output buffer created with shape: {self.output_buffer_array.shape}")

        # Load the human detector and pose estimator models
        logging.info("Loading human detector model...")
        self.detector = HumanDetector(**asdict(HD))
        logging.info("Loading human pose estimator model...")
        self.estimator = HumanPoseEstimator(**asdict(HPE))
        logging.info("Models loaded successfully")

        # Create 3D pose visualizer
        self.vis = MPLPosePrinter()
        logging.info("3D visualizer initialized")

        return True

    def updateModule(self):
        # Read image from input port (non-blocking)
        input_image = self.input_port.read(False)
        
        if input_image is None:
            # print("No image received")
            return True  # No image received, continue

        # Copy the received image to the input buffer
        self.input_buffer_image.copy(input_image)
        frame = np.copy(self.input_buffer_array)
        
        # Ensure frame is in BGR format for OpenCV processing
        frame = cv2.resize(frame, (self.image_width, self.image_height))

        # Display input frame for debugging
        cv2.imshow("Input Frame", frame)
        cv2.waitKey(1)
        
        # Run human detection
        det_res = self.detector.estimate(frame)
        bbox = det_res["bbox"]
        
        hpe_res = None
        if bbox is not None:
            # Run pose estimation if a person was detected
            hpe_res = self.estimator.estimate(frame, bbox, 0.0)
        
        # Prepare output image with bbox drawn
        output_frame = frame.copy()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Send annotated image to output port
        self.output_buffer_array[:, :] = output_frame
        output_image = self.output_port.prepare()
        output_image.copy(self.output_buffer_image)
        self.output_port.write()
        
        # Send pose data to pose output port
        pose_msg = yarp.Bottle()
        if hpe_res is not None:
            pose = hpe_res["pose"]
            edges = hpe_res["edges"]
            human_distance = hpe_res["human_distance"]
            human_position = hpe_res["human_position"]
            
            # Visualize 3D skeleton
            if pose is not None:
                pose_vis = pose - pose[0]  # Center at root joint
                self.vis.clear()
                self.vis.print_pose(pose_vis * 5, edges)
                self.vis.sleep(0.001)
            
            # Structure: [num_joints, x1, y1, z1, x2, y2, z2, ...]
            # pose_msg.addInt32(len(pose))
            for joint in pose:
                pose_msg.addFloat64(float(joint[0]))
                pose_msg.addFloat64(float(joint[1]))
                pose_msg.addFloat64(float(joint[2]))
            
            # # Add human distance
            # pose_msg.addFloat64(float(human_distance))
            
            # # Add human position
            # pose_msg.addFloat64(float(human_position[0]))
            # pose_msg.addFloat64(float(human_position[1]))
            # pose_msg.addFloat64(float(human_position[2]))
            
            logging.debug(f"Pose detected with {len(pose)} joints, distance: {human_distance:.2f}")
        else:
            # No pose detected, send empty bottle
            pose_msg.addInt32(0)
        
        self.pose_output_port.write(pose_msg)
        
        return True

    def getPeriod(self):
        # Return the period of the module in seconds
        return 0.03  # 30ms

    def interruptModule(self):
        # Handle module interruption
        logging.info("HPE module interrupted")
        self.input_port.interrupt()
        self.output_port.interrupt()
        self.pose_output_port.interrupt()
        return True

    def close(self):
        # Close ports and cleanup resources
        logging.info("Closing HPE module")
        self.input_port.close()
        self.output_port.close()
        self.pose_output_port.close()
        return True


if __name__ == "__main__":
    # Initialize YARP network
    yarp.Network.init()

    # Create and configure the module
    module = HumanPoseEstimationModule()
    rf = yarp.ResourceFinder()
    rf.configure(sys.argv)
    
    logging.info("Starting human pose estimation YARP module")
    
    # Run the module
    try:
        if not module.configure(rf):
            logging.error("Failed to configure the module")
        else:
            module.runModule()
    except KeyboardInterrupt:
        logging.info("Stopping HPE module")
    except Exception as e:
        logging.error(f"Error in HPE module: {e}", exc_info=True)
    finally:
        module.close()
        logging.info("HPE module closed")

    # Clean up the YARP network
    yarp.Network.fini()
