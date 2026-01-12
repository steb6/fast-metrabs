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
from utils.utils import project_pose_to_image, draw_skeleton_2d

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

        # Create 3D pose visualizer with absolute mode
        self.vis = MPLPosePrinter(absolute_mode=HPE.absolute_mode)
        logging.info("3D visualizer initialized")

        return True

    def updateModule(self):
        # Read image from input port (non-blocking)
        input_image = self.input_port.read(False)
        
        if input_image is None:
            return True  # No image received, continue

        # Copy the received image to the input buffer
        self.input_buffer_image.copy(input_image)
        frame = np.copy(self.input_buffer_array)
        
        # Ensure frame is in BGR format for OpenCV processing
        frame = cv2.resize(frame, (self.image_width, self.image_height))
        
        # Make a copy for visualization with reprojection
        frame_vis = frame.copy()
        
        # Run human detection
        det_res = self.detector.estimate(frame)
        bbox = det_res["bbox"]
        
        hpe_res = None
        if bbox is not None:
            # Run pose estimation if a person was detected
            hpe_res = self.estimator.estimate(frame, bbox, 0.0)
        
        # Prepare output image with bbox drawn
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            frame_vis = cv2.rectangle(frame_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Send pose data to pose output port
        pose_msg = yarp.Bottle()
        if hpe_res is not None:
            pose = hpe_res["pose"]  # Absolute or relative based on config
            pose_absolute_camera = hpe_res["pose_absolute_camera"]
            pose_relative = hpe_res["pose_relative"]
            edges = hpe_res["edges"]
            human_distance = hpe_res["human_distance"]
            human_position = hpe_res["human_position"]
            
            # Visualize 3D skeleton
            if pose is not None:
                self.vis.clear()
                if HPE.absolute_mode:
                    self.vis.print_pose(pose_absolute_camera, edges)
                    
                    # Show 2D predictions on bbone for HANDS ONLY
                    if HPE.visualize_projection:
                        pred2d_bbone = hpe_res["pred2d_bbone"]  # 32 joints in 256x256 space (BEFORE expand_joints)
                        bbone_image = hpe_res["bbone_image"]  # 256x256 transformed image
                        
                        # Visualize bbone with 2D hand predictions
                        bbone_vis = bbone_image.astype(np.uint8).copy()
                        
                        # Draw ALL joints with numbers to identify the correct hand indices
                        for idx, joint_2d in enumerate(pred2d_bbone):
                            pt = (int(joint_2d[0]), int(joint_2d[1]))
                            # Draw circle
                            cv2.circle(bbone_vis, pt, 4, (0, 255, 255), -1)  # Cyan in RGB
                            # Draw joint number
                            cv2.putText(bbone_vis, str(idx), (pt[0]+6, pt[1]-6), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        # Highlight hands - correct indices: 20=left wrist, 22=right wrist
                        hand_joints_32 = [20, 22]  # Left and right wrists in 32-joint skeleton
                        for hand_idx in hand_joints_32:
                            if hand_idx < len(pred2d_bbone):
                                hand_2d = pred2d_bbone[hand_idx]
                                pt = (int(hand_2d[0]), int(hand_2d[1]))
                                # Draw BIG circles for suspected hands
                                cv2.circle(bbone_vis, pt, 10, (255, 0, 0), 3)  # Blue border in RGB
                                cv2.putText(bbone_vis, f"H{hand_idx}", (pt[0]+12, pt[1]+12), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        # Show the bbone with hand annotations
                        cv2.imshow("2D Hand Predictions (BBox Space)", cv2.cvtColor(bbone_vis, cv2.COLOR_RGB2BGR))
                        
                        # Also draw 3D reprojection on main frame for comparison
                        joints_2d_reprojected = project_pose_to_image(pose_absolute_camera, self.estimator.K)
                        frame_vis = draw_skeleton_2d(frame_vis, joints_2d_reprojected, edges, 
                                                    color=(255, 255, 0), thickness=3, radius=6)
                else:
                    pose_vis = pose_relative
                    self.vis.print_pose(pose_vis * 5, edges)
                    
                self.vis.sleep(0.001)
            
            # Send pose data via YARP (using absolute camera frame coordinates)
            # Structure: [x1, y1, z1, x2, y2, z2, ...]
            for joint in pose_absolute_camera:
                pose_msg.addFloat64(float(joint[0]))
                pose_msg.addFloat64(float(joint[1]))
                pose_msg.addFloat64(float(joint[2]))
            
            logging.debug(f"Pose detected with {len(pose)} joints, pelvis position: {pose_absolute_camera[0]}")
        else:
            # No pose detected, send empty bottle
            pose_msg.addInt32(0)
        
        self.pose_output_port.write(pose_msg)

        # Display frame with reprojection for debugging
        cv2.imshow("Human detection", cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        
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
