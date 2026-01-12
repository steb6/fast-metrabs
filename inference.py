import numpy as np
from tqdm import tqdm
import cv2
import os
from dataclasses import asdict
from utils.visualizer import MPLPosePrinter
from utils.modules import HumanDetector, HumanPoseEstimator
from utils.utils import project_pose_to_image, draw_skeleton_2d
from configs import HPE, HD


if __name__ == "__main__":
    from configs import HPE
    import pycuda.autoinit

    # Video source selection
    # If you want to read from a video file instead of probing cameras, set VIDEO_PATH
    VIDEO_PATH = "input.mp4"  # e.g. "/path/to/video.mp4" — leave empty to use camera

    W = 640
    H = 480

    if VIDEO_PATH:
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

    # Create visualizer with absolute mode enabled
    vis = MPLPosePrinter(absolute_mode=HPE.absolute_mode)

    d = HumanDetector(**asdict(HD))

    h = HumanPoseEstimator(**asdict(HPE))

    for _ in tqdm(range(10000)):
        ret, img = cap.read()
        img = cv2.resize(img, (640, 480))
        
        # Make a copy for visualization with reprojection
        img_vis = img.copy()

        det_res = d.estimate(img)
        bbox = det_res["bbox"]

        hpe_res = None
        if bbox is not None:
            hpe_res = h.estimate(img, bbox, 0.0)

        if hpe_res is not None:
            p = hpe_res["pose"]  # This is now absolute if absolute_mode=True
            p_absolute_camera = hpe_res["pose_absolute_camera"]
            e = hpe_res["edges"]
            b = det_res["bbox"]

            if p is not None:
                # Print distance information
                if HPE.absolute_mode:
                    pelvis_distance = np.linalg.norm(p_absolute_camera[0])
                    print(f"Pelvis distance from camera: {pelvis_distance:.2f}m, Position: {p_absolute_camera[0]}")
                else:
                    print(np.sqrt(np.sum(np.square(np.array([0, 0, 0]) - np.array(p[0])))))
                
                # Visualize 3D pose
                vis.clear()
                if HPE.absolute_mode:
                    vis.print_pose(p_absolute_camera, e)
                else:
                    vis.print_pose(p * 5, e)  # Scale up for relative mode
                vis.sleep(0.001)
                
                # Reproject 3D skeleton to 2D for validation (if in absolute mode)
                if HPE.absolute_mode and HPE.visualize_projection:
                    joints_2d_reprojected = project_pose_to_image(p_absolute_camera, h.K)
                    img_vis = draw_skeleton_2d(img_vis, joints_2d_reprojected, e, color=(0, 255, 255), thickness=2, radius=4)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            img_vis = cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Show image with reprojected skeleton
        cv2.imshow("Pose Estimation with Reprojection", img_vis)
        cv2.waitKey(1)
