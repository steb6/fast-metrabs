import numpy as np
from tqdm import tqdm
import cv2
import os
from dataclasses import asdict
from utils.visualizer import MPLPosePrinter
from utils.modules import HumanDetector, HumanPoseEstimator
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

    vis = MPLPosePrinter()

    d = HumanDetector(**asdict(HD))

    h = HumanPoseEstimator(**asdict(HPE))

    for _ in tqdm(range(10000)):
        ret, img = cap.read()
        img = cv2.resize(img, (640, 480))

        det_res = d.estimate(img)
        bbox = det_res["bbox"]

        if bbox is not None:
            hpe_res = h.estimate(img, bbox, 0.0)

        if hpe_res is not None:
            p = hpe_res["pose"]
            e = hpe_res["edges"]
            b = det_res["bbox"]

            if p is not None:
                print(np.sqrt(np.sum(np.square(np.array([0, 0, 0]) - np.array(p[0])))))
                p = p - p[0]
                vis.clear()
                vis.print_pose(p*5, e)
                vis.sleep(0.001)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        cv2.imshow("after", img)
        cv2.waitKey(1)
