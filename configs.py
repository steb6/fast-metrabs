import os
from logging import INFO
from human_pose_estimator import HumanPoseEstimator, HumanDetector
# from utils.concurrency.generic_node_fps import GenericNodeFPS
# from utils.concurrency.py_queue import PyQueue
# from utils.concurrency.utils.signals import Signals
from comfort import BaseConfig
import numpy as np
from pathlib import Path

input_type = "skeleton"  # rgb, skeleton or hybrid
seq_len = 8 if input_type != "skeleton" else 16
base_dir = os.path.join('action_rec', 'hpe', 'weights', 'engines', 'docker')


class HPE(BaseConfig):
    model = HumanPoseEstimator

    class Args:
        image_transformation_engine_path = (Path("assets") / 'image_transformation1.engine').as_posix()
        bbone_engine_path = (Path("assets") / 'bbone1.engine').as_posix()
        heads_engine_path = (Path("assets") / 'heads1.engine').as_posix()
        expand_joints_path = (Path("assets") / '32_to_122.npy').as_posix()
        skeleton_types_path = (Path("assets") / 'skeleton_types.pkl').as_posix()
        skeleton = 'smpl+head_30'

        # D435i (got from andrea)
        fx = 382.691528320312
        fy = 381.886566162109
        ppx = 317.998718261719
        ppy = 244.468139648438

        width = 640
        height = 480

        necessary_percentage_visible_joints = 0.3


class Logging(BaseConfig):
    class Logger:
        class Params:
            level = INFO  # Minimum logging level or list of logging levels
            recurring = False

    debug = True
    # options: rgb depth mask 'fps center hands partial scene reconstruction transform
    keys = {'bbox': None, 'rgb': None}  # Debugging


class HD(BaseConfig):
    model = HumanDetector

    class Args:
        yolo_thresh = 0.3
        nms_thresh = 0.7
        yolo_engine_path = (Path("assets") / 'yolo.engine').as_posix()