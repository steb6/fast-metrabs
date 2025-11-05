import os
from logging import INFO
from utils.modules import HumanPoseEstimator, HumanDetector
from comfort import BaseConfig
from pathlib import Path

input_type = "skeleton"  # rgb, skeleton or hybrid
seq_len = 8 if input_type != "skeleton" else 16


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


class HD(BaseConfig):
    model = HumanDetector

    class Args:
        yolo_thresh = 0.3
        nms_thresh = 0.7
        yolo_engine_path = (Path("assets") / 'yolo.engine').as_posix()