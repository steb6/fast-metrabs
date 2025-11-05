from dataclasses import dataclass
from pathlib import Path


@dataclass
class HPEConfig:
    """Configuration for Human Pose Estimator"""
    image_transformation_engine_path: str = (Path("assets") / 'image_transformation1.engine').as_posix()
    bbone_engine_path: str = (Path("assets") / 'bbone1.engine').as_posix()
    heads_engine_path: str = (Path("assets") / 'heads1.engine').as_posix()
    expand_joints_path: str = (Path("assets") / '32_to_122.npy').as_posix()
    skeleton_types_path: str = (Path("assets") / 'skeleton_types.pkl').as_posix()
    skeleton: str = 'smpl+head_30'

    # D435i (got from andrea)
    fx: float = 382.691528320312
    fy: float = 381.886566162109
    ppx: float = 317.998718261719
    ppy: float = 244.468139648438

    width: int = 640
    height: int = 480

    necessary_percentage_visible_joints: float = 0.3


@dataclass
class HDConfig:
    """Configuration for Human Detector"""
    yolo_thresh: float = 0.3
    nms_thresh: float = 0.7
    yolo_engine_path: str = (Path("assets") / 'yolo.engine').as_posix()


# Create default instances
HPE = HPEConfig()
HD = HDConfig()