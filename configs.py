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

    # Camera intrinsics - Webcam (Calibrated 2025-12-22)
    # Calibration error: 0.662 pixels
    fx: float = 271.038926
    fy: float = 258.575834
    ppx: float = 250.792272
    ppy: float = 233.271941
    
    # Camera intrinsics - RealSense D435i (original from Andrea)
    # Uncomment these and comment webcam values above to use RealSense
    # fx: float = 382.691528320312
    # fy: float = 381.886566162109
    # ppx: float = 317.998718261719
    # ppy: float = 244.468139648438

    width: int = 640
    height: int = 480

    necessary_percentage_visible_joints: float = 0.3
    
    # Absolute pose estimation settings
    absolute_mode: bool = True  # If True, output absolute 3D coordinates; if False, output relative (pelvis-centered)
    
    # Camera extrinsics: camera pose in world coordinates
    # Rotation matrix (3x3) from camera frame to world frame
    # Default: Identity (camera frame = world frame)
    extrinsics_rotation: list = None  # Will be converted to 3x3 numpy array, format: [[r11,r12,r13], [r21,r22,r23], [r31,r32,r33]]
    # Translation vector (3x1) from camera frame to world frame (in meters)
    extrinsics_translation: list = None  # Will be converted to 3x1 numpy array, format: [tx, ty, tz]
    
    # Visualization settings for absolute mode
    visualize_projection: bool = True  # If True, reproject 3D skeleton to 2D image for validation


@dataclass
class HDConfig:
    """Configuration for Human Detector"""
    yolo_thresh: float = 0.3
    nms_thresh: float = 0.7
    yolo_engine_path: str = (Path("assets") / 'yolo.engine').as_posix()


# Create default instances
HPE = HPEConfig()
HD = HDConfig()