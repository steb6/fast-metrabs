import pickle
from utils.utils import homography, is_within_fov, reconstruct_absolute, postprocess_yolo_output
import einops
import numpy as np
from utils.trt_runner import Runner
import cv2
import copy


class HumanDetector:
    def __init__(self, yolo_thresh=None, nms_thresh=None, yolo_engine_path=None):

        self.yolo_thresh = yolo_thresh
        self.nms_thresh = nms_thresh
        self.yolo = Runner(yolo_engine_path)  # model_config.yolo_engine_path

    def estimate(self, rgb):

        # Preprocess for yolo
        square_img = cv2.resize(rgb, (256, 256), fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
        yolo_in = copy.deepcopy(square_img)
        yolo_in = cv2.cvtColor(yolo_in, cv2.COLOR_BGR2RGB)
        yolo_in = np.transpose(yolo_in, (2, 0, 1)).astype(np.float32)
        yolo_in = np.expand_dims(yolo_in, axis=0)
        yolo_in = yolo_in / 255.0

        # Yolo
        outputs = self.yolo(yolo_in)
        boxes, confidences = outputs[0].reshape(1, 4032, 1, 4), outputs[1].reshape(1, 4032, 80)
        bboxes_batch = postprocess_yolo_output(boxes, confidences, self.yolo_thresh, self.nms_thresh)

        # Get only the bounding box with the human with highest probability
        box = bboxes_batch[0]  # Remove batch dimension
        humans = []
        for e in box:  # For each object in the image
            if e[5] == 0:  # If it is a human
                humans.append(e)
        if len(humans) > 0:
            # humans.sort(key=lambda x: x[4], reverse=True)  # Sort with decreasing probability
            humans.sort(key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)  # Sort with decreasing area  # TODO TEST
            human = humans[0]
        else:
            return {"bbox": None}

        # Preprocess for BackBone
        x1 = int(human[0] * rgb.shape[1]) if int(human[0] * rgb.shape[1]) > 0 else 0
        y1 = int(human[1] * rgb.shape[0]) if int(human[1] * rgb.shape[0]) > 0 else 0
        x2 = int(human[2] * rgb.shape[1]) if int(human[2] * rgb.shape[1]) > 0 else 0
        y2 = int(human[3] * rgb.shape[0]) if int(human[3] * rgb.shape[0]) > 0 else 0

        return {"rgb": rgb, "bbox": (x1, y1, x2, y2)}


class HumanPoseEstimator:
    def __init__(self, skeleton=None, image_transformation_engine_path=None, bbone_engine_path=None,
                 heads_engine_path=None, skeleton_types_path=None, expand_joints_path=None, fx=None, fy=None, ppx=None,
                 ppy=None, necessary_percentage_visible_joints=None, width=None, height=None, absolute_mode=True,
                 extrinsics_rotation=None, extrinsics_translation=None, visualize_projection=True):

        # Intrinsics and K matrix of Camera
        self.K = np.zeros((3, 3), np.float32)
        self.K[0][0] = fx
        self.K[0][2] = ppx
        self.K[1][1] = fy
        self.K[1][2] = ppy
        self.K[2][2] = 1

        # Load conversions
        self.skeleton = skeleton
        self.expand_joints = np.load(expand_joints_path)
        with open(skeleton_types_path, "rb") as input_file:
            self.skeleton_types = pickle.load(input_file)

        # Load modules
        self.image_transformation = Runner(image_transformation_engine_path)
        self.bbone = Runner(bbone_engine_path)
        self.heads = Runner(heads_engine_path)

        self.necessary_percentage_visible_joints = necessary_percentage_visible_joints
        
        # Absolute pose estimation settings
        self.absolute_mode = absolute_mode
        self.visualize_projection = visualize_projection
        
        # Camera extrinsics: transformation from camera frame to world frame
        # Default to identity (camera frame = world frame)
        if extrinsics_rotation is not None:
            self.R_cam_to_world = np.array(extrinsics_rotation, dtype=np.float32)
        else:
            self.R_cam_to_world = np.eye(3, dtype=np.float32)
        
        if extrinsics_translation is not None:
            self.t_cam_to_world = np.array(extrinsics_translation, dtype=np.float32).reshape(3, 1)
        else:
            self.t_cam_to_world = np.zeros((3, 1), dtype=np.float32)
    
    def update_extrinsics(self, rotation=None, translation=None):
        """Update camera extrinsics (useful for reading from YARP port later)"""
        if rotation is not None:
            self.R_cam_to_world = np.array(rotation, dtype=np.float32)
        if translation is not None:
            self.t_cam_to_world = np.array(translation, dtype=np.float32).reshape(3, 1)

    def estimate(self, rgb, bbox, yarp_read_time):

        x1, y1, x2, y2 = bbox

        new_K, homo_inv = homography(x1, x2, y1, y2, self.K, 256)

        # Apply homography
        H = self.K @ np.linalg.inv(new_K @ homo_inv)
        bbone_in = self.image_transformation(rgb.astype(int), H.astype(np.float32))

        bbone_in = bbone_in[0].reshape(1, 256, 256, 3)  # [..., ::-1]
        bbone_in_ = (bbone_in / 255.0).astype(np.float32)

        # BackBone
        outputs = self.bbone(bbone_in_)

        # Heads
        logits = self.heads(outputs[0])

        # Get logits 3d
        logits = logits[0].reshape(1, 8, 8, 288)
        _, logits2d, logits3d = np.split(logits, [0, 32], axis=3)
        current_format = 'b h w (d j)'
        logits3d = einops.rearrange(logits3d, f'{current_format} -> b h w d j', j=32)  # 5, 8, 8, 9, 32

        # 3D Softmax
        heatmap_axes = (2, 1, 3)
        max_along_axis = logits3d.max(axis=heatmap_axes, keepdims=True)
        exponential = np.exp(logits3d - max_along_axis)
        denominator = np.sum(exponential, axis=heatmap_axes, keepdims=True)
        res = exponential / denominator

        # 3D Decode Heatmap
        result = []
        for ax in heatmap_axes:
            other_heatmap_axes = tuple(other_ax for other_ax in heatmap_axes if other_ax != ax)
            summed_over_other_heatmap_axes = np.sum(res, axis=other_heatmap_axes, keepdims=True)
            coords = np.linspace(0.0, 1.0, res.shape[ax])
            decoded = np.tensordot(summed_over_other_heatmap_axes, coords, axes=[[ax], [0]])
            result.append(np.squeeze(np.expand_dims(decoded, ax), axis=heatmap_axes))
        pred3d = np.stack(result, axis=-1)

        # 2D Softmax
        heatmap_axes = (2, 1)
        max_along_axis = logits2d.max(axis=heatmap_axes, keepdims=True)
        exponential = np.exp(logits2d - max_along_axis)
        denominator = np.sum(exponential, axis=heatmap_axes, keepdims=True)
        res = exponential / denominator

        # Decode heatmap
        result = []
        for ax in heatmap_axes:
            other_heatmap_axes = tuple(other_ax for other_ax in heatmap_axes if other_ax != ax)
            summed_over_other_heatmap_axes = np.sum(res, axis=other_heatmap_axes, keepdims=True)
            coords = np.linspace(0.0, 1.0, res.shape[ax])
            decoded = np.tensordot(summed_over_other_heatmap_axes, coords, axes=[[ax], [0]])
            result.append(np.squeeze(np.expand_dims(decoded, ax), axis=heatmap_axes))
        pred2d = np.stack(result, axis=-1) * 255

        # Get absolute position (if desired)
        is_predicted_to_be_in_fov = is_within_fov(pred2d)

        # If less than 1/3 of the joints is visible, then the resulting pose will be weird
        # n < 30
        # if is_predicted_to_be_in_fov.sum() < is_predicted_to_be_in_fov.size*self.necessary_percentage_visible_joints:
        #     return None

        # Move the skeleton into estimated absolute position if necessary
        pred3d = reconstruct_absolute(pred2d, pred3d, new_K[None, ...], is_predicted_to_be_in_fov, weak_perspective=False)

        # Save pred2d BEFORE any transformation (in 256x256 bbone space) WITHOUT removing batch dim yet
        pred2d_bbone_raw = pred2d.copy()  # Keep batch dimension for now
        bbone_image = bbone_in[0].copy()  # For visualization
        
        # Go back in original space (without augmentation and homography)
        pred3d = pred3d @ homo_inv
        
        # Get correct skeleton
        pred3d = (pred3d.swapaxes(1, 2) @ self.expand_joints).swapaxes(1, 2)
        
        if self.skeleton is not None:
            pred3d = pred3d[:, self.skeleton_types[self.skeleton]['indices']]
            edges = self.skeleton_types[self.skeleton]['edges']
        else:
            edges = None

        pred3d = pred3d[0]  # Remove batch dimension
        pred2d = pred2d[0]  # Remove batch dimension for later use
        pred2d_bbone = pred2d_bbone_raw[0]  # Now extract for visualization

        # Store absolute camera-frame coordinates before any transformation
        pred3d_absolute_camera = pred3d.copy()

        human_distance = np.sqrt(
            np.sum(np.square(np.array([0, 0, 0]) - np.array(pred3d[0])))) * 2.5
        human_position = pred3d[0, :]
        human_pixels2 = self.K@human_position

        # Transform to world frame if extrinsics are provided
        pred3d_absolute_world = (self.R_cam_to_world @ pred3d_absolute_camera.T).T + self.t_cam_to_world.T

        # For backward compatibility: compute relative pose (pelvis-centered)
        pred3d_relative = pred3d - pred3d[0, :]
        
        # Choose output based on mode
        if self.absolute_mode:
            pred3d_output = pred3d_absolute_camera  # Use camera-frame absolute coordinates
        else:
            pred3d_output = pred3d_relative  # Use relative (pelvis-centered) coordinates

        # Compute human occupancy
        pred3d_0 = pred3d_output
        x_min, x_max, z_min, z_max = min(pred3d_0[:, 0]), max(pred3d_0[:, 0]), min(pred3d_0[:, 2]), max(pred3d_0[:, 2])
        human_occupancy = (x_min, x_max, z_min, z_max)
        index_min = np.argmin(pred3d_0[:, 0])
        index_max = np.argmax(pred3d_0[:, 0])
        x_min_pixel, y_min_pixel, x_max_pixel, y_max_pixel = pred2d[0,0],pred2d[0,1],pred2d[index_max,0],pred2d[index_max,1]
        human_pixels = (x_min_pixel,y_min_pixel,x_max_pixel,y_max_pixel)

        return {"pose": pred3d_output,
                "pose_absolute_camera": pred3d_absolute_camera,
                "pose_absolute_world": pred3d_absolute_world,
                "pose_relative": pred3d_relative,
                "pred2d_bbone": pred2d_bbone,  # 2D predictions in 256x256 bbone space (32 joints, BEFORE expand)
                "bbone_image": bbone_image,  # The 256x256 transformed image
                "edges": edges,
                "human_distance": human_distance,
                "human_position": human_position,
                "human_occupancy": human_occupancy,
                "human_pixels": human_pixels,
                "yarp_read_time": yarp_read_time}
