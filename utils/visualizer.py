import matplotlib.pyplot as plt
import numpy as np


class MPLPosePrinter:
    def __init__(self, absolute_mode=False, **args):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=180, azim=0, vertical_axis='y')

        self.absolute_mode = absolute_mode
        
        if absolute_mode:
            # For absolute mode, use larger, dynamic limits
            self.ax.set_xlim3d([-2.0, 2.0])
            self.ax.set_ylim3d([0.0, 4.0])  # Y is typically up, expect positive values
            self.ax.set_zlim3d([0.0, 4.0])  # Z is depth, expect positive values
        else:
            # For relative mode, use smaller symmetric limits
            self.ax.set_xlim3d([-2.0, 2.0])
            self.ax.set_ylim3d([-2.0, 2.0])
            self.ax.set_zlim3d([-2.0, 2.0])

        plt.grid(False)
        plt.axis('off')
        plt.title('3D Human Pose - Absolute Mode' if absolute_mode else '3D Human Pose - Relative Mode')

    def print_pose(self, pose, edges, color='b', auto_scale=None):
        """
        Print 3D pose.
        
        Args:
            pose: 3D pose array, shape (N_joints, 3) or (1, N_joints, 3)
            edges: List of edges for skeleton
            color: Color for drawing
            auto_scale: If True, automatically adjust axis limits based on pose. 
                       If None, uses absolute_mode setting.
        """
        if auto_scale is None:
            auto_scale = self.absolute_mode
            
        self.ax.plot([0, 0], [-1, 1], [0, 0])
        if len(pose.shape) == 2:
            pose = pose[None]
        
        if edges is not None:
            for p in pose:
                for edge in edges:
                    a = p[edge[0]]
                    b = p[edge[1]]
                    self.ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color)
        
        # Auto-scale axes in absolute mode
        if auto_scale and len(pose) > 0:
            all_points = pose.reshape(-1, 3)
            padding = 0.5  # meters
            
            x_min, x_max = all_points[:, 0].min() - padding, all_points[:, 0].max() + padding
            y_min, y_max = all_points[:, 1].min() - padding, all_points[:, 1].max() + padding
            z_min, z_max = all_points[:, 2].min() - padding, all_points[:, 2].max() + padding
            
            # Ensure minimum range for visibility
            x_range = max(x_max - x_min, 1.0)
            y_range = max(y_max - y_min, 1.0)
            z_range = max(z_max - z_min, 1.0)
            
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            z_center = (z_min + z_max) / 2
            
            max_range = max(x_range, y_range, z_range) / 2
            
            self.ax.set_xlim3d([x_center - max_range, x_center + max_range])
            self.ax.set_ylim3d([y_center - max_range, y_center + max_range])
            self.ax.set_zlim3d([z_center - max_range, z_center + max_range])
        
        plt.draw()
        plt.show()

    def clear(self):
        self.ax.clear()
        
        if self.absolute_mode:
            self.ax.set_xlim3d([-2.0, 2.0])
            self.ax.set_ylim3d([0.0, 4.0])
            self.ax.set_zlim3d([0.0, 4.0])
        else:
            self.ax.set_xlim3d([-2.0, 2.0])
            self.ax.set_ylim3d([-2.0, 2.0])
            self.ax.set_zlim3d([-2.0, 2.0])
            
        plt.grid(False)
        plt.axis('off')

    @staticmethod
    def sleep(t):
        plt.pause(t)
