import matplotlib.pyplot as plt


class MPLPosePrinter:
    def __init__(self, **args):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=180, azim=0, vertical_axis='y')

        # Setting the axes properties
        self.ax.set_xlim3d([-2.0, 2.0])
        self.ax.set_ylim3d([-2.0, 2.0])
        self.ax.set_zlim3d([-2.0, 2.0])

        plt.grid(False)
        plt.axis('off')
        plt.title('3D Human Pose')

    def print_pose(self, pose, edges, color='b'):
        self.ax.plot([0, 0], [-1, 1], [0, 0])
        if len(pose.shape) == 2:
            pose = pose[None]
        # pose_flat = pose.reshape(-1, 3)
        # self.sc._offsets3d = (pose_flat[:, 0], pose_flat[:, 1], pose_flat[:, 2])
        if edges is not None:
            for p in pose:
                for edge in edges:
                    a = p[edge[0]]
                    b = p[edge[1]]
                    self.ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color)
        plt.draw()
        plt.show()

    def clear(self):
        self.ax.clear()
        self.ax.set_xlim3d([-2.0, 2.0])
        self.ax.set_ylim3d([-2.0, 2.0])
        self.ax.set_zlim3d([-2.0, 2.0])
        plt.grid(False)
        plt.axis('off')

    @staticmethod
    def sleep(t):
        plt.pause(t)
