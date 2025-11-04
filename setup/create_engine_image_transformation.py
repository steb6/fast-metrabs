import numpy as np
from setup.create_engine import create_engine


if __name__ == "__main__":

    # Image Transformation
    i = {"frame": np.ones(shape=(480, 640, 3), dtype=np.int32),
         "H": np.ones(shape=(1, 3, 3), dtype=np.float32)}
    create_engine(  # p,
        'assets/image_transformation1.onnx',
        'assets/image_transformation1.engine',
        i)