import numpy as np
from setup.create_engine import create_engine


if __name__ == "__main__":
    # YOLO
    i = {"input": np.ones(shape=(1, 3, 256, 256), dtype=np.float32)}
    create_engine(
        'assets/yolo.onnx',
        'assets/yolo.engine',
        i)