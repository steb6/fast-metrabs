import numpy as np
from setup.create_engine import create_engine

if __name__ == "__main__":

    # BackBone
    i = {"images": np.ones(shape=(1, 256, 256, 3), dtype=np.float32)}
    create_engine(
        'assets/bbone1.onnx',
        'assets/bbone1.engine',
        i)