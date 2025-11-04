import numpy as np
from setup.create_engine import create_engine

if __name__ == "__main__":

    # Heads
    i = {"input": np.ones(shape=(81920,), dtype=np.float32)}
    create_engine(
        'assets/heads1.onnx',
        'assets/heads1.engine',
        i)