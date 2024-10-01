import numpy as np


# Function from https://github.com/SeanChenxy/HandMesh/blob/main/utils/augmentation.py to solve import errors
def get_m1to1_gaussian_rand(scale):
    r = 2
    while r < -1 or r > 1:
        r = np.random.normal(scale=scale)

    return r
