import re
from pyparsing import C
from solver import *
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os


def load_velodyne_points(filename):
    """Load point cloud from KITTI .bin file"""
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # Returns [x, y, z]


def main_loop(all_points):

    C = [np.eye(4)]

    for i in range(len(all_points) - 1):
        X = all_points[i]
        Y_gt = all_points[i + 1]

        T = ICP(X, Y_gt)
        C.append(C[-1] @ T)

    return C




def main():
    folder = "dataset/sequences/00/velodyne/"
    files = sorted(os.listdir(folder))
    all_points = []
    for file in files:
        if file.endswith(".bin"):
            points = load_velodyne_points(os.path.join(folder, file))
            all_points.append(points)

    transformations = main_loop(all_points)
    