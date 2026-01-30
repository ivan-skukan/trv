# main.py
from solver import *
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os


def load_velodyne_points(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]


def voxel_downsample(points, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    return np.asarray(pcd.points)


def main_loop(all_points):
    C = [np.eye(4)]
    T_prev = np.eye(4)

    for i in range(len(all_points) - 1):
        print(f"Aligning point cloud {i} to {i + 1}")
        X = all_points[i]
        Y = all_points[i + 1]

        T, ok = ICP(X, Y, T_init=T_prev)
        if ok:
            T_prev = T
            C.append(C[-1] @ T)
        else:
            C.append(C[-1].copy())

    return C


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "dataset", "sequences", "00", "velodyne")

    voxel_size = 0.2
    ransac_iters = RANSAC_ITERATIONS
    ransac_thresh = RANSAC_THRESHOLD

    files = sorted(os.listdir(folder))
    all_points = []
    for file in files:
        if file.endswith(".bin"):
            points = load_velodyne_points(os.path.join(folder, file))
            if voxel_size > 0:
                points = voxel_downsample(points, voxel_size)
            print(f"Loaded {file} with shape {points.shape}")
            all_points.append(points)

    print(f"Loaded {len(all_points)} point clouds.")
    transformations = main_loop(all_points)

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(
        results_dir,
        f"estimated_voxel{voxel_size:.2f}_ransac{ransac_iters}_th{ransac_thresh:.2f}.txt",
    )
    with open(output_file, "w") as f:
        for T in transformations:
            line = " ".join(f"{v:.9e}" for v in T[:3, :4].reshape(-1))
            f.write(line + "\n")
    print(f"Transformations saved to {output_file}")

    positions = np.array([T[:3, 3] for T in transformations])

    gt_path = os.path.join(base_dir, "dataset", "sequences", "gt", "00", "00.txt")
    gt_positions = None
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            gt_lines = [line.strip() for line in f if line.strip()]
        gt_mats = []
        for line in gt_lines:
            vals = np.fromstring(line, sep=" ")
            if vals.size == 12:
                T_gt = np.eye(4)
                T_gt[:3, :4] = vals.reshape(3, 4)
                gt_mats.append(T_gt)
        if gt_mats:
            gt_positions = np.array([T[:3, 3] for T in gt_mats])

    plt.figure(figsize=(6, 6))
    plt.plot(positions[:, 1], positions[:, 0], "-o", markersize=3, label="Estimated")
    if gt_positions is not None:
        plt.plot(gt_positions[:, 1], gt_positions[:, 0], "-", linewidth=2, label="Ground truth")
    plt.title("Estimated trajectory (y-x)")
    plt.xlabel("y (left)")
    plt.ylabel("x (forward)")
    plt.axis("equal")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
