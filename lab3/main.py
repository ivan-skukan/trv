from solver import *
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os


def load_velodyne_points(filename):
    """Load point cloud from KITTI .bin file"""
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # Returns [x, y, z]

def voxel_downsample(points, voxel_size):
    """Downsample a point cloud using a voxel grid."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd.points)

def main_loop(all_points):

    C = [np.eye(4)]
    T_prev = np.eye(4)

    for i in range(len(all_points) - 1):
        print(f"Aligning point cloud {i} to {i + 1}")
        X = all_points[i]
        Y_gt = all_points[i + 1]

        T = ICP(X, Y_gt, T_init=T_prev)
        T_prev = T
        C.append(C[-1] @ T)

    return C


def main():
    # Resolve dataset path relative to this file so it works from any CWD.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "dataset", "sequences", "00", "velodyne")
    voxel_size = 0.2  # meters
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
    
    # Store transformations in KITTI pose format (one 3x4 row-major per line).
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(
        results_dir,
        f"estimated_voxel{voxel_size:.2f}_ransac{ransac_iters}_th{ransac_thresh:.2f}.txt",
    )
    with open(output_file, "w") as f:
        for i, T in enumerate(transformations):
            line = " ".join(f"{v:.9e}" for v in T[:3, :4].reshape(-1))
            f.write(line + "\n")
            print(f"Transformation {i}:\n{T}\n")
    print(f"Transformations saved to {output_file}")
    

    positions = np.array([T[:3, 3] for T in transformations])
    plt.figure(figsize=(6, 6))
    plt.plot(positions[:, 1], positions[:, 0], "-o", markersize=3)
    plt.title("Estimated trajectory (y-x)")
    plt.xlabel("y (left)")
    plt.ylabel("x (forward)")
    plt.axis("equal")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()
