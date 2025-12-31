import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def load_velodyne_points(filename):
    """Load point cloud from KITTI .bin file"""
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    print("Shape:")
    print(points.shape)
    return points  # Returns [x, y, z, reflectance]

def visualize_point_cloud_o3d(points, downsample_voxel=0.1):
    """Fast interactive 3D visualization using Open3D"""
    points_xyz = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    
    # Optional downsampling for speed
    if downsample_voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel)
    
    print(f"Displaying {len(pcd.points)} points after downsampling")
    o3d.visualization.draw_geometries([pcd])

def visualize_bird_eye_view(points, title="Bird's Eye View"):
    """Top-down 2D view using matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Color by height (z-axis)
    scatter = ax.scatter(x, y, c=z, cmap='jet', s=0.5, alpha=0.6)
    
    ax.set_xlabel('X (forward)')
    ax.set_ylabel('Y (left)')
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Height (Z)')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Example: Load a point cloud from sequence 00
    bin_file = "dataset/sequences/00/velodyne/000000.bin"
    
    print(f"Loading point cloud from {bin_file}...")
    points = load_velodyne_points(bin_file)
    
    print(f"Loaded {points.shape[0]} points")
    print(f"X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # Fast 3D visualization
    visualize_point_cloud_o3d(points, downsample_voxel=0.1)
    
    # Bird's eye view
    visualize_bird_eye_view(points, "KITTI Point Cloud - Bird's Eye View")
