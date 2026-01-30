# main.py
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from solver import (
    load_calib,
    detect_corners,
    track_points,
    filter_stereo_matches,
    triangulate_points,
    filter_depth,
    solve_pnp_ransac,
    se3_from_rt,
)


def load_image(sequence_dir, cam, idx):
    img_path = os.path.join(sequence_dir, f"image_{cam}", f"{idx:06d}.png")
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


def main():
    max_corners = 2500
    quality = 0.01
    min_distance = 7

    fb_thresh = 1.0          
    max_y_diff = 1.5         
    z_min = 0.5              
    z_max = 80.0

    pnp_reproj = 2.0
    pnp_min_inliers = 40
    pnp_min_ratio = 0.35

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sequence = "00"
    sequence_dir = os.path.join(base_dir, "dataset", "sequences", sequence)
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    P0, P1, K = load_calib(sequence_dir)

    # Maintain camera->world pose so pose[:3,3] is camera position in world.
    pose_cw = np.eye(4, dtype=np.float64)
    poses_cw = [pose_cw.copy()]

    idx = 0
    img_left_prev = load_image(sequence_dir, "0", idx)
    img_right_prev = load_image(sequence_dir, "1", idx)
    if img_left_prev is None or img_right_prev is None:
        print("No images found.")
        return

    while True:
        idx += 1
        img_left = load_image(sequence_dir, "0", idx)
        img_right = load_image(sequence_dir, "1", idx)
        print(f"Processing frame {idx}...")

        if img_left is None or img_right is None:
            break

        pts_l_prev = detect_corners(
            img_left_prev,
            max_corners=max_corners,
            quality=quality,
            min_distance=min_distance,
        )

        pts_r_prev, m_stereo = track_points(
            img_left_prev, img_right_prev, pts_l_prev, fb_thresh=fb_thresh
        )
        pts_l, m_temp = track_points(
            img_left_prev, img_left, pts_l_prev, fb_thresh=fb_thresh
        )

        mask = m_stereo & m_temp
        pts_l_prev = pts_l_prev[mask]
        pts_r_prev = pts_r_prev[mask]
        pts_l = pts_l[mask]

        if len(pts_l_prev) < 80:
            print(f"Frame {idx}: not enough tracked points ({len(pts_l_prev)})")
            img_left_prev, img_right_prev = img_left, img_right
            poses_cw.append(pose_cw.copy())
            continue

        pts_l_prev, pts_r_prev, smask = filter_stereo_matches(
            pts_l_prev, pts_r_prev, max_y_diff=max_y_diff
        )
        pts_l = pts_l[smask]

        if len(pts_l_prev) < 40:
            print(f"Frame {idx}: not enough stereo-consistent matches ({len(pts_l_prev)})")
            img_left_prev, img_right_prev = img_left, img_right
            poses_cw.append(pose_cw.copy())
            continue

        points_3d = triangulate_points(P0, P1, pts_l_prev, pts_r_prev)
        points_3d, dmask = filter_depth(points_3d, z_min=z_min, z_max=z_max)
        pts_l = pts_l[dmask]

        if len(points_3d) < 30:
            print(f"Frame {idx}: not enough valid 3D points ({len(points_3d)})")
            img_left_prev, img_right_prev = img_left, img_right
            poses_cw.append(pose_cw.copy())
            continue

        rvec, tvec, inliers = solve_pnp_ransac(
            points_3d,
            pts_l,
            K,
            reproj_err=pnp_reproj,
            min_inliers=pnp_min_inliers,
            min_inlier_ratio=pnp_min_ratio,
        )
        if rvec is None:
            print(f"Frame {idx}: PnP rejected/failed")
            img_left_prev, img_right_prev = img_left, img_right
            poses_cw.append(pose_cw.copy())
            continue

        T_pc = se3_from_rt(rvec, tvec)                  # prev -> curr
        pose_cw = pose_cw @ np.linalg.inv(T_pc)         # camera -> world

        poses_cw.append(pose_cw.copy())

        img_left_prev, img_right_prev = img_left, img_right

    out_cw = os.path.join(results_dir, f"stereo_{sequence}_cw.txt")
    out_wc = os.path.join(results_dir, f"stereo_{sequence}_wc.txt")
    with open(out_cw, "w") as f1, open(out_wc, "w") as f2:
        for Tcw in poses_cw:
            line_cw = " ".join(f"{v:.9e}" for v in Tcw[:3, :4].reshape(-1))
            Twc = np.linalg.inv(Tcw)
            line_wc = " ".join(f"{v:.9e}" for v in Twc[:3, :4].reshape(-1))
            f1.write(line_cw + "\n")
            f2.write(line_wc + "\n")

    print(f"Wrote {len(poses_cw)} poses to:\n  {out_cw}\n  {out_wc}")

    positions = np.array([T[:3, 3] for T in poses_cw])
    plt.figure(figsize=(6, 6))
    plt.plot(positions[:, 0], positions[:, 2], "-", linewidth=1.5)
    plt.title("Estimated trajectory (camera->world), X-Z plane")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.axis("equal")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
