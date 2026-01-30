# solver.py
import os
import numpy as np
import cv2


def load_calib(sequence_dir):
    """
    Load P0, P1 projection matrices from calib.txt (KITTI odometry format).
    Returns: P0 (3x4), P1 (3x4), K (3x3)
    """
    calib_path = os.path.join(sequence_dir, "calib.txt")
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Missing calib.txt at {calib_path}")

    data = {}
    with open(calib_path, "r") as f:
        for line in f:
            if ":" in line:
                key, vals = line.split(":", 1)
                data[key.strip()] = np.fromstring(vals, sep=" ")

    P0 = data["P0"].reshape(3, 4).astype(np.float64)
    P1 = data["P1"].reshape(3, 4).astype(np.float64)

    K = P0[:, :3].copy()
    return P0, P1, K


def detect_corners(image, max_corners=2000, quality=0.01, min_distance=7):
    """Detect good corners to track (Shi-Tomasi). Returns Nx2 float32."""
    corners = cv2.goodFeaturesToTrack(
        image,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_distance,
        blockSize=7,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2).astype(np.float32)


def track_points(img1, img2, pts1, fb_thresh=1.0):
    """
    Track points from img1->img2 using KLT with forward-backward consistency.
    Returns:
      pts2: Nx2 float32
      mask: N bool
    """
    if len(pts1) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=bool)

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    pts1_ = pts1.reshape(-1, 1, 2).astype(np.float32)
    pts2, st_fwd, _ = cv2.calcOpticalFlowPyrLK(img1, img2, pts1_, None, **lk_params)
    if pts2 is None or st_fwd is None:
        return np.empty((0, 2), dtype=np.float32), np.zeros((len(pts1),), dtype=bool)

    pts1_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(img2, img1, pts2, None, **lk_params)
    if pts1_back is None or st_bwd is None:
        return pts2.reshape(-1, 2), np.zeros((len(pts1),), dtype=bool)

    st_fwd = st_fwd.reshape(-1).astype(bool)
    st_bwd = st_bwd.reshape(-1).astype(bool)

    fb_err = np.linalg.norm(pts1_back.reshape(-1, 2) - pts1, axis=1)
    mask = st_fwd & st_bwd & (fb_err <= float(fb_thresh))

    return pts2.reshape(-1, 2).astype(np.float32), mask


def filter_stereo_matches(pts_left, pts_right, max_y_diff=1.0):
    """
    Keep stereo matches that satisfy rectified epipolar constraint and positive disparity.
    Returns filtered (pts_left_f, pts_right_f, mask_bool).
    """
    pts_left = np.asarray(pts_left, dtype=np.float32)
    pts_right = np.asarray(pts_right, dtype=np.float32)
    y_diff = np.abs(pts_left[:, 1] - pts_right[:, 1])
    disparity = pts_left[:, 0] - pts_right[:, 0]
    mask = (y_diff <= float(max_y_diff)) & (disparity > 0.0)
    return pts_left[mask], pts_right[mask], mask


def triangulate_points(P0, P1, pts_left, pts_right):
    """
    Triangulate points using projection matrices.
    Inputs: pts_left/right as Nx2 (float).
    Output: Nx3 in the left camera frame (k-1).
    """
    pts_left = np.asarray(pts_left, dtype=np.float64)
    pts_right = np.asarray(pts_right, dtype=np.float64)

    pts4d = cv2.triangulatePoints(P0, P1, pts_left.T, pts_right.T)  # 4xN
    pts3d = (pts4d[:3] / pts4d[3]).T  # Nx3
    return pts3d.astype(np.float64)


def filter_depth(points_3d, z_min=0.1, z_max=80.0):
    """Filter 3D points by depth (z forward in camera frame). Returns (points_f, mask)."""
    points_3d = np.asarray(points_3d, dtype=np.float64)
    z = points_3d[:, 2]
    mask = (z > float(z_min)) & (z < float(z_max)) & np.isfinite(z)
    return points_3d[mask], mask


def solve_pnp_ransac(points_3d, points_2d, K,
                     reproj_err=2.0, iters=200, conf=0.999,
                     min_inliers=40, min_inlier_ratio=0.35):
    """
    Robust PnP (EPNP in RANSAC) with inlier gating and refinement (ITERATIVE) on inliers.
    Returns (rvec, tvec, inliers_idx) or (None,None,None).
    """
    points_3d = np.asarray(points_3d, dtype=np.float32)
    points_2d = np.asarray(points_2d, dtype=np.float32)
    K = np.asarray(K, dtype=np.float64)

    n = len(points_3d)
    if n < 20:
        return None, None, None

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d,
        points_2d,
        K,
        None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=float(reproj_err),
        iterationsCount=int(iters),
        confidence=float(conf),
    )
    if not ok or inliers is None:
        return None, None, None

    inliers = inliers.reshape(-1)
    nin = len(inliers)
    if nin < int(min_inliers) or nin / float(n) < float(min_inlier_ratio):
        return None, None, None

    obj_in = points_3d[inliers]
    img_in = points_2d[inliers]

    ok2, rvec2, tvec2 = cv2.solvePnP(
        obj_in,
        img_in,
        K,
        None,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if ok2:
        rvec, tvec = rvec2, tvec2

    return rvec, tvec, inliers


def se3_from_rt(rvec, tvec):
    """Build a 4x4 SE3 matrix from rvec/tvec."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.astype(np.float64)
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T
