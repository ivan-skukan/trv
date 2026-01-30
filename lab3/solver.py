# solver.py
import numpy as np
from scipy.spatial import KDTree

RANSAC_ITERATIONS = 200
RANSAC_THRESHOLD = 0.2


def T_minimizer(p, q):
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("p and q must be (N,3) with matching shapes")

    p_centroid = np.mean(p, axis=0)
    q_centroid = np.mean(q, axis=0)

    p_centered = p - p_centroid
    q_centered = q - q_centroid

    H = p_centered.T @ q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = q_centroid - R @ p_centroid
    return R, t


def RANSAC_alignment(src_pts, dst_pts):
    if src_pts.shape != dst_pts.shape or src_pts.ndim != 2 or src_pts.shape[1] != 3:
        raise ValueError("src_pts and dst_pts must be (N,3) with matching shapes")

    N = src_pts.shape[0]
    if N < 3:
        return np.eye(4), np.zeros((0,), dtype=bool)

    best_T = np.eye(4)
    best_inliers = np.zeros((N,), dtype=bool)
    max_inliers = 0

    thr = float(RANSAC_THRESHOLD)
    iters = int(RANSAC_ITERATIONS)

    for _ in range(iters):
        idx = np.random.choice(N, 3, replace=False)
        p = src_pts[idx]
        q = dst_pts[idx]

        R, t = T_minimizer(p, q)
        src_tf = (R @ src_pts.T).T + t
        d = np.linalg.norm(src_tf - dst_pts, axis=1)

        inliers = d < thr
        nin = int(np.sum(inliers))
        if nin > max_inliers:
            max_inliers = nin
            best_inliers = inliers
            best_T[:3, :3] = R
            best_T[:3, 3] = t

    if max_inliers >= 3:
        R, t = T_minimizer(src_pts[best_inliers], dst_pts[best_inliers])
        best_T[:3, :3] = R
        best_T[:3, 3] = t

    return best_T, best_inliers


def closest_points(tree, X):
    distances, indices = tree.query(X)
    return distances, indices


def ICP(X, Y, max_iterations=20, epsilon=1e-6, T_init=None, min_inliers=200, min_inlier_ratio=0.05):
    T = np.eye(4) if T_init is None else T_init.copy()
    tree = KDTree(Y)

    last_error = float("inf")
    ok = True

    for _ in range(int(max_iterations)):
        X_tf = (T[:3, :3] @ X.T).T + T[:3, 3]
        _, nn_idx = closest_points(tree, X_tf)
        Y_match = Y[nn_idx]

        dT, inliers = RANSAC_alignment(X_tf, Y_match)
        nin = int(np.sum(inliers))
        if nin < int(min_inliers) or nin / float(len(X_tf)) < float(min_inlier_ratio):
            ok = False
            break

        T = dT @ T

        X_tf_new = (T[:3, :3] @ X.T).T + T[:3, 3]
        _, nn_idx_new = closest_points(tree, X_tf_new)
        Y_match_new = Y[nn_idx_new]
        error = float(np.mean(np.linalg.norm(X_tf_new - Y_match_new, axis=1)))

        if abs(last_error - error) < float(epsilon):
            break
        last_error = error

    return T, ok
