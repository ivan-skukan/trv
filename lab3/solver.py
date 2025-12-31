from os import error
from flask.config import T
import numpy as np
from scipy.spatial import KDTree


def T_minimizer(p, q):
    """Extracts rotation matrix R and translation vector t from known correspondences."""
    assert p.shape == q.shape and p.shape[1] == 3, "Input point sets must be of shape (N, 3)"

    p_centroid = np.mean(p, axis=0)
    q_centroid = np.mean(q, axis=0)

    p_centered = p - p_centroid
    q_centered = q - q_centroid

    H = p_centered.T @ q_centered

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    
    t = q_centroid - R @ p_centroid

    return R, t

def RANSAC_allignment(Y, Y_gt):
    """RANSAC to find the best transformation aligning Y to Y_gt."""
    max_inliers = 0
    best_T = np.eye(4)
    num_iterations = 100 # idk lol
    threshold = 0.1 # also dont know lol
    N = Y.shape[0]
    for _ in range(num_iterations):
        indices = np.random.choice(N, 3, replace=False)
        p_sample = Y[indices]
        q_sample = Y_gt[indices]

        R, t = T_minimizer(p_sample, q_sample)

        Y_transformed = (R @ Y.T).T + t

        distances = np.linalg.norm(Y_transformed - Y_gt, axis=1)
        inliers = distances < threshold
        num_inliers = np.sum(inliers)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_T[:3, :3] = R
            best_T[:3, 3] = t

    return best_T

def closest_points(X, Y):
    """Finds the closest points in Y for each point in X."""
    tree = KDTree(Y)
    distances, indices = tree.query(X)
    Y_closest = Y[indices]
    return Y_closest


def ICP(X, Y_gt, max_iterations=1000, epsilon=1e-6):
    """Iterative Closest Point algorithm to align point cloud X to Y_gt."""
    convergence = False
    last_error = float('inf')
    T = np.eye(4)

    while not convergence and max_iterations > 0:
        Y = (T[:3, :3] @ X.T).T + T[:3, 3]
        Y_closest = closest_points(Y, Y_gt)
        dT = RANSAC_allignment(Y, Y_closest)
        T = dT @ T
        
        error = np.mean(np.linalg.norm(T[:3, :3] @ X.T + T[:3, 3][:, np.newaxis] - Y_closest.T, axis=0))
        print(f"Current error: {error}")
        if abs(last_error - error) < epsilon:
            convergence = True
        last_error = error
        max_iterations -= 1
    return T
