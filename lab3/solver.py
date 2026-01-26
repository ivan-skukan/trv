import numpy as np
from scipy.spatial import KDTree

RANSAC_ITERATIONS = 200
RANSAC_THRESHOLD = 0.2


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
    best_inliers = None
    best_T = np.eye(4)
    num_iterations = RANSAC_ITERATIONS
    threshold = RANSAC_THRESHOLD
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
            best_inliers = inliers
            best_T[:3, :3] = R
            best_T[:3, 3] = t

    if best_inliers is not None and np.sum(best_inliers) >= 3:
        R, t = T_minimizer(Y[best_inliers], Y_gt[best_inliers])
        best_T[:3, :3] = R
        best_T[:3, 3] = t

    return best_T

def closest_points(X, Y):
    """Finds the closest points in Y for each point in X."""
    tree = KDTree(Y)
    distances, indices = tree.query(X)
    Y_closest = Y[indices]
    return Y_closest


def ICP(X, Y_gt, max_iterations=20, epsilon=1e-6, T_init=None):
    """Iterative Closest Point algorithm to align point cloud X to Y_gt."""
    convergence = False
    last_error = float('inf')
    T = np.eye(4) if T_init is None else T_init.copy()

    while not convergence and max_iterations > 0:
        Y = (T[:3, :3] @ X.T).T + T[:3, 3]
        Y_closest = closest_points(Y, Y_gt)
        dT = RANSAC_allignment(Y, Y_closest)
        print("dT norm:", np.linalg.norm(dT[:3, :3] - np.eye(3)), np.linalg.norm(dT[:3, 3]))
        T = dT @ T
        
        Y_new = (T[:3, :3] @ X.T).T + T[:3, 3]
        Y_closest = closest_points(Y_new, Y_gt)
        error = np.mean(np.linalg.norm(Y_new - Y_closest, axis=1))
        print(f"Current error: {error}")
        if abs(last_error - error) < epsilon:
            convergence = True
        last_error = error
        max_iterations -= 1
    return T
