import numpy as np
from scipy.spatial import cKDTree

def load_pointcloud(npy_path):
    """Load 2D pointcloud from .npy file. Returns (N, 2) array."""
    points = np.load(npy_path)
    assert points.ndim == 2 and points.shape[1] == 2
    return points


def load_pointcloud_transposed(npy_path):
    """Load pointcloud and return as (2, N) for fast distance queries."""
    return load_pointcloud(npy_path).T

def build_pointcloud_index(npy_path=None, points=None):
    """
    Build KD-tree index for nearest-neighbor queries.

    Returns:
        points_Nx2: np.ndarray of shape (N, 2)
        tree: scipy.spatial.cKDTree
    """
    if points is None:
        if npy_path is None:
            raise ValueError("Either npy_path or points must be provided.")
        points = load_pointcloud(npy_path)

    tree = cKDTree(points)
    return points, tree


def find_nearest_point(points_2xN, query_xy):
    """
    Find the nearest point in points_2xN to query_xy.

    Args:
        points_2xN: (2, N) array of XY points
        query_xy: (2,) array or list [x, y]

    Returns:
        nearest_relative: (2,) relative offset from query to nearest point
        distance: scalar distance to nearest point
    """
    query = np.asarray(query_xy, dtype=points_2xN.dtype).reshape(2, 1)
    diffs = points_2xN - query
    dist2 = diffs[0] * diffs[0] + diffs[1] * diffs[1]
    idx = np.argmin(dist2)
    return diffs[:, idx], float(np.sqrt(dist2[idx]))

def find_nearest_point_kdtree(points_Nx2, tree, query_xy):
    """
    Find nearest point using KD-tree.

    Args:
        points_Nx2: (N, 2) array
        tree: cKDTree built from points_Nx2
        query_xy: (2,) array-like

    Returns:
        nearest_relative: (2,) relative offset from query to nearest point
        distance: scalar distance to nearest point
    """
    query = np.asarray(query_xy, dtype=np.float64)
    dist, idx = tree.query(query, k=1)
    nearest = points_Nx2[idx]
    nearest_relative = nearest - query
    return nearest_relative, float(dist)