import numpy as np


def load_pointcloud(npy_path):
    """Load 2D pointcloud from .npy file. Returns (N, 2) array."""
    points = np.load(npy_path)
    assert points.ndim == 2 and points.shape[1] == 2
    return points


def load_pointcloud_transposed(npy_path):
    """Load pointcloud and return as (2, N) for fast distance queries."""
    return load_pointcloud(npy_path).T


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
    query = np.array(query_xy).reshape(2, 1)
    diffs = points_2xN - query
    dists = np.linalg.norm(diffs, axis=0)
    idx = np.argmin(dists)
    return diffs[:, idx], dists[idx]
