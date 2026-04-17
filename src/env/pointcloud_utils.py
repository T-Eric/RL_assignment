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

def angular_diff(a,b):
    # angular distance between two angles in radians
    diff=a-b
    diff=(diff+np.pi)%(2*np.pi)-np.pi
    return np.abs(diff)

def compute_directional_distances(
    points_Nx2,
    query_xy,
    num_dirs=16,
    max_range=80.0,
    sector_half_angle=None,):
    """
    Compute directional obstacle distances around query point.

    Args:
        points_Nx2: (N, 2) obstacle points
        query_xy: (2,) current position
        num_dirs: number of angular sectors
        max_range: max sensing range
        sector_half_angle: optional half-angle for each sector;
            default = pi / num_dirs

    Returns:
        dists: (num_dirs,) float32 array
    """
    query = np.asarray(query_xy, dtype=np.float64).reshape(1, 2)
    rel = points_Nx2 - query                      # (N, 2)
    dist = np.linalg.norm(rel, axis=1)            # (N,)
    angle = np.arctan2(rel[:, 1], rel[:, 0])      # (N,)

    # optionally ignore very far points
    valid = dist <= max_range
    rel = rel[valid]
    dist = dist[valid]
    angle = angle[valid]

    if sector_half_angle is None:
        sector_half_angle = np.pi / num_dirs

    centers = np.linspace(0.0, 2 * np.pi, num_dirs, endpoint=False)
    dists = np.full(num_dirs, max_range, dtype=np.float32)

    if len(dist) == 0:
        return dists

    for k, c in enumerate(centers):
        diff = angular_diff(angle, c)
        mask = diff <= sector_half_angle
        if np.any(mask):
            dists[k] = np.min(dist[mask])

    return dists