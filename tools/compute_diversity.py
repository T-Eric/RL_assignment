"""
Diversity computation via Dynamic Time Warping (DTW).

This is the OFFICIAL evaluation function — both the baseline and student
submissions are scored with exactly this code.  Do NOT modify.

Usage as library:
    from compute_diversity import compute_dtw, compute_diversity_score

Usage as CLI:
    python compute_diversity.py --trajs_dir <path> --initials_path <path>
"""

import os
import json
import argparse
import numpy as np
from itertools import combinations
from multiprocessing import Pool, cpu_count


# ======================================================================
# Trajectory normalization
# ======================================================================

STEP_SIZE = 2.0 * np.sqrt(2)  # fixed per-step Euclidean distance (±2 m in x and y)


def resample_trajectory(traj, step_size=STEP_SIZE):
    """
    Resample a trajectory to uniform arc-length intervals.

    This ensures DTW scores are comparable regardless of the original
    temporal resolution of the trajectory.
    """
    traj = np.asarray(traj, dtype=np.float64)
    if len(traj) < 2:
        return traj

    diffs = np.diff(traj, axis=0)
    seg_len = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = cum_len[-1]

    if total_len < step_size:
        return traj

    n_samples = int(total_len / step_size) + 1
    sample_at = np.linspace(0, total_len, n_samples)

    resampled = np.column_stack([
        np.interp(sample_at, cum_len, traj[:, 0]),
        np.interp(sample_at, cum_len, traj[:, 1]),
    ])
    return resampled


# ======================================================================
# Core DTW implementation
# ======================================================================

def compute_dtw(traj_a, traj_b):
    """
    Compute the DTW distance between two 2D trajectories.

    Args:
        traj_a: np.ndarray of shape (N, 2) — sequence of (x, y) points
        traj_b: np.ndarray of shape (M, 2) — sequence of (x, y) points

    Returns:
        float: DTW distance (sum of aligned Euclidean distances)
    """
    traj_a = np.asarray(traj_a, dtype=np.float64)
    traj_b = np.asarray(traj_b, dtype=np.float64)
    n, m = len(traj_a), len(traj_b)

    # Two-row DP for memory efficiency
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr = np.full(m + 1, np.inf)
        diffs = traj_a[i - 1] - traj_b  # (m, 2)
        costs = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
        for j in range(1, m + 1):
            curr[j] = costs[j - 1] + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return float(prev[m])


# ======================================================================
# Aggregate diversity score
# ======================================================================

def _compute_one_initial(trajs_np_list):
    """Mean pairwise DTW for one initial's trajectory list."""
    resampled = [resample_trajectory(t) for t in trajs_np_list]
    dists = []
    for a, b in combinations(range(len(resampled)), 2):
        dists.append(compute_dtw(resampled[a], resampled[b]))
    return float(np.mean(dists)) if dists else 0.0


def compute_diversity_score(trajs_per_initial, n_workers=None):
    """
    Compute overall diversity score.

    Args:
        trajs_per_initial: dict  {initial_id: [traj_1, traj_2, ...]}
            Each traj is an np.ndarray of shape (T, 2).
        n_workers: number of parallel workers (default: cpu_count)

    Returns:
        overall: float — mean of per-initial mean pairwise DTW
        per_initial: dict {initial_id: float}
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 16)

    keys = sorted(trajs_per_initial.keys())
    inputs = [trajs_per_initial[k] for k in keys]

    with Pool(n_workers) as pool:
        scores = pool.map(_compute_one_initial, inputs)

    per_initial = {k: s for k, s in zip(keys, scores)}
    overall = float(np.mean(scores)) if scores else 0.0
    return overall, per_initial


# ======================================================================
# CLI for standalone evaluation
# ======================================================================

def load_trajs_from_dir(trajs_dir, initials_path, trajs_per_initial=20):
    """
    Load trajectories from directory structure:
        trajs_dir/
            initial_0/
                traj_0.txt  (each line: x y)
                traj_1.txt
                ...
    """
    with open(initials_path) as f:
        initials = json.load(f)

    result = {}
    for init in initials:
        iid = init["initial_id"]
        init_dir = os.path.join(trajs_dir, f"initial_{iid}")
        if not os.path.isdir(init_dir):
            print(f"WARNING: missing directory for initial_{iid}")
            continue

        trajs = []
        for j in range(trajs_per_initial):
            traj_path = os.path.join(init_dir, f"traj_{j}.txt")
            if os.path.exists(traj_path):
                trajs.append(np.loadtxt(traj_path))
        if len(trajs) >= trajs_per_initial:
            result[iid] = trajs[:trajs_per_initial]
        else:
            print(f"WARNING: initial_{iid} has only {len(trajs)} trajs "
                  f"(need {trajs_per_initial})")

    return result


def main():
    parser = argparse.ArgumentParser(description="Compute trajectory diversity")
    parser.add_argument("--trajs_dir", required=True,
                        help="Directory with initial_X/traj_Y.txt files")
    parser.add_argument("--initials_path", required=True,
                        help="Path to eval_initials_100.json")
    parser.add_argument("--trajs_per_initial", type=int, default=20)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    print("Loading trajectories...")
    trajs = load_trajs_from_dir(args.trajs_dir, args.initials_path,
                                args.trajs_per_initial)
    print(f"Loaded {len(trajs)} initials with {args.trajs_per_initial} trajs each")

    print("Computing diversity...")
    overall, per_initial = compute_diversity_score(trajs, args.workers)

    print(f"\n{'=' * 50}")
    print(f"Overall Diversity Score: {overall:.2f}")
    scores = list(per_initial.values())
    print(f"Per-initial: min={min(scores):.2f}, max={max(scores):.2f}, "
          f"median={np.median(scores):.2f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
