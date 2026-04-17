"""
One-time script to compute baseline diversity from existing trajectory data.
NOT part of the student assignment; used only to generate baseline files.
"""

import os
import json
import numpy as np
import re
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

TRAJ_DIR = "/home/songjunru/UAV/UAV/saved_data/grid_16_200-300_1767751107.488123/success_trajs"
CENTERS_PATH = "/home/songjunru/UAV/RL_assignment/data/cluster_centers.json"
INITIALS_JSONL = "/home/songjunru/UAV/get_initial/initials_16/initial_samples_200-300.jsonl"
OUTPUT_DIR = "/home/songjunru/UAV/RL_assignment/data"

SUCCESS_RADIUS = 30.0
NUM_INITIALS = 100
TRAJS_PER_INITIAL = 20
SEED = 42


def parse_numpy_line(line):
    line = line.strip().strip("[]")
    parts = re.split(r'[,\s]+', line)
    parts = [p for p in parts if p]
    return [float(x) for x in parts]


def compute_dtw_numpy(traj_a_np, traj_b_np):
    """Numpy-optimized DTW using two-row rolling buffer."""
    n, m = len(traj_a_np), len(traj_b_np)
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr = np.full(m + 1, np.inf)
        diffs = traj_a_np[i - 1] - traj_b_np  # (m, 2)
        costs = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)  # (m,)
        for j in range(1, m + 1):
            curr[j] = costs[j - 1] + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[m]


def compute_dtw_for_initial(trajs_np_list):
    """Compute mean pairwise DTW for one initial's trajectories."""
    n = len(trajs_np_list)
    dtw_distances = []
    for a, b in combinations(range(n), 2):
        d = compute_dtw_numpy(trajs_np_list[a], trajs_np_list[b])
        dtw_distances.append(d)
    return float(np.mean(dtw_distances)) if dtw_distances else 0.0


def main():
    np.random.seed(SEED)

    with open(CENTERS_PATH) as f:
        centers = json.load(f)

    initials_list = []
    with open(INITIALS_JSONL) as f:
        for line in f:
            initials_list.append(json.loads(line))

    # --- Phase 1: Parse all trajectory files ---
    print("Phase 1: Parsing trajectory files...")
    files = sorted(os.listdir(TRAJ_DIR), key=lambda x: int(x.replace('.txt', '')))

    groups = defaultdict(list)

    for fname in tqdm(files, desc="Parsing trajectories"):
        filepath = os.path.join(TRAJ_DIR, fname)
        with open(filepath) as f:
            lines = f.readlines()

        target_id = int(lines[1].strip())
        target_id_str = str(target_id)

        if target_id_str not in centers:
            continue

        cx, cy = centers[target_id_str]

        waypoints = []
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            try:
                vals = parse_numpy_line(line)
                waypoints.append((vals[0], vals[1]))
            except:
                continue

        truncated = None
        for i, (x, y) in enumerate(waypoints):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist <= SUCCESS_RADIUS:
                truncated = waypoints[:i+1]
                break

        if truncated is not None and len(truncated) >= 2:
            init_vals = parse_numpy_line(lines[0])
            key = (round(init_vals[0], 2), round(init_vals[1], 2), target_id)
            groups[key].append(truncated)

    print(f"Unique initials with valid trajectories: {len(groups)}")
    eligible = {k: v for k, v in groups.items() if len(v) >= TRAJS_PER_INITIAL}
    print(f"Initials with >= {TRAJS_PER_INITIAL} valid trajs: {len(eligible)}")

    eligible_keys = list(eligible.keys())
    np.random.shuffle(eligible_keys)
    selected_keys = sorted(eligible_keys[:NUM_INITIALS])
    print(f"Selected {len(selected_keys)} initials")

    initials_lookup = {}
    for init in initials_list:
        lk = (round(init["x_start"], 2), round(init["y_start"], 2), init["target_id"])
        initials_lookup[lk] = init

    # --- Phase 2: Save trajectories and prepare DTW inputs ---
    print("\nPhase 2: Saving trajectories...")
    baseline_trajs_dir = os.path.join(OUTPUT_DIR, "baseline_trajs")
    os.makedirs(baseline_trajs_dir, exist_ok=True)

    eval_initials = []
    dtw_inputs = []

    for i, key in enumerate(tqdm(selected_keys, desc="Saving trajectories")):
        x_start, y_start, target_id = key
        target_center = centers[str(target_id)]

        init_info = initials_lookup.get(key, {
            "x_start": x_start, "y_start": y_start,
            "target_id": target_id,
            "target_center_x": target_center[0],
            "target_center_y": target_center[1],
        })

        eval_initials.append({
            "initial_id": i,
            "x_start": init_info.get("x_start", x_start),
            "y_start": init_info.get("y_start", y_start),
            "target_id": target_id,
            "target_center_x": target_center[0],
            "target_center_y": target_center[1],
            "distance": init_info.get("distance", None),
        })

        trajs = eligible[key][:TRAJS_PER_INITIAL]

        init_dir = os.path.join(baseline_trajs_dir, f"initial_{i}")
        os.makedirs(init_dir, exist_ok=True)
        trajs_np = []
        for j, traj in enumerate(trajs):
            traj_path = os.path.join(init_dir, f"traj_{j}.txt")
            with open(traj_path, 'w') as f:
                for x, y in traj:
                    f.write(f"{x} {y}\n")
            trajs_np.append(np.array(traj))
        dtw_inputs.append(trajs_np)

    # --- Phase 3: Parallel DTW computation ---
    n_workers = min(cpu_count(), 16)
    print(f"\nPhase 3: Computing DTW diversity ({n_workers} workers)...")

    with Pool(n_workers) as pool:
        all_dtw_scores = list(tqdm(
            pool.imap(compute_dtw_for_initial, dtw_inputs),
            total=len(dtw_inputs), desc="DTW per initial"
        ))

    overall_diversity = float(np.mean(all_dtw_scores))
    print(f"\n{'='*50}")
    print(f"Baseline Diversity Score: {overall_diversity:.2f}")
    print(f"Per-initial DTW: min={min(all_dtw_scores):.2f}, max={max(all_dtw_scores):.2f}, "
          f"median={np.median(all_dtw_scores):.2f}")
    print(f"{'='*50}")

    with open(os.path.join(OUTPUT_DIR, "eval_initials_100.json"), 'w') as f:
        json.dump(eval_initials, f, indent=2)

    diversity_info = {
        "overall_diversity": overall_diversity,
        "num_initials": len(selected_keys),
        "trajs_per_initial": TRAJS_PER_INITIAL,
        "success_radius_m": SUCCESS_RADIUS,
        "dtw_method": "full DTW with Euclidean distance on (x, y) coordinates",
        "per_initial_dtw": {f"initial_{i}": float(s) for i, s in enumerate(all_dtw_scores)},
        "seed": SEED,
    }
    with open(os.path.join(OUTPUT_DIR, "baseline_diversity.json"), 'w') as f:
        json.dump(diversity_info, f, indent=2)

    print(f"\nAll done! Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
