"""
Evaluation script for student submissions.

Checks:
  1. All 100 initials have >= 20 successful trajectories
  2. Each trajectory reaches within 30m of target center
  3. Computes diversity score and compares with baseline

Usage:
    python evaluate_submission.py \
        --submission_dir <path_to_student_trajs> \
        --initials_path ../data/eval_initials_100.json \
        --baseline_path ../data/baseline_diversity.json
"""

import os
import json
import argparse
import numpy as np

from compute_diversity import compute_dtw, compute_diversity_score


def validate_trajectory(traj, target_center, success_radius=30.0):
    """Check if trajectory reaches within success_radius of target center."""
    for point in traj:
        dist = np.sqrt((point[0] - target_center[0])**2 +
                       (point[1] - target_center[1])**2)
        if dist <= success_radius:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_dir", required=True,
                        help="Directory: initial_X/traj_Y.txt")
    parser.add_argument("--initials_path", required=True)
    parser.add_argument("--baseline_path", required=True)
    parser.add_argument("--trajs_per_initial", type=int, default=20)
    parser.add_argument("--success_radius", type=float, default=30.0)
    args = parser.parse_args()

    with open(args.initials_path) as f:
        initials = json.load(f)
    with open(args.baseline_path) as f:
        baseline = json.load(f)

    baseline_score = baseline["overall_diversity"]
    required_trajs = args.trajs_per_initial

    print(f"Evaluating submission: {args.submission_dir}")
    print(f"Baseline diversity: {baseline_score:.2f}")
    print(f"Required: {required_trajs} valid trajs per initial\n")

    # --- Check completeness ---
    all_trajs = {}
    missing_initials = []
    insufficient_initials = []
    invalid_trajs_count = 0
    total_trajs = 0

    for init in initials:
        iid = init["initial_id"]
        target_center = [init["target_center_x"], init["target_center_y"]]
        init_dir = os.path.join(args.submission_dir, f"initial_{iid}")

        if not os.path.isdir(init_dir):
            missing_initials.append(iid)
            continue

        valid_trajs = []
        for j in range(required_trajs):
            traj_path = os.path.join(init_dir, f"traj_{j}.txt")
            if not os.path.exists(traj_path):
                continue
            traj = np.loadtxt(traj_path)
            total_trajs += 1
            if traj.ndim == 2 and traj.shape[1] == 2:
                if validate_trajectory(traj, target_center, args.success_radius):
                    valid_trajs.append(traj)
                else:
                    invalid_trajs_count += 1
            else:
                invalid_trajs_count += 1

        if len(valid_trajs) >= required_trajs:
            all_trajs[iid] = valid_trajs[:required_trajs]
        else:
            insufficient_initials.append((iid, len(valid_trajs)))

    # --- Report ---
    print("=" * 60)
    print("COMPLETENESS CHECK")
    print("=" * 60)
    print(f"Total initials expected: {len(initials)}")
    print(f"Missing initials: {len(missing_initials)}")
    print(f"Insufficient trajs: {len(insufficient_initials)}")
    print(f"Valid initials: {len(all_trajs)} / {len(initials)}")
    print(f"Total trajectories loaded: {total_trajs}")
    print(f"Invalid trajectories: {invalid_trajs_count}")

    if missing_initials:
        print(f"\nMissing initial IDs: {missing_initials[:10]}{'...' if len(missing_initials) > 10 else ''}")
    if insufficient_initials:
        print(f"\nInsufficient initial IDs (id, count): {insufficient_initials[:10]}")

    completeness_pass = len(all_trajs) >= len(initials)
    print(f"\nCompleteness: {'PASS' if completeness_pass else 'FAIL'}")

    # --- Diversity ---
    if len(all_trajs) > 0:
        print(f"\n{'=' * 60}")
        print("DIVERSITY EVALUATION")
        print("=" * 60)
        overall, per_initial = compute_diversity_score(all_trajs)
        scores = list(per_initial.values())
        print(f"Your diversity score: {overall:.2f}")
        print(f"Baseline diversity:   {baseline_score:.2f}")
        print(f"Per-initial: min={min(scores):.2f}, max={max(scores):.2f}, "
              f"median={np.median(scores):.2f}")

        diversity_pass = overall > baseline_score
        print(f"\nDiversity > Baseline: {'PASS' if diversity_pass else 'FAIL'}")
    else:
        print("\nNo valid initials for diversity evaluation.")
        diversity_pass = False

    # --- Final ---
    print(f"\n{'=' * 60}")
    print(f"FINAL RESULT: {'PASS' if completeness_pass and diversity_pass else 'FAIL'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
