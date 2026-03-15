"""
Visualize trajectories on top of the 2D point cloud.

Usage:
    python visualize_trajs.py \
        --pointcloud ../data/pointcloud_2d.npy \
        --trajs_dir ../data/baseline_trajs \
        --initials_path ../data/eval_initials_100.json \
        --initial_ids 0 25 50 75 \
        --output visualization.png
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def visualize(pointcloud_path, trajs_dir, initials_path, initial_ids,
              output_path, trajs_per_initial=20):
    pcd = np.load(pointcloud_path)
    with open(initials_path) as f:
        initials = json.load(f)

    init_map = {init["initial_id"]: init for init in initials}

    n = len(initial_ids)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(9 * cols, 8 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax_idx, iid in enumerate(initial_ids):
        ax = axes[ax_idx]
        init = init_map.get(iid)
        if init is None:
            ax.set_title(f"Initial {iid}: NOT FOUND")
            continue

        cx, cy = init["target_center_x"], init["target_center_y"]
        sx, sy = init["x_start"], init["y_start"]

        ax.scatter(pcd[:, 0], pcd[:, 1], s=0.3, c="gray", alpha=0.3,
                   rasterized=True)

        cmap = plt.cm.Blues(np.linspace(0.3, 1.0, trajs_per_initial))
        init_dir = os.path.join(trajs_dir, f"initial_{iid}")
        for j in range(trajs_per_initial):
            traj_path = os.path.join(init_dir, f"traj_{j}.txt")
            if os.path.exists(traj_path):
                traj = np.loadtxt(traj_path)
                ax.plot(traj[:, 0], traj[:, 1], color=cmap[j],
                        linewidth=0.8, alpha=0.7)

        ax.plot(sx, sy, "go", markersize=8, label="Start")
        circle = plt.Circle((cx, cy), 30, fill=False, color="red",
                             linestyle="--", linewidth=1.5)
        ax.add_patch(circle)
        ax.plot(cx, cy, "r*", markersize=12, label="Target (30m)")

        dist = init.get("distance") or 0
        ax.set_title(f"Initial {iid}  (dist={dist:.0f}m)", fontsize=11)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.2)

    for ax_idx in range(n, len(axes)):
        axes[ax_idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointcloud", required=True)
    parser.add_argument("--trajs_dir", required=True)
    parser.add_argument("--initials_path", required=True)
    parser.add_argument("--initial_ids", type=int, nargs="+", default=[0, 25, 50, 75])
    parser.add_argument("--output", default="visualization.png")
    parser.add_argument("--trajs_per_initial", type=int, default=20)
    args = parser.parse_args()

    visualize(args.pointcloud, args.trajs_dir, args.initials_path,
              args.initial_ids, args.output, args.trajs_per_initial)


if __name__ == "__main__":
    main()
