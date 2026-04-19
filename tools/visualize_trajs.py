"""
Visualize trajectories on top of the 2D point cloud.

Usage:
    python tools/visualize_trajs.py \
    --pointcloud pointcloud_2d.npy \
    --trajs_dir submission_v1 \
    --initials_path data/eval_initials_20.json \
    --initial_ids 0 1 2 3 \
    --output vis.png \
    --show_fail \
    --fail_alpha 0.08

Optional:
    --show_fail          # whether to draw failed trajectories
    --fail_alpha 0.15    # transparency for failed trajectories
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def visualize(
    pointcloud_path,
    trajs_dir,
    initials_path,
    initial_ids,
    output_path,
    trajs_per_initial=20,
    show_fail=False,
    fail_alpha=0.15,
):
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

        ax.scatter(
            pcd[:, 0], pcd[:, 1],
            s=0.3, c="gray", alpha=0.3,
            rasterized=True
        )

        init_dir = os.path.join(trajs_dir, f"initial_{iid}")
        if not os.path.isdir(init_dir):
            ax.set_title(f"Initial {iid}: trajectory dir NOT FOUND")
            continue

        # ---------- 1. 收集文件 ----------
        success_files = []
        for j in range(trajs_per_initial):
            path = os.path.join(init_dir, f"traj_{j}.txt")
            if os.path.exists(path):
                success_files.append(path)

        fail_files = sorted(
            f for f in os.listdir(init_dir)
            if f.startswith("fail_traj_") and f.endswith(".txt")
        )

        num_success = len(success_files)

        # ---------- 2. 决定是否画 fail ----------
        draw_fail = show_fail or (num_success == 0)

        # ---------- 3. 画成功轨迹 ----------
        cmap = plt.cm.Blues(np.linspace(0.3, 1.0, max(1, num_success)))

        for idx, traj_path in enumerate(success_files):
            traj = np.loadtxt(traj_path)
            traj = np.atleast_2d(traj)
            if traj.shape[0] >= 2:
                ax.plot(
                    traj[:, 0], traj[:, 1],
                    color=cmap[min(idx, len(cmap)-1)],
                    linewidth=0.8,
                    alpha=0.7,
                )

        # ---------- 4. 画失败轨迹（条件触发） ----------
        if draw_fail:
            for fail_file in fail_files:
                fail_traj = np.loadtxt(os.path.join(init_dir, fail_file))
                fail_traj = np.atleast_2d(fail_traj)
                if fail_traj.shape[0] >= 2:
                    ax.plot(
                        fail_traj[:, 0], fail_traj[:, 1],
                        color="red",
                        linewidth=0.8,
                        alpha=fail_alpha,
                    )

        # ---------- 5. 起点 + 终点 ----------
        ax.plot(sx, sy, "go", markersize=8, label="Start")
        circle = plt.Circle(
            (cx, cy), 30,
            fill=False,
            color="red",
            linestyle="--",
            linewidth=1.5
        )
        ax.add_patch(circle)
        ax.plot(cx, cy, "r*", markersize=12, label="Target (30m)")

        dist = init.get("distance") or 0

        #  标注 success 数量
        ax.set_title(
            f"Initial {iid} (dist={dist:.0f}m) | success={num_success}",
            fontsize=11
        )

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

    parser.add_argument(
        "--show_fail",
        action="store_true",
        help="whether to visualize failed trajectories",
    )
    parser.add_argument(
        "--fail_alpha",
        type=float,
        default=0.15,
        help="alpha for failed trajectories",
    )

    args = parser.parse_args()

    visualize(
        args.pointcloud,
        args.trajs_dir,
        args.initials_path,
        args.initial_ids,
        args.output,
        args.trajs_per_initial,
        show_fail=args.show_fail,
        fail_alpha=args.fail_alpha,
    )


if __name__ == "__main__":
    main()