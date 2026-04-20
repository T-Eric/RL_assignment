import numpy as np
from tqdm import tqdm
from src.env.pointcloud_utils import compute_directional_distances
import json
import matplotlib.pyplot as plt
import os


def detect_open_regions(
    pointcloud,
    num_dirs=16,
    max_range=80.0,
    open_thresh=20.0,
    open_ratio_thresh=0.6,
    cell_size=2.5,
):
    min_xy = pointcloud.min(axis=0)
    max_xy = pointcloud.max(axis=0)

    xs = np.arange(min_xy[0], max_xy[0], cell_size)
    ys = np.arange(min_xy[1], max_xy[1], cell_size)

    open_cells = []

    for x in tqdm(xs):
        for y in ys:
            xy = np.array([x, y], dtype=np.float32)

            dists = compute_directional_distances(
                pointcloud,
                xy,
                num_dirs=num_dirs,
                max_range=max_range,
            )

            open_ratio = np.mean(dists > open_thresh)

            if open_ratio > open_ratio_thresh:
                open_cells.append((float(x), float(y)))

    return open_cells


def main():
    pcd_path = "pointcloud_2d.npy"
    if not os.path.exists(pcd_path):
        pcd_path = os.path.join(os.path.dirname(__file__), "../pointcloud_2d.npy")
    pcd = np.load(pcd_path)

    print("开始检测开阔区域...")
    open_cells = detect_open_regions(
        pcd,
        num_dirs=16,
        max_range=80.0,
        open_thresh=10.0,
        open_ratio_thresh=0.4,
        cell_size=2.5,
    )

    with open("open_regions.json", "w") as f:
        json.dump(open_cells, f)

    plt.figure(figsize=(10, 10))
    plt.scatter(pcd[:, 0], pcd[:, 1], s=0.2, alpha=0.25, label="pointcloud")

    if len(open_cells) > 0:
        ox = np.array(open_cells)
        plt.scatter(ox[:, 0], ox[:, 1], c="blue", s=4, alpha=0.6, label="open")

    plt.legend()
    plt.title("Detected Open Regions")
    plt.gca().set_aspect("equal")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("open_regions.png", dpi=180)
    print("检测完成，结果已保存为 open_regions.json、open_regions.png")


if __name__ == "__main__":
    main()