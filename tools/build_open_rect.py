"""
Build inner-packed open rectangles from open-region cells.

This script:
1. loads open_regions.json (list of open cells)
2. rasterizes them into a binary grid
3. optionally erodes the mask to remove thin/uncertain boundary parts
4. repeatedly extracts the largest all-ones axis-aligned rectangle
5. removes that rectangle from the mask
6. appends manually specified open rectangles
7. saves rectangles to json and renders a visualization

Usage example:
    python tools/build_open_rect.py \
      --open_regions open_regions.json \
      --pointcloud pointcloud_2d.npy \
      --output_json open_rects.json \
      --output_png open_rects.png \
      --cell_size 2.5 \
      --erode_kernel 3 \
      --min_rect_area_cells 80 \
      --min_rect_width 20 \
      --min_rect_height 20 \
      --max_rects 20
"""

import argparse
import json
from typing import List, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# Manual open-rectangle container
# Edit this list in the future if you want more hard-coded areas.
# Each item is [xmin, xmax, ymin, ymax].
# ============================================================
MANUAL_OPEN_RECTS = [
    [-1300.0, -1205.0, -550.0, 50.0],
]


def make_manual_rect(xmin: float, xmax: float, ymin: float, ymax: float, region_id=None) -> Dict:
    """
    Build a full rectangle entry from four coordinates.
    """
    xmin = float(xmin)
    xmax = float(xmax)
    ymin = float(ymin)
    ymax = float(ymax)

    if xmax <= xmin:
        raise ValueError(f"Invalid manual rect: xmax <= xmin ({xmax} <= {xmin})")
    if ymax <= ymin:
        raise ValueError(f"Invalid manual rect: ymax <= ymin ({ymax} <= {ymin})")

    width = xmax - xmin
    height = ymax - ymin

    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "width": width,
        "height": height,
        "center": [float((xmin + xmax) / 2.0), float((ymin + ymax) / 2.0)],
        "area_cells": None,
        "width_cells": None,
        "height_cells": None,
        "region_id": region_id,
        "source": "manual",
    }


def load_open_cells(path: str) -> np.ndarray:
    with open(path, "r") as f:
        data = json.load(f)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Invalid open_regions file: {path}")
    return arr


def build_grid(open_cells: np.ndarray, cell_size: float):
    """
    Map open cells to a dense binary grid.

    Returns:
        grid: (H, W) uint8
        min_ix, min_iy: integer offsets so that
            world_x = (ix + min_ix) * cell_size
            world_y = (iy + min_iy) * cell_size
    """
    ij = np.floor(open_cells / cell_size + 1e-8).astype(np.int64)

    min_ix = int(ij[:, 0].min())
    min_iy = int(ij[:, 1].min())

    local = ij.copy()
    local[:, 0] -= min_ix
    local[:, 1] -= min_iy

    h = int(local[:, 0].max()) + 1
    w = int(local[:, 1].max()) + 1

    grid = np.zeros((h, w), dtype=np.uint8)
    for i, j in local:
        grid[i, j] = 1

    return grid, min_ix, min_iy


def erode_binary_grid(grid: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Simple binary erosion with an all-ones square kernel.
    Keeps cell=1 only if the whole neighborhood is 1.
    """
    if kernel_size <= 1:
        return grid.copy()

    pad = kernel_size // 2
    h, w = grid.shape
    out = np.zeros_like(grid)

    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            patch = grid[i - pad:i + pad + 1, j - pad:j + pad + 1]
            if np.all(patch == 1):
                out[i, j] = 1
    return out


def largest_rectangle_in_histogram(heights: List[int]):
    """
    Standard monotonic-stack largest rectangle in histogram.

    Returns:
        area, left, right, height
    where [left, right] is inclusive in column index.
    """
    stack = []
    best_area = 0
    best_left = 0
    best_right = -1
    best_height = 0

    extended = heights + [0]
    for idx, h in enumerate(extended):
        start = idx
        while stack and stack[-1][1] > h:
            left_idx, prev_h = stack.pop()
            area = prev_h * (idx - left_idx)
            if area > best_area:
                best_area = area
                best_left = left_idx
                best_right = idx - 1
                best_height = prev_h
            start = left_idx
        if not stack or stack[-1][1] < h:
            stack.append((start, h))

    return best_area, best_left, best_right, best_height


def extract_largest_all_ones_rectangle(mask: np.ndarray):
    """
    Find the largest all-ones axis-aligned rectangle in a binary mask.

    Returns:
        rect dict with keys:
            top, bottom, left, right, area, width, height
        or None if no ones exist.
    """
    h, w = mask.shape
    heights = [0] * w

    best = None
    best_area = 0

    for r in range(h):
        for c in range(w):
            if mask[r, c] == 1:
                heights[c] += 1
            else:
                heights[c] = 0

        area, left, right, rect_h = largest_rectangle_in_histogram(heights)
        if area > best_area and rect_h > 0:
            bottom = r
            top = r - rect_h + 1
            best_area = area
            best = {
                "top": int(top),
                "bottom": int(bottom),
                "left": int(left),
                "right": int(right),
                "area": int(area),
                "width_cells": int(right - left + 1),
                "height_cells": int(rect_h),
            }

    return best


def rect_grid_to_world(rect: Dict, cell_size: float, min_ix: int, min_iy: int):
    """
    Convert grid-index rectangle to world-coordinate rectangle.
    """
    top = rect["top"]
    bottom = rect["bottom"]
    left = rect["left"]
    right = rect["right"]

    xmin = (top + min_ix) * cell_size
    xmax = (bottom + 1 + min_ix) * cell_size
    ymin = (left + min_iy) * cell_size
    ymax = (right + 1 + min_iy) * cell_size

    width = xmax - xmin
    height = ymax - ymin

    return {
        "xmin": float(xmin),
        "xmax": float(xmax),
        "ymin": float(ymin),
        "ymax": float(ymax),
        "width": float(width),
        "height": float(height),
        "center": [float((xmin + xmax) / 2.0), float((ymin + ymax) / 2.0)],
        "area_cells": int(rect["area"]),
        "width_cells": int(rect["width_cells"]),
        "height_cells": int(rect["height_cells"]),
        "source": "auto",
    }


def remove_rectangle_from_mask(mask: np.ndarray, rect: Dict):
    mask[rect["top"]:rect["bottom"] + 1, rect["left"]:rect["right"] + 1] = 0


def extract_rectangles(
    mask: np.ndarray,
    cell_size: float,
    min_ix: int,
    min_iy: int,
    min_rect_area_cells: int,
    min_rect_width: float,
    min_rect_height: float,
    max_rects: int,
):
    """
    Repeatedly extract largest inner rectangles.
    """
    work = mask.copy()
    rects = []

    for _ in range(max_rects):
        rect = extract_largest_all_ones_rectangle(work)
        if rect is None:
            break
        if rect["area"] < min_rect_area_cells:
            break

        world_rect = rect_grid_to_world(rect, cell_size, min_ix, min_iy)

        if world_rect["width"] < min_rect_width or world_rect["height"] < min_rect_height:
            remove_rectangle_from_mask(work, rect)
            continue

        rects.append(world_rect)
        remove_rectangle_from_mask(work, rect)

    return rects, work


def build_manual_rects():
    """
    Convert MANUAL_OPEN_RECTS into full entries.
    """
    manual_rects = []
    for coords in MANUAL_OPEN_RECTS:
        if len(coords) != 4:
            raise ValueError(f"Manual rect must have 4 values, got: {coords}")
        xmin, xmax, ymin, ymax = coords
        manual_rects.append(make_manual_rect(xmin, xmax, ymin, ymax))
    return manual_rects


def add_reference_grid(ax, xlim, ylim, spacing=10.0, color="green", alpha=0.35, size=4):
    """
    Add green reference dots every `spacing` meters.
    """
    xs = np.arange(np.floor(xlim[0] / spacing) * spacing, xlim[1] + spacing, spacing)
    ys = np.arange(np.floor(ylim[0] / spacing) * spacing, ylim[1] + spacing, spacing)

    gx, gy = np.meshgrid(xs, ys)
    ax.scatter(
        gx.ravel(), gy.ravel(),
        s=size, c=color, alpha=alpha, marker="o", linewidths=0,
        label="10m ref grid"
    )


def visualize(
    pointcloud_path: str,
    open_cells: np.ndarray,
    rects: List[Dict],
    output_png: str,
):
    pcd = np.load(pointcloud_path)

    fig, ax = plt.subplots(figsize=(10, 10))

    # reference dots first, so they stay in the background
    xlim = (pcd[:, 0].min(), pcd[:, 0].max())
    ylim = (pcd[:, 1].min(), pcd[:, 1].max())
    add_reference_grid(ax, xlim=xlim, ylim=ylim, spacing=10.0)

    ax.scatter(pcd[:, 0], pcd[:, 1], s=0.2, alpha=0.2, color="gray", label="pointcloud")
    ax.scatter(open_cells[:, 0], open_cells[:, 1], s=2, alpha=0.15, color="blue", label="open cells")

    for rect in rects:
        edgecolor = "green" if rect.get("source") == "manual" else "red"
        textcolor = "darkgreen" if rect.get("source") == "manual" else "darkred"

        patch = Rectangle(
            (rect["xmin"], rect["ymin"]),
            rect["width"],
            rect["height"],
            fill=False,
            edgecolor=edgecolor,
            linewidth=2.0,
        )
        ax.add_patch(patch)
        ax.text(
            rect["center"][0],
            rect["center"][1],
            str(rect["region_id"]),
            color=textcolor,
            fontsize=8,
            ha="center",
            va="center",
        )

    ax.set_title("Open Rectangles (max inner rectangles + manual rects)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180, bbox_inches="tight")
    print(f"Saved visualization to {output_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open_regions", required=True)
    parser.add_argument("--pointcloud", required=True)
    parser.add_argument("--output_json", default="open_rects.json")
    parser.add_argument("--output_png", default="open_rects.png")

    parser.add_argument("--cell_size", type=float, default=2.5)
    parser.add_argument("--erode_kernel", type=int, default=3)

    parser.add_argument("--min_rect_area_cells", type=int, default=80)
    parser.add_argument("--min_rect_width", type=float, default=20.0)
    parser.add_argument("--min_rect_height", type=float, default=20.0)
    parser.add_argument("--max_rects", type=int, default=20)

    args = parser.parse_args()

    open_cells = load_open_cells(args.open_regions)
    print(f"Loaded {len(open_cells)} open cells from {args.open_regions}")

    grid, min_ix, min_iy = build_grid(open_cells, args.cell_size)
    print(f"Built grid with shape {grid.shape}")

    eroded = erode_binary_grid(grid, args.erode_kernel)
    print(f"Eroded with kernel size {args.erode_kernel}")

    auto_rects, remaining = extract_rectangles(
        mask=eroded,
        cell_size=args.cell_size,
        min_ix=min_ix,
        min_iy=min_iy,
        min_rect_area_cells=args.min_rect_area_cells,
        min_rect_width=args.min_rect_width,
        min_rect_height=args.min_rect_height,
        max_rects=args.max_rects,
    )

    manual_rects = build_manual_rects()

    # assign region ids after combining
    rects = auto_rects + manual_rects
    for i, rect in enumerate(rects):
        rect["region_id"] = i

    with open(args.output_json, "w") as f:
        json.dump(rects, f, indent=2)
    print(f"Saved {len(rects)} rectangles to {args.output_json}")

    visualize(
        pointcloud_path=args.pointcloud,
        open_cells=open_cells,
        rects=rects,
        output_png=args.output_png,
    )


if __name__ == "__main__":
    main()