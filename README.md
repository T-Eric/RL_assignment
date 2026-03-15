# Assignment: Diverse Trajectory Collection via Reinforcement Learning

## Overview

Training vision-language navigation (VLN) models for autonomous drones requires large datasets of diverse navigation trajectories. In this assignment, you will implement a reinforcement learning agent that navigates a simulated urban environment and collects trajectories that are both **successful** (reaching the target) and **diverse** (taking meaningfully different paths).

You are given 100 start-target pairs in a point-cloud-based 2D urban scene. Your task is to produce at least 20 successful trajectories per pair whose aggregate diversity exceeds the provided baseline. You are free to choose any RL algorithm, reward formulation, or exploration strategy you see fit.

## Motivation

A standard RL agent trained with a typical goal-reaching reward will converge to a single near-optimal policy that repeatedly follows the same path for a given start-target pair. The figure below shows 20 trajectories collected by such a vanilla agent for 4 representative start-target pairs:

<p align="center">
<img src="examples/baseline_visualization.png" width="800">
</p>

The trajectories are nearly identical — the agent has found one "good" route and exploits it. This is a well-known mode collapse problem in policy optimization. For downstream applications such as VLN model training, this lack of diversity means the model only learns a narrow set of navigation behaviors.

**Your challenge is to design methods that break this pattern and produce meaningfully different trajectories while still reaching the target.** Possible directions include (but are not limited to) reward shaping, population-based training, latent-conditioned policies, entropy regularization, or any creative approach you can devise.

## Environment

The environment is a 2D navigation task derived from the [OpenFly](https://github.com/SHAILAB-IPEC/OpenFly-Platform) `airsim_16` scene. The drone flies at a fixed altitude of 15 m, and the scene is represented as a set of 2D obstacle points.

| Property | Value |
|---|---|
| Scene extent | x: [-1200, -700], y: [-600, 100] (meters) |
| Obstacle map | 2D point cloud you produce in the Setup step below |
| Action space | 2D continuous displacement (dx, dy), suggested range [-2, 2] m per axis |
| Success condition | Drone reaches within **30 m** of the target building center |
| Collision condition | Drone reaches within **2 m** of any obstacle point |
| Episode horizon | Up to 300 steps |

**Observation and reward design are up to you.** The point cloud and target location provide sufficient information to construct any observation and reward you need. At minimum, you can query the relative vector to the target center and the relative vector to the nearest obstacle point at each step.

## Setup

### 1. Python Environment

```bash
conda create -n rl_assignment python=3.10 -y
conda activate rl_assignment
pip install torch numpy matplotlib gym
```

### 2. Point Cloud Preparation

You need to produce a 2D obstacle point cloud for the `airsim_16` scene. The raw 3D point cloud is available from the OpenFly project.

**Step 1 — Download the raw point cloud.**

The OpenFly project hosts scene data on HuggingFace. Download the `airsim_16` PCD file from [https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly](https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly), following the instructions under [OpenFly README — Scene data files](https://github.com/SHAILAB-IPEC/OpenFly-Platform#toolchain).

**Step 2 — Process with CloudCompare.**

1. Open CloudCompare and import the PCD file (`File → Open`).
2. **Downsample**: `Edit → Subsample`, select **Space** mode, set minimum point spacing to **0.5–1.0 m**.
3. **Crop XY region**: `Edit → Segment` (scissors icon). Draw a polygon enclosing x ∈ [-1200, -700], y ∈ [-600, 100]. Confirm to keep the selected points.
4. **Filter by height**: switch to a side view and repeat the segment step to keep only z ∈ [10, 20] (the flight-altitude layer at ~15 m).
5. **Export**: `File → Save As` → choose ASCII format (`.txt`). Keep only the X and Y columns.

**Step 3 — Load into Python.**

```python
import numpy as np
points = np.loadtxt("exported_cloud.txt", usecols=(0, 1))  # (N, 2)
np.save("pointcloud_2d.npy", points)
```

### 3. Verify Setup

```python
import numpy as np, json
pcd = np.load("pointcloud_2d.npy")
with open("data/eval_initials_100.json") as f:
    initials = json.load(f)
print(f"Point cloud: {pcd.shape}, Initials: {len(initials)}")
```

## Task

### Requirements

1. **Implement an RL agent** (e.g., PPO, SAC, or any algorithm of your choice) that learns to navigate from a start position to a target building center.

2. **Collect trajectories** for the 100 evaluation start-target pairs defined in `data/eval_initials_100.json`. For each pair, collect at least **20 successful trajectories**.

3. **Maximize trajectory diversity.** Your aggregate diversity score (measured by DTW; see below) must exceed the baseline score of **2311.29**.

### Diversity Metric

Diversity is measured via Dynamic Time Warping (DTW) distance between trajectory pairs:

1. Each trajectory is first **resampled** to uniform arc-length intervals of 2√2 m (matching the fixed step size). This ensures scores are independent of the original temporal resolution.
2. For each start-target pair, compute the mean pairwise DTW distance across all C(20, 2) = 190 trajectory pairs.
3. The overall score is the mean of these per-pair scores across all 100 pairs.

The official evaluation code is provided in `tools/compute_diversity.py`. **Do not modify this file** — it will be used for grading.

### Evaluation Initials

Each entry in `data/eval_initials_100.json` specifies:

```json
{
  "initial_id": 0,
  "x_start": -960.6,
  "y_start": -324.55,
  "target_center_x": -936.12,
  "target_center_y": -153.19,
  "distance": 245.7
}
```

Start-to-target distances range from approximately 200 m to 300 m.

## Provided Files

```
data/
  eval_initials_100.json     100 evaluation start-target pairs
  baseline_trajs/            Baseline trajectories (100 x 20)
  baseline_diversity.json    Baseline diversity score and metadata

tools/
  compute_diversity.py       Official diversity evaluation (do not modify)
  evaluate_submission.py     Full submission checker
  visualize_trajs.py         Trajectory visualization utility

examples/
  baseline_visualization.png Example visualization of baseline trajectories
```

## Submission Format

Submit a directory with the following structure:

```
submission/
  initial_0/
    traj_0.txt
    traj_1.txt
    ...
    traj_19.txt
  initial_1/
    ...
  ...
  initial_99/
    ...
```

Each `traj_X.txt` is a plain-text file where each line contains a space-separated `x y` coordinate pair representing one waypoint of the trajectory.

### Self-Evaluation

```bash
cd tools

# Diversity score
python compute_diversity.py \
    --trajs_dir ../submission \
    --initials_path ../data/eval_initials_100.json

# Full check (completeness + diversity vs. baseline)
python evaluate_submission.py \
    --submission_dir ../submission \
    --initials_path ../data/eval_initials_100.json \
    --baseline_path ../data/baseline_diversity.json
```

### Visualization

```bash
python tools/visualize_trajs.py \
    --pointcloud pointcloud_2d.npy \
    --trajs_dir submission \
    --initials_path data/eval_initials_100.json \
    --initial_ids 0 25 50 75 \
    --output my_trajs.png
```

## Deliverables

1. **Source code** — Your complete implementation including environment, RL algorithm, training script, and any utilities.

2. **Trajectory data** — 100 directories, each containing 20 trajectory files as specified above.

3. **Report** (PDF, 3–5 pages) —
   - Method description: algorithm choice, environment design, reward formulation, and any techniques used to promote diversity.
   - Training analysis: learning curves and key hyperparameters.
   - Results: diversity score, comparison with baseline, and trajectory visualizations for at least 3 start-target pairs.

## Grading

| Component | Weight |
|---|---|
| Implementation quality | 20% |
| Completeness (20 successful trajectories per pair) | 30% |
| Diversity exceeds baseline | 30% |
| Report | 20% |

## Bonus: Vision-Language-Action Model (+20%)

Use your collected trajectories to train a VLA model that predicts navigation actions from first-person-view (FPV) images:

1. Replay your trajectories in the AirSim simulator to capture FPV images at each waypoint.
2. Train a vision-language model (e.g., fine-tune Qwen2.5-VL with LoRA) that takes an FPV image, current pose, and target coordinate as input and predicts the next action.
3. Report the action prediction error on both in-distribution and held-out start-target pairs.

## References

- [OpenFly Platform](https://github.com/SHAILAB-IPEC/OpenFly-Platform)
- J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.

---

## Instructor Reference Materials

> **This section is for the instructor only and should be removed before distributing the assignment to students.**

The `_instructor/` directory contains optional reference materials that may be provided to students at the instructor's discretion:

```
_instructor/
  cluster_centers.json     Building center coordinates {id: [cx, cy]}
  pointcloud_2d.npy        Pre-processed 2D point cloud (99993, 2) — can be
                            distributed if students have difficulty with the
                            CloudCompare processing step.

  env/
    uav_env.py             Complete environment implementation with:
                            - Point cloud loading and nearest-point queries
                            - 4D observation: [nearest_obstacle_rel, target_rel]
                            - Basic reward: distance progress + success bonus
                              (+200) + collision penalty (-100) + step cost (-0.5)
                            - Auto-reset with success-weighted initial sampling
    pointcloud_utils.py    Point cloud loading and nearest-point query utilities

  ppo/
    ppo.py                 Standard PPO with clipped surrogate + value loss
    storage.py             Dict-based rollout storage with GAE

  model.py                 Policy network: 3-layer MLP (4→128) + actor/critic heads,
                            continuous actions via tanh-squashed Gaussian

  train.py                 Training entry point: loads initials, runs PPO loop,
                            saves successful trajectories in the submission format

  vla_example.py           Simplified VLA fine-tuning example (Qwen2.5-VL + LoRA),
                            including ActionTokenizer, data formatting, collate
                            function, and FPV capture hints for AirSim
```

**Baseline statistics:**
- Diversity score: **2311.29** (mean pairwise DTW across 100 initials × 20 trajectories, resampled to 2√2 m intervals)
- Per-initial DTW range: 180 – 12977 (median 1181)
- full details in `data/baseline_diversity.json`
