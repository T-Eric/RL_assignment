"""
Simplified UAV Navigation Environment for RL trajectory collection.

The drone navigates in a 2D plane at fixed altitude (15m) using point cloud
data as the map. The goal is to reach within 30m of a target building center
while avoiding collisions with obstacles (point cloud points).

Observation: 4D sensor vector
  - [0:2]: relative offset to nearest obstacle point (dx, dy)
  - [2:4]: relative offset to target center (dx, dy)

Reward: distance-based progress + success bonus + collision penalty

Action: 2D continuous (dx, dy), each in [-action_limit, action_limit]
"""

import gym
import copy
import torch
import numpy as np
from .pointcloud_utils import load_pointcloud_transposed, find_nearest_point, build_pointcloud_index, find_nearest_point_kdtree


class UAVNavEnv(gym.Env):

    DEFAULT_PARAMS = {
        "max_steps": 300,
        "success_radius": 30.0,
        "collision_threshold": 2.0,
        "action_limit": [2.0, 2.0],
    }

    def __init__(self, pointcloud_path, env_params=None, save_dir=None,
                 device=None, initials=None):
        super().__init__()
        params = {**self.DEFAULT_PARAMS, **(env_params or {})}

        self.max_steps = params["max_steps"]
        self.success_radius = params["success_radius"]
        self.collision_threshold = params["collision_threshold"]
        self.action_limit = np.array(params["action_limit"])
        self.device = device or torch.device("cpu")
        self.save_dir = save_dir

        self.initials = initials or []
        self.initial_index = 0
        self.success_counts = [0] * max(len(self.initials), 1)

        # Load point cloud: (2, N)
        print(f"Loading point cloud from {pointcloud_path}...")
        # self.all_points = load_pointcloud_transposed(pointcloud_path)
        self.all_points, self.kd_tree = build_pointcloud_index(npy_path=pointcloud_path)
        print(f"Loaded {self.all_points.shape[1]} points")

        self.observation_shape = {"sensor": (4,)}
        self.action_shape = (2,)

        self.curr_pose = None
        self.target_center = None
        self.step_count = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, initial_pose, target_center):
        """
        Reset environment for a new episode.

        Args:
            initial_pose: np.array [x, y, 0.0]
            target_center: np.array [cx, cy] — target building center
        """
        self.curr_pose = np.array(initial_pose, dtype=np.float64)
        self.target_center = np.array(target_center, dtype=np.float64)
        self.step_count = 0
        self.episode_reward = 0.0
        self.initial_pose = initial_pose.copy()
        self._prev_target_dist = None

        obs = self._get_obs()
        sensor = obs["sensor"].cpu().numpy()
        self._prev_target_dist = np.linalg.norm(sensor[2:4])
        return obs

    def step(self, action):
        dx = action.squeeze(0).cpu()[0].item()
        dy = action.squeeze(0).cpu()[1].item()
        x, y, _ = self.curr_pose
        self.curr_pose = np.array([x + dx, y + dy, np.arctan2(dy, dx)])

        obs = self._get_obs()
        sensor = obs["sensor"].cpu().numpy()

        nearest_obs_rel = sensor[:2]
        target_rel = sensor[2:4]
        obstacle_dist = np.linalg.norm(nearest_obs_rel)
        target_dist = np.linalg.norm(target_rel)

        # --- Check termination conditions ---
        done = False
        info = {}

        # Collision
        if obstacle_dist <= self.collision_threshold:
            done = True
            info["won"] = False

        # Success
        if target_dist <= self.success_radius:
            done = True
            info["won"] = True

        self.step_count += 1
        if self.step_count >= self.max_steps and not done:
            done = True
            info["won"] = False

        # --- Reward computation ---
        reward = 0.0

        # Progress toward target (positive when getting closer)
        if self._prev_target_dist is not None:
            reward += (self._prev_target_dist - target_dist)

        # Collision penalty
        if obstacle_dist <= self.collision_threshold:
            reward -= 100.0

        # Success bonus
        if info.get("won", False):
            reward += 200.0

        # Step penalty
        reward -= 0.5

        self._prev_target_dist = target_dist

        self.episode_reward += reward

        if done:
            info["episode"] = {"r": self.episode_reward}
            info["initial_pose"] = copy.deepcopy(self.initial_pose)
            info["target_center"] = copy.deepcopy(self.target_center)

            # Auto-reset: sample next initial
            self._sample_next_initial(info.get("won", False))
            obs = self.reset(
                initial_pose=np.array([
                    self.initials[self.initial_index]["x_start"],
                    self.initials[self.initial_index]["y_start"],
                    0.0
                ]),
                target_center=np.array([
                    self.initials[self.initial_index]["target_center_x"],
                    self.initials[self.initial_index]["target_center_y"],
                ])
            )

        return obs, torch.tensor([reward]), [done], [info]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        xy = np.array([self.curr_pose[0], self.curr_pose[1]])

        nearest_obs_rel, _ = find_nearest_point_kdtree(self.all_points, self.kd_tree, xy)
        target_rel = self.target_center - xy

        sensor = np.concatenate([nearest_obs_rel, target_rel]).astype(np.float32)

        return {
            "sensor": torch.tensor(sensor, device=self.device),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_next_initial(self, was_success):
        if was_success:
            self.success_counts[self.initial_index] += 1

        probs = 1.0 / (np.log(np.array(self.success_counts) + 2))
        probs = probs / probs.sum()
        self.initial_index = np.random.choice(len(self.initials), p=probs)
