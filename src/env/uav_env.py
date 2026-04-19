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
from .pointcloud_utils import load_pointcloud_transposed, find_nearest_point, build_pointcloud_index, find_nearest_point_kdtree, compute_directional_distances


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

        # observation augmentation
        self.num_dirs = params.get("num_dirs", 16)
        self.max_obs_range = params.get("max_obs_range", 80.0)

        # visitation rewards
        self.cell_size = params.get("cell_size", 5.0)

        self.use_episode_vis = params.get("use_episode_vis", False)
        self.use_history_vis = params.get("use_history_vis", False)

        self.visit_bonus = params.get("visit_bonus", 1.0)
        self.cell_repeat_penalty = params.get("cell_repeat_penalty", 0.0)
        # self.history_bonus_coef=params.get("history_bonus_coef", 0.0)
        self.global_history_bonus_coef = params.get(
            "global_history_bonus_coef", 0.0)
        self.latent_history_bonus_coef = params.get(
            "latent_history_bonus_coef", 0.0)

        # latent z
        self.latent_dim = params.get("latent_dim", 4)
        self.curr_latent = None

        # soft collision / safety shaping
        self.use_soft_collision = params.get("use_soft_collision", False)
        self.safe_distance = params.get("safe_distance", 6.0)
        self.safe_penalty_coef = params.get("safe_penalty_coef", 0.1)
        
        # latent-wise route bias
        self.use_route_bias = params.get("use_route_bias", False)
        self.route_bias_scale = params.get("route_bias_scale", 40.0) # goal bias
        self.route_bias_table = self._build_route_bias_table()

        # Load point cloud: (2, N)
        print(f"Loading point cloud from {pointcloud_path}...")
        # self.all_points = load_pointcloud_transposed(pointcloud_path)
        self.all_points, self.kd_tree = build_pointcloud_index(
            npy_path=pointcloud_path)
        print(f"Loaded {self.all_points.shape[0]} points")

        # obs shape: +latent
        base_sensor_dim = self.num_dirs + 4
        if self.use_route_bias and self.latent_dim > 0:
            base_sensor_dim += 2

        if self.latent_dim > 0:
            self.observation_shape = {
                "sensor": (base_sensor_dim,),
                "latent": (self.latent_dim,),
            }
        else:
            self.observation_shape = {
                "sensor": (base_sensor_dim,),
            }
        self.action_shape = (2,)

        self.curr_pose = None
        self.target_center = None
        self.step_count = 0

        # visitation conditions
        self.curr_initial_id = None
        self.visited_cells_ep = set()
        
        # global per-initial coverage
        self.global_coverage_maps = {
            init["initial_id"]: {} for init in self.initials
        }

        # latent-specific per-initial coverage
        if self.latent_dim > 0:
            self.latent_coverage_maps = {
                init["initial_id"]: {
                    z: {} for z in range(self.latent_dim)
                }
                for init in self.initials
            }
        else:
            self.latent_coverage_maps = {
                init["initial_id"]: {}
                for init in self.initials
            }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, initial_pose, target_center, initial_id=None, latent=None):
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

        # visitation condition reset
        self.curr_initial_id = initial_id
        self.curr_latent = latent
        self.visited_cells_ep = set()

        # episode summary stats
        self.path_length = 0.0
        self.turn_sum = 0.0
        self.prev_heading = None

        start_xy = np.array(
            [initial_pose[0], initial_pose[1]], dtype=np.float64)
        start_cell = self._xy_to_cell(start_xy)
        self.visited_cells_ep.add(start_cell)  # not to reward on start cell

        obs = self._get_obs()
        sensor = obs["sensor"].cpu().numpy()
        self._prev_target_dist = np.linalg.norm(
            sensor[self.num_dirs+2:self.num_dirs+4])
        return obs

    def step(self, action):
        dx = action.squeeze(0).cpu()[0].item()
        dy = action.squeeze(0).cpu()[1].item()

        x, y, _ = self.curr_pose
        step_len = float(np.sqrt(dx * dx + dy * dy))
        curr_heading = float(np.arctan2(dy, dx))

        self.curr_pose = np.array([x + dx, y + dy, curr_heading])
        curr_xy = np.array(
            [self.curr_pose[0], self.curr_pose[1]], dtype=np.float64)
        curr_cell = self._xy_to_cell(curr_xy)

        # summary stats update
        self.path_length += step_len
        if self.prev_heading is not None:
            dtheta = curr_heading - self.prev_heading
            # wrap to [-pi, pi]
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
            self.turn_sum += abs(dtheta)
        self.prev_heading = curr_heading

        obs = self._get_obs()
        sensor = obs["sensor"].cpu().numpy()

        dir_dists = sensor[:self.num_dirs]
        nearest_obs_rel = sensor[self.num_dirs:self.num_dirs+2]
        target_rel = sensor[self.num_dirs+2:self.num_dirs+4]
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

        # soft collision / safety shaping
        if self.use_soft_collision and obstacle_dist < self.safe_distance:
            reward -= self.safe_penalty_coef * \
                (self.safe_distance - obstacle_dist) ** 2

        # visitation rewards
        visit_reward = 0.0
        if self.use_episode_vis:
            if curr_cell not in self.visited_cells_ep:
                visit_reward += self.visit_bonus
                self.visited_cells_ep.add(curr_cell)
            else:
                visit_reward -= self.cell_repeat_penalty


        # historical coverage reward
        history_reward = 0.0

        if self.use_history_vis and self.curr_initial_id is not None:
            # global per-initial history reward
            global_map = self.global_coverage_maps[self.curr_initial_id]
            global_count = global_map.get(curr_cell, 0)
            history_reward += self.global_history_bonus_coef / \
                np.sqrt(1.0 + global_count)
            # latent-specific per-initial history reward
            if self.curr_latent is not None:
                latent_id = int(np.argmax(self.curr_latent))
                latent_map = self.latent_coverage_maps[self.curr_initial_id][latent_id]
                latent_count = latent_map.get(curr_cell, 0)
                history_reward += self.latent_history_bonus_coef / \
                    np.sqrt(1.0 + latent_count)

        self._prev_target_dist = target_dist

        reward += visit_reward+history_reward

        self.episode_reward += reward

        if done:
            info["episode"] = {"r": self.episode_reward}
            info["initial_pose"] = copy.deepcopy(self.initial_pose)
            info["target_center"] = copy.deepcopy(self.target_center)

            # episode summary stats
            final_xy = np.array(
                [self.curr_pose[0], self.curr_pose[1]], dtype=np.float64)
            start_xy = np.array(
                [self.initial_pose[0], self.initial_pose[1]], dtype=np.float64)
            target_xy = np.array(
                [self.target_center[0], self.target_center[1]], dtype=np.float64)

            disp = final_xy - start_xy
            goal_rel_final = target_xy - final_xy
            net_disp = np.linalg.norm(disp)
            straightness = net_disp / (self.path_length + 1e-6)
            visited_count = float(len(self.visited_cells_ep))

            summary = np.array([
                disp[0] / 300.0,
                disp[1] / 300.0,
                goal_rel_final[0] / 300.0,
                goal_rel_final[1] / 300.0,
                self.path_length / 300.0,
                net_disp / 300.0,
                straightness,
                self.turn_sum / np.pi,
                visited_count / 100.0,
            ], dtype=np.float32)

            info["episode_summary"] = summary

            if self.curr_latent is not None:
                info["latent_id"] = int(np.argmax(self.curr_latent))

            if info.get("won", False):
                # coverage_map = self.coverage_maps[self.curr_initial_id]
                # for cell in self.visited_cells_ep:
                #     coverage_map[cell] = coverage_map.get(cell, 0) + 1
                
                # 1) update global per-initial map
                global_map = self.global_coverage_maps[self.curr_initial_id]
                for cell in self.visited_cells_ep:
                    global_map[cell] = global_map.get(cell, 0) + 1

                # 2) update latent-specific per-initial map
                if self.curr_latent is not None:
                    latent_id = int(np.argmax(self.curr_latent))
                    latent_map = self.latent_coverage_maps[self.curr_initial_id][latent_id]
                    for cell in self.visited_cells_ep:
                        latent_map[cell] = latent_map.get(cell, 0) + 1

            # Auto-reset: sample next initial
            # self._sample_next_initial(info.get("won", False))

            # Don't auto-reset: do it in the training loop to allow logging final state
            # TODO: deleted auto reset! Redo it in training loop
            # obs = self.reset(
            #     initial_pose=np.array([
            #         self.initials[self.initial_index]["x_start"],
            #         self.initials[self.initial_index]["y_start"],
            #         0.0
            #     ]),
            #     target_center=np.array([
            #         self.initials[self.initial_index]["target_center_x"],
            #         self.initials[self.initial_index]["target_center_y"],
            #     ])
            # )

        return obs, torch.tensor([reward], device=self.device), [done], [info]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------


    def _get_obs(self):
        xy = np.array([self.curr_pose[0], self.curr_pose[1]])

        nearest_obs_rel, _ = find_nearest_point_kdtree(
            self.all_points, self.kd_tree, xy)
        target_rel = self.target_center - xy

        directional_dists = compute_directional_distances(
            self.all_points,
            xy,
            num_dirs=self.num_dirs,
            max_range=self.max_obs_range,
        )

        parts = [
            directional_dists.astype(np.float32),
            nearest_obs_rel.astype(np.float32),
            target_rel.astype(np.float32),
        ]

        if self.use_route_bias and self.latent_dim > 0:
            if self.curr_latent is None:
                latent_id = 0
            else:
                latent_id = int(np.argmax(self.curr_latent))

            bias_dir = self.route_bias_table[latent_id]   # unit vector in R^2
            biased_target_rel = target_rel + self.route_bias_scale * bias_dir
            parts.append(biased_target_rel.astype(np.float32))

        sensor = np.concatenate(parts).astype(np.float32)

        obs = {
            "sensor": torch.tensor(sensor, device=self.device),
        }

        if self.latent_dim > 0:
            if self.curr_latent is None:
                latent = np.zeros(self.latent_dim, dtype=np.float32)
            else:
                latent = self.curr_latent.astype(np.float32)
            obs["latent"] = torch.tensor(latent, device=self.device)

        return obs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_next_initial(self, was_success):
        if was_success:
            self.success_counts[self.initial_index] += 1

        probs = 1.0 / (np.log(np.array(self.success_counts) + 2))
        probs = probs / probs.sum()
        self.initial_index = np.random.choice(len(self.initials), p=probs)

    # ------------------------------------------------------------------
    # Visitation tracking
    # ------------------------------------------------------------------
    def _xy_to_cell(self, xy):
        x, y = xy
        cx = int(np.floor(x / self.cell_size))
        cy = int(np.floor(y / self.cell_size))
        return (cx, cy)

    # uniformly distributed bias directions
    def _build_route_bias_table(self):
        if self.latent_dim <= 0:
            return None

        table = np.zeros((self.latent_dim, 2), dtype=np.float32)
        for z in range(self.latent_dim):
            theta = 2.0 * np.pi * z / self.latent_dim
            table[z, 0] = np.cos(theta)
            table[z, 1] = np.sin(theta)
        return table
