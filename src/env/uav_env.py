"""
Simplified UAV Navigation Environment for RL trajectory collection.

The drone navigates in a 2D plane at fixed altitude (15m) using point cloud
data as the map. The goal is to reach within success_radius of a target
building center while avoiding collisions with obstacle points.

Observation:
  sensor = [directional obstacle distances, nearest obstacle rel xy, target rel xy,
            optional route-biased target rel xy, optional stuck flag + escape dir]
  latent = one-hot latent vector (optional)

Action:
  2D continuous displacement (dx, dy), each in [-action_limit, action_limit]
"""

import copy
import gym
import numpy as np
import torch

from .pointcloud_utils import (
    build_pointcloud_index,
    compute_directional_distances,
    find_nearest_point_kdtree,
)


class UAVNavEnv(gym.Env):
    DEFAULT_PARAMS = {
        # core task
        "max_steps": 300,
        "success_radius": 30.0,
        "collision_threshold": 2.0,
        "action_limit": [2.0, 2.0],

        # observation
        "num_dirs": 16,
        "max_obs_range": 80.0,

        # episode visitation
        "cell_size": 5.0,
        "use_episode_vis": False,
        "visit_bonus": 1.0,
        "cell_repeat_penalty": 0.0,

        # history visitation
        "use_history_vis": False,
        "global_history_bonus_coef": 0.0,
        "latent_history_bonus_coef": 0.0,

        # position-aware history
        "history_decay_beta": 0.5,
        "history_start_relax_radius": 45.0,
        "history_goal_relax_radius": 45.0,
        "use_latent_pseudo_goal": True,
        "latent_goal_shift_scale": 35.0,
        "use_inter_latent_repulsion": False,
        "inter_latent_repulsion_coef": 0.0,
        "failure_history_weight": 0.0,
        "history_pos_weight_cap": 1.0,

        # initial sampling
        "initial_success_target": 60.0,
        "initial_sampling_gamma": 0.8,
        "zero_success_boost": 1.0,

        # latent
        "latent_dim": 4,

        # safety
        "use_soft_collision": False,
        "safe_distance": 6.0,
        "safe_penalty_coef": 0.1,

        # near-goal shaping
        "goal_relax_outer_radius": 80.0,
        "goal_relax_inner_radius": 40.0,
        "goal_soft_collision_min_scale": 0.25,
        "goal_progress_max_scale": 1.8,

        # route bias
        "use_route_bias": False,
        "route_bias_scale": 20.0,

        # stuck escape
        "use_stuck_escape": False,
        "escape_lookahead": 12.0,
        "stuck_window": 12,
        "stuck_progress_threshold": 6.0,
        "stuck_unique_ratio_threshold": 0.35,
        "escape_open_length": 15.0,
        "escape_open_weight": 1.0,
        "escape_goal_weight": 0.8,
    }

    def __init__(self, pointcloud_path, env_params=None, save_dir=None, device=None, initials=None):
        super().__init__()
        params = {**self.DEFAULT_PARAMS, **(env_params or {})}

        # core task
        self.max_steps = params["max_steps"]
        self.success_radius = params["success_radius"]
        self.collision_threshold = params["collision_threshold"]
        self.action_limit = np.array(params["action_limit"], dtype=np.float32)

        # infra
        self.device = device or torch.device("cpu")
        self.save_dir = save_dir
        self.initials = initials or []
        self.initial_index = 0
        self.success_counts = [0] * max(len(self.initials), 1)

        # observation
        self.num_dirs = params["num_dirs"]
        self.max_obs_range = params["max_obs_range"]

        # episode visitation
        self.cell_size = params["cell_size"]
        self.use_episode_vis = params["use_episode_vis"]
        self.visit_bonus = params["visit_bonus"]
        self.cell_repeat_penalty = params["cell_repeat_penalty"]

        # history visitation
        self.use_history_vis = params["use_history_vis"]
        self.global_history_bonus_coef = params["global_history_bonus_coef"]
        self.latent_history_bonus_coef = params["latent_history_bonus_coef"]

        # position-aware history
        self.history_decay_beta = params["history_decay_beta"]
        self.history_start_relax_radius = params["history_start_relax_radius"]
        self.history_goal_relax_radius = params["history_goal_relax_radius"]
        self.use_latent_pseudo_goal = params["use_latent_pseudo_goal"]
        self.latent_goal_shift_scale = params["latent_goal_shift_scale"]
        self.use_inter_latent_repulsion = params["use_inter_latent_repulsion"]
        self.inter_latent_repulsion_coef = params["inter_latent_repulsion_coef"]
        self.failure_history_weight = params["failure_history_weight"]
        self.history_pos_weight_cap = params["history_pos_weight_cap"]

        # initial sampling
        self.initial_success_target = params["initial_success_target"]
        self.initial_sampling_gamma = params["initial_sampling_gamma"]
        self.zero_success_boost = params["zero_success_boost"]

        # latent
        self.latent_dim = params["latent_dim"]
        self.curr_latent = None

        # safety
        self.use_soft_collision = params["use_soft_collision"]
        self.safe_distance = params["safe_distance"]
        self.safe_penalty_coef = params["safe_penalty_coef"]

        # near-goal shaping
        self.goal_relax_outer_radius = params["goal_relax_outer_radius"]
        self.goal_relax_inner_radius = params["goal_relax_inner_radius"]
        self.goal_soft_collision_min_scale = params["goal_soft_collision_min_scale"]
        self.goal_progress_max_scale = params["goal_progress_max_scale"]

        # route bias
        self.use_route_bias = params["use_route_bias"]
        self.route_bias_scale = params["route_bias_scale"]
        self.route_bias_table = self._build_route_bias_table()

        # stuck escape
        self.use_stuck_escape = params["use_stuck_escape"]
        self.escape_lookahead = params["escape_lookahead"]
        self.stuck_window = params["stuck_window"]
        self.stuck_progress_threshold = params["stuck_progress_threshold"]
        self.stuck_unique_ratio_threshold = params["stuck_unique_ratio_threshold"]
        self.escape_open_length = params["escape_open_length"]
        self.escape_open_weight = params["escape_open_weight"]
        self.escape_goal_weight = params["escape_goal_weight"]

        print(f"Loading point cloud from {pointcloud_path}...")
        self.all_points, self.kd_tree = build_pointcloud_index(npy_path=pointcloud_path)
        print(f"Loaded {self.all_points.shape[0]} points")

        # observation shape
        base_sensor_dim = self.num_dirs + 4  # directional_dists + nearest_obs_rel + target_rel

        if self.use_route_bias and self.latent_dim > 0:
            base_sensor_dim += 2

        if self.use_stuck_escape:
            base_sensor_dim += 3  # stuck flag + escape dir (2D)

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

        # runtime state
        self.curr_pose = None
        self.target_center = None
        self.step_count = 0
        self.curr_initial_id = None

        self.visited_cells_ep = set()
        self._ep_rewarded_cells = set()

        # per-initial coverage
        self.global_coverage_maps = {
            init["initial_id"]: {} for init in self.initials
        }

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
        self.curr_pose = np.array(initial_pose, dtype=np.float64)
        self.target_center = np.array(target_center, dtype=np.float64)
        self.initial_pose = np.array(initial_pose, dtype=np.float64)
        self.curr_initial_id = initial_id
        self.curr_latent = latent

        self.step_count = 0
        self.episode_reward = 0.0
        self._prev_target_dist = None

        # episode visitation
        self.visited_cells_ep = set()
        self._ep_rewarded_cells = set()

        start_xy = np.array([self.initial_pose[0], self.initial_pose[1]], dtype=np.float64)
        start_cell = self._xy_to_cell(start_xy)
        self.visited_cells_ep.add(start_cell)
        self._ep_rewarded_cells.add(start_cell)

        # summary stats
        self.path_length = 0.0
        self.turn_sum = 0.0
        self.prev_heading = None

        # stuck-history buffer
        self.recent_positions = []
        self.recent_target_dists = []
        self.recent_cells = []

        # diagnostics
        self.min_goal_dist = float(np.linalg.norm(self.target_center - start_xy))
        self.min_obstacle_dist = float("inf")

        # reward decomposition accumulators
        self.progress_reward_sum = 0.0
        self.success_bonus_sum = 0.0
        self.collision_penalty_sum = 0.0
        self.step_penalty_sum = 0.0
        self.soft_collision_penalty_sum = 0.0
        self.visit_reward_sum = 0.0
        self.history_reward_sum = 0.0
        self.inter_latent_repulsion_sum = 0.0

        obs = self._get_obs()

        # IMPORTANT: true-goal distance only
        self._prev_target_dist = np.linalg.norm(self.target_center - start_xy)
        return obs

    def step(self, action):
        dx = action.squeeze(0).cpu()[0].item()
        dy = action.squeeze(0).cpu()[1].item()

        x, y, _ = self.curr_pose
        step_len = float(np.sqrt(dx * dx + dy * dy))
        curr_heading = float(np.arctan2(dy, dx))

        self.curr_pose = np.array([x + dx, y + dy, curr_heading], dtype=np.float64)
        curr_xy = np.array([self.curr_pose[0], self.curr_pose[1]], dtype=np.float64)
        curr_cell = self._xy_to_cell(curr_xy)

        # summary stats
        self.path_length += step_len
        if self.prev_heading is not None:
            dtheta = curr_heading - self.prev_heading
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
            self.turn_sum += abs(dtheta)
        self.prev_heading = curr_heading

        # keep full visited set regardless of whether episode-vis reward is enabled
        self.visited_cells_ep.add(curr_cell)

        # build obs after state update
        obs = self._get_obs()
        sensor = obs["sensor"].cpu().numpy()

        nearest_obs_rel = sensor[self.num_dirs:self.num_dirs + 2]
        obstacle_dist = float(np.linalg.norm(nearest_obs_rel))

        true_target_rel = self.target_center - curr_xy
        target_dist = float(np.linalg.norm(true_target_rel))

        self.min_goal_dist = min(self.min_goal_dist, target_dist)
        self.min_obstacle_dist = min(self.min_obstacle_dist, obstacle_dist)

        # update history buffers for stuck detection
        self.recent_positions.append(curr_xy.copy())
        self.recent_target_dists.append(target_dist)
        self.recent_cells.append(curr_cell)

        if len(self.recent_positions) > self.stuck_window:
            self.recent_positions.pop(0)
            self.recent_target_dists.pop(0)
            self.recent_cells.pop(0)

        # ------------------------------------------------------------------
        # termination
        # ------------------------------------------------------------------
        done = False
        info = {}

        if obstacle_dist <= self.collision_threshold:
            done = True
            info["won"] = False
            info["term_type"] = "collision"

        if target_dist <= self.success_radius:
            done = True
            info["won"] = True
            info["term_type"] = "success"

        self.step_count += 1
        if self.step_count >= self.max_steps and not done:
            done = True
            info["won"] = False
            info["term_type"] = "timeout"

        # ------------------------------------------------------------------
        # reward decomposition
        # ------------------------------------------------------------------
        reward = 0.0

        progress_reward = 0.0
        collision_penalty = 0.0
        success_bonus = 0.0
        step_penalty = -0.5
        soft_collision_penalty = 0.0
        visit_reward = 0.0
        history_reward = 0.0
        inter_latent_repulsion = 0.0

        progress_scale = self._goal_proximity_scale(
            target_dist=target_dist,
            outer_radius=self.goal_relax_outer_radius,
            inner_radius=self.goal_relax_inner_radius,
            near_value=self.goal_progress_max_scale,
            far_value=1.0,
        )
        if self._prev_target_dist is not None:
            progress_reward = progress_scale * (self._prev_target_dist - target_dist)

        if obstacle_dist <= self.collision_threshold:
            collision_penalty = -100.0

        if info.get("won", False):
            success_bonus = 200.0

        reward += progress_reward
        reward += collision_penalty
        reward += success_bonus
        reward += step_penalty

        safe_scale = self._goal_proximity_scale(
            target_dist=target_dist,
            outer_radius=self.goal_relax_outer_radius,
            inner_radius=self.goal_relax_inner_radius,
            near_value=self.goal_soft_collision_min_scale,
            far_value=1.0,
        )
        if self.use_soft_collision and obstacle_dist < self.safe_distance:
            soft_collision_penalty = -(
                safe_scale
                * self.safe_penalty_coef
                * (self.safe_distance - obstacle_dist) ** 2
            )
            reward += soft_collision_penalty

        # episode-level anti-loop reward
        if self.use_episode_vis:
            if curr_cell not in self._ep_rewarded_cells:
                visit_reward += self.visit_bonus
                self._ep_rewarded_cells.add(curr_cell)
            else:
                visit_reward -= self.cell_repeat_penalty

        # history visitation reward
        if self.use_history_vis and self.curr_initial_id is not None:
            latent_id = self._get_latent_id()

            # global history
            global_map = self.global_coverage_maps[self.curr_initial_id]
            global_count = global_map.get(curr_cell, 0.0)
            w_global = self._compute_history_position_weight(curr_cell, use_latent_goal=False)
            history_reward += (
                w_global
                * self.global_history_bonus_coef
                * self._history_novelty(global_count)
            )

            # latent-specific history
            if latent_id is not None:
                latent_map = self.latent_coverage_maps[self.curr_initial_id][latent_id]
                latent_count = latent_map.get(curr_cell, 0.0)
                w_latent = self._compute_history_position_weight(curr_cell, use_latent_goal=True)
                history_reward += (
                    w_latent
                    * self.latent_history_bonus_coef
                    * self._history_novelty(latent_count)
                )

                # inter-latent repulsion
                if self.use_inter_latent_repulsion and self.inter_latent_repulsion_coef > 0.0:
                    other_latent_count = self._get_other_latent_overlap_count(
                        self.curr_initial_id, curr_cell, latent_id
                    )
                    inter_latent_repulsion = -(
                        w_latent
                        * self.inter_latent_repulsion_coef
                        * np.log(1.0 + other_latent_count)
                    )

        reward += visit_reward
        reward += history_reward
        reward += inter_latent_repulsion

        self._prev_target_dist = target_dist
        self.episode_reward += reward

        # accumulate diagnostics
        self.progress_reward_sum += float(progress_reward)
        self.success_bonus_sum += float(success_bonus)
        self.collision_penalty_sum += float(collision_penalty)
        self.step_penalty_sum += float(step_penalty)
        self.soft_collision_penalty_sum += float(soft_collision_penalty)
        self.visit_reward_sum += float(visit_reward)
        self.history_reward_sum += float(history_reward)
        self.inter_latent_repulsion_sum += float(inter_latent_repulsion)

        if done:
            info["episode"] = {"r": float(self.episode_reward)}
            info["initial_pose"] = copy.deepcopy(self.initial_pose)
            info["target_center"] = copy.deepcopy(self.target_center)
            info["initial_id"] = self.curr_initial_id

            # episode summary features for latent discriminator
            final_xy = np.array([self.curr_pose[0], self.curr_pose[1]], dtype=np.float64)
            start_xy = np.array([self.initial_pose[0], self.initial_pose[1]], dtype=np.float64)
            target_xy = np.array([self.target_center[0], self.target_center[1]], dtype=np.float64)

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

            # detailed diagnostics
            info["final_goal_dist"] = float(target_dist)
            info["min_goal_dist"] = float(self.min_goal_dist)
            info["final_obstacle_dist"] = float(obstacle_dist)
            info["min_obstacle_dist"] = float(self.min_obstacle_dist)
            info["episode_len"] = int(self.step_count)

            info["reward_breakdown"] = {
                "progress": float(self.progress_reward_sum),
                "success_bonus": float(self.success_bonus_sum),
                "collision_penalty": float(self.collision_penalty_sum),
                "step_penalty": float(self.step_penalty_sum),
                "soft_collision_penalty": float(self.soft_collision_penalty_sum),
                "visit_reward": float(self.visit_reward_sum),
                "history_reward": float(self.history_reward_sum),
                "inter_latent_repulsion": float(self.inter_latent_repulsion_sum),
            }

            # update history maps
            update_weight = 1.0 if info.get("won", False) else self.failure_history_weight

            if update_weight > 0.0 and self.curr_initial_id is not None:
                global_map = self.global_coverage_maps[self.curr_initial_id]
                for cell in self.visited_cells_ep:
                    global_map[cell] = global_map.get(cell, 0.0) + update_weight

                latent_id = self._get_latent_id()
                if latent_id is not None:
                    latent_map = self.latent_coverage_maps[self.curr_initial_id][latent_id]
                    for cell in self.visited_cells_ep:
                        latent_map[cell] = latent_map.get(cell, 0.0) + update_weight

        return obs, torch.tensor([reward], device=self.device), [done], [info]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        xy = np.array([self.curr_pose[0], self.curr_pose[1]], dtype=np.float64)

        nearest_obs_rel, _ = find_nearest_point_kdtree(
            self.all_points, self.kd_tree, xy
        )
        true_target_rel = self.target_center - xy

        directional_dists = compute_directional_distances(
            self.all_points,
            xy,
            num_dirs=self.num_dirs,
            max_range=self.max_obs_range,
        )

        obs_target_rel = true_target_rel.astype(np.float32)

        # stuck-aware escape target
        stuck_flag = 0.0
        escape_dir = np.array([0.0, 0.0], dtype=np.float32)
        used_escape_target = False

        if self.use_stuck_escape and self._check_stuck():
            stuck_flag = 1.0
            escape_dir = self._compute_escape_direction(
                directional_dists=directional_dists,
                target_rel=true_target_rel,
            )
            if escape_dir is None:
                escape_dir = np.array([0.0, 0.0], dtype=np.float32)
            else:
                escape_dir = np.asarray(escape_dir, dtype=np.float32)

            escape_target = xy + self.escape_lookahead * escape_dir
            obs_target_rel = (escape_target - xy).astype(np.float32)
            used_escape_target = True

        parts = [
            directional_dists.astype(np.float32),
            nearest_obs_rel.astype(np.float32),
            obs_target_rel.astype(np.float32),
        ]

        if self.use_route_bias and self.latent_dim > 0:
            if not used_escape_target:
                latent_id = 0 if self.curr_latent is None else int(np.argmax(self.curr_latent))
                bias_dir = self.route_bias_table[latent_id]
                biased_target_rel = obs_target_rel + self.route_bias_scale * bias_dir
                parts.append(biased_target_rel.astype(np.float32))
            else:
                parts.append(obs_target_rel.astype(np.float32))

        if self.use_stuck_escape:
            parts.append(np.array([stuck_flag], dtype=np.float32))
            parts.append(escape_dir.astype(np.float32))

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
    # Sampling / utilities
    # ------------------------------------------------------------------

    def _sample_next_initial(self, was_success):
        if was_success:
            self.success_counts[self.initial_index] += 1

        success_counts = np.array(self.success_counts, dtype=np.float32)
        mask = success_counts < self.initial_success_target

        if not np.any(mask):
            self.initial_index = np.random.randint(len(self.initials))
            return

        gamma = self.initial_sampling_gamma
        weights = np.zeros_like(success_counts, dtype=np.float32)
        weights[mask] = 1.0 / ((success_counts[mask] + 1.0) ** gamma)

        if self.zero_success_boost > 1.0:
            zero_mask = (success_counts == 0.0) & mask
            weights[zero_mask] *= self.zero_success_boost

        weight_sum = weights.sum()
        if weight_sum <= 1e-8:
            weights[mask] = 1.0
            weight_sum = weights.sum()

        probs = weights / weight_sum
        self.initial_index = int(np.random.choice(len(self.initials), p=probs))

    def _xy_to_cell(self, xy):
        x, y = xy
        cx = int(np.floor(x / self.cell_size))
        cy = int(np.floor(y / self.cell_size))
        return (cx, cy)

    def _build_route_bias_table(self):
        if self.latent_dim <= 0:
            return None

        table = np.zeros((self.latent_dim, 2), dtype=np.float32)
        for z in range(self.latent_dim):
            theta = 2.0 * np.pi * z / self.latent_dim
            table[z, 0] = np.cos(theta)
            table[z, 1] = np.sin(theta)
        return table

    def _goal_proximity_scale(self, target_dist, outer_radius, inner_radius, near_value, far_value):
        if target_dist >= outer_radius:
            return far_value
        if target_dist <= inner_radius:
            return near_value

        ratio = (target_dist - inner_radius) / (outer_radius - inner_radius)
        return near_value + (far_value - near_value) * ratio

    def _check_stuck(self):
        if len(self.recent_positions) < self.stuck_window:
            return False

        progress = self.recent_target_dists[0] - self.recent_target_dists[-1]
        cond_progress = progress < self.stuck_progress_threshold

        unique_ratio = len(set(self.recent_cells)) / float(self.stuck_window)
        cond_cells = unique_ratio < self.stuck_unique_ratio_threshold

        return cond_progress and cond_cells

    def _compute_escape_direction(self, directional_dists, target_rel):
        num_dirs = len(directional_dists)

        target_norm = np.linalg.norm(target_rel)
        if target_norm < 1e-6:
            goal_dir = np.array([0.0, 0.0], dtype=np.float32)
        else:
            goal_dir = (target_rel / target_norm).astype(np.float32)

        best_score = -1e9
        best_vec = np.array([0.0, 0.0], dtype=np.float32)

        for k in range(num_dirs):
            theta = 2.0 * np.pi * k / num_dirs
            u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

            d_k = directional_dists[k]
            open_score = min(d_k / self.escape_open_length, 1.0)
            goal_align = float(np.dot(goal_dir, u))

            score = (
                self.escape_open_weight * open_score
                + self.escape_goal_weight * goal_align
            )

            if score > best_score:
                best_score = score
                best_vec = u

        return best_vec

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def _cell_to_center_xy(self, cell):
        cx, cy = cell
        x = (cx + 0.5) * self.cell_size
        y = (cy + 0.5) * self.cell_size
        return np.array([x, y], dtype=np.float64)

    def _get_latent_id(self):
        if self.curr_latent is None:
            return None
        return int(np.argmax(self.curr_latent))

    def _get_latent_pseudo_goal(self):
        if (not self.use_latent_pseudo_goal) or (self.curr_latent is None) or (self.route_bias_table is None):
            return self.target_center

        latent_id = self._get_latent_id()
        bias_dir = self.route_bias_table[latent_id].astype(np.float64)
        return self.target_center + self.latent_goal_shift_scale * bias_dir

    def _compute_history_position_weight(self, cell, use_latent_goal=False):
        cell_xy = self._cell_to_center_xy(cell)
        start_xy = np.array([self.initial_pose[0], self.initial_pose[1]], dtype=np.float64)

        if use_latent_goal:
            goal_xy = self._get_latent_pseudo_goal().astype(np.float64)
        else:
            goal_xy = self.target_center.astype(np.float64)

        d_start = np.linalg.norm(cell_xy - start_xy)
        d_goal = np.linalg.norm(cell_xy - goal_xy)

        w_start = min(d_start / self.history_start_relax_radius, 1.0)
        w_goal = min(d_goal / self.history_goal_relax_radius, 1.0)

        w = min(w_start, w_goal)
        w = min(w, self.history_pos_weight_cap)
        return float(w)

    def _history_novelty(self, count):
        return 1.0 / ((1.0 + count) ** self.history_decay_beta)

    def _get_other_latent_overlap_count(self, initial_id, cell, curr_latent_id):
        if self.latent_dim <= 0 or curr_latent_id is None:
            return 0.0

        total = 0.0
        latent_maps = self.latent_coverage_maps[initial_id]
        for z, z_map in latent_maps.items():
            if z == curr_latent_id:
                continue
            total += z_map.get(cell, 0.0)
        return total