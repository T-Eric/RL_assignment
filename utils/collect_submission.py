import os
import json
import argparse
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.env.uav_env import UAVNavEnv
from src.model import Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointcloud_path", type=str, required=True)
    parser.add_argument("--initials_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--success_radius", type=float, default=30.0)
    parser.add_argument("--collision_threshold", type=float, default=2.0)
    parser.add_argument("--action_limit", type=float, default=2.0)

    parser.add_argument("--trajs_per_initial", type=int, default=100)
    parser.add_argument("--max_attempts_per_initial", type=int, default=10000)
    parser.add_argument("--max_fail_save", type=int, default=20)

    # observation / reward / latent params
    parser.add_argument("--num_dirs", type=int, default=16)
    parser.add_argument("--max_obs_range", type=float, default=80.0)
    parser.add_argument("--cell_size", type=float, default=5.0)

    parser.add_argument("--use_episode_vis", action="store_true")
    parser.add_argument("--use_history_vis", action="store_true")
    parser.add_argument("--visit_bonus", type=float, default=1.0)
    parser.add_argument("--cell_repeat_penalty", type=float, default=0.1)

    parser.add_argument("--global_history_bonus_coef", type=float, default=0.1)
    parser.add_argument("--latent_history_bonus_coef", type=float, default=0.2)

    parser.add_argument("--use_soft_collision", action="store_true")
    parser.add_argument("--safe_distance", type=float, default=4.0)
    parser.add_argument("--safe_penalty_coef", type=float, default=0.02)

    parser.add_argument("--use_route_bias", action="store_true")
    parser.add_argument("--route_bias_scale", type=float, default=20.0)

    parser.add_argument("--latent_dim", type=int, default=4)

    # optional current-train params for shape compatibility
    parser.add_argument("--goal_relax_outer_radius", type=float, default=80.0)
    parser.add_argument("--goal_relax_inner_radius", type=float, default=40.0)
    parser.add_argument("--goal_soft_collision_min_scale", type=float, default=0.25)
    parser.add_argument("--goal_progress_max_scale", type=float, default=1.8)

    parser.add_argument("--use_stuck_escape", action="store_true")
    parser.add_argument("--escape_lookahead", type=float, default=12.0)
    parser.add_argument("--stuck_window", type=int, default=12)
    parser.add_argument("--stuck_progress_threshold", type=float, default=6.0)
    parser.add_argument("--stuck_unique_ratio_threshold", type=float, default=0.35)
    parser.add_argument("--escape_open_length", type=float, default=15.0)
    parser.add_argument("--escape_open_weight", type=float, default=1.0)
    parser.add_argument("--escape_goal_weight", type=float, default=0.8)

    # collect
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)

    # collect-time open excursion
    parser.add_argument("--use_open_excursion", action="store_true")
    parser.add_argument("--open_rects_path", type=str, default=None)
    parser.add_argument("--open_excursion_trigger_prob", type=float, default=0.2)
    parser.add_argument("--open_excursion_min_target_dist", type=float, default=60.0)
    parser.add_argument("--open_excursion_length", type=float, default=18.0)
    parser.add_argument("--open_excursion_max_steps", type=int, default=12)
    parser.add_argument("--open_excursion_abort_obstacle_dist", type=float, default=3.0)
    
    parser.add_argument("--open_excursion_subgoal_radius", type=float, default=10.0)
    parser.add_argument("--open_excursion_rect_margin", type=float, default=5.0)

    return parser.parse_args()

def sample_point_inside_rect(rect, margin=10.0):
    """
    Sample a point strictly inside an open rect, leaving some margin from boundaries.
    """
    xmin = rect["xmin"] + margin
    xmax = rect["xmax"] - margin
    ymin = rect["ymin"] + margin
    ymax = rect["ymax"] - margin

    # fallback if rect is too small for the requested margin
    if xmax <= xmin:
        xmin = rect["xmin"]
        xmax = rect["xmax"]
    if ymax <= ymin:
        ymin = rect["ymin"]
        ymax = rect["ymax"]

    x = np.random.uniform(xmin, xmax)
    y = np.random.uniform(ymin, ymax)
    return np.array([x, y], dtype=np.float64)

def get_current_open_rect(xy, rects):
    for rect in rects:
        if point_in_rect(xy, rect):
            return rect
    return None

def save_traj_txt(path, traj):
    arr = np.asarray(traj, dtype=np.float64)
    np.savetxt(path, arr, fmt="%.6f")


def sample_onehot_latent(latent_dim, latent_id=None):
    if latent_dim <= 0:
        return None
    z = np.zeros(latent_dim, dtype=np.float32)
    if latent_id is None:
        idx = np.random.randint(latent_dim)
    else:
        idx = int(latent_id) % latent_dim
    z[idx] = 1.0
    return z


def build_env_and_policy(args, device, initials):
    env_params = {
        "max_steps": args["max_steps"],
        "success_radius": args["success_radius"],
        "collision_threshold": args["collision_threshold"],
        "action_limit": [args["action_limit"], args["action_limit"]],
        "num_dirs": args["num_dirs"],
        "max_obs_range": args["max_obs_range"],
        "cell_size": args["cell_size"],
        "use_episode_vis": args["use_episode_vis"],
        "use_history_vis": args["use_history_vis"],
        "visit_bonus": args["visit_bonus"],
        "cell_repeat_penalty": args["cell_repeat_penalty"],
        "global_history_bonus_coef": args["global_history_bonus_coef"],
        "latent_history_bonus_coef": args["latent_history_bonus_coef"],
        "latent_dim": args["latent_dim"],
        "use_soft_collision": args["use_soft_collision"],
        "safe_distance": args["safe_distance"],
        "safe_penalty_coef": args["safe_penalty_coef"],
        "use_route_bias": args["use_route_bias"],
        "route_bias_scale": args["route_bias_scale"],

        # shape-compatible current-train params
        "goal_relax_outer_radius": args["goal_relax_outer_radius"],
        "goal_relax_inner_radius": args["goal_relax_inner_radius"],
        "goal_soft_collision_min_scale": args["goal_soft_collision_min_scale"],
        "goal_progress_max_scale": args["goal_progress_max_scale"],

        "use_stuck_escape": args["use_stuck_escape"],
        "escape_lookahead": args["escape_lookahead"],
        "stuck_window": args["stuck_window"],
        "stuck_progress_threshold": args["stuck_progress_threshold"],
        "stuck_unique_ratio_threshold": args["stuck_unique_ratio_threshold"],
        "escape_open_length": args["escape_open_length"],
        "escape_open_weight": args["escape_open_weight"],
        "escape_goal_weight": args["escape_goal_weight"],
    }

    env = UAVNavEnv(
        pointcloud_path=args["pointcloud_path"],
        env_params=env_params,
        save_dir=None,
        device=device,
        initials=initials,
    )

    obs_dim = env.observation_shape["sensor"][0]
    latent_dim = env.observation_shape["latent"][0] if "latent" in env.observation_shape else 0

    actor_critic = Policy(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        action_dim=2,
        action_limit=(args["action_limit"], args["action_limit"]),
    )

    ckpt = torch.load(args["checkpoint_path"], map_location=device)
    actor_critic.load_state_dict(ckpt)
    actor_critic.to(device)
    actor_critic.eval()

    return env, actor_critic


def load_open_rects(path):
    if path is None:
        return []
    with open(path, "r") as f:
        rects = json.load(f)
    return rects


def point_in_rect(xy, rect):
    x, y = float(xy[0]), float(xy[1])
    return (
        rect["xmin"] <= x <= rect["xmax"]
        and rect["ymin"] <= y <= rect["ymax"]
    )


def get_current_open_rect_id(xy, rects):
    for rect in rects:
        if point_in_rect(xy, rect):
            return rect["region_id"]
    return None


def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def sample_excursion_direction(true_target_rel, latent=None):
    """
    Side/front-biased excursion.
    Not full 360 random: we bias toward left/right detours relative to goal direction.
    Optionally let latent weakly bias left/right preference.
    """
    g = unit(true_target_rel.astype(np.float32))
    if np.linalg.norm(g) < 1e-8:
        g = np.array([1.0, 0.0], dtype=np.float32)

    left = np.array([-g[1], g[0]], dtype=np.float32)
    right = np.array([g[1], -g[0]], dtype=np.float32)

    # decide left/right with mild latent preference
    p_left = 0.5
    if latent is not None:
        latent_id = int(np.argmax(latent))
        if latent_id == 0:
            p_left = 0.8
        elif latent_id == 1:
            p_left = 0.6
        elif latent_id == 2:
            p_left = 0.4
        elif latent_id == 3:
            p_left = 0.2

    use_left = (np.random.rand() < p_left)
    side = left if use_left else right

    # angle offset from goal direction toward side direction
    # choose in [45°, 100°]
    theta_deg = np.random.uniform(45.0, 100.0)
    theta = np.deg2rad(theta_deg)

    # convex combination on unit circle approximation
    d = np.cos(theta) * g + np.sin(theta) * side
    d = unit(d)
    return d


def patch_obs_for_excursion(
    obs,
    env,
    excursion_target_rel,
    use_route_bias,
    use_stuck_escape,
):
    """
    Modify a copied obs dict:
      - replace target_rel by excursion_target_rel
      - disable route bias by setting biased_target_rel = excursion_target_rel
      - if stuck-escape aux dims exist, set them to neutral zeros
    Sensor layout:
      [dirs][nearest2][target2][biased_target2 if route_bias][stuck_flag+escape_dir if stuck_escape]
    """
    patched = {}
    for k, v in obs.items():
        patched[k] = v.clone()

    sensor = patched["sensor"]

    idx_target_start = env.num_dirs + 2
    idx_target_end = env.num_dirs + 4
    sensor[idx_target_start:idx_target_end] = torch.tensor(
        excursion_target_rel.astype(np.float32),
        device=sensor.device
    )

    cursor = idx_target_end

    if use_route_bias and env.latent_dim > 0:
        # disable route bias during excursion by duplicating excursion target
        sensor[cursor:cursor + 2] = torch.tensor(
            excursion_target_rel.astype(np.float32),
            device=sensor.device
        )
        cursor += 2

    if use_stuck_escape:
        # neutralize stuck escape aux channels during open excursion
        sensor[cursor] = 0.0           # stuck_flag
        sensor[cursor + 1] = 0.0       # escape_dir x
        sensor[cursor + 2] = 0.0       # escape_dir y

    return patched


def rollout_one_episode(
    env,
    actor_critic,
    init_item,
    deterministic=False,
    latent=None,
    open_rects=None,
    use_open_excursion=False,
    open_excursion_trigger_prob=0.2,
    open_excursion_min_target_dist=60.0,
    open_excursion_length=18.0,   # 保留兼容，但强制子目标模式不再依赖它
    open_excursion_max_steps=12,
    open_excursion_abort_obstacle_dist=3.0,
    open_excursion_subgoal_radius=15.0,
    open_excursion_rect_margin=10.0,
    use_route_bias=False,
    use_stuck_escape=False,
):
    initial_pose = np.array(
        [init_item["x_start"], init_item["y_start"], 0.0], dtype=np.float64
    )
    target_center = np.array(
        [init_item["target_center_x"], init_item["target_center_y"]], dtype=np.float64
    )

    obs = env.reset(
        initial_pose=initial_pose,
        target_center=target_center,
        initial_id=init_item["initial_id"],
        latent=latent,
    )

    traj = [[env.curr_pose[0], env.curr_pose[1]]]
    done = False
    won = False
    ep_reward = 0.0
    step_count = 0

    # collect-time excursion controller state
    triggered_rects = set()
    excursion_active = False
    excursion_steps = 0
    excursion_target = None

    while not done:
        curr_xy = np.array([env.curr_pose[0], env.curr_pose[1]], dtype=np.float64)
        true_target_rel = env.target_center - curr_xy
        target_dist = np.linalg.norm(true_target_rel)

        # detect current rect
        current_rect = None
        current_rect_id = None
        if use_open_excursion and open_rects:
            current_rect = get_current_open_rect(curr_xy, open_rects)
            if current_rect is not None:
                current_rect_id = current_rect["region_id"]

        # trigger a new rect-subgoal excursion
        if (
            use_open_excursion
            and (not excursion_active)
            and (current_rect_id is not None)
            and (current_rect_id not in triggered_rects)
            and (target_dist > open_excursion_min_target_dist)
            and (np.random.rand() < open_excursion_trigger_prob)
        ):
            excursion_target = sample_point_inside_rect(
                current_rect,
                margin=open_excursion_rect_margin,
            )
            excursion_active = True
            excursion_steps = 0
            triggered_rects.add(current_rect_id)

        # choose observation passed to policy
        obs_for_policy = obs
        if excursion_active:
            excursion_target_rel = (excursion_target - curr_xy).astype(np.float32)
            obs_for_policy = patch_obs_for_excursion(
                obs=obs,
                env=env,
                excursion_target_rel=excursion_target_rel,
                use_route_bias=use_route_bias,
                use_stuck_escape=use_stuck_escape,
            )

        with torch.no_grad():
            _, action, _ = actor_critic.act(
                {k: v.unsqueeze(0) for k, v in obs_for_policy.items()},
                deterministic=deterministic,
            )

        obs, reward, done_list, infos = env.step(action)
        done = done_list[0]
        info = infos[0]
        ep_reward += reward.item()
        step_count += 1

        traj.append([env.curr_pose[0], env.curr_pose[1]])

        # update excursion lifecycle
        if excursion_active:
            excursion_steps += 1
            new_xy = np.array([env.curr_pose[0], env.curr_pose[1]], dtype=np.float64)
            excursion_dist = np.linalg.norm(excursion_target - new_xy)

            sensor_np = obs["sensor"].cpu().numpy()
            nearest_obs_rel = sensor_np[env.num_dirs:env.num_dirs + 2]
            obstacle_dist = np.linalg.norm(nearest_obs_rel)

            if (
                excursion_dist < open_excursion_subgoal_radius
                or obstacle_dist < open_excursion_abort_obstacle_dist
                or excursion_steps >= open_excursion_max_steps
            ):
                excursion_active = False
                excursion_steps = 0
                excursion_target = None

        if done:
            won = info.get("won", False)

    return {
        "won": won,
        "traj": traj,
        "reward": ep_reward,
        "steps": step_count,
    }


def split_evenly(items, num_splits):
    chunks = [[] for _ in range(num_splits)]
    for i, item in enumerate(items):
        chunks[i % num_splits].append(item)
    return chunks


def worker_collect(worker_rank, args_dict, init_subset):
    seed = args_dict["seed"] + worker_rank
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cpu")
    env, actor_critic = build_env_and_policy(args_dict, device, init_subset)

    open_rects = []
    if args_dict["use_open_excursion"]:
        open_rects = load_open_rects(args_dict["open_rects_path"])

    summary = []

    for init_item in init_subset:
        iid = init_item["initial_id"]
        init_dir = os.path.join(args_dict["output_dir"], f"initial_{iid}")
        os.makedirs(init_dir, exist_ok=True)

        success_count = 0
        fail_count = 0
        attempts = 0

        print(f"[worker {worker_rank}] initial_{iid}: target {args_dict['trajs_per_initial']}")

        while success_count < args_dict["trajs_per_initial"] and attempts < args_dict["max_attempts_per_initial"]:
            attempts += 1

            latent = None
            if args_dict["latent_dim"] > 0:
                latent_id = (attempts - 1) % args_dict["latent_dim"]
                latent = sample_onehot_latent(args_dict["latent_dim"], latent_id=latent_id)

            result = rollout_one_episode(
                env=env,
                actor_critic=actor_critic,
                init_item=init_item,
                deterministic=args_dict["deterministic"],
                latent=latent,
                open_rects=open_rects,
                use_open_excursion=args_dict["use_open_excursion"],
                open_excursion_trigger_prob=args_dict["open_excursion_trigger_prob"],
                open_excursion_min_target_dist=args_dict["open_excursion_min_target_dist"],
                open_excursion_length=args_dict["open_excursion_length"],
                open_excursion_max_steps=args_dict["open_excursion_max_steps"],
                open_excursion_abort_obstacle_dist=args_dict["open_excursion_abort_obstacle_dist"],
                use_route_bias=args_dict["use_route_bias"],
                use_stuck_escape=args_dict["use_stuck_escape"],
                open_excursion_subgoal_radius=args_dict["open_excursion_subgoal_radius"],
                open_excursion_rect_margin=args_dict["open_excursion_rect_margin"],
            )

            if result["won"]:
                traj_path = os.path.join(init_dir, f"traj_{success_count}.txt")
                save_traj_txt(traj_path, result["traj"])
                success_count += 1

                if success_count % 5 == 0 or success_count == args_dict["trajs_per_initial"]:
                    print(
                        f"[worker {worker_rank}] initial_{iid}: "
                        f"success={success_count}/{args_dict['trajs_per_initial']}, "
                        f"attempts={attempts}"
                    )
            else:
                fail_count += 1
                if fail_count <= args_dict["max_fail_save"]:
                    traj_path = os.path.join(init_dir, f"fail_traj_{fail_count}.txt")
                    save_traj_txt(traj_path, result["traj"])

        summary.append({
            "worker_rank": worker_rank,
            "initial_id": iid,
            "success_count": success_count,
            "attempts": attempts,
        })

        if success_count < args_dict["trajs_per_initial"]:
            print(
                f"[worker {worker_rank}] WARNING initial_{iid}: "
                f"{success_count}/{args_dict['trajs_per_initial']} successes only"
            )

    return summary


def main():
    args = parse_args()

    with open(args.initials_path, "r") as f:
        initials = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    args_dict = vars(args).copy()

    if args.use_open_excursion and args.open_rects_path is None:
        raise ValueError("--use_open_excursion requires --open_rects_path")

    num_workers = min(args.num_workers, len(initials))
    chunks = split_evenly(initials, num_workers)

    all_summary = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for worker_rank, init_subset in enumerate(chunks):
            futures.append(
                executor.submit(worker_collect, worker_rank, args_dict, init_subset)
            )

        for fut in as_completed(futures):
            worker_summary = fut.result()
            all_summary.extend(worker_summary)

    all_summary = sorted(all_summary, key=lambda x: x["initial_id"])

    summary_path = os.path.join(args.output_dir, "collection_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summary, f, indent=2)

    print("\nCollection complete!")
    print(f"Saved submission to: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()