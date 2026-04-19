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
    parser.add_argument("--route_bias_scale", type=float, default=40.0)

    parser.add_argument("--latent_dim", type=int, default=4)

    # collection
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)

    return parser.parse_args()


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
        latent_dim=latent_dim,  # critical: must match training
        action_dim=2,
        action_limit=(args["action_limit"], args["action_limit"]),
    )

    ckpt = torch.load(args["checkpoint_path"], map_location=device)
    actor_critic.load_state_dict(ckpt)
    actor_critic.to(device)
    actor_critic.eval()

    return env, actor_critic


def rollout_one_episode(env, actor_critic, init_item, deterministic=False, latent=None):
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

    while not done:
        with torch.no_grad():
            _, action, _ = actor_critic.act(
                {k: v.unsqueeze(0) for k, v in obs.items()},
                deterministic=deterministic,
            )

        obs, reward, done_list, infos = env.step(action)
        done = done_list[0]
        info = infos[0]
        ep_reward += reward.item()
        step_count += 1

        traj.append([env.curr_pose[0], env.curr_pose[1]])

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

    # CPU collection is usually better for multi-process rollout.
    device = torch.device("cpu")

    env, actor_critic = build_env_and_policy(args_dict, device, init_subset)

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
                # cycle by attempts, not successes, so all latents get tried fairly
                latent_id = (attempts - 1) % args_dict["latent_dim"]
                latent = sample_onehot_latent(args_dict["latent_dim"], latent_id=latent_id)

            result = rollout_one_episode(
                env=env,
                actor_critic=actor_critic,
                init_item=init_item,
                deterministic=args_dict["deterministic"],
                latent=latent,
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