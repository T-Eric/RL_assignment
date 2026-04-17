# choose initials to run several rollouts
# to judge the performance of the trained policy
# 1. successful rate and traj len
# 2. failure reasons (collision? timeout?)
# 3. rollout visualization and overlaying
# 4. deterministic and stochastic

# my_method/play_policy.py

import os
import json
import argparse
import numpy as np
import torch

from _instructor.env.uav_env import UAVNavEnv
from _instructor.model import Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointcloud_path", type=str, required=True)
    parser.add_argument("--initials_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)

    parser.add_argument("--initial_id", type=int, required=True,
                        help="Which initial_id to test")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="How many rollouts to run")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic action selection")

    parser.add_argument("--save_dir", type=str, default=None,
                        help="Optional directory to save trajectories")
    parser.add_argument("--save_failures", action="store_true",
                        help="Whether to save failed trajectories too")

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--success_radius", type=float, default=30.0)
    parser.add_argument("--collision_threshold", type=float, default=2.0)
    parser.add_argument("--action_limit", type=float, default=2.0)

    return parser.parse_args()


def save_traj(path, traj):
    arr = np.asarray(traj, dtype=np.float64)
    np.savetxt(path, arr, fmt="%.6f")


def classify_failure(traj, init_item, collision_threshold):
    """
    粗略判断失败类型：
    - 如果末点距离目标 > success_radius 且轨迹长度达到 max_steps，通常是 timeout
    - 这里更稳妥的做法是依赖 env 的 info["won"] + step_count 是否到顶
    但 env 没直接给 failure_type，所以这里只做辅助，不作强结论
    """
    # 这个函数先留空，主逻辑里通过 step_count 推断
    return "unknown"


def rollout_one_episode(env, actor_critic, init_item, deterministic=False):
    initial_pose = np.array(
        [init_item["x_start"], init_item["y_start"], 0.0], dtype=np.float64
    )
    target_center = np.array(
        [init_item["target_center_x"], init_item["target_center_y"]], dtype=np.float64
    )

    obs = env.reset(initial_pose=initial_pose, target_center=target_center)

    traj = [[env.curr_pose[0], env.curr_pose[1]]]
    ep_reward = 0.0
    step_count = 0
    done = False
    won = False

    while not done:
        with torch.no_grad():
            value, action, action_log_prob = actor_critic.act(
                {k: v.unsqueeze(0) for k, v in obs.items()},
                deterministic=deterministic,
            )

        prev_step_count = env.step_count
        obs, reward, done_list, infos = env.step(action)
        done = done_list[0]
        info = infos[0]

        ep_reward += reward.item()
        step_count += 1

        # 注意：即使 env 在 done 后自动 reset，这里记录的 curr_pose 仍然是 step 后的位置
        # 如果你担心自动 reset 影响，也可以改 env；当前版本一般仍可正常记录到终点前最后位置
        traj.append([env.curr_pose[0], env.curr_pose[1]])

        if done:
            won = info.get("won", False)

            # 用 step_count 是否达到 max_steps 粗略判断 timeout
            if won:
                outcome = "success"
            else:
                if step_count >= env.max_steps:
                    outcome = "timeout"
                else:
                    outcome = "collision"

    return {
        "won": won,
        "outcome": outcome,
        "traj": traj,
        "reward": ep_reward,
        "steps": step_count,
    }


def main():
    args = parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    with open(args.initials_path, "r") as f:
        initials = json.load(f)

    init_item = None
    for item in initials:
        if item["initial_id"] == args.initial_id:
            init_item = item
            break
    if init_item is None:
        raise ValueError(f"initial_id={args.initial_id} not found in {args.initials_path}")

    env_params = {
        "max_steps": args.max_steps,
        "success_radius": args.success_radius,
        "collision_threshold": args.collision_threshold,
        "action_limit": [args.action_limit, args.action_limit],
    }

    env = UAVNavEnv(
        pointcloud_path=args.pointcloud_path,
        env_params=env_params,
        save_dir=None,
        device=device,
        initials=initials,
    )

    actor_critic = Policy(
        obs_dim=4,
        action_dim=2,
        action_limit=(args.action_limit, args.action_limit),
    )
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    actor_critic.load_state_dict(ckpt)
    actor_critic.to(device)
    actor_critic.eval()

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    results = []
    n_success = 0
    n_collision = 0
    n_timeout = 0

    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Initial ID : {args.initial_id}")
    print(f"Episodes   : {args.num_episodes}")
    print(f"Mode       : {'deterministic' if args.deterministic else 'stochastic'}")
    print("=" * 70)

    for ep in range(args.num_episodes):
        result = rollout_one_episode(
            env=env,
            actor_critic=actor_critic,
            init_item=init_item,
            deterministic=args.deterministic,
        )
        results.append(result)

        if result["outcome"] == "success":
            n_success += 1
        elif result["outcome"] == "collision":
            n_collision += 1
        else:
            n_timeout += 1

        print(
            f"[Episode {ep:03d}] outcome={result['outcome']:9s} "
            f"steps={result['steps']:3d} reward={result['reward']:8.2f}"
        )

        if args.save_dir is not None:
            should_save = result["won"] or args.save_failures
            if should_save:
                suffix = result["outcome"]
                path = os.path.join(args.save_dir, f"ep_{ep:03d}_{suffix}.txt")
                save_traj(path, result["traj"])

    success_rate = n_success / args.num_episodes
    mean_steps = np.mean([r["steps"] for r in results]) if results else 0.0
    mean_reward = np.mean([r["reward"] for r in results]) if results else 0.0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"success   : {n_success}/{args.num_episodes} ({success_rate:.2%})")
    print(f"collision : {n_collision}/{args.num_episodes}")
    print(f"timeout   : {n_timeout}/{args.num_episodes}")
    print(f"mean steps: {mean_steps:.2f}")
    print(f"mean rew. : {mean_reward:.2f}")

    if args.save_dir is not None:
        summary_path = os.path.join(args.save_dir, "summary.json")
        summary = {
            "checkpoint_path": args.checkpoint_path,
            "initial_id": args.initial_id,
            "num_episodes": args.num_episodes,
            "deterministic": args.deterministic,
            "success": n_success,
            "collision": n_collision,
            "timeout": n_timeout,
            "success_rate": success_rate,
            "mean_steps": mean_steps,
            "mean_reward": mean_reward,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()