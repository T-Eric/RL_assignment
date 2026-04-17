"""
PPO Training script for UAV trajectory collection.

Usage:
    python train.py --pointcloud_path ../data/pointcloud_2d.npy \
                    --initials_path ../data/eval_initials_100.json
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from collections import deque

import torch

from env.uav_env import UAVNavEnv
from ppo.ppo import PPO
from ppo.storage import DictRolloutStorage
from model import Policy

# episode_level utils
def sample_onehot_latent(latent_dim):
    if latent_dim <= 0:
        return None
    z = np.zeros(latent_dim, dtype=np.float32)
    idx = np.random.randint(latent_dim)
    z[idx] = 1.0
    return z


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointcloud_path", type=str, required=True)
    parser.add_argument("--initials_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--max_iter", type=int, default=100000)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--num_steps", type=int, default=256,
                        help="rollout length per update")
    parser.add_argument("--ppo_epoch", type=int, default=4)
    parser.add_argument("--num_mini_batch", type=int, default=4)
    parser.add_argument("--clip_param", type=float, default=0.1)
    parser.add_argument("--entropy_coef", type=float, default=0.02)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--gpu", type=int, default=0)
    
    parser.add_argument("--num_dirs", type=int, default=16)
    parser.add_argument("--max_obs_range", type=float, default=80.0)
    # visitation rewards
    parser.add_argument("--cell_size", type=float, default=5.0)
    parser.add_argument("--use_episode_vis", action="store_true")
    parser.add_argument("--use_history_vis", action="store_true")
    parser.add_argument("--visit_bonus", type=float, default=1.0)
    parser.add_argument("--cell_repeat_penalty", type=float, default=0.1)
    parser.add_argument("--history_bonus_coef", type=float, default=0.5)
    
    parser.add_argument("--use_soft_collision", action="store_true")
    parser.add_argument("--safe_distance", type=float, default=6.0)
    parser.add_argument("--safe_penalty_coef", type=float, default=0.1)
    
    parser.add_argument("--latent_dim", type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load initials
    with open(args.initials_path) as f:
        initials = json.load(f)
    print(f"Loaded {len(initials)} initials")

    # Experiment directory
    if args.save_dir is None:
        args.save_dir = os.path.join("saved_data", f"run_{int(time.time())}")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "controllers"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "success_trajs"), exist_ok=True)

    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Create environment
    env_params = {
        "max_steps": args.max_steps,
        "success_radius": 30.0,
        "collision_threshold": 2.0,
        "action_limit": [2.0, 2.0],
        
        "num_dirs": args.num_dirs,
        "max_obs_range": args.max_obs_range,

        "cell_size": args.cell_size,
        "use_episode_vis": args.use_episode_vis,
        "use_history_vis": args.use_history_vis,
        "visit_bonus": args.visit_bonus,
        "cell_repeat_penalty": args.cell_repeat_penalty,
        "history_bonus_coef": args.history_bonus_coef,
        
        "latent_dim": args.latent_dim,
        
        "use_soft_collision": args.use_soft_collision,
        "safe_distance": args.safe_distance,
        "safe_penalty_coef": args.safe_penalty_coef,
    }
    env = UAVNavEnv(
        pointcloud_path=args.pointcloud_path,
        env_params=env_params,
        save_dir=args.save_dir,
        device=device,
        initials=initials,
    )

    # Create policy and PPO agent
    obs_dim=env.observation_shape["sensor"][0]
    actor_critic = Policy(obs_dim=obs_dim, action_dim=2, action_limit=(2.0, 2.0))
    actor_critic.to(device)

    agent = PPO(
        actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
        args.value_loss_coef, args.entropy_coef,
        lr=args.lr, eps=1e-5, max_grad_norm=args.max_grad_norm,
    )

    # Rollout storage
    rollouts = DictRolloutStorage(
        args.num_steps, 1, env.observation_shape, env.action_shape,
        actor_critic.recurrent_hidden_state_size,
    )

    # Initial reset
    init0 = initials[0]
    latent0=sample_onehot_latent(args.latent_dim)
    obs = env.reset(
        initial_pose=np.array([init0["x_start"], init0["y_start"], 0.0]),
        target_center=np.array([init0["target_center_x"], init0["target_center_y"]]),
        initial_id=init0["initial_id"],
        latent=latent0,
    )
    for key in obs:
        rollouts.obs[key][0].copy_(obs[key])
    rollouts.to(device)

    # Trajectory buffer for saving successful trajectories
    # traj = [obs["sensor"].cpu().squeeze().numpy()[:2].tolist()]  # start as [obs_x, obs_y]
    # Actually save current pose
    traj = [[env.curr_pose[0], env.curr_pose[1]]]

    episode_rewards = deque(maxlen=50)
    start_time = time.time()

    print(f"\nStarting training (max_iter={args.max_iter})...\n")

    for j in range(args.max_iter):
        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(
                    {k: rollouts.obs[k][step] for k in rollouts.obs})

            obs_next, reward, done, infos = env.step(action)
            traj.append([env.curr_pose[0], env.curr_pose[1]])

            for info in infos:
                if "episode" in info:
                    episode_rewards.append(info["episode"]["r"])

                    if info.get("won", False):
                        n_success = len(os.listdir(
                            os.path.join(args.save_dir, "success_trajs")))
                        traj_path = os.path.join(
                            args.save_dir, "success_trajs", f"{n_success}.txt")
                        with open(traj_path, "w") as f:
                            ip = info["initial_pose"]
                            tc = info["target_center"]
                            f.write(f"{ip[0]} {ip[1]}\n")
                            f.write(f"{tc[0]} {tc[1]}\n")
                            for px, py in traj:
                                f.write(f"{px} {py}\n")
                                
                    # manual reset
                    env._sample_next_initial(info.get("won", False))
                    next_init = env.initials[env.initial_index]
                    next_latent=sample_onehot_latent(args.latent_dim)
                    obs_next = env.reset(
                        initial_pose=np.array([
                            next_init["x_start"],
                            next_init["y_start"],
                            0.0
                        ]),
                        target_center=np.array([
                            next_init["target_center_x"],
                            next_init["target_center_y"],
                        ]),
                        initial_id=next_init["initial_id"],
                        latent=next_latent,
                    )
                
                    traj = [[env.curr_pose[0], env.curr_pose[1]]]

            masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done]).to(device)
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info else [1.0] for info in infos]
            ).to(device)
            rhs = torch.zeros(1, actor_critic.recurrent_hidden_state_size).to(device)
            rollouts.insert(obs_next, rhs, action, action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                {k: rollouts.obs[k][-1] for k in rollouts.obs}).detach()

        rollouts.compute_returns(next_value, True, args.gamma, args.gae_lambda)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) > 0:
            elapsed = time.time() - start_time
            total_steps = (j + 1) * args.num_steps
            n_success = len(os.listdir(os.path.join(args.save_dir, "success_trajs")))
            print(f"[Iter {j:6d}] steps={total_steps:8d}  "
                  f"reward={np.mean(episode_rewards):7.1f}  "
                  f"success_trajs={n_success}  "
                  f"v_loss={value_loss:.4f}  "
                  f"elapsed={elapsed:.0f}s")

            with open(os.path.join(args.save_dir, "train_log.txt"), "a") as f:
                f.write(f"{j}\t{np.mean(episode_rewards):.4f}\t{n_success}\n")

        if j % args.save_interval == 0 and j > 0:
            torch.save(actor_critic.state_dict(),
                       os.path.join(args.save_dir, "controllers", f"{j}_controller.pt"))

    torch.save(actor_critic.state_dict(),
               os.path.join(args.save_dir, "controllers", "final_controller.pt"))
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
