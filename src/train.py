import argparse
import json
import os
import time
from collections import Counter, deque

import numpy as np
import torch
import torch.nn.functional as F

from env.uav_env import UAVNavEnv
from model import LatentDiscriminator, Policy
from ppo.ppo import PPO
from ppo.storage import DictRolloutStorage


def sample_onehot_latent_with_id(latent_dim, latent_id):
    if latent_dim <= 0:
        return None
    z = np.zeros(latent_dim, dtype=np.float32)
    z[latent_id] = 1.0
    return z


def parse_args():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--pointcloud_path", type=str, required=True)
    parser.add_argument("--initials_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)

    # optimization
    parser.add_argument("--max_iter", type=int, default=100000)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_param", type=float, default=0.1)
    parser.add_argument("--entropy_coef", type=float, default=0.02)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    # rollout / parallelism
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=128, help="rollout length per update")
    parser.add_argument("--ppo_epoch", type=int, default=3)
    parser.add_argument("--num_mini_batch", type=int, default=8)

    # logging / saving
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--traj_flush_interval", type=int, default=50)
    parser.add_argument("--disc_batch_size", type=int, default=32)
    parser.add_argument("--stats_window", type=int, default=200)

    # observation
    parser.add_argument("--num_dirs", type=int, default=16)
    parser.add_argument("--max_obs_range", type=float, default=80.0)

    # episode visitation
    parser.add_argument("--cell_size", type=float, default=5.0)
    parser.add_argument("--use_episode_vis", action="store_true")
    parser.add_argument("--visit_bonus", type=float, default=1.0)
    parser.add_argument("--cell_repeat_penalty", type=float, default=0.1)

    # history visitation
    parser.add_argument("--use_history_vis", action="store_true")
    parser.add_argument("--global_history_bonus_coef", type=float, default=0.3)
    parser.add_argument("--latent_history_bonus_coef", type=float, default=0.5)
    parser.add_argument("--history_decay_beta", type=float, default=0.5)
    parser.add_argument("--history_start_relax_radius", type=float, default=45.0)
    parser.add_argument("--history_goal_relax_radius", type=float, default=45.0)
    parser.add_argument("--use_latent_pseudo_goal", action="store_true")
    parser.add_argument("--latent_goal_shift_scale", type=float, default=35.0)
    parser.add_argument("--use_inter_latent_repulsion", action="store_true")
    parser.add_argument("--inter_latent_repulsion_coef", type=float, default=0.15)
    parser.add_argument("--failure_history_weight", type=float, default=0.08)
    parser.add_argument("--history_pos_weight_cap", type=float, default=1.0)

    # initial sampling
    parser.add_argument("--initial_success_target", type=float, default=60.0)
    parser.add_argument("--initial_sampling_gamma", type=float, default=0.8)
    parser.add_argument("--zero_success_boost", type=float, default=1.0)

    # safety / goal shaping
    parser.add_argument("--use_soft_collision", action="store_true")
    parser.add_argument("--safe_distance", type=float, default=6.0)
    parser.add_argument("--safe_penalty_coef", type=float, default=0.1)
    parser.add_argument("--goal_relax_outer_radius", type=float, default=80.0)
    parser.add_argument("--goal_relax_inner_radius", type=float, default=40.0)
    parser.add_argument("--goal_soft_collision_min_scale", type=float, default=0.25)
    parser.add_argument("--goal_progress_max_scale", type=float, default=1.8)

    # route bias / latent
    parser.add_argument("--use_route_bias", action="store_true")
    parser.add_argument("--route_bias_scale", type=float, default=20.0)
    parser.add_argument("--latent_dim", type=int, default=4)

    # discriminator
    parser.add_argument("--disc_bonus_coef", type=float, default=5.0)
    parser.add_argument("--disc_loss_coef", type=float, default=1.0)
    parser.add_argument("--disc_lr", type=float, default=1e-4)

    # stuck escape
    parser.add_argument("--use_stuck_escape", action="store_true")
    parser.add_argument("--escape_lookahead", type=float, default=12.0)
    parser.add_argument("--stuck_window", type=int, default=12)
    parser.add_argument("--stuck_progress_threshold", type=float, default=6.0)
    parser.add_argument("--stuck_unique_ratio_threshold", type=float, default=0.35)
    parser.add_argument("--escape_open_length", type=float, default=15.0)
    parser.add_argument("--escape_open_weight", type=float, default=1.0)
    parser.add_argument("--escape_goal_weight", type=float, default=0.8)

    return parser.parse_args()


def make_env(args, initials, device):
    env_params = {
        "max_steps": args.max_steps,
        "success_radius": 30.0,
        "collision_threshold": 2.0,
        "action_limit": [2.0, 2.0],

        "num_dirs": args.num_dirs,
        "max_obs_range": args.max_obs_range,

        "cell_size": args.cell_size,
        "use_episode_vis": args.use_episode_vis,
        "visit_bonus": args.visit_bonus,
        "cell_repeat_penalty": args.cell_repeat_penalty,

        "use_history_vis": args.use_history_vis,
        "global_history_bonus_coef": args.global_history_bonus_coef,
        "latent_history_bonus_coef": args.latent_history_bonus_coef,
        "history_decay_beta": args.history_decay_beta,
        "history_start_relax_radius": args.history_start_relax_radius,
        "history_goal_relax_radius": args.history_goal_relax_radius,
        "use_latent_pseudo_goal": args.use_latent_pseudo_goal,
        "latent_goal_shift_scale": args.latent_goal_shift_scale,
        "use_inter_latent_repulsion": args.use_inter_latent_repulsion,
        "inter_latent_repulsion_coef": args.inter_latent_repulsion_coef,
        "failure_history_weight": args.failure_history_weight,
        "history_pos_weight_cap": args.history_pos_weight_cap,

        "initial_success_target": args.initial_success_target,
        "initial_sampling_gamma": args.initial_sampling_gamma,
        "zero_success_boost": args.zero_success_boost,

        "latent_dim": args.latent_dim,

        "use_soft_collision": args.use_soft_collision,
        "safe_distance": args.safe_distance,
        "safe_penalty_coef": args.safe_penalty_coef,

        "goal_relax_outer_radius": args.goal_relax_outer_radius,
        "goal_relax_inner_radius": args.goal_relax_inner_radius,
        "goal_soft_collision_min_scale": args.goal_soft_collision_min_scale,
        "goal_progress_max_scale": args.goal_progress_max_scale,

        "use_route_bias": args.use_route_bias,
        "route_bias_scale": args.route_bias_scale,

        "use_stuck_escape": args.use_stuck_escape,
        "escape_lookahead": args.escape_lookahead,
        "stuck_window": args.stuck_window,
        "stuck_progress_threshold": args.stuck_progress_threshold,
        "stuck_unique_ratio_threshold": args.stuck_unique_ratio_threshold,
        "escape_open_length": args.escape_open_length,
        "escape_open_weight": args.escape_open_weight,
        "escape_goal_weight": args.escape_goal_weight,
    }

    return UAVNavEnv(
        pointcloud_path=args.pointcloud_path,
        env_params=env_params,
        save_dir=args.save_dir,
        device=device,
        initials=initials,
    )


def stack_obs(obs_list, device):
    keys = obs_list[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([obs[k] for obs in obs_list], dim=0).to(device)
    return out


def get_init_and_latent(env, latent_cursor_by_initial, latent_dim):
    next_init = env.initials[env.initial_index]
    if latent_dim > 0:
        init_id = next_init["initial_id"]
        latent_id = latent_cursor_by_initial[init_id]
        latent = sample_onehot_latent_with_id(latent_dim, latent_id)
        latent_cursor_by_initial[init_id] = (latent_id + 1) % latent_dim
    else:
        latent = None
    return next_init, latent


def flush_success_trajs(pending_success_trajs, out_dir, success_counter):
    for traj_record in pending_success_trajs:
        traj_path = os.path.join(out_dir, f"{success_counter}.txt")
        with open(traj_path, "w") as f:
            ip = traj_record["initial_pose"]
            tc = traj_record["target_center"]
            f.write(f"{ip[0]} {ip[1]}\n")
            f.write(f"{tc[0]} {tc[1]}\n")
            for px, py in traj_record["traj"]:
                f.write(f"{px} {py}\n")
        success_counter += 1
    pending_success_trajs.clear()
    return success_counter


def maybe_update_discriminator(
    disc,
    disc_optimizer,
    summary_buffer,
    label_buffer,
    batch_size,
    loss_coef,
    device,
):
    if disc is None or len(summary_buffer) < batch_size:
        return None

    summaries = torch.tensor(
        np.array(summary_buffer[:batch_size], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    labels = torch.tensor(
        np.array(label_buffer[:batch_size], dtype=np.int64),
        dtype=torch.long,
        device=device,
    )

    logits = disc(summaries)
    disc_loss = F.cross_entropy(logits, labels)

    disc_optimizer.zero_grad()
    (loss_coef * disc_loss).backward()
    disc_optimizer.step()

    del summary_buffer[:batch_size]
    del label_buffer[:batch_size]
    return float(disc_loss.item())


def safe_mean(values):
    return float(np.mean(values)) if len(values) > 0 else 0.0


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    with open(args.initials_path) as f:
        initials = json.load(f)
    print(f"Loaded {len(initials)} initials")

    latent_cursor_by_initial = {
        init["initial_id"]: 0 for init in initials
    }

    if args.save_dir is None:
        args.save_dir = os.path.join("saved_data", f"run_{int(time.time())}")

    os.makedirs(args.save_dir, exist_ok=True)
    controllers_dir = os.path.join(args.save_dir, "controllers")
    success_dir = os.path.join(args.save_dir, "success_trajs")
    os.makedirs(controllers_dir, exist_ok=True)
    os.makedirs(success_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    envs = [make_env(args, initials, device) for _ in range(args.num_envs)]

    obs_dim = envs[0].observation_shape["sensor"][0]
    latent_dim = envs[0].observation_shape["latent"][0] if "latent" in envs[0].observation_shape else 0

    actor_critic = Policy(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        action_dim=2,
        action_limit=(2.0, 2.0),
    ).to(device)

    if args.resume_checkpoint is not None:
        print(f"Loading checkpoint from {args.resume_checkpoint}")
        state_dict = torch.load(args.resume_checkpoint, map_location=device)
        actor_critic.load_state_dict(state_dict, strict=True)

    disc = None
    disc_optimizer = None
    if latent_dim > 1:
        disc = LatentDiscriminator(summary_dim=9, latent_dim=latent_dim).to(device)
        disc_optimizer = torch.optim.Adam(disc.parameters(), lr=args.disc_lr)

    agent = PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=1e-5,
        max_grad_norm=args.max_grad_norm,
    )

    rollouts = DictRolloutStorage(
        args.num_steps,
        args.num_envs,
        envs[0].observation_shape,
        envs[0].action_shape,
        actor_critic.recurrent_hidden_state_size,
    )
    rollouts.to(device)

    # initial reset
    obs_list = []
    curr_trajs = []

    for env_idx, env in enumerate(envs):
        init0 = initials[env_idx % len(initials)]

        if latent_dim > 0:
            init0_id = init0["initial_id"]
            latent0_id = latent_cursor_by_initial[init0_id]
            latent0 = sample_onehot_latent_with_id(latent_dim, latent0_id)
            latent_cursor_by_initial[init0_id] = (latent0_id + 1) % latent_dim
        else:
            latent0 = None

        obs = env.reset(
            initial_pose=np.array([init0["x_start"], init0["y_start"], 0.0]),
            target_center=np.array([init0["target_center_x"], init0["target_center_y"]]),
            initial_id=init0["initial_id"],
            latent=latent0,
        )
        obs_list.append(obs)
        curr_trajs.append([[env.curr_pose[0], env.curr_pose[1]]])

    obs_batch = stack_obs(obs_list, device)
    for key in obs_batch:
        rollouts.obs[key][0].copy_(obs_batch[key])

    # training statistics
    episode_rewards = deque(maxlen=args.stats_window)

    term_type_window = deque(maxlen=args.stats_window)
    min_goal_dist_window = deque(maxlen=args.stats_window)
    final_goal_dist_window = deque(maxlen=args.stats_window)
    episode_len_window = deque(maxlen=args.stats_window)

    rb_progress = deque(maxlen=args.stats_window)
    rb_success = deque(maxlen=args.stats_window)
    rb_collision = deque(maxlen=args.stats_window)
    rb_step = deque(maxlen=args.stats_window)
    rb_soft_collision = deque(maxlen=args.stats_window)
    rb_visit = deque(maxlen=args.stats_window)
    rb_history = deque(maxlen=args.stats_window)
    rb_repulsion = deque(maxlen=args.stats_window)

    start_time = time.time()

    success_counter = 0
    pending_success_trajs = []

    total_success_episodes = 0
    total_collision_episodes = 0
    total_timeout_episodes = 0

    global_success_by_initial = {
        init["initial_id"]: 0 for init in initials
    }

    disc_summary_buffer = []
    disc_label_buffer = []
    last_disc_loss = None

    print(f"\nStarting training with {args.num_envs} envs (max_iter={args.max_iter})...\n")

    for j in range(args.max_iter):
        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(
                    {k: rollouts.obs[k][step] for k in rollouts.obs}
                )

            next_obs_list = []
            reward_list = []
            done_list = []
            bad_mask_list = []

            for env_idx, env in enumerate(envs):
                obs_next, reward, done, infos = env.step(action[env_idx:env_idx + 1])
                curr_trajs[env_idx].append([env.curr_pose[0], env.curr_pose[1]])

                info = infos[0]
                d = done[0]
                episode_return_for_log = None

                if "episode" in info:
                    episode_return_for_log = info["episode"]["r"]

                    # episode-level stats
                    term_type = info.get("term_type", "unknown")
                    term_type_window.append(term_type)

                    if term_type == "success":
                        total_success_episodes += 1
                    elif term_type == "collision":
                        total_collision_episodes += 1
                    elif term_type == "timeout":
                        total_timeout_episodes += 1

                    if "min_goal_dist" in info:
                        min_goal_dist_window.append(info["min_goal_dist"])
                    if "final_goal_dist" in info:
                        final_goal_dist_window.append(info["final_goal_dist"])
                    if "episode_len" in info:
                        episode_len_window.append(info["episode_len"])

                    rb = info.get("reward_breakdown", None)
                    if rb is not None:
                        rb_progress.append(rb.get("progress", 0.0))
                        rb_success.append(rb.get("success_bonus", 0.0))
                        rb_collision.append(rb.get("collision_penalty", 0.0))
                        rb_step.append(rb.get("step_penalty", 0.0))
                        rb_soft_collision.append(rb.get("soft_collision_penalty", 0.0))
                        rb_visit.append(rb.get("visit_reward", 0.0))
                        rb_history.append(rb.get("history_reward", 0.0))
                        rb_repulsion.append(rb.get("inter_latent_repulsion", 0.0))

                    # discriminator update signal / bonus
                    if disc is not None and "episode_summary" in info and "latent_id" in info:
                        disc_summary_buffer.append(info["episode_summary"])
                        disc_label_buffer.append(info["latent_id"])

                        if info.get("won", False):
                            with torch.no_grad():
                                summary = torch.tensor(
                                    info["episode_summary"],
                                    dtype=torch.float32,
                                    device=device,
                                ).unsqueeze(0)
                                logits = disc(summary)
                                log_probs = torch.log_softmax(logits, dim=-1)
                                disc_bonus = args.disc_bonus_coef * log_probs[0, info["latent_id"]]
                                reward = reward + disc_bonus.view(1)
                                episode_return_for_log += disc_bonus.item()

                    if episode_return_for_log is not None:
                        episode_rewards.append(episode_return_for_log)

                    # successful trajectory bookkeeping
                    if info.get("won", False):
                        pending_success_trajs.append({
                            "initial_pose": info["initial_pose"],
                            "target_center": info["target_center"],
                            "traj": curr_trajs[env_idx].copy(),
                        })
                        if "initial_id" in info:
                            global_success_by_initial[info["initial_id"]] += 1

                    # manual reset
                    env._sample_next_initial(info.get("won", False))
                    next_init, next_latent = get_init_and_latent(
                        env, latent_cursor_by_initial, latent_dim
                    )

                    obs_next = env.reset(
                        initial_pose=np.array([next_init["x_start"], next_init["y_start"], 0.0]),
                        target_center=np.array([
                            next_init["target_center_x"],
                            next_init["target_center_y"],
                        ]),
                        initial_id=next_init["initial_id"],
                        latent=next_latent,
                    )
                    curr_trajs[env_idx] = [[env.curr_pose[0], env.curr_pose[1]]]

                next_obs_list.append(obs_next)
                reward_list.append(reward)
                done_list.append(d)
                bad_mask_list.append(0.0 if "bad_transition" in info else 1.0)

            obs_next_batch = stack_obs(next_obs_list, device)
            reward_batch = torch.stack(reward_list, dim=0).to(device)
            masks = torch.tensor(
                [[0.0] if d else [1.0] for d in done_list],
                dtype=torch.float32,
                device=device,
            )
            bad_masks = torch.tensor(
                [[bm] for bm in bad_mask_list],
                dtype=torch.float32,
                device=device,
            )
            rhs = torch.zeros(args.num_envs, actor_critic.recurrent_hidden_state_size, device=device)

            rollouts.insert(
                obs_next_batch,
                rhs,
                action,
                action_log_prob,
                value,
                reward_batch,
                masks,
                bad_masks,
            )

            last_disc_loss = maybe_update_discriminator(
                disc=disc,
                disc_optimizer=disc_optimizer,
                summary_buffer=disc_summary_buffer,
                label_buffer=disc_label_buffer,
                batch_size=args.disc_batch_size,
                loss_coef=args.disc_loss_coef,
                device=device,
            )

            if len(pending_success_trajs) >= args.traj_flush_interval:
                success_counter = flush_success_trajs(
                    pending_success_trajs,
                    success_dir,
                    success_counter,
                )

        with torch.no_grad():
            next_value = actor_critic.get_value(
                {k: rollouts.obs[k][-1] for k in rollouts.obs}
            ).detach()

        rollouts.compute_returns(next_value, True, args.gamma, args.gae_lambda)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) > 0:
            elapsed = time.time() - start_time
            total_steps = (j + 1) * args.num_steps * args.num_envs
            current_success_total = success_counter + len(pending_success_trajs)

            term_counter = Counter(term_type_window)
            success_values = np.array(list(global_success_by_initial.values()), dtype=np.float32)
            num_initials_ge_1 = int((success_values >= 1).sum())
            num_initials_ge_10 = int((success_values >= 10).sum())
            median_success = float(np.median(success_values))
            mean_success = float(np.mean(success_values))

            log_msg = (
                f"[Iter {j:6d}] "
                f"steps={total_steps:9d}  "
                f"reward={safe_mean(episode_rewards):7.1f}  "
                f"success_trajs={current_success_total:6d}  "
                f"succWin={term_counter.get('success', 0):4d}  "
                f"collWin={term_counter.get('collision', 0):4d}  "
                f"timeWin={term_counter.get('timeout', 0):4d}  "
                f"succTot={total_success_episodes:6d}  "
                f"init>=1={num_initials_ge_1:3d}  "
                f"init>=10={num_initials_ge_10:3d}  "
                f"medSucc={median_success:5.1f}  "
                f"meanSucc={mean_success:5.1f}  "
                f"minGoal={safe_mean(min_goal_dist_window):6.1f}  "
                f"finalGoal={safe_mean(final_goal_dist_window):6.1f}  "
                f"epLen={safe_mean(episode_len_window):6.1f}  "
                f"v_loss={value_loss:.4f}  "
                f"entropy={dist_entropy:.4f}"
            )

            if len(rb_progress) > 0:
                log_msg += (
                    f"  r_prog={safe_mean(rb_progress):6.1f}"
                    f"  r_succ={safe_mean(rb_success):6.1f}"
                    f"  r_coll={safe_mean(rb_collision):6.1f}"
                    f"  r_step={safe_mean(rb_step):6.1f}"
                    f"  r_soft={safe_mean(rb_soft_collision):6.1f}"
                    f"  r_visit={safe_mean(rb_visit):6.1f}"
                    f"  r_hist={safe_mean(rb_history):6.1f}"
                    f"  r_rep={safe_mean(rb_repulsion):6.1f}"
                )

            log_msg += f"  elapsed={elapsed:.0f}s"
            if last_disc_loss is not None:
                log_msg += f"  disc_loss={last_disc_loss:.4f}"

            print(log_msg)

            with open(os.path.join(args.save_dir, "train_log.txt"), "a") as f:
                f.write(
                    f"{j}\t"
                    f"{safe_mean(episode_rewards):.4f}\t"
                    f"{current_success_total}\t"
                    f"{term_counter.get('success', 0)}\t"
                    f"{term_counter.get('collision', 0)}\t"
                    f"{term_counter.get('timeout', 0)}\t"
                    f"{num_initials_ge_1}\t"
                    f"{num_initials_ge_10}\t"
                    f"{median_success:.2f}\t"
                    f"{mean_success:.2f}\t"
                    f"{safe_mean(min_goal_dist_window):.2f}\t"
                    f"{safe_mean(final_goal_dist_window):.2f}\t"
                    f"{safe_mean(episode_len_window):.2f}\t"
                    f"{safe_mean(rb_progress):.2f}\t"
                    f"{safe_mean(rb_success):.2f}\t"
                    f"{safe_mean(rb_collision):.2f}\t"
                    f"{safe_mean(rb_step):.2f}\t"
                    f"{safe_mean(rb_soft_collision):.2f}\t"
                    f"{safe_mean(rb_visit):.2f}\t"
                    f"{safe_mean(rb_history):.2f}\t"
                    f"{safe_mean(rb_repulsion):.2f}\t"
                    f"{value_loss:.6f}\t"
                    f"{dist_entropy:.6f}\n"
                )

        if j % args.save_interval == 0 and j > 0:
            if pending_success_trajs:
                success_counter = flush_success_trajs(
                    pending_success_trajs,
                    success_dir,
                    success_counter,
                )
            torch.save(
                actor_critic.state_dict(),
                os.path.join(controllers_dir, f"{j}_controller.pt"),
            )

    if pending_success_trajs:
        success_counter = flush_success_trajs(
            pending_success_trajs,
            success_dir,
            success_counter,
        )

    while disc is not None and len(disc_summary_buffer) >= args.disc_batch_size:
        last_disc_loss = maybe_update_discriminator(
            disc=disc,
            disc_optimizer=disc_optimizer,
            summary_buffer=disc_summary_buffer,
            label_buffer=disc_label_buffer,
            batch_size=args.disc_batch_size,
            loss_coef=args.disc_loss_coef,
            device=device,
        )

    torch.save(
        actor_critic.state_dict(),
        os.path.join(controllers_dir, "final_controller.pt"),
    )
    print("\nTraining complete!")


if __name__ == "__main__":
    main()