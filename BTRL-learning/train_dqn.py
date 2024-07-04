import os

import numpy as np
import matplotlib.pyplot as plt
import random
import gymnasium as gym
import time
import torch
from torch.utils.tensorboard import SummaryWriter

import envs
from misc import ReplayBuffer
from dqn import DQN


def main():
    # HYPERPARAMETERS
    class PARAMS:
        env_id = "LavaGoalConveyerAcceleration-lava-v0"
        total_timesteps = 30_000
        lr = 0.0005
        buffer_size = 1e6
        gamma = 0.99
        tau = 1
        target_freq = 1_000
        batch_size = 1024
        hidden_dim = 32
        start_epsilon = 1.0
        end_epsilon = 0.05
        exp_fraction = 0.5
        learning_start = 5_000
        seed = 0
        load_cp_dqn = ""
        load_cp_con = ""
        con_thresh = 0.1

    # DIR FOR LOGGING
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = f"runs/{PARAMS.env_id}/{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    writer = SummaryWriter(f"{exp_dir}")

    # write args to txt file in exp_dir
    with open(f"{exp_dir}/args.txt", "w") as f:
        for arg in vars(PARAMS):
            f.write(f"{arg}: {getattr(PARAMS, arg)}\n")

    # SEEDING
    random.seed(PARAMS.seed)
    np.random.seed(PARAMS.seed)
    torch.manual_seed(PARAMS.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ENVIRONMENT
    env = gym.make(PARAMS.env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if PARAMS.load_cp_con:
        raise NotImplementedError("Constrained model is not implemented yet.")
    else:
        con_model = None

    # MODEL
    dqn = DQN(
        action_dim=action_dim,
        state_dim=state_dim,
        hidden_dim=PARAMS.hidden_dim,
        device=device,
        lr=PARAMS.lr,
        gamma=PARAMS.gamma,
        load_cp=PARAMS.load_cp_dqn,
        con_model=con_model,
        con_thresh=PARAMS.con_thresh,
    )

    replay_buffer = ReplayBuffer(
        buffer_size=int(PARAMS.buffer_size),
        observation_space=env.observation_space,
        action_space=env.action_space,
        handle_timeout_termination=False
    )

    # TRAINING
    epsilon_vals = np.linspace(PARAMS.start_epsilon, PARAMS.end_epsilon, int(PARAMS.exp_fraction * PARAMS.total_timesteps))
    obs, info = env.reset()
    episodes_done, ep_len, ep_reward_sum, ep_acc_violations = 0, 0, 0, 0
    loss_hist = []
    avg_q_hist = []
    ep_reward_hist = []
    ep_len_hist = []
    ep_acc_violations_hist = []
    for global_step in range(PARAMS.total_timesteps):
        epsilon = epsilon_vals[min(global_step, len(epsilon_vals) - 1)]
        writer.add_scalar("epsilon", epsilon, global_step
                          )
        action = dqn.act(obs, epsilon)
        next_obs, reward, done, trunc, info = env.step(action)
        replay_buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            infos=info)

        obs = next_obs
        ep_len += 1
        ep_reward_sum += reward

        if (done or trunc):
            obs, info = env.reset()
            writer.add_scalar("episode/length", ep_len, global_step)
            writer.add_scalar("episode/reward_sum", ep_reward_sum, global_step)
            writer.add_scalar("episode/acc_violations", ep_acc_violations, global_step)
            ep_reward_hist.append(ep_reward_sum)
            ep_len_hist.append(ep_len)
            ep_acc_violations_hist.append(ep_acc_violations)

            ep_len, ep_reward_sum, ep_acc_violations = 0, 0, 0
            episodes_done += 1

        if global_step > PARAMS.learning_start:
            batch = replay_buffer.sample(PARAMS.batch_size)
            loss, avg_q = dqn.update(
                state_batch=batch.observations,
                action_batch=batch.actions,
                reward_batch=batch.rewards,
                next_state_batch=batch.next_observations,
                done_batch=batch.dones,
            )
            writer.add_scalar("train/q_loss", loss, global_step)
            writer.add_scalar("train/avg_q", avg_q, global_step)
            loss_hist.append(loss)
            avg_q_hist.append(avg_q)

            if global_step % PARAMS.target_freq == 0:
                dqn.target_update(PARAMS.tau)

    # SAVE MODEL AND DATA
    dqn.save_model(exp_dir)
    replay_buffer.save(f"{exp_dir}/replay_buffer.npz")

    # PLOT TRAINING CURVES
    titles = ["Loss", "Avg Q", "Episode Reward", "Episode Length", "Episode Acc Violations"]
    graphs = [loss_hist, avg_q_hist, ep_reward_hist, ep_len_hist, ep_acc_violations_hist]
    for y_data, title in zip(graphs, titles):
        plt.plot(y_data)
        plt.title(title)
        plt.savefig(f"{exp_dir}/{title}.png")
        plt.close()


if __name__ == "__main__":
    main()
