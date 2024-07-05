import os
import yaml
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
from plotting import plot_value_2D, plot_discrete_actions


def main():
    # HYPERPARAMETERS
    params = {
        "env_id": "LavaGoalConveyerAcceleration-lava-v0",
        "total_timesteps": 300_000,
        "lr": 0.0005,
        "buffer_size": 1e6,
        "gamma": 0.99,
        "tau": 1,
        "target_freq": 10_000,
        "batch_size": 1024,
        "hidden_dim": 32,
        "start_epsilon": 1.0,
        "end_epsilon": 0.05,
        "exp_fraction": 0.5,
        "learning_start": 50_000,
        "seed": 1,
        "load_cp_dqn": "",
        "load_cp_con": "",
        "con_thresh": 0.1,
    }

    # DIR FOR LOGGING
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = f"runs/{params['env_id']}/{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    writer = SummaryWriter(f"{exp_dir}")

    # SAVE PARAMS AS  YAML
    with open(f"{exp_dir}/params.yaml", "w") as f:
        yaml.dump(params, f)

    # SEEDING
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ENVIRONMENT
    env = gym.make(params["env_id"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if params["load_cp_con"]:
        raise NotImplementedError("Constrained model is not implemented yet.")
    else:
        con_model = None

    # MODEL
    dqn = DQN(
        action_dim=action_dim,
        state_dim=state_dim,
        hidden_dim=params["hidden_dim"],
        device=device,
        lr=params["lr"],
        gamma=params["gamma"],
        load_cp=params["load_cp_dqn"],
        con_model=con_model,
        con_thresh=params["con_thresh"],
    )

    replay_buffer = ReplayBuffer(
        buffer_size=int(params["buffer_size"]),
        observation_space=env.observation_space,
        action_space=env.action_space,
        handle_timeout_termination=False
    )

    # TRAINING
    epsilon_vals = np.linspace(params["start_epsilon"], params["end_epsilon"], int(params["exp_fraction"] * params["total_timesteps"]))
    obs, info = env.reset()
    episodes_done, ep_len, ep_reward_sum, ep_acc_violations = 0, 0, 0, 0
    loss_hist = []
    avg_q_hist = []
    ep_reward_hist = []
    ep_len_hist = []
    ep_acc_violations_hist = []
    for global_step in range(params["total_timesteps"]):
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

            print(f"Episode {episodes_done} | Length: {ep_len} | Reward: {ep_reward_sum} | Acc Violations: {ep_acc_violations} | {global_step} / {params['total_timesteps']} steps")

            ep_len, ep_reward_sum, ep_acc_violations = 0, 0, 0
            episodes_done += 1

        if global_step > params["learning_start"]:
            batch = replay_buffer.sample(params["batch_size"])
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

            if global_step % params["target_freq"] == 0:
                dqn.target_update(params["tau"])

    # SAVE MODEL AND DATA
    dqn.save_model(exp_dir)
    replay_buffer.save(f"{exp_dir}/replay_buffer.npz")

    # PLOT TRAINING CURVES
    titles = ["Loss Q", "Avg Q", "Episode Reward", "Episode Length", "Episode Acc Violations"]
    graphs = [loss_hist, avg_q_hist, ep_reward_hist, ep_len_hist, ep_acc_violations_hist]
    for y_data, title in zip(graphs, titles):
        plt.plot(y_data)
        plt.title(title)
        plt.savefig(f"{exp_dir}/{title}.png")
        plt.close()

    for vel in [
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([-2.0, 0.0]),
        np.array([0.0, 2.0]),
        np.array([0.0, -2.0]),
    ]:
        value_function = "min"
        plot_value_2D(
            dqn=dqn.q_net,
            velocity=vel,
            value_function=value_function,
            env=env,
            x_lim=env.x_range,
            x_steps=env.x_range[-1] + 1,
            y_lim=env.y_range,
            y_steps=env.y_range[-1] + 1,
            device=device,
            save_path=f"{exp_dir}/vf:{value_function}_velocity:{vel}.png"
        )

    for eval_state in env.eval_states:
        plot_discrete_actions(
            dqn=dqn.q_net,
            state=eval_state,
            action_map=env.action_map,
            device=device,
            save_path=f"{exp_dir}/qf_state:{eval_state}.png",
        )


if __name__ == "__main__":
    main()
