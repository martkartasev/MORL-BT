import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import random
import gymnasium as gym
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import envs
from misc import ReplayBuffer
from dqn import DQN
from plotting import create_plots_numpy_env


def setup_numpy_env(env_id):
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    obs, info = env.reset()
    episodes_done, ep_len, ep_reward_sum, ep_acc_violations = 0, 0, 0, 0
    loss_hist = []
    avg_q_hist = []
    ep_reward_hist = []
    ep_len_hist = []
    ep_acc_violations_hist = []

    logging_dict = {
        "episodes_done": episodes_done,
        "ep_len": ep_len,
        "ep_reward_sum": ep_reward_sum,
        "ep_acc_violations": ep_acc_violations,
        "loss_hist": loss_hist,
        "avg_q_hist": avg_q_hist,
        "ep_reward_hist": ep_reward_hist,
        "ep_len_hist": ep_len_hist,
        "ep_acc_violations_hist": ep_acc_violations_hist
    }

    return env, state_dim, action_dim, obs, info, logging_dict


def setup_unity_env(unity_scene_dir):
    engine = EngineConfigurationChannel()
    engine.set_configuration_parameters(time_scale=2)  # Can speed up simulation between steps with this
    engine.set_configuration_parameters(quality_level=0)
    engine.set_configuration_parameters(width=1000, height=1000)
    print("Creating unity env (instance started?)...")
    env = UnityEnvironment(
        file_name=f"envs/{unity_scene_dir}/myBuild-MORL-BT.x86_64",  # comment out to connect to unity editor instance
        no_graphics=False,  # Can disable graphics if needed
        # base_port=10001,  # for starting multiple envs
        side_channels=[engine])
    print("Unity env ready")

    action_dim = 25
    # state_dim = 9  # for flat env with pos, acc, goal
    state_dim = 17  # for flat env with pos, acc, goal, trigger, button
    env.single_action_space = gym.spaces.Discrete(action_dim)
    env.single_observation_space = gym.spaces.Box(
        low=np.array([-np.inf] * state_dim),
        high=np.array([np.inf] * state_dim),
        dtype=np.float32
    )
    n_agents = 16  # number of agents in the unity scene

    env.reset()  # init unity env and all agents within
    (decision_steps, terminal_steps) = env.get_steps("BridgeEnv?team=0")
    obs = decision_steps.obs[0]
    info = {}

    episodes_done = 0
    loss_hist = []
    avg_q_hist = []
    ep_reward_hist = []
    ep_len_hist = []
    ep_acc_violations_hist = []
    ep_reward_sum = np.zeros((n_agents, 1))
    ep_len = np.zeros((n_agents, 1))
    ep_acc_violations = np.zeros((n_agents, 1))

    return env, state_dim, action_dim, obs, info, episodes_done, ep_len, ep_reward_sum, ep_acc_violations, loss_hist, avg_q_hist, ep_reward_hist, ep_len_hist, ep_acc_violations_hist


def env_interaction_numpy_env(
        dqn,
        obs,
        epsilon,
        env,
        replay_buffer,
        writer,
        global_step,
        params,
        logging_dict
):
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
    logging_dict["ep_len"] += 1
    logging_dict["ep_reward_sum"] += reward

    if (done or trunc):
        obs, info = env.reset()
        writer.add_scalar("episode/length", logging_dict["ep_len"], global_step)
        writer.add_scalar("episode/reward_sum", logging_dict["ep_reward_sum"], global_step)
        writer.add_scalar("episode/acc_violations", logging_dict["ep_acc_violations"], global_step)
        logging_dict["ep_reward_hist"].append(logging_dict["ep_reward_sum"])
        logging_dict["ep_len_hist"].append(logging_dict["ep_len"])
        logging_dict["ep_acc_violations_hist"].append(logging_dict["ep_acc_violations"])

        print(
            f"Episode {logging_dict['episodes_done']} | "
            f"Length: {logging_dict['ep_len']} | "
            f"Reward: {logging_dict['ep_reward_sum']} | "
            f"Acc Violations: {logging_dict['ep_acc_violations']} | "
            f"{global_step} / {params['total_timesteps']} steps")

        logging_dict["ep_len"] = 0
        logging_dict["ep_reward_sum"] = 0
        logging_dict["ep_acc_violations"] = 0
        logging_dict["episodes_done"] += 1
        
    return obs


def main():
    # HYPERPARAMETERS
    params = {
        "which_env": "numpy",  # "unity" or "numpy"
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
        "load_cp_dqn": "runs/LavaGoalConveyerAcceleration-lava-v0/2024-07-04-21-02-36_goodRepr/q_net.pth",
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

    # ENVIRONMENT SETUP
    if params["which_env"] == "numpy":
        env, state_dim, action_dim, obs, info, logging_dict = setup_numpy_env(params["env_id"])
    elif params["which_env"] == "unity":
        raise NotImplementedError("Unity environment is not implemented yet.")
    else:
        raise ValueError(f"which_env must be 'numpy' or 'unity' but got '{params['which_env']}'")

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
    for global_step in range(params["total_timesteps"]):
        epsilon = epsilon_vals[min(global_step, len(epsilon_vals) - 1)]
        writer.add_scalar("epsilon", epsilon, global_step)

        # one-step interaction with the environment
        obs = env_interaction_numpy_env(
            dqn=dqn,
            obs=obs,
            epsilon=epsilon,
            env=env,
            replay_buffer=replay_buffer,
            writer=writer,
            global_step=global_step,
            params=params,
            logging_dict=logging_dict
        )

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
            logging_dict["loss_hist"].append(loss)
            logging_dict["avg_q_hist"].append(avg_q)

            if global_step % params["target_freq"] == 0:
                dqn.target_update(params["tau"])

    # SAVE MODEL AND DATA
    dqn.save_model(exp_dir)
    replay_buffer.save(f"{exp_dir}/replay_buffer.npz")

    # PLOT TRAINING CURVES
    titles = ["Loss Q", "Avg Q", "Episode Reward", "Episode Length", "Episode Acc Violations"]
    graphs = [
        logging_dict["loss_hist"],
        logging_dict["avg_q_hist"],
        logging_dict["ep_reward_hist"],
        logging_dict["ep_len_hist"],
        logging_dict["ep_acc_violations_hist"]
        ]
    for y_data, title in zip(graphs, titles):
        plt.plot(y_data)
        plt.title(title)
        plt.savefig(f"{exp_dir}/{title}.png")
        plt.close()

    if params["which_env"] == "numpy":
        create_plots_numpy_env(
            network=dqn.q_net,
            env=env,
            device=device,
            save_dir=f"{exp_dir}"
        )

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
