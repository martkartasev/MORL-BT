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
from envs.unity_misc import rewards_flat_acc_env, done_check_flat_acc_env
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
        file_name=f"envs/unity_builds/{unity_scene_dir}/myBuild-MORL-BT.x86_64",  # comment out to connect to unity editor instance
        no_graphics=False,  # Can disable graphics if needed
        # base_port=10001,  # for starting multiple envs
        side_channels=[engine])
    print("Unity env ready")

    action_dim = 25
    # state_dim = 9  # for flat env with pos, acc, goal
    state_dim = 17  # for flat env with pos, acc, goal, trigger, button
    env.action_space = gym.spaces.Discrete(action_dim)
    env.observation_space = gym.spaces.Box(
        low=np.array([-np.inf] * state_dim),
        high=np.array([np.inf] * state_dim),
        dtype=np.float32
    )
    n_agents = 16  # number of agents in the unity scene

    env.reset()  # init unity env and all agents within

    episodes_done = 0
    loss_hist = []
    avg_q_hist = []
    ep_reward_hist = []
    ep_len_hist = []
    ep_acc_violations_hist = []
    ep_reward_sum = np.zeros((n_agents, 1))
    ep_len = np.zeros((n_agents, 1))
    ep_acc_violations = np.zeros((n_agents, 1))

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

    return env, state_dim, action_dim, logging_dict


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
        writer.add_scalar("episode/length", logging_dict["ep_len"], logging_dict["episodes_done"])
        writer.add_scalar("episode/reward_sum", logging_dict["ep_reward_sum"], logging_dict["episodes_done"])
        writer.add_scalar("episode/acc_violations", logging_dict["ep_acc_violations"], logging_dict["episodes_done"])
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


def env_interaction_unity_env(
        dqn,
        epsilon,
        env,
        replay_buffer,
        writer,
        params,
        logging_dict
):
    (decision_steps, terminal_steps) = env.get_steps("BridgeEnv?team=0")
    obs = decision_steps.obs[0]  # Strange structure, but this is how you get the observations array
    nr_agents = len(decision_steps)  # this many agents need to take an action

    if nr_agents > 0:
        unity_actions = np.zeros((nr_agents, 3))
        for i in decision_steps.agent_id:
            rl_action = dqn.act(obs[i], epsilon)
            logging_dict["ep_len"][i] += 1

            screenshot_action = 1 if params["unity_take_screenshots"] else 0
            reset_action = 0 if params["unity_max_ep_len"] > logging_dict["ep_len"][i] else 1
            unity_actions[i] = [rl_action, reset_action, screenshot_action]

            if screenshot_action:
                raise NotImplementedError("Taking screenshots is not implemented yet.")

    else:
        unity_actions = np.zeros((0, 3))  # we still need to pass an empty action tuple even if no agent acts

    action_tuple = ActionTuple()
    action_tuple.add_discrete(unity_actions)
    env.set_actions("BridgeEnv?team=0", action_tuple)
    env.step()
    (next_decision_steps, next_terminal_steps) = env.get_steps("BridgeEnv?team=0")

    if len(next_decision_steps.agent_id) == 0 and len(next_terminal_steps.agent_id) > 0:
        # some episode has ended, we need to handle that
        for idx, j in enumerate(next_terminal_steps.agent_id):
            if j in decision_steps.agent_id:
                # find those agents that did a step and now are done, aka those that did a valid (s, a, s') transition
                agent_obs = obs[j][:]
                next_obs = next_terminal_steps.obs[0][idx][:]
                action = unity_actions[j][0]

                # compute reward
                rew = np.array([rewards_flat_acc_env(agent_obs, task=params["unity_task"])])
                logging_dict["ep_reward_sum"][j] += rew

                # check done
                done = done_check_flat_acc_env(obs)
                if done:
                    print(f"Agent {j} is done (terminal state)!")
                else:
                    print(f"Agent {j} is done (episode trunaction)! Return: {np.around(logging_dict['ep_reward_sum'][j], decimals=3)}")

                writer.add_scalar("episode/reward_sum", logging_dict['ep_reward_sum'][j], logging_dict["episodes_done"])
                writer.add_scalar("episode/length", logging_dict['ep_len'][j], logging_dict["episodes_done"])
                logging_dict["ep_len_hist"].append(logging_dict["ep_len"][j].copy())
                logging_dict["ep_reward_hist"].append(logging_dict['ep_reward_sum'][j].copy())
                logging_dict['ep_reward_sum'][j] = 0
                logging_dict["ep_len"][j] = 0
                logging_dict["episodes_done"] += 1

                termination = np.array([int(done)])
                info = None
                replay_buffer.add(agent_obs, next_obs, action, rew, termination, info)
            else:
                # happens after reset, we have a "next obs" but not yet a "current obs"
                # transition will be added to the buffer after the next env interaction
                pass

    elif len(next_decision_steps.agent_id) > 0 and len(next_terminal_steps.agent_id) == 0 and len(decision_steps.agent_id) == len(next_decision_steps.agent_id):
        # all agents have taken a step and none has terminates
        for j in next_decision_steps.agent_id:
            agent_obs = obs[j][:]
            next_obs = next_decision_steps.obs[0][j][:]
            action = unity_actions[j][0]

            # compute reward
            rew = np.array([rewards_flat_acc_env(agent_obs, task=params["unity_task"])])
            logging_dict["ep_reward_sum"][j] += rew

            # check done ( should not be done here, since those cases are handeled above...
            done = done_check_flat_acc_env(obs)
            if done:
                raise RuntimeWarning(f"Agent {j} is done, but should not be done here!")

            termination = np.array([int(done)])
            info = None
            replay_buffer.add(agent_obs, next_obs, action, rew, termination, info)

    else:
        # this seems to happen after reset, we have next obs but not yet current obs
        pass


def main():
    # HYPERPARAMETERS
    which_env = "unity"
    params = {
        "which_env": which_env,  # "unity" or "numpy"
        # "env_id": "LavaGoalConveyerAcceleration-lava-v0",
        "env_id": "flat-acc-button",  # name of the folder containing the unity scene binaries
        "unity_take_screenshots": False,
        "unity_max_ep_len": 1000,
        "unity_task": "fetch_trigger",
        "total_timesteps": 200_000,
        "lr": 0.0005,
        "buffer_size": 1e6,
        "gamma": 0.99,
        "tau": 1,
        "target_freq": 10_000,
        "batch_size": 256 if which_env == "unity" else 1024,
        "hidden_dim": 64 if which_env == "unity" else 32,
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
    exp_dir = f"runs/{params['env_id']}"
    if params["which_env"] == "unity":
        exp_dir += f"_{params['unity_task']}"

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir += timestamp

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
        env, state_dim, action_dim, logging_dict = setup_unity_env(params["env_id"])
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
        if params["which_env"] == "numpy":
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
        elif params["which_env"] == "unity":
            obs = env_interaction_unity_env(
                dqn=dqn,
                epsilon=epsilon,
                env=env,
                replay_buffer=replay_buffer,
                writer=writer,
                params=params,
                logging_dict=logging_dict
            )
        else:
            raise ValueError(f"which_env must be 'numpy' or 'unity' but got '{params['which_env']}'")

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

        if global_step % 10_000 == 0:
            print(f"Step {global_step} / {params['total_timesteps']}, saving model and buffer...")
            dqn.save_model(exp_dir)
            replay_buffer.save(f"{exp_dir}/replay_buffer.npz")

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
