import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import random
import gymnasium as gym
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# from mlagents_envs.base_env import ActionTuple
# from mlagents_envs.environment import UnityEnvironment
# from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import envs
from envs.unity_misc import rewards_flat_acc_env, done_check_flat_acc_env, unity_state_predicate_check, unity_state_predicate_names
from misc import ReplayBuffer
from dqn import DQN
from plotting import create_plots_numpy_env, plot_unity_q_vals


def setup_numpy_env(params, device, exp_dir):
    env_id = params["env_id"]
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    obs, info = env.reset()
    episodes_done, ep_len, ep_reward_sum = 0, 0, 0
    loss_hist = []
    avg_q_hist = []
    ep_reward_hist = []
    ep_len_hist = []
    ep_state_predicates = np.zeros(len(env.state_predicate_names))
    ep_state_predicate_hist = []

    logging_dict = {
        "episodes_done": episodes_done,
        "ep_len": ep_len,
        "ep_reward_sum": ep_reward_sum,
        "ep_state_predicates": ep_state_predicates,
        "loss_hist": loss_hist,
        "avg_q_hist": avg_q_hist,
        "ep_reward_hist": ep_reward_hist,
        "ep_len_hist": ep_len_hist,
        "ep_state_predicate_hist": ep_state_predicate_hist
    }

    avoid_lava_dqn = DQN(
        action_dim=action_dim,
        state_dim=state_dim,
        hidden_arch=params["numpy_env_lava_dqn_arch"],
        hidden_activation=params["hidden_activation"],
        device=device,
        lr=params["lr"],
        gamma=params["gamma"],
        load_cp=params["numpy_env_lava_dqn_cp"],
        con_model_load_cp="",  # highest prio, no constraint...
        con_thresh=np.inf,  # value does not matter without constraint...
        model_name="avoid_lava",
    )

    reach_goal_dqn = DQN(
        action_dim=action_dim,
        state_dim=state_dim,
        hidden_arch=params["numpy_env_goal_dqn_arch"],
        hidden_activation=params["hidden_activation"],
        device=device,
        lr=params["lr"],
        gamma=params["gamma"],
        load_cp=params["numpy_env_goal_dqn_cp"],
        con_model_load_cp=params["numpy_env_lava_feasibility_dqn_cp"],
        con_model_arch=params["numpy_env_lava_feasibility_dqn_arch"],
        con_thresh=params["numpy_env_feasibility_thresh"],
        model_name="reach_goal",
    )

    if "lava" in env_id:
        dqns = [avoid_lava_dqn]
    elif "goal" in env_id:
        avoid_lava_dqn.save_model(exp_dir)
        dqns = [avoid_lava_dqn, reach_goal_dqn]
    else:
        raise ValueError(f"Unknown env-id '{env_id}', not sure which DQNs to use...")

    return env, state_dim, action_dim, obs, info, logging_dict, dqns


def setup_unity_env(unity_scene_dir, take_screenshots=False):
    engine = EngineConfigurationChannel()
    engine.set_configuration_parameters(time_scale=2)  # Can speed up simulation between steps with this
    engine.set_configuration_parameters(quality_level=0)
    engine.set_configuration_parameters(width=1000, height=1000)
    print("Creating unity env (instance started?)...")
    if take_screenshots:
        env = UnityEnvironment(
            # file_name=f"envs/unity_builds/{unity_scene_dir}/myBuild-MORL-BT.x86_64",
            # comment out to connect to unity editor instance
            no_graphics=False,  # Can disable graphics if needed
            # base_port=10001,  # for starting multiple envs
            side_channels=[engine])
    else:
        env = UnityEnvironment(
            file_name=f"envs/unity_builds/{unity_scene_dir}/myBuild-MORL-BT.x86_64",  # comment out to connect to unity editor instance
            no_graphics=False,  # Can disable graphics if needed
            # base_port=10001,  # for starting multiple envs
            side_channels=[engine])
    print("Unity env ready")

    dqns = []
    raise NotImplementedError("Unity env DQN setup for BT not implemented yet...")

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

    env.state_predicate_names = unity_state_predicate_names

    env.check_state_predicates = unity_state_predicate_check

    env.reset()  # init unity env and all agents within

    episodes_done = 0
    loss_hist = []
    avg_q_hist = []
    ep_reward_hist = []
    ep_len_hist = []
    ep_state_predicate_hist = []
    ep_reward_sum = np.zeros((n_agents, 1))
    ep_len = np.zeros((n_agents, 1))
    ep_state_predicate = np.zeros((n_agents, len(env.state_predicate_names)))

    logging_dict = {
        "episodes_done": episodes_done,
        "ep_len": ep_len,
        "ep_reward_sum": ep_reward_sum,
        "ep_state_predicates": ep_state_predicate,
        "loss_hist": loss_hist,
        "avg_q_hist": avg_q_hist,
        "ep_reward_hist": ep_reward_hist,
        "ep_len_hist": ep_len_hist,
        "ep_state_predicate_hist": ep_state_predicate_hist
    }

    return env, state_dim, action_dim, logging_dict, dqns


def env_interaction_numpy_env(
        dqns,
        obs,
        epsilon,
        env,
        replay_buffer,
        writer,
        global_step,
        params,
        logging_dict
):

    # BT is just if-else, and only active if we are training the goal reach DQN
    if len(dqns) == 2:
        agent_x = obs[0]
        agent_y = obs[1]
        # if env.conveyer_x_min < agent_x < env.lava_x_max and env.conveyer_y_min < agent_y < env.lava_y_max:
        if env.lava_x_min < agent_x < env.lava_x_max and env.lava_y_min < agent_y < env.lava_y_max:
            print("Agent in lava or on conveyer, using avoid DQN")
            dqn_idx = 0
        else:
            dqn_idx = 1
    else:
        dqn_idx = 0

    action = dqns[dqn_idx].act(obs, epsilon)
    next_obs, reward, done, trunc, info = env.step(action)

    if dqn_idx == len(dqns) - 1:
        # only add transition to the replay buffer when the DQN we are currently learning is used...
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
    logging_dict["ep_state_predicates"] += info["state_predicates"]

    if (done or trunc):
        obs, info = env.reset()
        writer.add_scalar("episode/length", logging_dict["ep_len"], logging_dict["episodes_done"])
        writer.add_scalar("episode/reward_sum", logging_dict["ep_reward_sum"], logging_dict["episodes_done"])
        # writer.add_scalar("episode/acc_violations", logging_dict["ep_acc_violations"], logging_dict["episodes_done"])
        for i, state_predicate in enumerate(env.state_predicate_names):
            writer.add_scalar(f"episode/{state_predicate}", logging_dict["ep_state_predicates"][i], logging_dict["episodes_done"])
        logging_dict["ep_reward_hist"].append(logging_dict["ep_reward_sum"])
        logging_dict["ep_len_hist"].append(logging_dict["ep_len"])
        logging_dict["ep_state_predicate_hist"].append(logging_dict["ep_state_predicates"])

        print(
            f"Episode {logging_dict['episodes_done']} | "
            f"Length: {logging_dict['ep_len']} | "
            f"Reward: {logging_dict['ep_reward_sum']} | "
            f"{global_step} / {params['total_timesteps']} steps")

        logging_dict["ep_len"] = 0
        logging_dict["ep_reward_sum"] = 0
        logging_dict["ep_state_predicates"] = np.zeros(len(env.state_predicate_names))
        logging_dict["episodes_done"] += 1
        
    return obs


def env_interaction_unity_env(
        dqns,
        epsilon,
        env,
        device,
        exp_dir,
        global_step,
        replay_buffer,
        writer,
        params,
        logging_dict
):
    raise NotImplementedError("Unity env interaction for BT with mutiple DQNs not implemented yet...")

    (decision_steps, terminal_steps) = env.get_steps("BridgeEnv?team=0")
    obs = decision_steps.obs[0]  # Strange structure, but this is how you get the observations array
    nr_agents = len(decision_steps)  # this many agents need to take an action

    if nr_agents > 0:
        unity_actions = np.zeros((nr_agents, 3))
        for i in decision_steps.agent_id:
            rl_action, q_vals = dqn.act(obs[i], epsilon, ret_vals=True)
            logging_dict["ep_len"][i] += 1

            screenshot_action = 0
            reset_action = 0 if params["unity_max_ep_len"] > logging_dict["ep_len"][i] else 1

            if params["unity_take_screenshots"]:
                if -3 < obs[i][0] < 3:
                # if True:
                    screenshot_action = 1
                    os.makedirs(f"{exp_dir}/imgs/Q", exist_ok=True)
                    plot_unity_q_vals(
                        obs[i],
                        dqn.q_net,
                        device,
                        save_path=f"{exp_dir}/imgs/Q/{global_step}_Q.png",
                        title=f"Q values: xyz={obs[i][0:3]}",
                    )
                    if dqn.con_model is not None:
                        os.makedirs(f"{exp_dir}/imgs/ACC", exist_ok=True)
                        plot_unity_q_vals(
                            obs[i],
                            dqn.con_model,
                            device,
                            save_path=f"{exp_dir}/imgs/ACC/{global_step}_reachabilityQ.png",
                            title=f"feasibility Q values, xzy={obs[i][0:3]}",
                            vmin=0,
                            vmax=1,
                            con_thresh=dqn.con_thresh,
                        )

            unity_actions[i] = [rl_action, reset_action, screenshot_action]

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
                logging_dict["ep_len"][j] += 1

                state_predicates = env.check_state_predicates(agent_obs)
                logging_dict["ep_state_predicates"][j] += state_predicates

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
                logging_dict["ep_state_predicate_hist"].append(logging_dict["ep_state_predicates"][j].copy())
                logging_dict['ep_reward_sum'][j] = 0
                logging_dict["ep_len"][j] = 0
                logging_dict["episodes_done"] += 1
                logging_dict["ep_state_predicates"][j] = np.zeros(len(env.state_predicate_names))

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
            logging_dict["ep_len"][j] += 1

            state_predicates = env.check_state_predicates(agent_obs)
            logging_dict["ep_state_predicates"][j] += state_predicates

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
    which_env = "numpy"  # "unity" or "numpy
    # which_env = "unity"  # "unity" or "numpy
    params = {
        "which_env": which_env,
        # "env_id": "LavaGoalConveyerAcceleration-lava-v0",
        # "env_id": "LavaGoalConveyerAcceleration-lava-noConveyer-v0",
        # "env_id": "SimpleAccEnv-lava-v0",
        # "env_id": "SimpleAccEnv-withConveyer-lava-v0",
        # "env_id": "SimpleAccEnv-goal-v0",
        "env_id": "SimpleAccEnv-withConveyer-goal-v0",
        # "env_id": "flat-acc-button",  # name of the folder containing the unity scene binaries
        # "env_id": "flat-acc",  # name of the folder containing the unity scene binaries
        "unity_take_screenshots": True,
        "unity_max_ep_len": 1000,
        "unity_task": "fetch_trigger",
        # "unity_task": "reach_goal",
        "total_timesteps": 100_000,
        "lr": 0.0005,
        "buffer_size": 1e6,
        "gamma": 0.99,
        "tau": 0.005,
        "target_freq": 1,
        "batch_size": 256,
        "hidden_activation": nn.ReLU,
        "start_epsilon": 1.0,
        "end_epsilon": 0.05,
        "exp_fraction": 0.5,
        "learning_start": 10_000,
        "seed": 1,
        "numpy_env_lava_dqn_cp": "runs/SimpleAccEnv-withConveyer-lava-v0/2024-07-13-11-08-28_netArch/avoid_lava_net.pth",
        "numpy_env_lava_dqn_arch": [32, 32, 16, 16],
        # "numpy_env_lava_feasibility_dqn_cp": "",
        "numpy_env_lava_feasibility_dqn_cp": "runs/SimpleAccEnv-withConveyer-goal-v0/2024-07-11-11-06-24_250k/feasibility_2024-07-12-12-18-19_hiddenArch-32-16_hardTarget/feasibility_dqn.pt",
        "numpy_env_lava_feasibility_dqn_arch": [32, 32, 16, 16],
        "numpy_env_feasibility_thresh": 0.1,
        "numpy_env_goal_dqn_cp": "",
        # "numpy_env_goal_dqn_cp": "runs/SimpleAccEnv-withConveyer-goal-v0/2024-07-13-11-45-59_BT_withCon/reach_goal_net.pth",
        # "numpy_env_goal_dqn_cp": "runs/SimpleAccEnv-withConveyer-goal-v0/2024-07-13-12-46-08_BT_noCon/reach_goal_net.pth",
        # "numpy_env_goal_dqn_cp": "runs/SimpleAccEnv-withConveyer-goal-v0/2024-07-13-16-53-55_BT_withCon_batch256_clipGrad1_hiddenArch256/reach_goal_net.pth",
        "numpy_env_goal_dqn_arch": [256, 256],
    }

    # DIR FOR LOGGING
    exp_dir = f"runs/{params['env_id']}"
    if params["which_env"] == "unity":
        exp_dir += f"_{params['unity_task']}"

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir += f"/{timestamp}"

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
        env, state_dim, action_dim, obs, info, logging_dict, dqns = setup_numpy_env(params=params, device=device, exp_dir=exp_dir)
    elif params["which_env"] == "unity":
        env, state_dim, action_dim, logging_dict, dqns = setup_unity_env(params["env_id"], take_screenshots=params["unity_take_screenshots"])
    else:
        raise ValueError(f"which_env must be 'numpy' or 'unity' but got '{params['which_env']}'")

    learn_dqn = dqns[-1]  # we always only learn the last DQN, all other DQNs in list must be trained already

    # REPLAY BUFFER
    replay_buffer = ReplayBuffer(
        buffer_size=int(params["buffer_size"]),
        observation_space=env.observation_space,
        action_space=env.action_space,
        handle_timeout_termination=False
    )

    # TRAINING
    epsilon_vals = np.linspace(params["start_epsilon"], params["end_epsilon"], int(params["exp_fraction"] * params["total_timesteps"]))
    for global_step in range(params["total_timesteps"]):
        # if params["load_cp_dqn"]:
        #     epsilon = epsilon_vals[-1]
        # else:
        epsilon = epsilon_vals[min(global_step, len(epsilon_vals) - 1)]
        writer.add_scalar("epsilon", epsilon, global_step)

        # one-step interaction with the environment
        if params["which_env"] == "numpy":
            obs = env_interaction_numpy_env(
                dqns=dqns,
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
                dqns=dqns,
                epsilon=epsilon,
                env=env,
                device=device,
                exp_dir=exp_dir,
                global_step=global_step,
                replay_buffer=replay_buffer,
                writer=writer,
                params=params,
                logging_dict=logging_dict
            )
        else:
            raise ValueError(f"which_env must be 'numpy' or 'unity' but got '{params['which_env']}'")

        if global_step > params["learning_start"]:
            batch = replay_buffer.sample(params["batch_size"])
            loss, avg_q = learn_dqn.update(
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
                learn_dqn.target_update(params["tau"])

        if global_step % 10_000 == 0:
            print(f"Step {global_step} / {params['total_timesteps']}, saving model and buffer...")
            learn_dqn.save_model(exp_dir)
            replay_buffer.save(f"{exp_dir}/replay_buffer.npz")

    # SAVE MODEL AND DATA
    learn_dqn.save_model(exp_dir)
    replay_buffer.save(f"{exp_dir}/replay_buffer.npz")

    # PLOT TRAINING CURVES
    titles = ["Loss Q", "Avg Q", "Episode Reward", "Episode Length"]
    graphs = [
        logging_dict["loss_hist"],
        logging_dict["avg_q_hist"],
        logging_dict["ep_reward_hist"],
        logging_dict["ep_len_hist"],
        ]
    for y_data, title in zip(graphs, titles):
        plt.plot(y_data)
        plt.title(title)
        plt.savefig(f"{exp_dir}/{title}.png")
        plt.close()

    state_predicate_occurances = np.asarray(logging_dict["ep_state_predicate_hist"])
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    for i, state_predicate in enumerate(env.state_predicate_names):
        y_data = state_predicate_occurances[:, i]
        # apply some smoothing
        y_data_smoothed = np.convolve(y_data, np.ones(10) / 10, mode="same")
        plt.plot(y_data_smoothed, label=state_predicate, color=colors[i])
        plt.plot(y_data, alpha=0.1, color=colors[i])
        plt.title(f"{state_predicate} Occurances")
        plt.savefig(f"{exp_dir}/state_predicate_{state_predicate}.png")
        plt.close()

    if params["which_env"] == "numpy":
        create_plots_numpy_env(
            dqn=learn_dqn,
            env=env,
            device=device,
            save_dir=f"{exp_dir}"
        )

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
