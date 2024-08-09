import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import random
import gymnasium as gym
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# from mlagents_envs.base_env import ActionTuple
# from mlagents_envs.environment import UnityEnvironment
# from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import envs
from envs.unity_misc import rewards_flat_acc_env, done_check_flat_acc_env, unity_state_predicate_check, unity_state_predicate_names
from envs.simple_acc_env import action_to_acc
from misc import ReplayBuffer
from dqn import DQN
from plotting import create_plots_numpy_env, plot_unity_q_vals, plot_multiple_rollouts


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
    eval_reward_hist = []
    eval_state_predicate_hist = []
    eval_episodes_times = []

    logging_dict = {
        "episodes_done": episodes_done,
        "ep_len": ep_len,
        "ep_reward_sum": ep_reward_sum,
        "ep_state_predicates": ep_state_predicates,
        "loss_hist": loss_hist,
        "avg_q_hist": avg_q_hist,
        "ep_reward_hist": ep_reward_hist,
        "ep_len_hist": ep_len_hist,
        "ep_state_predicate_hist": ep_state_predicate_hist,
        "eval_reward_hist": eval_reward_hist,
        "eval_state_predicate_hist": eval_state_predicate_hist,
        "eval_episodes_times": eval_episodes_times,
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
        con_model_load_cps=[],  # highest prio, no constraint...
        model_name="avoid_lava",
        batch_norm=params["numpy_env_lava_dqn_batchNorm"],
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
        con_model_load_cps=[
            params["numpy_env_lava_feasibility_dqn_cp"]
        ],
        con_model_arches=[
            params["numpy_env_lava_feasibility_dqn_arch"]
        ],
        con_threshes=[
            params["numpy_env_lava_feasibility_thresh"]
        ],
        con_batch_norms=[
            params["numpy_env_lava_feasibility_batchNorm"]
        ],
        model_name="reach_goal",
        batch_norm=params["numpy_env_goal_dqn_batchNorm"],
    )

    if "lava" in env_id:
        dqns = [avoid_lava_dqn]
    elif "goal" in env_id or "battery" in env_id:
        assert params["numpy_env_lava_dqn_cp"] != "", "Pre-trained avoid_lava DQN load path must be given"
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
        logging_dict,
        device,
        with_plot=False,
        save_plot_path="",
        eval_ep=False
):

    # BT is just if-else, and only active if we are training the goal reach DQN
    agent_x = obs[0]
    agent_y = obs[1]
    if len(dqns) == 1:
        dqn_idx = 0
    elif len(dqns) == 2:  # avoid lava and next task (reach goal, or go left)
        if env.lava_x_min < agent_x < env.lava_x_max and env.lava_y_min < agent_y < env.lava_y_max:
            print("Agent in lava, using avoid DQN")
            dqn_idx = 0
        else:
            dqn_idx = 1
    else:
        raise NotImplementedError("More than 2 DQNs given, Implement BT here!")

    action = dqns[dqn_idx].act(obs, epsilon)
    next_obs, reward, done, trunc, info = env.step(action)

    punish_reward = reward
    if params["with_lava_reward_punish"]:
        if env.lava_x_min < next_obs[0] < env.lava_x_max and env.lava_y_min < next_obs[1] < env.lava_y_max:
            punish_reward -= 50

    if with_plot:
        with torch.no_grad():
            # figure out how many subplots we need
            n_dqns = len(dqns)
            n_con_models = 0
            for dqn in dqns:
                n_con_models += len(dqn.con_models)
            n_subplots = n_dqns + n_con_models + 1

            state_fig, state_axs = plt.subplots(nrows=1, ncols=n_subplots, figsize=(20, 5))

            subplot_idx = 0
            for dqn_plt_idx, dqn in enumerate(dqns):
                q_vals = dqn.q_net(torch.tensor(obs).float().to(device)).detach().cpu().numpy()
                # plot q_vals
                for a in range(env.action_space.n):
                    acc = action_to_acc(a)
                    point = state_axs[subplot_idx].scatter(acc[0], acc[1], s=800, c=q_vals[a], vmin=q_vals.min(), vmax=q_vals.max())
                plt.colorbar(point, ax=state_axs[subplot_idx])

                state_axs[subplot_idx].set_title(f"Q-values {dqn_plt_idx}: ({'active' if dqn_idx == dqn_plt_idx else 'inactive'})")
                subplot_idx += 1

                for con_plt_idx, con_model in enumerate(dqn.con_models):
                    con_model.eval()
                    con_q_vals = con_model(torch.tensor(obs).unsqueeze(0).float().to(device)).squeeze().detach().cpu().numpy()
                    con_mask = dqns[dqn_plt_idx].compute_mask(torch.tensor(obs).unsqueeze(0).float().to(device))
                    # plot con_q_vals
                    for a in range(env.action_space.n):
                        acc = action_to_acc(a)
                        point = state_axs[subplot_idx].scatter(acc[0], acc[1], s=800, c=con_q_vals[a], vmin=con_q_vals.min(), vmax=con_q_vals.max())
                        if con_mask[a]:
                            state_axs[subplot_idx].scatter(acc[0], acc[1], s=800, c="r", marker="x")

                    plt.colorbar(point, ax=state_axs[subplot_idx])

                    state_axs[subplot_idx].set_title(f"Q-values {dqn_plt_idx}, con: {con_plt_idx}")
                    subplot_idx += 1

            # plot env
            lava_rect = plt.Rectangle(
                (env.lava_x_min, env.lava_y_min),
                env.lava_x_max - env.lava_x_min,
                env.lava_y_max - env.lava_y_min,
                color="orange",
                alpha=1
            )
            state_axs[-1].add_patch(lava_rect)
            conveyer_rect = plt.Rectangle(
                (env.conveyer_x_min, env.conveyer_y_min),
                env.conveyer_x_max - env.conveyer_x_min,
                env.conveyer_y_max - env.conveyer_y_min,
                color="gray",
                alpha=1
            )
            state_axs[-1].add_patch(conveyer_rect)
            state_axs[-1].quiver(obs[0], obs[1], obs[2], obs[3], color="r")  # current state
            state_axs[-1].set_xlim(env.x_min - 0.1, env.x_max + 0.1)
            state_axs[-1].set_ylim(env.y_min - 0.1, env.y_max + 0.1)
            state_axs[-1].set_title(f"Env: {obs}")

            if save_plot_path:
                if not os.path.exists(os.path.dirname(save_plot_path)):
                    os.makedirs(os.path.dirname(save_plot_path))
                plt.savefig(save_plot_path)

            plt.close()

    if dqn_idx == len(dqns) - 1 and not eval_ep:
        # only add transition to the replay buffer when the DQN we are currently learning is used and we are not doing eval run
        replay_buffer.add(
            obs=obs,
            action=action,
            reward=punish_reward,  # use reward with punishment for learning
            next_obs=next_obs,
            done=done,
            infos=info)

    obs = next_obs
    logging_dict["ep_len"] += 1
    logging_dict["ep_reward_sum"] += reward  # log (MDP) reward without punishment
    logging_dict["ep_state_predicates"] += info["state_predicates"]

    if (done or trunc):
        obs, info = env.reset(
            options={"x": env.x_max / 2 + np.random.uniform(-8, 8), "y": 1} if 'goal' in params["env_id"] else {}
            # options={}
        )
        if not eval_ep:
            # only log non-eval episodes
            writer.add_scalar("episode/length", logging_dict["ep_len"], logging_dict["episodes_done"])
            writer.add_scalar("episode/reward_sum", logging_dict["ep_reward_sum"], logging_dict["episodes_done"])
            for i, state_predicate in enumerate(env.state_predicate_names):
                writer.add_scalar(f"episode/{state_predicate}", logging_dict["ep_state_predicates"][i], logging_dict["episodes_done"])
                
        logging_dict["ep_reward_hist"].append(logging_dict["ep_reward_sum"])
        logging_dict["ep_len_hist"].append(logging_dict["ep_len"])
        logging_dict["ep_state_predicate_hist"].append(logging_dict["ep_state_predicates"])

        print(
            f"Episode {logging_dict['episodes_done']} | "
            f"Length: {logging_dict['ep_len']} | "
            f"Reward: {logging_dict['ep_reward_sum']} | "
            f"Loss: {logging_dict['loss_hist'][-1] if len(logging_dict['loss_hist']) > 0 else None} | "
            f"{global_step} / {params['total_timesteps']} steps")

        logging_dict["ep_len"] = 0
        logging_dict["ep_reward_sum"] = 0
        logging_dict["ep_state_predicates"] = np.zeros(len(env.state_predicate_names))
        logging_dict["episodes_done"] += 1
        
    return obs, reward, done, trunc, info


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


def main(args):
    # HYPERPARAMETERS
    which_env = "numpy"  # "unity" or "numpy
    # which_env = "unity"  # "unity" or "numpy
    params = {
        "which_env": which_env,
        # "env_id": "SimpleAccEnv-wide-withConveyer-lava-v0",
        # "env_id": "SimpleAccEnv-wide-withConveyer-goal-v0",
        # "env_id": "SimpleAccEnv-wide-withConveyer-sum-v0",
        # "env_id": "SimpleAccEnv-wide-withConveyer-left-v0",
        # "env_id": "flat-acc-button",  # name of the folder containing the unity scene binaries
        # "env_id": "flat-acc",  # name of the folder containing the unity scene binaries
        "env_id": args.env_id,
        "unity_take_screenshots": True,
        "unity_max_ep_len": 1000,
        "unity_task": "fetch_trigger",
        # "unity_task": "reach_goal",
        "no_train_only_plot": False,
        "total_timesteps": args.total_steps,
        "lr": 0.0005,
        "buffer_size": 1e6,
        "gamma": 0.995,
        # "tau": 0.001,
        # "target_freq": 1,
        "tau": 1,
        "target_freq": 10000,
        "batch_size": 2048,
        "hidden_activation": nn.ReLU,
        "start_epsilon": 1.0,
        "end_epsilon": 0.05,
        "exp_fraction": 0.5,
        "learning_start": args.learning_starts,
        "seed": args.seed,
        "with_lava_reward_punish": args.punishACC,
        # "numpy_env_lava_dqn_cp": "",
        "numpy_env_lava_dqn_cp": args.lava_dqn_path,
        "numpy_env_lava_dqn_arch": [32, 32, 16, 16],
        "numpy_env_lava_dqn_batchNorm": False,
        # "numpy_env_lava_dqn_arch": [256, 256],
        "numpy_env_lava_feasibility_dqn_cp": args.lava_constraint_feasibility_path,
        "numpy_env_lava_feasibility_dqn_arch": [64, 64, 32, 32],
        "numpy_env_lava_feasibility_thresh": 0.05,
        "numpy_env_lava_feasibility_batchNorm": True,
        "numpy_env_goal_dqn_cp": "",
        # "numpy_env_goal_dqn_cp": "runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-24-22-19-06_withLavaFeasibility/reach_left_net.pth",
        # "numpy_env_goal_dqn_cp": "runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-16-16-27-29_slowLava_trainedWithoutCon/reach_goal_net.pth",
        "numpy_env_goal_dqn_arch": [32, 32, 16, 16],
        "numpy_env_goal_dqn_batchNorm": False,
    }

    # DIR FOR LOGGING
    exp_dir = f"runs/{params['env_id']}"
    if params["which_env"] == "unity":
        exp_dir += f"_{params['unity_task']}"

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir += f"/{timestamp}"
    exp_dir += f"_{args.exp_name}"

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
    epsilon_vals = np.linspace(params["start_epsilon"], params["end_epsilon"], int(params["exp_fraction"] * (params["total_timesteps"] - params["learning_start"])))
    episodes_since_eval = 1_000_000
    # episodes_since_eval = 5
    for global_step in range(params["total_timesteps"]):
        if params["no_train_only_plot"]:
            # we are only creating plots and collecting trajectory data...
            continue

        if global_step > params["learning_start"]:
            epsilon = epsilon_vals[min(global_step - params["learning_start"], len(epsilon_vals) - 1)]
        else:
            epsilon = params["start_epsilon"]
        writer.add_scalar("epsilon", epsilon, global_step)

        # one-step interaction with the environment
        if params["which_env"] == "numpy":
            obs, _, done, trunc, _ = env_interaction_numpy_env(
                dqns=dqns,
                obs=obs,
                epsilon=epsilon,
                env=env,
                replay_buffer=replay_buffer,
                writer=writer,
                global_step=global_step,
                params=params,
                logging_dict=logging_dict,
                device=device,
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

        # include one eval episode every n episodes...
        if global_step > params["learning_start"]:
            if (done or trunc):
                episodes_since_eval -= 1
                if episodes_since_eval <= 0:
                    with torch.no_grad():
                        episodes_since_eval = 5
                        eval_obs, eval_info = env.reset(options={
                            "x": env.x_max / 2 + np.random.uniform(-4, 4),
                            "y": 1
                        })
                        eval_logging_dict = {
                            "episodes_done": 0,
                            "ep_len": 0,
                            "ep_reward_sum": 0,
                            "ep_state_predicates": np.zeros(len(env.state_predicate_names)),
                            "loss_hist": [],
                            "avg_q_hist": [],
                            "ep_reward_hist": [],
                            "ep_len_hist": [],
                            "ep_state_predicate_hist": []
                        }
                        eval_done, eval_trunc = False, False
                        while not (eval_done or eval_trunc):
                            eval_obs, eval_reward, eval_done, eval_trunc, _ = env_interaction_numpy_env(
                                dqns=dqns,
                                obs=eval_obs,
                                epsilon=params["end_epsilon"],
                                env=env,
                                replay_buffer=replay_buffer,
                                writer=writer,
                                global_step=global_step,
                                params=params,
                                logging_dict=eval_logging_dict,
                                device=device,
                                eval_ep=True
                            )

                        # save reward and predicate from eval episodes to main logging dict
                        logging_dict["eval_reward_hist"].append(eval_logging_dict["ep_reward_hist"][-1])
                        logging_dict["eval_state_predicate_hist"].append(eval_logging_dict["ep_state_predicate_hist"][-1])
                        logging_dict["eval_episodes_times"].append(logging_dict["episodes_done"])

                        obs, info = env.reset()  # reset for next regular, non-eval episode...

    # SAVE MODEL AND DATA
    learn_dqn.save_model(exp_dir)
    replay_buffer.save(f"{exp_dir}/replay_buffer.npz")

    # save logging data
    np.savez(
        f"{exp_dir}/logging_data.npz",
        loss_hist=logging_dict["loss_hist"],
        avg_q_hist=logging_dict["avg_q_hist"],
        train_reward_hist=logging_dict["ep_reward_hist"],
        train_len_hist=logging_dict["ep_len_hist"],
        train_state_predicate_hist=logging_dict["ep_state_predicate_hist"],
        eval_reward_hist=logging_dict["eval_reward_hist"],
        eval_state_predicate_hist=logging_dict["eval_state_predicate_hist"],
        eval_ep_times=logging_dict["eval_episodes_times"],
    )

    # PLOT TRAINING CURVES
    if not params["no_train_only_plot"]:
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
            dqns=dqns,
            env=env,
            device=device,
            save_dir=f"{exp_dir}",
            plot_eval_states=True,
            plot_value_function=True,
            n_rollouts=10
        )

        # PLOT TRAJECTORIES
        trajectory_data = []
        rewards = []
        state_predicates = []
        for j in range(100):
            obs, info = env.reset(options={"x": env.x_max / 2 + np.random.uniform(-4, 4), "y": 1} if 'goal' in params["env_id"] else {})
            done, trunc = False, False
            trajectory = [obs[:2]]
            episodes_done, ep_len, ep_reward_sum = 0, 0, 0
            loss_hist = []
            avg_q_hist = []
            ep_reward_hist = []
            ep_len_hist = []
            ep_state_predicates = np.zeros(len(env.state_predicate_names))
            ep_state_predicate_hist = []

            eval_logging_dict = {
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

            while not (done or trunc):
                new_obs, reward, done, trunc, info = env_interaction_numpy_env(
                    dqns=dqns,
                    obs=obs,
                    epsilon=params["end_epsilon"],
                    env=env,
                    replay_buffer=replay_buffer,
                    writer=writer,
                    global_step=global_step,
                    params=params,
                    logging_dict=eval_logging_dict,
                    with_plot=True if j < 3 else False,
                    save_plot_path=f"{exp_dir}/bt_rollouts/{j}/{eval_logging_dict['ep_len']}.png",
                    device=device,
                    eval_ep=True
                )

                trajectory.append(new_obs[:2])
                obs = new_obs

            trajectory_data.append(np.array(trajectory)[:-1, :])  # remove last obs, since it is new reset obs already...
            rewards.append(eval_logging_dict["ep_reward_hist"][-1])
            state_predicates.append(eval_logging_dict["ep_state_predicate_hist"][-1])

        trajectory_data = np.array(trajectory_data)
        rewards = np.array(rewards)
        state_predicates = np.array(state_predicates)

        plot_multiple_rollouts(
            traj_data=trajectory_data,
            save_path=f"{exp_dir}/trajectories.png",
            xlim=[env.x_min - 0.1, env.x_max + 0.1],
            ylim=[env.y_min - 0.1, env.y_max + 0.1],
            show=False
        )
        np.savez(
            f"{exp_dir}/trajectories.npz",
            trajectories=trajectory_data,
            rewards=rewards,
            state_predicates=state_predicates,
            state_predicate_names=env.state_predicate_names
        )

    env.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--total_steps", type=int, default=1_000_000, help="Total number of training steps")
    parser.add_argument("-s", "--seed", type=int, default=0, help="The random seed for this run")
    parser.add_argument("-l", "--learning_starts", type=int, default=200_000, help="Do this many random actions before learning starts")
    parser.add_argument("-e", "--exp_name", type=str, default="refactorMLP_maxVel:1.5_200epLen_batch:2048_200kRandom", help="Additional string to append to the experiment directory")
    parser.add_argument('--punishACC', default=False, action=argparse.BooleanOptionalAction, help="Agent receives reward penalty for ACC violation")

    # parser.add_argument("-ldqnp", "--lava_dqn_path", type=str, default="", help="Path to load the lava avoiding DQN policy from.")
    # parser.add_argument("-ldqnp", "--lava_dqn_path", type=str, default="runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-29-10-03-55_withBattery/avoid_lava_net.pth", help="Path to load the lava avoiding DQN policy from.")
    # parser.add_argument("-ldqnp", "--lava_dqn_path", type=str, default="runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-08-01-14-52-27_withBattery_refactorMLP_slowAlways_500epLength_200kRandom/avoid_lava_net.pth", help="Path to load the lava avoiding DQN policy from.")
    parser.add_argument("-ldqnp", "--lava_dqn_path", type=str, default="runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-08-03-19-53-26_withBattery_refactorMLP_maxVel:1.5_200kRandom_200epLen/avoid_lava_net.pth", help="Path to load the lava avoiding DQN policy from.")

    # parser.add_argument("-lfcp", "--lava_constraint_feasibility_path", type=str, default="", help="Path to load Lava feasibility constraint network from.")
    # parser.add_argument("-lfcp", "--lava_constraint_feasibility_path", type=str, default="runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-29-10-03-55_withBattery/feasibility_2024-07-29-10-36-22/feasibility_dqn.pt", help="Path to load Lava feasibility constraint network from.")
    # parser.add_argument("-lfcp", "--lava_constraint_feasibility_path", type=str, default="runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-08-01-14-52-27_withBattery_refactorMLP_slowAlways_500epLength_200kRandom/feasibility_2024-08-01-15-25-20_1k_lrDecay/feasibility_dqn.pt", help="Path to load Lava feasibility constraint network from.")
    parser.add_argument("-lfcp", "--lava_constraint_feasibility_path", type=str, default="runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-08-03-19-53-26_withBattery_refactorMLP_maxVel:1.5_200kRandom_200epLen/feasibility_2024-08-04-09-41-43_1k_lrDecay/feasibility_dqn.pt", help="Path to load Lava feasibility constraint network from.")

    # parser.add_argument("-i", "--env_id", type=str, default="SimpleAccEnv-wide-withConveyer-goal-v0", help="Which gym env to train on.")
    parser.add_argument("-i", "--env_id", type=str, default="SimpleAccEnv-wide-withConveyer-battery-v0", help="Which gym env to train on.")
    # parser.add_argument("-i", "--env_id", type=str, default="SimpleAccEnv-wide-withConveyer-lava-v0", help="Which gym env to train on.")

    args = parser.parse_args()
    print(args)

    main(args)
