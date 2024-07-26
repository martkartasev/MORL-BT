from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
# This is a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

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

import envs
from envs.unity_misc import rewards_flat_acc_env, done_check_flat_acc_env, unity_state_predicate_check, unity_state_predicate_names
from envs.simple_acc_env import action_to_acc
from plotting import create_plots_numpy_env, plot_unity_q_vals, plot_multiple_rollouts


def setup_numpy_env(params, device, exp_dir):
    env_id = params["env_id"]
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    def mask_fn(env: gym.Env) -> np.ndarray:
        # Do whatever you'd like in this function to return the action mask
        # for the current env. In this example, we assume the env has a
        # helpful method we can rely on.
        # return env.valid_action_mask()
        # TODO, integrate feasibility estimator
        return np.ones([env.action_space.n], dtype=bool)

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking

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

    ppos = [
        MaskablePPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1)
    ]

    return env, state_dim, action_dim, obs, info, logging_dict, ppos, mask_fn


def main(args):
    # HYPERPARAMETERS
    which_env = "numpy"  # "unity" or "numpy
    # which_env = "unity"  # "unity" or "numpy
    params = {
        "which_env": which_env,
        # "env_id": "SimpleAccEnv-wide-withConveyer-lava-v0",
        "env_id": "SimpleAccEnv-wide-withConveyer-goal-v0",
        # "env_id": "SimpleAccEnv-wide-withConveyer-sum-v0",
        # "env_id": "SimpleAccEnv-wide-withConveyer-left-v0",
        "no_train_only_plot": False,
        "total_timesteps": args.total_steps,
        "seed": args.seed,
        "with_lava_reward_punish": False,
        "numpy_env_lava_cp": "",
        # "numpy_env_lava_cp": "runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-16-03-00-37_good/avoid_lava_net.pth",
        # "numpy_env_lava_cp": "runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-25-14-23-05_100kRandom_squareReset/avoid_lava_net.pth",
        # "numpy_env_lava_dqn_arch": [32, 32, 16, 16],
        # "numpy_env_lava_feasibility_dqn_cp": "",
        "numpy_env_lava_feasibility_dqn_cp": "runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-25-16-24-08_200kRandom_squareResetMultipleReings/feasibility_2024-07-25-17-29-29/feasibility_dqn.pt",
        "numpy_env_lava_feasibility_dqn_arch": [32, 32, 32, 16],
        "numpy_env_lava_feasibility_thresh": 0.05,
        "numpy_env_goal_cp": "",
        "numpy_env_goal_arch": [32, 32, 16, 16],
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

    # SAVE PARAMS AS YAML
    with open(f"{exp_dir}/params.yaml", "w") as f:
        yaml.dump(params, f)

    # SEEDING
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ENVIRONMENT SETUP
    env, state_dim, action_dim, obs, info, logging_dict, ppos, mask_fn = setup_numpy_env(params=params, device=device, exp_dir=exp_dir)

    ppo = ppos[-1]  # we always only learn the last DQN, all other DQNs in list must be trained already

    # PPO STUFF
    ppo.learn(params["total_timesteps"])
    ppo.save(f"{exp_dir}/ppo")

    del ppo  # remove to demonstrate saving and loading
    ppo = MaskablePPO.load(f"{exp_dir}/ppo")

    # PLOT TRAJECTORIES
    trajectory_data = []
    rewards = []
    state_predicates = []
    for j in range(100):
        obs, info = env.reset(options={
            "x": env.x_max / 2 + np.random.uniform(-4, 4),
            "y": 1
        })
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
            action = ppo.predict(obs, action_masks=np.ones([env.action_space.n], dtype=bool))[0]
            new_obs, reward, done, trunc, info = env.step(action)

            # TODO, log state predicates

            trajectory.append(new_obs[:2])
            obs = new_obs

        trajectory_data.append(np.array(trajectory)[:-1, :])  # remove last obs, since it is new reset obs already...

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--total_steps", type=int, default=200_000, help="Total number of training steps")
    parser.add_argument("-s", "--seed", type=int, default=0, help="The random seed for this run")
    parser.add_argument("-e", "--exp_name", type=str, default="ppo", help="Additional string to append to the experiment directory")
    args = parser.parse_args()
    print(args)

    main(args)

