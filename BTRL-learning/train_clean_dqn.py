# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from plotting import create_plots_numpy_env, plot_unity_q_vals, plot_multiple_rollouts


import envs
from networks import MLP

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    # env_id: str = "SimpleAccEnv-wide-withConveyer-lava-v0"
    env_id: str = "SimpleAccEnv-wide-withConveyer-shapedSum-v0"

    """the id of the environment"""
    total_timesteps: int = 300_000
    """total timesteps of the experiments"""
    learning_rate: float = 0.0005
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(10e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1
    """the target network update rate"""
    target_network_frequency: int = 10000
    """the timesteps it takes to update the target network"""
    batch_size: int = 4096
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 200_000
    """timestep to start learning"""
    train_frequency: int = 1
    """the frequency of training"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
# class QNetwork(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
#             nn.ReLU(),
#             nn.Linear(120, 84),
#             nn.ReLU(),
#             nn.Linear(84, env.single_action_space.n),
#         )
# 
#     def forward(self, x):
#         return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = MLP(
        input_size=envs.single_observation_space.shape[0],
        output_size=envs.single_action_space.n,
        hidden_arch=[32, 32, 16, 16]
    ).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    # target_network = QNetwork(envs).to(device)
    target_network = MLP(
        input_size=envs.single_observation_space.shape[0],
        output_size=envs.single_action_space.n,
        hidden_arch=[32, 32, 16, 16]
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    episodes_done, ep_len, ep_reward_sum = 0, 0, 0
    loss_hist = []
    avg_q_hist = []
    ep_reward_hist = []
    ep_len_hist = []
    ep_state_predicates = np.zeros(len(envs.envs[0].state_predicate_names))
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

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if args.num_envs > 1:
            raise ValueError("The following lines only log running episode stats assuming only a single environment...")

        logging_dict["ep_len"] += 1
        logging_dict["ep_reward_sum"] += rewards[0]  # log (MDP) reward without punishment
        if not "final_info" in infos:
            # avoid keyError here and add state predicates from last step in the following if block
            logging_dict["ep_state_predicates"] += np.array(infos["state_predicates"][0], dtype=np.int64)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    logging_dict["ep_state_predicates"] += np.array(info["state_predicates"], dtype=np.int64)

                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                    logging_dict["ep_len_hist"].append(info["episode"]["l"][0])
                    logging_dict["ep_reward_hist"].append(info["episode"]["r"][0])
                    logging_dict["ep_state_predicate_hist"].append(logging_dict["ep_state_predicates"])

                    logging_dict["ep_len"] = 0
                    logging_dict["ep_reward_sum"] = 0
                    logging_dict["ep_state_predicates"] = np.zeros(len(envs.envs[0].state_predicate_names))
                    logging_dict["episodes_done"] += 1

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                logging_dict["loss_hist"].append(loss.item())
                logging_dict["avg_q_hist"].append(old_val.mean().item())

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    # save logging data
    np.savez(
        f"runs/{run_name}/logging_data.npz",
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
    img_dir = f"runs/{run_name}/imgs"
    os.makedirs(img_dir, exist_ok=True)
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
        plt.savefig(f"{img_dir}/{title}.png")
        plt.close()

        state_predicate_occurances = np.asarray(logging_dict["ep_state_predicate_hist"])
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
        for i, state_predicate in enumerate(envs.envs[0].state_predicate_names):
            y_data = state_predicate_occurances[:, i]
            # apply some smoothing
            y_data_smoothed = np.convolve(y_data, np.ones(10) / 10, mode="same")
            plt.plot(y_data_smoothed, label=state_predicate, color=colors[i])
            plt.plot(y_data, alpha=0.1, color=colors[i])
            plt.title(f"{state_predicate} Occurances")
            plt.savefig(f"{img_dir}/state_predicate_{state_predicate}.png")
            plt.close()

        create_plots_numpy_env(
            dqns=[q_network],
            env=envs.envs[0],
            device=device,
            save_dir=f"{img_dir}",
            plot_eval_states=True,
            plot_value_function=True,
            n_rollouts=10
        )

    # PLOT TRAJECTORIES
    env = gym.make(args.env_id)
    trajectory_data = []
    rewards = []
    state_predicates = []
    for j in range(100):
        print(f"Running plotting episode {j}")
        battery = 0.1 if j % 2 == 0 else 0.9  # alternate between low and high battery episodes for plotting
        reset_options = {
            "x": env.x_max / 2 + np.random.uniform(-4, 4),
            "y": 1,
            "battery": battery
        }

        obs, info = env.reset(
            options=reset_options
        )
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
            q_values = q_network(torch.Tensor(obs).to(device))
            action = torch.argmax(q_values).cpu().item()
            new_obs, reward, done, trunc, info = env.step(action)

            eval_logging_dict["ep_reward_sum"] += reward
            eval_logging_dict["ep_state_predicates"] += np.array(info["state_predicates"], dtype=np.int64)

            trajectory.append(new_obs[:2])
            obs = new_obs

        trajectory_data.append(np.array(trajectory)[:-1, :])  # remove last obs, since it is new reset obs already...
        rewards.append(eval_logging_dict["ep_reward_sum"])
        state_predicates.append(eval_logging_dict["ep_state_predicates"])

    # append last obs to each trajectory to make them all the same length
    max_len = max([len(traj) for traj in trajectory_data])
    trajectory_data_same_len = []
    for traj in trajectory_data:
        while len(traj) < max_len:
            traj = np.vstack([traj, traj[-1]])
        trajectory_data_same_len.append(traj)
    trajectory_data = np.array(trajectory_data_same_len)

    rewards = np.array(rewards)
    state_predicates = np.array(state_predicates)

    plot_multiple_rollouts(
        traj_data=trajectory_data,
        save_path=f"{img_dir}/trajectories.png",
        xlim=[env.x_min - 0.1, env.x_max + 0.1],
        ylim=[env.y_min - 0.1, env.y_max + 0.1],
        show=False
    )
    np.savez(
        f"runs/{run_name}/trajectories.npz",
        trajectories=trajectory_data,
        rewards=rewards,
        state_predicates=state_predicates,
        state_predicate_names=env.state_predicate_names
    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()