import numpy as np
import matplotlib.pyplot as plt
import torch
from envs.simple_acc_env import action_to_acc, SimpleAccEnv
from networks import MLP
import yaml

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Flag from https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
def plot_value_2D(
        dqn,
        env,
        device,
        velocity,
        battery=None,
        value_function="max",
        x_lim=[0, 6],
        x_steps=7,
        y_lim=[0, 6],
        y_steps=7,
        resolution=100,
        threshold=None,
        save_path=""
):
    value_fig = plt.figure(figsize=(x_steps, y_steps))

    x = np.linspace(int(x_lim[0]), int(x_lim[1]), int(x_steps * resolution))
    y = np.linspace(int(y_lim[1]), int(y_lim[0]), int(y_steps * resolution))
    X, Y = np.meshgrid(x, y)
    agent_pos = torch.from_numpy(np.array([X.flatten(), Y.flatten()]).T).to(device)

    agent_vel = torch.from_numpy(velocity).unsqueeze(0).to(device)
    agent_vel = agent_vel.repeat(agent_pos.shape[0], 1)

    if battery is not None:
        battery_tensor = torch.tensor(battery).unsqueeze(0).to(device)
        battery_tensor = battery_tensor.repeat(agent_pos.shape[0], 1)
        q_inp = torch.concatenate([agent_pos, agent_vel, battery_tensor], dim=1)
    else:
        q_inp = torch.concatenate([agent_pos, agent_vel], dim=1)

    q_inp = torch.Tensor(q_inp).to(device).float()
    q_values = dqn(q_inp)

    if value_function == "max":
        state_values = q_values.max(dim=1).values.cpu().detach().numpy()
    elif value_function == "min":
        state_values = q_values.min(dim=1).values.cpu().detach().numpy()
    elif value_function == "mean":
        state_values = q_values.mean(dim=1).cpu().detach().numpy()
    else:
        raise ValueError(f"value_function must be 'min', 'max', or 'mean' but got '{value_function}'")

    state_values = state_values.reshape(X.shape)

    cbar = value_fig.gca().imshow(state_values, cmap="magma", interpolation="nearest", extent=[0, int(x_lim[1]), 0, int(y_lim[1])])
    cbar = value_fig.colorbar(cbar)

    plt.xlabel("Env. x")
    plt.ylabel("Env. y")
    title = f"State values, {value_function} Q"
    if velocity is not None:
        title += f", velocity {velocity}"

    plt.title(title)
    plt.xticks(range(int(x_lim[0]), int(x_lim[1]) + 1))
    plt.yticks(range(int(y_lim[0]), int(y_lim[1]) + 1))

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(value_fig)


def plot_discrete_actions(dqn, state, device, action_map, save_path=""):
    q_vals = dqn(torch.Tensor(state).to(device).unsqueeze(0)).detach().cpu().numpy().flatten()
    for a_idx in range(len(q_vals)):
        print(f"Action {action_map[a_idx]}: {q_vals[a_idx]}")
        acceleration = action_map[a_idx]
        plt.scatter(
            acceleration[0],
            acceleration[1],
            s=100,
            c=q_vals[a_idx],
            cmap="viridis",
            vmin=min(q_vals),
            vmax=max(q_vals),
        )

    plt.xlabel("Acceleration x")
    plt.ylabel("Acceleration y")
    plt.colorbar(label="Q-value")
    plt.title(f"Q-values for state {state}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def create_plots_numpy_env(
        dqns,
        env,
        device,
        save_dir,
        n_rollouts=10,
        plot_value_function=True,
        plot_eval_states=True,
):
    dqn = dqns[-1]  # plot currently learning dqn
    if plot_value_function:
        # plot value function with different velocities
        for vel in [
            np.array([0.0, 0.0]),
            np.array([2.0, 0.0]),
            np.array([-2.0, 0.0]),
            np.array([0.0, 2.0]),
            np.array([0.0, -2.0]),
        ]:
            for batt in [0.1, 0.5, 1.0]:
                for value_function in ["max", "mean", "min"]:
                    # value_function = "min"
                    plot_value_2D(
                        dqn=dqn.q_net,
                        velocity=vel,
                        value_function=value_function,
                        battery=batt,
                        env=env,
                        x_lim=env.x_range,
                        x_steps=env.x_range[-1] + 1,
                        y_lim=env.y_range,
                        y_steps=env.y_range[-1] + 1,
                        device=device,
                        save_path=f"{save_dir}/vf-{value_function}_vel{vel}_batt{batt}.png"
                    )

    if plot_eval_states:
        # plot Q-function in particular states
        for eval_state in env.eval_states:
            # plot_discrete_actions(
            #     dqn=network,
            #     state=eval_state,
            #     action_map=env.action_map,
            #     device=device,
            #     save_path=f"{save_dir}/qf_state:{eval_state}.png",
            # )
            q_values = dqn.q_net(torch.Tensor(eval_state).to(device).unsqueeze(0)).detach().cpu().numpy().flatten()
            for a in range(env.action_space.n):
                acc = action_to_acc(a)
                plt.scatter(acc[0], acc[1], c=q_values[a], s=800, cmap="plasma", vmin=q_values.min(), vmax=q_values.max())
                plt.text(acc[0], acc[1], f"{a}, {acc}", fontsize=8, ha='center', va='center')

            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            plt.title(f"State: {eval_state}")
            plt.colorbar()
            plt.savefig(f"{save_dir}/eval_state_{eval_state}.png")
            # plt.show()
            plt.close()

    # plot rolouts
    if False:
        for i in range(n_rollouts):
            print(f"lotting Rollout {i}")

            rollout_base_dir = f"{save_dir}/rollouts"
            rollout_dir = f"{rollout_base_dir}/{i}"
            os.makedirs(rollout_dir, exist_ok=True)

            rect = plt.Rectangle(
                (env.lava_x_min, env.lava_y_min),
                env.lava_x_max - env.lava_x_min,
                env.lava_y_max - env.lava_y_min,
                fill=True,
                color='orange',
                alpha=0.5
            )
            plt.gca().add_patch(rect)
            if env.with_conveyer:
                conveyer_rect = plt.Rectangle(
                    (env.conveyer_x_min, env.conveyer_y_min),
                    env.conveyer_x_max - env.conveyer_x_min,
                    env.conveyer_y_max - env.conveyer_y_min,
                    fill=True,
                    color='gray',
                    alpha=0.5
                )
                plt.gca().add_patch(conveyer_rect)

            reset_options = {
                "y": 1,
                "x": 5 + np.random.uniform(-1, 1),
            }
            obs, _ = env.reset(options=reset_options)
            # obs, _ = env.reset(options={})
            trajectory = [obs[:2]]
            ep_reward = 0
            ep_len = 0
            done, trunc = False, False
            while not (done or trunc):
                print(obs)
                q_val_fig, q_val_axs = plt.subplots(1, 4, figsize=(20, 5))

                lava_q_vals = dqns[0].q_net(torch.from_numpy(obs).float().to(device)).detach().cpu().numpy()
                action = np.argmax(lava_q_vals)

                # plot lava q vals
                for a in range(env.action_space.n):
                    acc = action_to_acc(a)
                    point = q_val_axs[0].scatter(acc[0], acc[1], s=800, c=lava_q_vals[a], vmin=lava_q_vals.min(), vmax=lava_q_vals.max())
                q_val_axs[0].set_title("Lava Q-vals")
                plt.colorbar(point, ax=q_val_axs[0])

                if len(dqns) > 1:
                    goal_q_vals = dqns[1].q_net(torch.from_numpy(obs).float().to(device)).detach().cpu().numpy()
                    # plot goal q vals
                    for a in range(env.action_space.n):
                        acc = action_to_acc(a)
                        point = q_val_axs[2].scatter(acc[0], acc[1], s=800, c=goal_q_vals[a], vmin=goal_q_vals.min(), vmax=goal_q_vals.max())
                    q_val_axs[2].set_title("Goal Q-vals")
                    plt.colorbar(point, ax=q_val_axs[2])

                    if dqns[1].con_model is not None:
                        con_q_vals = dqns[1].con_model(torch.from_numpy(obs).float()).detach().cpu().numpy()
                        # forbidden_mask = con_q_vals > con_thresh
                        best_con_action_value = con_q_vals.min()
                        forbidden_mask = con_q_vals > best_con_action_value + dqn.con_thresh

                        # plot con q vals
                        for a in range(env.action_space.n):
                            acc = action_to_acc(a)
                            point = q_val_axs[1].scatter(acc[0], acc[1], s=800, c=con_q_vals[a], vmin=con_q_vals.min(), vmax=con_q_vals.max())
                            if forbidden_mask[a]:
                                q_val_axs[1].scatter(acc[0], acc[1], s=200, c="r", marker="x")

                        q_val_axs[1].set_title("Feasibility Q-vals")
                        plt.colorbar(point, ax=q_val_axs[1])

                        if not False in forbidden_mask:
                            print(f"ALL ACTIONS ARE FORBIDDEN IN STATE {obs}!")

                        goal_q_vals[forbidden_mask] -= np.inf

                    action = np.argmax(goal_q_vals)

                # plot env
                rect = plt.Rectangle(
                    (env.lava_x_min, env.lava_y_min),
                    env.lava_x_max - env.lava_x_min,
                    env.lava_y_max - env.lava_y_min,
                    fill=True,
                    color='orange',
                    alpha=0.5
                )
                q_val_axs[3].add_patch(rect)
                if env.with_conveyer:
                    conveyer_rect = plt.Rectangle(
                        (env.conveyer_x_min, env.conveyer_y_min),
                        env.conveyer_x_max - env.conveyer_x_min,
                        env.conveyer_y_max - env.conveyer_y_min,
                        fill=True,
                        color='gray',
                        alpha=0.5
                    )
                    q_val_axs[3].add_patch(conveyer_rect)

                q_val_fig.suptitle(f"State: {obs}, action: {action}, acc: {action_to_acc(action)}")

                q_val_axs[3].quiver(obs[0], obs[1], obs[2], obs[3], color="r")  # current state
                q_val_axs[3].plot(np.array(trajectory)[:, 0], np.array(trajectory)[:, 1], 'o-', c="r", alpha=0.5)
                q_val_axs[3].set_xlim(env.x_min - 0.1, env.x_max + 0.1)
                q_val_axs[3].set_ylim(env.y_min - 0.1, env.y_max + 0.1)
                q_val_axs[3].set_title("Env")

                plt.savefig(f"{rollout_dir}/q_vals{ep_len}.png")
                plt.close()

                obs, reward, done, trunc, _ = env.step(action)
                ep_reward += reward
                ep_len += 1

                trajectory.append(obs[:2])
                if done:
                    break

            trajectory = np.array(trajectory)
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'o-')

            plt.xlim(env.x_min - 0.1, env.x_max + 0.1)
            plt.ylim(env.y_min - 0.1, env.y_max + 0.1)
            plt.savefig(f"{rollout_dir}/trajectory.png")
            plt.show()
            plt.close()


def plot_multiple_rollouts(
        traj_data,
        ax=None,
        alpha=0.1,
        color="r",
        figsize=(10, 5),
        save_path="",
        show=True,
        close=True,
        xlim=(0, 10),
        ylim=(0, 10),
        ls="-",
        label="",
        legend=False
):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])

    for i in range(traj_data.shape[0]):
        ax.plot(traj_data[i, :, 0], traj_data[i, :, 1], alpha=alpha, c=color, ls=ls, label=label if i == 0 else None)

    if legend:
        ax.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    if close:
        plt.close()


def plot_simple_acc_env(env, ax=None, show=True, save_path="", close=True):
    if ax is None:
        fig, ax = plt.subplots()

    # lava rect
    lava_rect = plt.Rectangle(
        (env.lava_x_min, env.lava_y_min),
        env.lava_x_max - env.lava_x_min,
        env.lava_y_max - env.lava_y_min,
        fill=True,
        color='orange',
        alpha=0.5
    )
    plt.gca().add_patch(lava_rect)
    # add text to lava rect
    plt.text(
        env.lava_x_min + (env.lava_x_max - env.lava_x_min) / 2,
        env.lava_y_min + (env.lava_y_max - env.lava_y_min) / 2,
        "Unsafe region",
        fontsize=15,
        horizontalalignment='center',
        verticalalignment='center',
    )

    conveyer_rect = plt.Rectangle(
        (env.conveyer_x_min, env.conveyer_y_min),
        env.conveyer_x_max - env.conveyer_x_min,
        env.conveyer_y_max - env.conveyer_y_min,
        fill=True,
        color='gray',
        alpha=0.5
    )
    plt.gca().add_patch(conveyer_rect)
    plt.text(
        env.conveyer_x_min + (env.conveyer_x_max - env.conveyer_x_min) / 2,
        env.conveyer_y_min + (env.conveyer_y_max - env.conveyer_y_min) / 2,
        "Treadmill",
        fontsize=15,
        horizontalalignment='center',
        verticalalignment='center',
        )

    # plot some arrows on the conveyer belt, going from left to right
    plt.quiver(2.5, 3.5, 1, 0, color="k", scale=0.5, scale_units="xy")
    plt.quiver(2.5, 5, 1, 0, color="k", scale=0.5, scale_units="xy")
    plt.quiver(2.5, 6.5, 1, 0, color="k", scale=0.5, scale_units="xy")

    plt.quiver(5., 3.5, 1, 0, color="k", scale=0.5, scale_units="xy")
    plt.quiver(5., 6.5, 1, 0, color="k", scale=0.5, scale_units="xy")

    plt.quiver(7.5, 3.5, 1, 0, color="k", scale=0.5, scale_units="xy")
    plt.quiver(7.5, 5, 1, 0, color="k", scale=0.5, scale_units="xy")
    plt.quiver(7.5, 6.5, 1, 0, color="k", scale=0.5, scale_units="xy")

    # goal
    plt.scatter(
        env.goal_x,
        env.goal_y,
        s=200,
        c='gold',
        zorder=10,
        marker='*',
        label="Goal"
    )

    # battery
    plt.scatter(
        env.battery_x,
        env.battery_y,
        s=200,
        c='green',
        zorder=10,
        marker='+',
        label="Charger"
    )

    ax.set_xlim(env.x_min - 0.1, env.x_max + 0.1)
    ax.set_ylim(env.y_min - 0.1, env.y_max + 0.1)

    ax.set_xticks([], [])
    ax.set_yticks([], [])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    if show:
        plt.show()

    if close:
        plt.close()


class EnvActuatorGrid5x5:
    def __init__(self):
        self.num_actions = 25

    def get_acceleration(self, action):
        """
        Get acceleration from discrete action, taken form unity env actuator
        @param action: int, discrete action
        """
        i = action % 5
        j = action // 5
        acceleration_x = (i - 2) / 2.0
        acceleration_z = (j - 2) / 2.0
        return (acceleration_x, 0.0, acceleration_z)

    def plot_action_acceleration_mapping(self):
        for action in range(25):
            acceleration = self.get_acceleration(action)
            action_row = action // 5
            row_markers = ["v", "s", "o", "d", "^"]
            action_col = action % 5
            colors = ["r", "g", "b", "c", "m", "y", "k", "w"]
            plt.scatter(
                acceleration[0],
                acceleration[2],
                label=f"Action {action}",
                marker=row_markers[action_row],
                color=colors[action_col])

        plt.xlabel("Acceleration x")
        plt.ylabel("Acceleration z")

        # put legend left and outside
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()


def plot_unity_q_vals(state, dqn, device, save_path="", title="", vmin=None, vmax=None, con_thresh=None):
    q_vals = dqn(torch.Tensor(state).to(device).unsqueeze(0)).detach().cpu().numpy().flatten()
    actuator = EnvActuatorGrid5x5()
    for action in range(q_vals.shape[0]):
        acceleration = actuator.get_acceleration(action)
        plt.scatter(
            acceleration[0],
            acceleration[2],
            s=800,
            c=q_vals[action],
            cmap='viridis',
            vmin=min(q_vals) if vmin is None else vmin,
            vmax=max(q_vals) if vmax is None else vmax,
        )
        if con_thresh is not None:
            allowed = q_vals[action] < con_thresh
            if not allowed:
                plt.scatter(
                    acceleration[0],
                    acceleration[2],
                    s=200,
                    c="red",
                    marker="x",
                )

        plt.text(acceleration[0], acceleration[2] + 0.05, str(action), fontsize=6, ha='center', va='center')

    # plt.title(f"Button: [{button_relative_position[0]:.3f}, {button_relative_position[2]:.3f}] , Trigger: [{trigger_relative_position[0]:.3f}, {trigger_relative_position[2]:.3f}]")
    plt.title(title)
    plt.xlabel("Acceleration x")
    plt.ylabel("Acceleration z")
    plt.colorbar(label="Q-value")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_bt_comp_rollouts(
        no_con_load_dir=r"runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-16-20-18-00_slowLava_trainedWithoutCon_len100/",
        con_load_dir=r"runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-16-20-38-15_slowLava_trainedWithCon_len100/",
        sum_load_dir=r"runs/SimpleAccEnv-wide-withConveyer-sum-v0/2024-07-19-13-45-26_MORL_sumWeight:0.5_withCon/",
        fontsize=15,
        method_names=["BT-DQN", "BT-MORL", "CBTRL (Ours)"],
        method_colors=["magenta", "yellow", "cyan"],
        method_ls=["--", ":", "-"],
):
    """
    Plot rollouts of different models in 2D numpy env
    :param no_con_load_dir: The goal-reach DQNs trained in BT but unconstrained
    :param con_load_dir: The goal-reach DQNs trained in BT with constraints (our method)
    :param sum_load_dir: The goal-reach DQN trained in MORL / sum-task fashion
    """
    # plot env
    fig, ax = plt.subplots(figsize=(10, 5))
    env = SimpleAccEnv(
        with_conveyer=True,
        x_max=20,
        conveyer_x_min=2,
        conveyer_x_max=10,
        lava_x_min=10,
        lava_x_max=18,
        goal_x=10,
    )
    plot_simple_acc_env(env, ax=ax, show=False, close=False)

    plt.rcParams.update({'font.size': fontsize})

    for load_idx, load_dir in enumerate([no_con_load_dir, sum_load_dir, con_load_dir]):
        # load and plot con data
        data = np.load(load_dir + "/trajectories.npz")
        plot_multiple_rollouts(
            traj_data=data["trajectories"],
            ax=ax,
            color=method_colors[load_idx],
            show=False,
            close=False,
            ls=method_ls[load_idx],
            alpha=0.25,
            label="")

    # create dummy line artists with alpha=1 for legend
    for idx in range(len(method_names)):
        plt.plot([-100, -100], [-100, -100], ls=method_ls[idx], c=method_colors[idx], label=method_names[idx])
    plt.legend(loc="lower left")

    plt.savefig("runs/2D-lava-con-noCon-rollouts.png", bbox_inches='tight')
    plt.show()
    plt.close()

    # reset font size
    plt.rcParams.update({'font.size': 12})


def plot_bt_comp_metrics(
        no_con_load_dirs=[],
        con_load_dirs=[],
        sum_load_dir=[],
        which_data="eval",  # eval or train
        fontsize=25,
        method_names=["BT-DQN", "BT-MORL", "CBTRL (Ours)"],
        method_colors=["magenta", "yellow", "cyan"],
        method_ls=["--", ":", "-"],
):
    """
    Plot avg. and std. metrics over time of multiple runs for 2D numpy env
    :param no_con_load_dirs: The directories containing the goal-reach DQNs trained in BT but unconstrained
    :param con_load_dirs: The directories containing the goal-reach DQNs trained in BT with constraints (our method)
    """
    # set fontsize
    plt.rcParams.update({'font.size': fontsize})

    # setup figure
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

    # iterate over all methods
    for idx, method_dirs in enumerate([no_con_load_dirs, sum_load_dir, con_load_dirs]):

        # load data from dirs
        eval_reward_hists = []
        train_reward_hists = []
        eval_predicate_hists = []
        eval_time_hists = []
        train_predicate_hists = []
        predicate_names = []

        # iterate over all repetitions
        for load_dir in method_dirs:
            data = np.load(load_dir + "/logging_data.npz")
            trajectories = np.load(load_dir + "/trajectories.npz")
            eval_reward_hists.append(data["eval_reward_hist"])
            train_reward_hists.append(data["train_reward_hist"])
            eval_predicate_hists.append(data["eval_state_predicate_hist"])
            eval_time_hists.append(data["eval_ep_times"])
            train_predicate_hists.append(data["train_state_predicate_hist"])
            predicate_names = trajectories["state_predicate_names"]

        # convert to np arrays
        if which_data == "eval":
            reward_hists = np.array(eval_reward_hists)
            predicate_hists = np.array(eval_predicate_hists)
        else:
            reward_hists = np.array(train_reward_hists)
            predicate_hists = np.array(train_predicate_hists)

        print(method_dirs)
        in_lava = predicate_hists[:, :, 0]
        at_goal = predicate_hists[:, :, 1]

        # compute mean and std for across repetitions
        mean_reward = np.mean(reward_hists, axis=0)
        std_reward = np.std(reward_hists, axis=0)

        mean_in_lava = np.mean(in_lava, axis=0)
        std_in_lava = np.std(in_lava, axis=0)

        mean_at_goal = np.mean(at_goal, axis=0)
        std_at_goal = np.std(at_goal, axis=0)

        # plot metrics
        n_x_ticks = 3
        lw = 1
        axs[0].plot(mean_reward, color=method_colors[idx], ls=method_ls[idx], lw=lw)
        axs[0].fill_between(range(len(mean_reward)), mean_reward - std_reward, mean_reward + std_reward, color=method_colors[idx], alpha=0.2)
        axs[0].set_ylabel("Reward")
        axs[0].set_xlabel("Episodes")
        axs[0].set_xlim(0, len(mean_reward) - 1)
        axs[0].set_ylim(-1200, -250)
        axs[0].set_xticks(np.linspace(0, len(mean_reward), n_x_ticks, dtype=np.int64))

        axs[1].plot(mean_in_lava, color=method_colors[idx], label=method_names[idx], ls=method_ls[idx], lw=lw)
        axs[1].fill_between(range(len(mean_in_lava)), mean_in_lava - std_in_lava, mean_in_lava + std_in_lava, color=method_colors[idx], alpha=0.2)
        axs[1].set_ylabel("Steps " + predicate_names[0].replace("_", " "))
        axs[1].set_xlabel("Episodes")
        axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=len(method_names))
        axs[1].set_xlim(0, len(mean_in_lava) - 1)
        axs[1].set_ylim(-1, 100)
        axs[1].set_xticks(np.linspace(0, len(mean_in_lava), n_x_ticks, dtype=np.int64))

        axs[2].plot(mean_at_goal, color=method_colors[idx], ls=method_ls[idx], lw=lw)
        axs[2].fill_between(range(len(mean_at_goal)), mean_at_goal - std_at_goal, mean_at_goal + std_at_goal, color=method_colors[idx], alpha=0.2)
        axs[2].set_ylabel("Steps " + predicate_names[1].replace("_", " "))
        axs[2].set_xlabel("Episodes")
        axs[2].set_xlim(0, len(mean_at_goal) - 1)
        axs[2].set_ylim(-1, 60)
        axs[2].set_xticks(np.linspace(0, len(mean_at_goal), n_x_ticks, dtype=np.int64))

    plt.tight_layout()
    plt.subplots_adjust(
        top=0.85,
        bottom=0.15,
        left=0.12,
        right=0.975,
        wspace=0.35
    )
    plt.savefig(f"runs/2D-lava-con-noCon-metrics.png")
    plt.show()
    plt.close()

    # reset fontsize
    plt.rcParams.update({'font.size': 12})


def plot_numpy_feasiblity_dqn(
        dqn_load_dir=r"runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-16-03-00-37_good/feasibility_2024-07-16-15-52-18/",
        file_name="feasibility_dqn.pt",
        state=np.array([6.2, 2.2, 0.8, 1.5, 1.0]),
        feasibility_thresh=0.05,
        value_function="min",
        cmap="viridis",
        value_resolution=500,
        fontsize=15,
):
    # set fontsize
    plt.rcParams.update({'font.size': fontsize})

    fig = plt.figure(figsize=(10, 5))
    plt.scatter(state[0], state[1], c="magenta", s=100, marker="^")
    # plt.scatter(state[0], state[1], c="magenta", s=40, marker="o")
    # plt.quiver(state[0], state[1], 0, 1, color="magenta")
    plt.gca().axis("off")
    plt.savefig("runs/marker")
    plt.close()

    # setup wide env
    env = SimpleAccEnv(
        with_conveyer=True,
        x_max=20,
        conveyer_x_min=2,
        conveyer_x_max=10,
        lava_x_min=10,
        lava_x_max=18,
        goal_x=10,
    )
    
    # read dqn params from dir
    params = yaml.load(open(f"{dqn_load_dir}/params.yaml", "r"), Loader=yaml.FullLoader)

    # older feasibility models did not have this...
    with_batchnorm = False
    if "with_batchNorm" in params:
        with_batchnorm = params["with_batchNorm"]

    # load trained model
    dqn = MLP(
        input_size=env.observation_space.shape[0],
        output_size=env.action_space.n,
        hidden_arch=params["hidden_arch"],
        hidden_activation=params["hidden_activation"],
        with_batchNorm=with_batchnorm,
    )
    dqn.load_state_dict(torch.load(f"{dqn_load_dir}/{file_name}"))
    dqn.eval()

    fig = plt.figure(figsize=(10, 5))

    # setup subplot grid: two plots ontop of each on the left, one large plot on the right
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])
    axs = [ax1, ax2, ax3]

    # setup input for value function plotting
    agent_x = np.linspace(env.x_min, env.x_max, value_resolution)
    agent_y = np.linspace(env.y_max, env.y_min, value_resolution)
    agent_x, agent_y = np.meshgrid(agent_x, agent_y)
    agent_x = agent_x.flatten()
    agent_y = agent_y.flatten()

    # plot with [0, 0] and [0, 2] (upward) velocity
    for vel_idx, vel_arr in enumerate([np.array([0, 0]), np.array([0, 2.0])]):
        agent_vel_x = np.full_like(agent_x, vel_arr[0])
        agent_vel_y = np.full_like(agent_y, vel_arr[1])
        battery = np.full_like(agent_y, 1.0)
        states = np.stack([agent_x, agent_y, agent_vel_x, agent_vel_y, battery], axis=1)

        q_values = dqn(torch.Tensor(states).to("cpu"))

        if value_function == "max":
            state_values = q_values.max(dim=1).values.cpu().detach().numpy()
        elif value_function == "min":
            state_values = q_values.min(dim=1).values.cpu().detach().numpy()
        elif value_function == "mean":
            state_values = q_values.mean(dim=1).cpu().detach().numpy()
        else:
            raise ValueError(f"value_function must be 'min', 'max', or 'mean' but got '{value_function}'")

        state_values = state_values.reshape(value_resolution, value_resolution)

        img = axs[vel_idx].imshow(state_values, cmap=cmap, interpolation="nearest", extent=[0, env.x_max, 0, env.y_max])

        # add indicator for state location
        axs[vel_idx].scatter(state[0], state[1], c="magenta", s=100, marker="^")
        # axs[vel_idx].scatter(state[0], state[1], c="magenta", s=40, marker="o")
        # axs[vel_idx].quiver(state[0], state[1], 0, 1, color="magenta")

        # plot rectangle and text for unsafe area
        lava_rect = plt.Rectangle(
            (env.lava_x_min, env.lava_y_min),
            env.lava_x_max - env.lava_x_min,
            env.lava_y_max - env.lava_y_min,
            fill=False,
            color='red',
            linewidth=2,
            alpha=1,
        )
        axs[vel_idx].add_patch(lava_rect)
        axs[vel_idx].text(
            env.lava_x_min + (env.lava_x_max - env.lava_x_min) / 2,
            env.lava_y_min + (env.lava_y_max - env.lava_y_min) / 2,
            "Unsafe area",
            fontsize=15,
            horizontalalignment='center',
            verticalalignment='center',
            color="red"
            )

        axs[vel_idx].set_xticks([], [])
        axs[vel_idx].set_yticks([], [])
        axs[vel_idx].set_title(f"Velocity: {vel_arr}")
        axs[vel_idx].set_xlabel("Env x")
        axs[vel_idx].set_ylabel("Env y")

    # plot feasiblity action-values and mask for particular state
    q_vals = dqn(torch.Tensor(state).unsqueeze(0)).detach().cpu().numpy().flatten()
    for a_idx in range(len(q_vals)):
        acceleration = action_to_acc(a_idx)
        point = axs[2].scatter(
            acceleration[0],
            acceleration[1],
            s=800,
            c=q_vals[a_idx],
            cmap=cmap,
            vmin=min(q_vals),
            vmax=max(q_vals),
        )

        if q_vals[a_idx] > q_vals.min() + feasibility_thresh:
            axs[2].scatter(acceleration[0], acceleration[1], s=800, c="r", marker="x")

    axs[2].set_xlim(-2.4, 2.4)
    axs[2].set_ylim(-2.4, 2.4)
    axs[2].set_title(f"Feasibility action mask, Y-velocity: {state[3]}")
    # axs[2].set_xlabel("Acceleration X")
    axs[2].set_ylabel("Acceleration Y")
    plt.colorbar(point, ax=axs[2])

    plt.tight_layout()
    plt.subplots_adjust(
        top=0.925,
        bottom=0.075,
        left=-0.03,
        right=0.99,
        wspace=0.05
    )
    plt.savefig("runs/feasiblity_estimator")
    plt.show()
    plt.close()

    # reset fontsize
    plt.rcParams.update({'font.size': 12})


if __name__ == "__main__":
    # env_actuator = EnvActuatorGrid5x5()
    # env_actuator.plot_action_acceleration_mapping()

    method_names = ["BT-DQN", "BT-Penalty", "CBTRL (Ours)"]
    method_colors = ["magenta", "red", "cyan"]
    method_ls = ["--", ":", "-"]

    plot_bt_comp_rollouts(
        con_load_dir=r"/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-18-10-22_noPunish_withConstraint_noEval_1",
        no_con_load_dir=r"/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-15-13-12_noPunish_noConstraint_noEval_2",
        sum_load_dir=r"/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-16-53-56_withPunish_noConstraint_noEval_2",
        method_names=method_names,
        method_colors=method_colors,
        method_ls=method_ls
    )

    plot_bt_comp_metrics(
        which_data="train",
        no_con_load_dirs=[
            "/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-14-52-57_noPunish_noConstraint_noEval_1",
            "/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-15-13-12_noPunish_noConstraint_noEval_2",
            "/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-15-33-37_noPunish_noConstraint_noEval_3",
            "/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-15-53-56_noPunish_noConstraint_noEval_4",
            "/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-16-14-18_noPunish_noConstraint_noEval_5",
        ],
        con_load_dirs=[
            "/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-18-10-22_noPunish_withConstraint_noEval_1",
            "/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-18-35-40_noPunish_withConstraint_noEval_2",
            "/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-19-01-19_noPunish_withConstraint_noEval_3",
            "/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-19-27-03_noPunish_withConstraint_noEval_4",
            "/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-19-52-26_noPunish_withConstraint_noEval_5",
        ],
        sum_load_dir=[
            "/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-16-34-46_withPunish_noConstraint_noEval_1",
            "/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-16-53-56_withPunish_noConstraint_noEval_2",
            "/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-17-13-07_withPunish_noConstraint_noEval_3",
            "/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-17-32-10_withPunish_noConstraint_noEval_4",
            "/home/finn/repos/MORL-BT/BTRL-learning/runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-17-51-14_withPunish_noConstraint_noEval_5",
        ],
        method_names=method_names,
        method_colors=method_colors,
        method_ls=method_ls
    )

    plot_numpy_feasiblity_dqn(
        # dqn_load_dir=r"runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-16-03-00-37_good/feasibility_2024-07-16-15-52-18/",
        # dqn_load_dir=r"runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-25-16-24-08_200kRandom_squareResetMultipleReings/feasibility_2024-07-25-17-29-29",
        dqn_load_dir=r"runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-29-10-03-55_withBattery/feasibility_2024-07-29-17-28-18",
    )


