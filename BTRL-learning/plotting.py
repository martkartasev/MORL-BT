import numpy as np
import matplotlib.pyplot as plt
import torch
from envs.simple_acc_env import action_to_acc

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Flag from https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
def plot_value_2D(
        dqn,
        env,
        device,
        velocity,
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
    agent_pos = np.array([X.flatten(), Y.flatten()]).T

    agent_vel = torch.from_numpy(velocity).unsqueeze(0).to(device)
    agent_vel = agent_vel.repeat(agent_pos.shape[0], 1)

    if env.observation_space.shape[0] == 6:
        goal_pos = torch.tensor([[env.goal_x, env.goal_y]])
        goal_pos = goal_pos.repeat(agent_pos.shape[0], 1)
        q_inp = np.concatenate([agent_pos, agent_vel, goal_pos], axis=1)
    else:
        q_inp = np.concatenate([agent_pos, agent_vel], axis=1)

    q_inp = torch.Tensor(q_inp).to(device)
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
        dqn,
        env,
        device,
        save_dir,
        n_rollouts=10
):

    # plot value function with different velocities
    for vel in [
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([-2.0, 0.0]),
        np.array([0.0, 2.0]),
        np.array([0.0, -2.0]),
    ]:
        for value_function in ["max", "mean", "min"]:
            # value_function = "min"
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
                save_path=f"{save_dir}/vf-{value_function}_vel{vel}.png"
            )

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
        plt.show()
        plt.close()

    # plot rolouts
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
            q_val_fig, q_val_axs = plt.subplots(1, 3, figsize=(15, 5))

            task_q_vals = dqn.q_net(torch.from_numpy(obs).float()).detach().cpu().numpy()

            # plot task q vals
            for a in range(env.action_space.n):
                acc = action_to_acc(a)
                point = q_val_axs[0].scatter(acc[0], acc[1], s=800, c=task_q_vals[a], vmin=task_q_vals.min(), vmax=task_q_vals.max())
            q_val_axs[0].set_title("Task Q-vals")
            plt.colorbar(point, ax=q_val_axs[0])

            if dqn.con_model is not None:
                con_q_vals = dqn.con_model(torch.from_numpy(obs).float()).detach().cpu().numpy()
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

                task_q_vals[forbidden_mask] -= np.inf

            # plot env
            rect = plt.Rectangle(
                (env.lava_x_min, env.lava_y_min),
                env.lava_x_max - env.lava_x_min,
                env.lava_y_max - env.lava_y_min,
                fill=True,
                color='orange',
                alpha=0.5
            )
            q_val_axs[2].add_patch(rect)
            if env.with_conveyer:
                conveyer_rect = plt.Rectangle(
                    (env.conveyer_x_min, env.conveyer_y_min),
                    env.conveyer_x_max - env.conveyer_x_min,
                    env.conveyer_y_max - env.conveyer_y_min,
                    fill=True,
                    color='gray',
                    alpha=0.5
                )
                q_val_axs[2].add_patch(conveyer_rect)

            action = np.argmax(task_q_vals)
            q_val_fig.suptitle(f"State: {obs}, action: {action}, acc: {action_to_acc(action)}")

            q_val_axs[2].quiver(obs[0], obs[1], obs[2], obs[3], color="r")  # current state
            plt.plot(np.array(trajectory)[:, 0], np.array(trajectory)[:, 1], 'o-', c="r", alpha=0.5)
            q_val_axs[2].set_xlim(env.x_min - 0.1, env.x_max + 0.1)
            q_val_axs[2].set_ylim(env.y_min - 0.1, env.y_max + 0.1)
            q_val_axs[2].set_title("Env")

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


if __name__ == "__main__":
    env_actuator = EnvActuatorGrid5x5()
    env_actuator.plot_action_acceleration_mapping()

