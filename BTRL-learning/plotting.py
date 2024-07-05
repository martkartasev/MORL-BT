import numpy as np
import matplotlib.pyplot as plt
import torch


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
    y = np.linspace(int(y_lim[0]), int(y_lim[1]), int(y_steps * resolution))
    X, Y = np.meshgrid(x, y)
    agent_pos = np.array([X.flatten(), Y.flatten()]).T

    goal_pos = torch.tensor([[env.goal_x, env.goal_y]])
    goal_pos = goal_pos.repeat(agent_pos.shape[0], 1)

    agent_vel = torch.tensor([velocity])
    agent_vel = agent_vel.repeat(agent_pos.shape[0], 1)
    q_inp = np.concatenate([agent_pos, agent_vel, goal_pos], axis=1)

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
        network,
        env,
        device,
        save_dir,
):

    # plot value function with different velocities
    for vel in [
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([-2.0, 0.0]),
        np.array([0.0, 2.0]),
        np.array([0.0, -2.0]),
    ]:
        value_function = "min"
        plot_value_2D(
            dqn=network,
            velocity=vel,
            value_function=value_function,
            env=env,
            x_lim=env.x_range,
            x_steps=env.x_range[-1] + 1,
            y_lim=env.y_range,
            y_steps=env.y_range[-1] + 1,
            device=device,
            save_path=f"{save_dir}/vf:{value_function}_velocity:{vel}.png"
        )

    # plot Q-function in particular states
    for eval_state in env.eval_states:
        plot_discrete_actions(
            dqn=network,
            state=eval_state,
            action_map=env.action_map,
            device=device,
            save_path=f"{save_dir}/qf_state:{eval_state}.png",
        )


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


def plot_unity_q_vals(state, dqn, device, save_path="", title="", vmin=None, vmax=None):
    q_vals = dqn(torch.Tensor(state).to(device).unsqueeze(0)).detach().cpu().numpy().flatten()
    actuator = EnvActuatorGrid5x5()
    for action in range(q_vals.shape[0]):
        acceleration = actuator.get_acceleration(action)
        plt.scatter(
            acceleration[0],
            acceleration[2],
            s=100,
            c=q_vals[action],
            cmap='viridis',
            vmin=min(q_vals) if vmin is None else vmin,
            vmax=max(q_vals) if vmax is None else vmax,
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

