import os

import yaml
import matplotlib.pyplot as plt
import numpy as np
from envs.simple_acc_env import SimpleAccEnv, action_to_acc
from dqn import DQN
from networks import MLP
import torch


def plot_q_state(q_values, state, env, cp_dir):
    for a in range(env.action_space.n):
        acc = action_to_acc(a)
        if "feasibility" in cp_dir:
            plt.scatter(acc[0], acc[1], c=q_values[a], s=800, cmap="plasma", vmin=0, vmax=1)
        else:
            plt.scatter(acc[0], acc[1], c=q_values[a], s=800, cmap="plasma", vmin=q_values.min(), vmax=q_values.max())
        plt.text(acc[0], acc[1], f"{a}, {acc}", fontsize=8, ha='center', va='center')

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.title(f"State: {state}")
    plt.colorbar()
    plt.savefig(f"{cp_dir}/eval_state_{state}.png")
    plt.show()
    plt.close()


def plot_cp(cp_dir="", squash_output=False):
    env = SimpleAccEnv()

    # plot eval states
    lava_rect = plt.Rectangle(
        (env.lava_x_min, env.lava_y_min),
        env.lava_x_max - env.lava_x_min,
        env.lava_y_max - env.lava_y_min,
        fill=True,
        color='orange',
        alpha=0.5
    )
    plt.gca().add_patch(lava_rect)
    for state in env.eval_states:
        obs = state
        env.reset(options={"x": obs[0], "y": obs[1], "vel_x": obs[2], "vel_y": obs[3]})
        if obs[2] != 0 or obs[3] != 0:
            plt.quiver(obs[0], obs[1], obs[2], obs[3])
        else:
            plt.scatter(obs[0], obs[1])

    plt.title("Env with eval states")
    plt.xlim(env.x_min - 0.1, env.x_max + 0.1)
    plt.ylim(env.y_min - 0.1, env.y_max + 0.1)
    plt.savefig(f"{cp_dir}/eval_states.png")
    plt.show()

    # load CP
    params = yaml.load(open(f"{cp_dir}/params.yaml", "r"), Loader=yaml.FullLoader)

    model = MLP(
        input_size=env.observation_space.shape[0],
        output_size=env.action_space.n,
        hidden_size=params["hidden_dim"],
        hidden_activation=params["hidden_activation"],
        squash_output=squash_output
    )
    if "feasibility" in cp_dir:
        model.load_state_dict(torch.load(f"{cp_dir}/feasibility_dqn.pt"))
    else:
        model.load_state_dict(torch.load(f"{cp_dir}/q_net.pth"))

    # plot value function with different velocities
    for vel in [
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([-2.0, 0.0]),
        np.array([0.0, 2.0]),
        np.array([0.0, -2.0]),
    ]:
        agent_x = np.linspace(env.x_min, env.x_max, 100)
        agent_y = np.linspace(env.y_max, env.y_min, 100)
        agent_x, agent_y = np.meshgrid(agent_x, agent_y)
        agent_x = agent_x.flatten()
        agent_y = agent_y.flatten()
        agent_vel_x = np.full_like(agent_x, vel[0])
        agent_vel_y = np.full_like(agent_y, vel[1])
        states = np.stack([agent_x, agent_y, agent_vel_x, agent_vel_y], axis=1)

        q_values = model(torch.Tensor(states).to("cpu"))

        for value_function in [torch.min, torch.max]:
            vf = value_function(q_values, dim=1).values.detach().cpu().numpy()
            vf = vf.reshape((100, 100))

            plt.imshow(vf, extent=[env.x_min, env.x_max, env.y_min, env.y_max])
            plt.colorbar()
            plt.title(f"Value function with velocity {vel}, {value_function.__name__}")
            plt.savefig(f"{cp_dir}/value_function_{vel}_{value_function.__name__}.png")
            plt.show()
            plt.close()

    for state in env.eval_states:
        q_values = model(torch.Tensor(state).to("cpu")).detach().cpu().numpy()
        plot_q_state(
            q_values=q_values,
            state=state,
            env=env,
            cp_dir=cp_dir
        )


def plot_rollouts(
        task_dqn_dir="",
        con_dqn_dir="",
        con_thresh=0.25,
        n_rollouts=1,
        con_squash_output=True,
        with_conveyer=False
):
    env = SimpleAccEnv(with_conveyer=with_conveyer)

    task_params = yaml.load(open(f"{task_dqn_dir}/params.yaml", "r"), Loader=yaml.FullLoader)
    task_dqn = MLP(
        input_size=env.observation_space.shape[0],
        output_size=env.action_space.n,
        hidden_size=task_params["hidden_dim"],
        hidden_activation=task_params["hidden_activation"],
        squash_output=False
    )
    task_dqn.load_state_dict(torch.load(f"{task_dqn_dir}/q_net.pth"))

    if con_dqn_dir:
        con_params = yaml.load(open(f"{con_dqn_dir}/params.yaml", "r"), Loader=yaml.FullLoader)
        con_dqn = MLP(
            input_size=env.observation_space.shape[0],
            output_size=env.action_space.n,
            hidden_size=con_params["hidden_dim"],
            hidden_activation=con_params["hidden_activation"],
            squash_output=con_squash_output
        )
        con_dqn.load_state_dict(torch.load(f"{con_dqn_dir}/feasibility_dqn.pt"))

    for i in range(n_rollouts):
        print(f"Rollout {i}")
        if con_dqn_dir:
            rollout_base_dir = f"{con_dqn_dir}/rollouts"
        else:
            rollout_base_dir = f"{task_dqn_dir}/rollouts"

        rollout_dir = f"{rollout_base_dir}/{i}"
        os.makedirs(rollout_dir, exist_ok=True)

        save_dict = {
            "task_dqn_dr": task_dqn_dir,
            "con_dqn_dir": con_dqn_dir,
            "con_thresh": con_thresh,

        }

        with open(f"{rollout_base_dir}/args.yaml", "w") as f:
            yaml.dump(save_dict, f)

        rect = plt.Rectangle(
            (env.lava_x_min, env.lava_y_min),
            env.lava_x_max - env.lava_x_min,
            env.lava_y_max - env.lava_y_min,
            fill=True,
            color='orange',
            alpha=0.5
        )
        plt.gca().add_patch(rect)
        if with_conveyer:
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
            "y": 1
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

            q_val_fig.suptitle(f"State: {obs}")
            task_q_vals = task_dqn(torch.from_numpy(obs).float()).detach().cpu().numpy()

            # plot task q vals
            for a in range(env.action_space.n):
                acc = action_to_acc(a)
                point = q_val_axs[0].scatter(acc[0], acc[1], s=800, c=task_q_vals[a], vmin=task_q_vals.min(), vmax=task_q_vals.max())
            q_val_axs[0].set_title("Task Q-vals")
            plt.colorbar(point, ax=q_val_axs[0])

            if con_dqn_dir:
                con_q_vals = con_dqn(torch.from_numpy(obs).float()).detach().cpu().numpy()
                forbidden_mask = con_q_vals > con_thresh

                # plot con q vals
                for a in range(env.action_space.n):
                    acc = action_to_acc(a)
                    point = q_val_axs[1].scatter(acc[0], acc[1], s=800, c=con_q_vals[a], vmin=0, vmax=1)
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
            if with_conveyer:
                conveyer_rect = plt.Rectangle(
                    (env.conveyer_x_min, env.conveyer_y_min),
                    env.conveyer_x_max - env.conveyer_x_min,
                    env.conveyer_y_max - env.conveyer_y_min,
                    fill=True,
                    color='gray',
                    alpha=0.5
                )
                q_val_axs[2].add_patch(conveyer_rect)
            q_val_axs[2].quiver(obs[0], obs[1], obs[2], obs[3], color="r")  # current state
            plt.plot(np.array(trajectory)[:, 0], np.array(trajectory)[:, 1], 'o-', c="r", alpha=0.5)
            q_val_axs[2].set_xlim(env.x_min - 0.1, env.x_max + 0.1)
            q_val_axs[2].set_ylim(env.y_min - 0.1, env.y_max + 0.1)
            q_val_axs[2].set_title("Env")

            plt.savefig(f"{rollout_dir}/q_vals{ep_len}.png")
            plt.close()

            action = np.argmax(task_q_vals)

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


if __name__ == "__main__":
    plot_cp(
        cp_dir=r"runs\SimpleAccEnv-withConveyer-goal-v0/2024-07-08-18-09-39",
    )
    plot_cp(
        cp_dir=r"runs/SimpleAccEnv-withConveyer-lava-v0/2024-07-08-17-45-38/feasibility_2024-07-08-18-04-25",
        squash_output=False,
    )

    plot_rollouts(
        task_dqn_dir=r"runs\SimpleAccEnv-withConveyer-goal-v0/2024-07-08-18-09-39",
        con_dqn_dir=r"runs/SimpleAccEnv-withConveyer-lava-v0/2024-07-08-17-45-38/feasibility_2024-07-08-18-04-25",
        # con_dqn_dir=r"",
        con_thresh=0.1,
        con_squash_output=False,
        n_rollouts=10,
        with_conveyer=True
    )

