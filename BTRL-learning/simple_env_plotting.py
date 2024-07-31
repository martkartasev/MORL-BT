import yaml
import matplotlib.pyplot as plt
import numpy as np
from envs.simple_acc_env import SimpleAccEnv, action_to_acc
from dqn import DQN
from networks import MLP
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Flag from https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a

def plot_q_state(q_values, state, env, cp_dir):
    for a in range(env.action_space.n):
        acc = action_to_acc(a)
        # if "feasibility" in cp_dir:
        #     plt.scatter(acc[0], acc[1], c=q_values[a], s=800, cmap="plasma", vmin=0, vmax=1)
        # else:
        #     plt.scatter(acc[0], acc[1], c=q_values[a], s=800, cmap="plasma", vmin=q_values.min(), vmax=q_values.max())
        plt.scatter(acc[0], acc[1], c=q_values[a], s=800, cmap="plasma", vmin=q_values.min(), vmax=q_values.max())
        plt.text(acc[0], acc[1], f"{a}, {acc}", fontsize=8, ha='center', va='center')

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.title(f"State: {state}")
    plt.colorbar()
    plt.savefig(f"{cp_dir}/eval_state_{state}.png")
    plt.show()
    plt.close()


def plot_cp(env, cp_dir="", cp_file="", squash_output=False, with_conveyer=False):

    # plot eval states
    lava_rect = plt.Rectangle(
        (env.lava_x_min, env.lava_y_min),
        env.lava_x_max - env.lava_x_min,
        env.lava_y_max - env.lava_y_min,
        fill=True,
        color='orange',
        alpha=0.5
    )

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

    plt.gca().add_patch(lava_rect)
    for state in env.eval_states:
        obs = state
        env.reset(options={"x": obs[0], "y": obs[1], "vel_x": obs[2], "vel_y": obs[3]})
        if obs[2] != 0 or obs[3] != 0:
            plt.quiver(obs[0], obs[1], obs[2], obs[3])
        else:
            plt.scatter(obs[0], obs[1])

    plt.scatter(env.goal_x, env.goal_y, c="gold", marker="*")
    plt.scatter(env.battery_x, env.battery_y,  c="green", marker="+")

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
        hidden_arch=params["hidden_arch"],
        hidden_activation=params["hidden_activation"],
        with_batchNorm=True,
    )
    model.load_state_dict(torch.load(f"{cp_dir}/{cp_file}"))
    model.eval()

    # plot value function with different velocities
    for vel in [
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([-2.0, 0.0]),
        np.array([0.0, 2.0]),
        np.array([0.0, -2.0]),
    ]:
        for batt in [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]:
            agent_x = np.linspace(env.x_min, env.x_max, 100)
            agent_y = np.linspace(env.y_max, env.y_min, 100)
            agent_x, agent_y = np.meshgrid(agent_x, agent_y)
            agent_x = agent_x.flatten()
            agent_y = agent_y.flatten()
            agent_vel_x = np.full_like(agent_x, vel[0])
            agent_vel_y = np.full_like(agent_y, vel[1])
            battery = np.ones([agent_x.shape[0]]) * batt
            states = np.stack([agent_x, agent_y, agent_vel_x, agent_vel_y, battery], axis=1)

            q_values = model(torch.Tensor(states).to("cpu"))

            for value_function in [torch.min]:
                vf = value_function(q_values, dim=1).values.detach().cpu().numpy()
                vf = vf.reshape((100, 100))

                plt.imshow(vf, extent=[env.x_min, env.x_max, env.y_min, env.y_max], vmin=0, vmax=1)
                plt.colorbar()
                plt.title(f"Value function with velocity {vel}, {value_function.__name__}, battery: {batt}")
                plt.savefig(f"{cp_dir}/value_function_{vel}_{value_function.__name__}_batt{batt}.png")
                plt.show()
                plt.close()

    for state in env.eval_states:
        q_values = model(torch.Tensor(state).unsqueeze(0).to("cpu")).squeeze().detach().cpu().numpy()
        plot_q_state(
            q_values=q_values,
            state=state,
            env=env,
            cp_dir=cp_dir
        )


def plot_rollouts(
        env,
        task_dqn_dir="",
        con_dqn_dirs=[],
        con_threshes=[],
        n_rollouts=1,
        with_conveyer=False
):
    task_params = yaml.load(open(f"{task_dqn_dir}/params.yaml", "r"), Loader=yaml.FullLoader)
    task_dqn = MLP(
        input_size=env.observation_space.shape[0],
        output_size=env.action_space.n,
        hidden_arch=task_params["numpy_env_goal_dqn_arch"],
        hidden_activation=task_params["hidden_activation"],
    )
    task_dqn.load_state_dict(torch.load(f"{task_dqn_dir}/reach_left_net.pth"))
    task_dqn.eval()

    con_dqns = []
    for con_dqn_dir in con_dqn_dirs:
        if con_dqn_dir:
            con_params = yaml.load(open(f"{con_dqn_dir}/params.yaml", "r"), Loader=yaml.FullLoader)
            con_dqn = MLP(
                input_size=env.observation_space.shape[0],
                output_size=env.action_space.n,
                hidden_arch=con_params["hidden_arch"],
                hidden_activation=con_params["hidden_activation"],
            )
            con_dqn.load_state_dict(torch.load(f"{con_dqn_dir}/feasibility_dqn.pt"))
            con_dqn.eval()
            con_dqns.append(con_dqn)

    for i in range(n_rollouts):
        print(f"Rollout {i}")
        if len(con_dqn_dirs) > 0:
            rollout_base_dir = f"{con_dqn_dirs[-1]}/rollouts"
        else:
            rollout_base_dir = f"{task_dqn_dir}/rollouts"

        rollout_dir = f"{rollout_base_dir}/{i}"
        os.makedirs(rollout_dir, exist_ok=True)

        save_dict = {
            "task_dqn_dr": task_dqn_dir,
            "con_dqn_dirs": con_dqn_dirs,
            "con_threshes": con_threshes,

        }

        with open(f"{rollout_base_dir}/args.yaml", "w") as f:
            yaml.dump(save_dict, f)

        reset_options = {
            "y": 1,
            "x": (env.x_max / 2) + np.random.uniform(-4, 4),
        }
        obs, _ = env.reset(options=reset_options)
        # obs, _ = env.reset(options={})
        trajectory = [obs[:2]]
        ep_reward = 0
        ep_len = 0
        done, trunc = False, False
        while not (done or trunc):
            print(obs)
            q_val_fig, q_val_axs = plt.subplots(1, 2 + len(con_dqns), figsize=(15, 5))

            task_q_vals = task_dqn(torch.from_numpy(obs).float()).detach().cpu().numpy()

            # plot task q vals
            for a in range(env.action_space.n):
                acc = action_to_acc(a)
                point = q_val_axs[0].scatter(acc[0], acc[1], s=800, c=task_q_vals[a], vmin=task_q_vals.min(), vmax=task_q_vals.max())
            q_val_axs[0].set_title("Task Q-vals")
            plt.colorbar(point, ax=q_val_axs[0])

            forbidden_mask_global = torch.zeros(task_q_vals.shape)
            con_colors = ["cyan",  "magenta"]
            for con_idx, con_dqn in enumerate(con_dqns):
                con_q_vals = con_dqn(torch.from_numpy(obs).float()).detach().cpu().numpy()
                # forbidden_mask = con_q_vals > con_thresh
                best_con_action_value = con_q_vals.min()
                forbidden_mask = con_q_vals > best_con_action_value + con_threshes[con_idx]

                if min(forbidden_mask_global + forbidden_mask) == 0:
                    forbidden_mask_global += forbidden_mask
                else:
                    print("Not applying lower prio mask, would result in empty action space!")

                # plot con q vals
                for a in range(env.action_space.n):
                    acc = action_to_acc(a)
                    point = q_val_axs[1 + con_idx].scatter(acc[0], acc[1], s=800, c=con_q_vals[a], vmin=con_q_vals.min(), vmax=con_q_vals.max())
                    if forbidden_mask[a]:
                        q_val_axs[1 + con_idx].scatter(acc[0], acc[1], s=200, c=con_colors[con_idx], marker="x")
                        q_val_axs[0].scatter(acc[0], acc[1], s=200, c=con_colors[con_idx], marker="x")

                q_val_axs[1 + con_idx].set_title(f"Feasibility Q-vals {con_idx}")
                plt.colorbar(point, ax=q_val_axs[1 + con_idx])

            forbidden_mask_global = forbidden_mask_global.clamp(0, 1).bool()

            if not False in forbidden_mask_global:
                print(f"ALL ACTIONS ARE FORBIDDEN IN STATE {obs}!")

            task_q_vals[forbidden_mask_global] -= np.inf

            # plot env
            rect = plt.Rectangle(
                (env.lava_x_min, env.lava_y_min),
                env.lava_x_max - env.lava_x_min,
                env.lava_y_max - env.lava_y_min,
                fill=True,
                color='orange',
                alpha=0.5
            )
            q_val_axs[-1].add_patch(rect)
            if with_conveyer:
                conveyer_rect = plt.Rectangle(
                    (env.conveyer_x_min, env.conveyer_y_min),
                    env.conveyer_x_max - env.conveyer_x_min,
                    env.conveyer_y_max - env.conveyer_y_min,
                    fill=True,
                    color='gray',
                    alpha=0.5
                )
                q_val_axs[-1].add_patch(conveyer_rect)

            action = np.argmax(task_q_vals)
            q_val_fig.suptitle(f"State: {obs}, action: {action}, acc: {action_to_acc(action)}")

            q_val_axs[-1].quiver(obs[0], obs[1], obs[2], obs[3], color="r")  # current state
            plt.plot(np.array(trajectory)[:, 0], np.array(trajectory)[:, 1], 'o-', c="r", alpha=0.5)
            q_val_axs[-1].set_xlim(env.x_min - 0.1, env.x_max + 0.1)
            q_val_axs[-1].set_ylim(env.y_min - 0.1, env.y_max + 0.1)
            q_val_axs[-1].set_title("Env")

            plt.tight_layout()
            plt.savefig(f"{rollout_dir}/q_vals{ep_len}.png")
            plt.close()

            obs, reward, done, trunc, _ = env.step(action)
            ep_reward += reward
            ep_len += 1

            trajectory.append(obs[:2])
            if done:
                break

        # plot final trajectory
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
            plt.gca().add_patch(conveyer_rect
                                )
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'o-')

        plt.xlim(env.x_min - 0.1, env.x_max + 0.1)
        plt.ylim(env.y_min - 0.1, env.y_max + 0.1)
        plt.tight_layout()
        plt.savefig(f"{rollout_dir}/trajectory.png", bbox_inches="tight")
        plt.show()
        plt.close()


if __name__ == "__main__":
    env = SimpleAccEnv(
        with_conveyer=True,
        x_max=20,
        conveyer_x_min=2,
        conveyer_x_max=10,
        lava_x_min=10,
        lava_x_max=18,
        goal_x=10,
        max_ep_len=150
    )
    # plot_cp(
    #     env=env,
    #     cp_dir=r"runs/SimpleAccEnv-withConveyer-goal-v0/2024-07-13-12-46-08_BT_noCon",
    #     cp_file="reach_goal_net.pth",
    #     with_conveyer=True,
    # )
    plot_cp(
        env=env,
        # cp_dir=r"runs/SimpleAccEnv-withConveyer-lava-v0/2024-07-14-19-08-39_250k_50krandom/feasibility_2024-07-14-21-50-23",
        # cp_dir=r"runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-16-03-00-37_good/feasibility_2024-07-25-10-33-03_lava_feasibilityDiscount:0.99_longTrain_LRDecay:0.9999_WeightDecay:0.0001_batch:256_BEST",
        # cp_dir=r"runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-25-16-24-08_200kRandom_squareResetMultipleReings/feasibility_2024-07-25-17-29-29",
        cp_dir=r"runs/SimpleAccEnv-wide-withConveyer-battery-v0/2024-07-30-12-08-57_1M/feasibility_2024-07-31-15-40-15_multiLoad_recursive_lessL2_EvenLargerModel_1k_lrDecay_veryLargeBatch",
        # cp_dir=r"runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-29-10-03-55_withBattery/feasibility_2024-07-29-17-28-18",
        # cp_dir=r"runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-29-10-03-55_withBattery/feasibility_2024-07-29-17-28-18",
        cp_file="feasibility_dqn.pt",
        with_conveyer=True,
    )

    # plot_rollouts(
    #     env=env,
    #     task_dqn_dir=r"runs/SimpleAccEnv-wide-withConveyer-goal-v0/2024-07-27-14-52-57_noPunish_noConstraint_noEval_1",
    #     con_dqn_dirs=[
    #         # r"runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-25-16-24-08_200kRandom_squareResetMultipleReings/feasibility_2024-07-25-17-29-29",
    #         r"runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-29-10-03-55_withBattery/feasibility_2024-07-29-10-36-22",
    #         # r"runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-16-03-00-37_good/feasibility_2024-07-24-11-57-59_goToLeft_withHighPrioCon_arch:32-32-16-16_batch:256_discount:0.99_L2norm:0.001_feasibilityDiscount"
    #     ],
    #     con_threshes=[
    #         0.1,
    #         0.05
    #     ],
    #     n_rollouts=10,
    #     with_conveyer=True
    # )

