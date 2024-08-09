import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
# from mlagents_envs.environment import UnityEnvironment
import yaml

from envs import LavaGoalConveyerAccelerationEnv, SimpleAccEnv
from networks import MLP
from plotting import plot_value_2D, plot_discrete_actions
import envs
import gymnasium as gym


def load_data_from_rb(load_dirs, n_obs, n_actions):
    obs = None
    actions = None
    next_obs = None
    dones = None
    for load_rb_dir in load_dirs:
        rb_path = f"{load_rb_dir}/replay_buffer.npz"
        print(f"Loading replay buffer from {rb_path}...")
        data = np.load(rb_path, allow_pickle=True)

        load_obs = data["all_states"]
        load_actions = data["all_actions"]
        load_next_obs = data["all_next_states"]
        load_dones = data["all_dones"]

        # shuffle the data obs, actions, next_obs, dones in the same way
        shuffle_idx = np.random.permutation(load_obs.shape[0])
        load_obs = load_obs[shuffle_idx]
        load_actions = load_actions[shuffle_idx]
        load_next_obs = load_next_obs[shuffle_idx]
        load_dones = load_dones[shuffle_idx]

        if len(load_obs.shape) == 3:
            load_obs = load_obs.squeeze(1)
            load_actions = load_actions.squeeze(1)
            load_next_obs = load_next_obs.squeeze(1)

        assert load_obs.shape[0] == load_actions.shape[0] == load_next_obs.shape[0] == load_dones.shape[0]
        assert load_obs.shape[1] == n_obs
        assert load_actions.shape[1] == 1
        assert load_next_obs.shape[1] == n_obs
        assert load_dones.shape[1] == 1

        if obs is None:
            obs = load_obs
            actions = load_actions
            next_obs = load_next_obs
            dones = load_dones
        else:
            obs = np.append(obs, load_obs, axis=0)
            actions = np.append(actions, load_actions, axis=0)
            next_obs = np.append(next_obs, load_next_obs, axis=0)
            dones = np.append(dones, load_dones, axis=0)

    # normalize data


    print(f""
          f"Done: obs.shape: {obs.shape}, "
          f"actions.shape: {actions.shape}, "
          f"next_obs.shape: {next_obs.shape}, "
          f"rewards.shape: {dones.shape}, dones.shape"
          )

    return data, obs, actions, next_obs, dones


def label_data(all_obs, label_fun):

    # label data for ACC violation
    print("Labeling data for ACC violation...")
    labels = np.ones((all_obs.shape[0], 1)) * np.inf
    for i in range(all_obs.shape[0]):
        state = all_obs[i]
        label = label_fun(state=state)
        labels[i] = int(label)

    assert not np.isinf(labels).any()
    print(f"Done: labels.shape: {labels.shape}, positive labels: {np.sum(labels)}")

    return labels


def create_training_plots(model, env, exp_dir, train_loss_hist=None, lr_hist=None, pred_mean_hist=None):
    if train_loss_hist is not None:
        plt.plot(train_loss_hist, label="train_loss")
        plt.ylabel("TD Loss")
        plt.xlabel("Updates")
        plt.legend()
        plt.savefig(f"{exp_dir}/feasibility_qf_loss.png")
        plt.close()

    if lr_hist is not None:
        plt.plot(lr_hist, label="lr")
        plt.ylabel("Learning Rate")
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig(f"{exp_dir}/feasibility_lr.png")
        plt.close()

    if pred_mean_hist is not None:
        plt.plot(pred_mean_hist, label="pred_mean")
        plt.ylabel("Mean Q-value")
        plt.xlabel("Updates")
        plt.legend()
        plt.savefig(f"{exp_dir}/feasibility_qf_mean.png")
        plt.close()

    if env is not None and isinstance(env, LavaGoalConveyerAccelerationEnv):
        for vel in [
            np.array([0.0, 0.0]),
            np.array([2.0, 0.0]),
            np.array([-2.0, 0.0]),
            np.array([0.0, 2.0]),
            np.array([0.0, -2.0]),
        ]:
            value_function = "min"
            plot_value_2D(
                dqn=model,
                velocity=vel,
                value_function=value_function,
                env=env,
                x_lim=env.x_range,
                x_steps=env.x_range[-1] + 1,
                y_lim=env.y_range,
                y_steps=env.y_range[-1] + 1,
                device="cpu",
                save_path=f"{exp_dir}/feasibility_vf:{value_function}_velocity:{vel}.png"
            )

        for eval_state in env.eval_states:
            plot_discrete_actions(
                dqn=model,
                state=eval_state,
                action_map=env.action_map,
                device="cpu",
                save_path=f"{exp_dir}/feasibility_qf_state:{eval_state}.png",
            )


def train_model(
        env,
        model,
        target_model,
        optimizer,
        scheduler,
        states,
        actions,
        labels,
        next_states,
        dones,
        batch_size,
        exp_dir,
        device,
        epochs=10,
        nuke_layer_every=1e6,
        gamma=0.99,
        polyak_tau=0.01,
        criterion=torch.nn.MSELoss(),
        higher_prio_constraint_nets=[],
        higher_prio_constraint_thresholds=[],
):

    assert len(higher_prio_constraint_thresholds) == len(higher_prio_constraint_nets)

    criterion = criterion()

    train_loss_hist = []
    lr_hist = []
    pred_mean_hist = []
    print("Training model...")

    batches_per_episode = states.shape[0] // batch_size
    # gamma = 0.99
    for epoch in range(epochs):
        model.train()
        target_model.train()
        train_loss = 0
        batches_done = 0
        for i in range(batches_per_episode):
            # sample batch
            batch_idx = np.random.choice(states.shape[0], batch_size)

            state_batch = torch.from_numpy(states[batch_idx]).to(device)
            action_batch = torch.from_numpy(actions[batch_idx]).to(device)
            reward_batch = torch.from_numpy(labels[batch_idx]).to(device)
            next_state_batch = torch.from_numpy(next_states[batch_idx]).to(device)
            done_batch = torch.from_numpy(dones[batch_idx]).to(device)

            # compute TD target
            with torch.no_grad():
                target_q_values = target_model(next_state_batch.float())
                # target_q_values = model(next_state_batch.float())

                # target_max = target_q_values.max(dim=1, keepdim=True)[0]
                # td_target = reward_batch + gamma * target_max * (1 - done_batch)

                for idx, net in enumerate(higher_prio_constraint_nets):
                    high_prio_vals = net(next_state_batch.float())
                    best_high_prio_vals = high_prio_vals.min(dim=1, keepdim=True)[0]  # TODO, consider higher prio when finding best?
                    high_prio_forbidden = high_prio_vals > best_high_prio_vals + higher_prio_constraint_thresholds[idx]

                    target_q_values[high_prio_forbidden] = torch.inf

                assert torch.all(target_q_values.min(dim=1).values < torch.inf)

                current_state_val = (1 - gamma) * reward_batch
                # current_state_val = reward_batch
                target_min = target_q_values.min(dim=1, keepdim=True)[0]
                future_val = torch.max(target_min.to(device), reward_batch)
                td_target = current_state_val + gamma * future_val

            q_values = model(state_batch.float().to(device))
            q_values = q_values.gather(dim=1, index=action_batch.to(torch.int64))
            loss = criterion(q_values.float(), td_target.float())
            pred_mean_hist.append(q_values.mean().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loss_hist.append(loss.item())
            lr_hist.append(optimizer.param_groups[0]["lr"])
            batches_done += 1

            # polyak target network update
            # tau = 0.01
            for target_param, param in zip(target_model.parameters(), model.parameters()):
                target_param.data.copy_(polyak_tau * param.data + (1.0 - polyak_tau) * target_param.data)

            if i % 100 == 0:
                print(f"Epoch {epoch}, batch {i} / {batches_per_episode}, loss: {np.around(loss.item(), 5)}, avg. q-values: {np.around(q_values.mean().item(), 3)}, lr={np.around(optimizer.param_groups[0]['lr'], 5)}")

        # train_loss_hist.append(train_loss / batches_done)
        lr_hist.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

        # hard target network update
        # target_model.load_state_dict(model.state_dict())

        if epoch % 50 == 0:
            # epoch_plot_dir = f"{exp_dir}/epoch_{epoch}"
            # os.makedirs(epoch_plot_dir, exist_ok=True)
            # create_training_plots(
            #     model=model,
            #     env=env,
            #     train_loss_hist=train_loss_hist,
            #     lr_hist=lr_hist,
            #     exp_dir=epoch_plot_dir,
            # )

            # save the model
            torch.save(model.state_dict(), f"{exp_dir}/feasibility_dqn.pt")

        if epoch % nuke_layer_every == 0 and epoch > 0:
            print(f"Nuking last FC layer...")
            model.network[-2].reset_parameters()
            target_model.network[-2].reset_parameters()

    print("Done: training model")

    print(f"Saving classifier to {exp_dir}/feasibility_dqn.pt")
    torch.save(model.state_dict(), f"{exp_dir}/feasibility_dqn.pt")

    return model, train_loss_hist, lr_hist, pred_mean_hist


def main():
    load_rb_dirs = [
        ## nice results with those two: vvvvv
        # "runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-29-10-03-55_withBattery",
        # "runs/SimpleAccEnv-wide-withConveyer-battery-v0/2024-07-30-12-08-57_1M"
        ## ^^^^^^^^^^
        # "runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-08-03-19-53-26_withBattery_refactorMLP_maxVel:1.5_200kRandom_200epLen",
        "runs/SimpleAccEnv-wide-withConveyer-battery-v0/2024-08-08-11-27-00_refactorMLP_maxVel:1.5_200epLen_batch:2048_200kRandom"
        
    ]
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = f"{load_rb_dirs[-1]}/feasibility_{timestamp}_1k_lrDecay_singleLoad_veryLargeBatch_recursive_thresh:005_modelEval"
    os.makedirs(exp_dir, exist_ok=True)

    # for numpy env
    # env = LavaGoalConveyerAccelerationEnv(task="lava")
    # n_obs = 6
    # n_actions = 9
    # def label_fun(state):
    #     return env.lava_x_range[0] < state[0] < env.lava_x_range[-1] and env.lava_y_range[0] < state[1] < env.lava_y_range[-1]

    env = gym.make("SimpleAccEnv-wide-withConveyer-lava-v0")
    
    n_obs = 5
    n_actions = 25
    def label_fun(state):
        # only lava
        # return env.lava_x_min <= state[0] <= env.lava_x_max and env.lava_y_min <= state[1] <= env.lava_y_max

        # only left
        # return state[0] > (env.x_max / 2)

        # only battery
        return state[4] <= 0
        
        # battery and lava
        # return (state[4] <= 0) or (env.lava_x_min <= state[0] <= env.lava_x_max and env.lava_y_min <= state[1] <= env.lava_y_max)

    # unity env
    # env = None
    # # n_obs = 9
    # n_obs = 17
    # n_actions = 25
    # def label_fun(state):
    #     return state[0] > 0.0

    print("Loading data...")
    data, obs, actions, next_obs, dones = load_data_from_rb(load_rb_dirs, n_obs, n_actions)

    print("Labeling transitions...")
    labels = label_data(all_obs=obs, label_fun=label_fun)

    params = {
        "optimizer_initial_lr": 0.001,
        "optimizer_weight_decay": 0.0001,
        "exponential_lr_decay": 0.9995,
        "batch_size": 16384,
        # "batch_size": 4096,
        "epochs": 1000,
        "nuke_layer_every": 1e9,
        "hidden_activation": torch.nn.ReLU,
        "hidden_arch": [64, 64, 32, 32],
        "criterion": torch.nn.MSELoss,
        "with_batchNorm": True,
        # "criterion": torch.nn.L1Loss,
        # "discount_gamma": 1.0,  # unlike traditional finite-horizon TD, feasibility discount must always be <1!
        "discount_gamma": 0.995,
        # "higher_prio_load_path": "runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-29-10-03-55_withBattery/feasibility_2024-07-29-17-28-18",
        # "higher_prio_load_path": "runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-31-17-15-32_withBattery_refactorMLP/feasibility_2024-07-31-19-37-15_1k_lrDecay_veryLargeBatch",
        "higher_prio_load_path": "runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-08-03-19-53-26_withBattery_refactorMLP_maxVel:1.5_200kRandom_200epLen/feasibility_2024-08-04-09-41-43_1k_lrDecay",
        # "higher_prio_load_path": "",
        "higher_prio_batchnorm": True,
        # "higher_prio_load_path": "",
        "higher_prio_arch": [64, 64, 32, 32],
        "higher_prio_threshold": 0.05,
        "polyak_tau": 0.01
    }

    # save params as yaml
    with open(f"{exp_dir}/params.yaml", "w") as f:
        yaml.dump(params, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Setting up model...")
    model = MLP(input_size=n_obs, output_size=n_actions, hidden_activation=params["hidden_activation"], hidden_arch=params["hidden_arch"], with_batchNorm=params["with_batchNorm"])
    model.to(device)
    target_model = MLP(input_size=n_obs, output_size=n_actions, hidden_activation=params["hidden_activation"], hidden_arch=params["hidden_arch"], with_batchNorm=params["with_batchNorm"])
    target_model.load_state_dict(model.state_dict())
    target_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["optimizer_initial_lr"], weight_decay=params["optimizer_weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["exponential_lr_decay"])
    
    # load higher_prio Model
    higher_prio_nets = []
    higher_prio_threshes = []
    if params["higher_prio_load_path"]:
        higher_prio_model = MLP(input_size=n_obs, output_size=n_actions, hidden_activation=params["hidden_activation"], hidden_arch=params["higher_prio_arch"], with_batchNorm=params["higher_prio_batchnorm"])
        higher_prio_model.load_state_dict(torch.load(f"{params['higher_prio_load_path']}/feasibility_dqn.pt"))
        higher_prio_model.to(device)
        higher_prio_model.eval()
        higher_prio_nets.append(higher_prio_model)
        higher_prio_threshes.append(params["higher_prio_threshold"])

    # train model
    model, train_loss_hist, lr_hist, pred_mean_hist = train_model(
        env=env,
        model=model,
        target_model=target_model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=params["criterion"],
        states=obs,
        actions=actions,
        labels=labels,
        next_states=next_obs,
        dones=dones,
        batch_size=params["batch_size"],
        exp_dir=exp_dir,
        epochs=params["epochs"],
        nuke_layer_every=params["nuke_layer_every"],
        gamma=params["discount_gamma"],
        higher_prio_constraint_nets=higher_prio_nets,
        higher_prio_constraint_thresholds=higher_prio_threshes,
        polyak_tau=params["polyak_tau"],
        device=device
    )

    create_training_plots(
        model=model,
        env=env,
        train_loss_hist=train_loss_hist,
        lr_hist=lr_hist,
        exp_dir=exp_dir,
        pred_mean_hist=pred_mean_hist,
    )


if __name__ == "__main__":
    main()

