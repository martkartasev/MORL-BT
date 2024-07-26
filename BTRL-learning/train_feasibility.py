import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
# from mlagents_envs.environment import UnityEnvironment
import yaml

from envs import LavaGoalConveyerAccelerationEnv, SimpleAccEnv
from networks import MLP
from plotting import plot_value_2D, plot_discrete_actions


def load_data_from_rb(rb_path, n_obs, n_actions):
    print(f"Loading replay buffer from {rb_path}...")
    data = np.load(rb_path, allow_pickle=True)

    obs = data["all_states"]
    actions = data["all_actions"]
    next_obs = data["all_next_states"]
    dones = data["all_dones"]

    # shuffle the data obs, actions, next_obs, dones in the same way
    shuffle_idx = np.random.permutation(obs.shape[0])
    obs = obs[shuffle_idx]
    actions = actions[shuffle_idx]
    next_obs = next_obs[shuffle_idx]
    dones = dones[shuffle_idx]

    if len(obs.shape) == 3:
        obs = obs.squeeze(1)
        actions = actions.squeeze(1)
        next_obs = next_obs.squeeze(1)

    assert obs.shape[0] == actions.shape[0] == next_obs.shape[0] == dones.shape[0]
    assert obs.shape[1] == n_obs
    assert actions.shape[1] == 1
    assert next_obs.shape[1] == n_obs
    assert dones.shape[1] == 1

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
                    best_high_prio_vals = high_prio_vals.min(dim=1).values  # TODO, consider higher prio when finding best?
                    high_prio_forbidden = high_prio_vals > best_high_prio_vals.unsqueeze(1) + higher_prio_constraint_thresholds[idx]

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
    # load_rb_dir = "runs/LavaGoalConveyerAcceleration-lava-v0/2024-07-05-09-56-06_veryGood_afterRefactor"
    # load_rb_dir = "runs/LavaGoalConveyerAcceleration-lava-noConveyer-v0/2024-07-05-14-53-05"
    # load_rb_dir = "runs/flat-acc-button_fetch_trigger/2024-07-05-11-46-34"
    # load_rb_dir = "runs/SimpleAccEnv-lava-v0/2024-07-07-11-18-06"
    # load_rb_dir = "runs/SimpleAccEnv-withConveyer-lava-v0/2024-07-08-17-45-38"
    # load_rb_dir = "runs/flat-acc_reach_goal/2024-07-05-19-37-30"
    # load_rb_dir = "runs/flat-acc-button_fetch_trigger/2024-07-09-20-42-07_trainAgain"
    # load_rb_dir = "runs/SimpleAccEnv-withConveyer-lava-v0/2024-07-11-20-12-40_250k"
    # load_rb_dir = "runs/SimpleAccEnv-withConveyer-lava-v0/2024-07-14-19-08-39_250k_50krandom"
    # load_rb_dir = "runs/SimpleAccEnv-withConveyer-goal-v0/2024-07-11-11-06-24_250k"
    load_rb_dir = "runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-25-16-24-08_200kRandom_squareResetMultipleReings"
    rb_path = f"{load_rb_dir}/replay_buffer.npz"
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = f"{load_rb_dir}/feasibility_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)

    # for numpy env
    # env = LavaGoalConveyerAccelerationEnv(task="lava")
    # n_obs = 6
    # n_actions = 9
    # def label_fun(state):
    #     return env.lava_x_range[0] < state[0] < env.lava_x_range[-1] and env.lava_y_range[0] < state[1] < env.lava_y_range[-1]

    env = SimpleAccEnv(
        with_conveyer=True,
        x_max=20,
        conveyer_x_min=2,
        conveyer_x_max=10,
        lava_x_min=10,
        lava_x_max=18,
        goal_x=10,
    )
    
    n_obs = 4
    n_actions = 25
    def label_fun(state):
        # only lava
        return env.lava_x_min <= state[0] <= env.lava_x_max and env.lava_y_min <= state[1] <= env.lava_y_max

        # only left
        # return state[0] > (env.x_max / 2)

        # combined estimator for right side of env OR being inside lava region
        # return (state[0] > (env.x_max / 2)) or (env.lava_x_min <= state[0] <= env.lava_x_max and env.lava_y_min <= state[1] <= env.lava_y_max)

    # unity env
    # env = None
    # # n_obs = 9
    # n_obs = 17
    # n_actions = 25
    # def label_fun(state):
    #     return state[0] > 0.0

    print("Loading data...")
    data, obs, actions, next_obs, dones = load_data_from_rb(rb_path, n_obs, n_actions)

    print("Labeling transitions...")
    labels = label_data(all_obs=obs, label_fun=label_fun)

    params = {
        "optimizer_initial_lr": 0.001,
        "optimizer_weight_decay": 0.0001,
        "exponential_lr_decay": 1.0,
        "batch_size": 2048,
        "epochs": 1000,
        "nuke_layer_every": 1e9,
        "hidden_activation": torch.nn.ReLU,
        "hidden_arch": [32, 32, 32, 16],
        "criterion": torch.nn.MSELoss,
        # "criterion": torch.nn.L1Loss,
        # "discount_gamma": 1.0,  # unlike traditional finite-horizon TD, feasibility discount must always be <1!
        "discount_gamma": 0.995,
        # "higher_prio_load_path": "runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-16-03-00-37_good/feasibility_2024-07-23-10-54-38_lava_repr",
        "higher_prio_load_path": "",
        "higher_prio_arch": [32, 32, 16, 16],
        "higher_prio_threshold": 0.1,
        "polyak_tau": 0.01
    }

    # save params as yaml
    with open(f"{exp_dir}/params.yaml", "w") as f:
        yaml.dump(params, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Setting up model...")
    model = MLP(input_size=n_obs, output_size=n_actions, hidden_activation=params["hidden_activation"], hidden_arch=params["hidden_arch"])
    model.to(device)
    target_model = MLP(input_size=n_obs, output_size=n_actions, hidden_activation=params["hidden_activation"], hidden_arch=params["hidden_arch"])
    target_model.load_state_dict(model.state_dict())
    target_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["optimizer_initial_lr"], weight_decay=params["optimizer_weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["exponential_lr_decay"])
    
    # load higher_prio Model
    higher_prio_nets = []
    higher_prio_threshes = []
    if params["higher_prio_load_path"]:
        higher_prio_model = MLP(input_size=n_obs, output_size=n_actions, hidden_activation=params["hidden_activation"], hidden_arch=params["higher_prio_arch"])
        higher_prio_model.load_state_dict(torch.load(f"{params['higher_prio_load_path']}/feasibility_dqn.pt"))
        higher_prio_model.to(device)
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

