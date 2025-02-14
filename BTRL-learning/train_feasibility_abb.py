import argparse
import os
import random
import time
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg.linalg
import torch
import yaml

import buffer
from networks import MLP


def load_mlagents_buffer(load_dir, max_obs=10000000):
    update_buffer = buffer.AgentBuffer()
    for direc in load_dir:
        replay_files = [f for f in os.listdir(direc) if isfile(join(direc, f)) and "extended_replay" in f]
        file = replay_files.pop(0)

        filename = os.path.join(direc, file)
        with open(filename, "rb+") as file_object:
            update_buffer.load_from_file(file_object)
            experiences = update_buffer.num_experiences

        obs = update_buffer._fields[(buffer.ObservationKeyPrefix.OBSERVATION, 0)].to_ndarray()
        next_obs = update_buffer._fields[(buffer.ObservationKeyPrefix.NEXT_OBSERVATION, 0)].to_ndarray()
        dones = update_buffer._fields[buffer.BufferKey.DONE].to_ndarray()
        actions = update_buffer._fields[buffer.BufferKey.DISCRETE_ACTION].to_ndarray()

        nr_files = int(2*max_obs / experiences)
        for i, file in enumerate(random.sample(replay_files, min(nr_files, len(replay_files)))):
            filename = os.path.join(direc, file)
            with open(filename, "rb+") as file_object:
                update_buffer.load_from_file(file_object)

            obs = np.append(obs, update_buffer._fields[(buffer.ObservationKeyPrefix.OBSERVATION, 0)].to_ndarray(), axis=0)
            next_obs = np.append(next_obs, update_buffer._fields[(buffer.ObservationKeyPrefix.NEXT_OBSERVATION, 0)].to_ndarray(), axis=0)
            dones = np.append(dones, update_buffer._fields[buffer.BufferKey.DONE].to_ndarray(), axis=0)
            actions = np.append(actions, update_buffer._fields[buffer.BufferKey.DISCRETE_ACTION].to_ndarray(), axis=0)
            if i % 10 == 0:
                print("Loaded {} out of {} files for the replay buffer. {} out of {} experiences loaded. ".format(i, nr_files, obs.shape[0], max_obs))

    return None, obs, actions, next_obs, dones


def train_model(
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
            # done_batch = torch.from_numpy(dones[batch_idx]).to(device)

            # compute TD target
            with torch.no_grad():
                target_q_values = target_model(next_state_batch.float())

                for idx, net in enumerate(higher_prio_constraint_nets):
                    high_prio_vals = net(next_state_batch.float())
                    best_high_prio_vals = high_prio_vals.min(dim=1, keepdim=True)[0]
                    # TODO, consider higher prio when finding best?
                    high_prio_forbidden = high_prio_vals > best_high_prio_vals + higher_prio_constraint_thresholds[idx]

                    target_q_values[high_prio_forbidden] = torch.inf

                assert torch.all(target_q_values.min(dim=1).values < torch.inf)

                current_state_val = (1 - gamma) * reward_batch
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
            for target_param, param in zip(target_model.parameters(), model.parameters()):
                target_param.data.copy_(polyak_tau * param.data + (1.0 - polyak_tau) * target_param.data)

            if i % 100 == 0:
                print(f"Epoch {epoch}, batch {i} / {batches_per_episode}, loss: {np.around(loss.item(), 5)}, avg. q-values: {np.around(q_values.mean().item(), 3)}, lr={np.around(optimizer.param_groups[0]['lr'], 5)}", flush=True)

        # train_loss_hist.append(train_loss / batches_done)
        lr_hist.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

        # hard target network update
        # target_model.load_state_dict(model.state_dict())

        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"{exp_dir}/feasibility_dqn_{epoch}.pt")

        # if epoch % nuke_layer_every == 0 and epoch > 0:
    print("Done: training model")

    print(f"Saving classifier to {exp_dir}/feasibility_dqn.pt")
    torch.save(model.state_dict(), f"{exp_dir}/feasibility_dqn.pt")

    print(f"Saving model as onnx to {exp_dir}/feasibility_dqn.onnx")
    torch_input = torch.randn(1, 31).to(device)
    torch.onnx.export(model,
                      torch_input,
                      f"{exp_dir}/feasibility_dqn.onnx",
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=15,  # the ONNX version to export the model to
                      do_constant_folding=True, )

    return model, train_loss_hist, lr_hist, pred_mean_hist


def label_data(all_obs, label_fun):
    # label data for ACC violation
    print("Labeling data for ACC violation...")
    labels = np.ones((all_obs.shape[0], 1)) * np.inf
    for i in range(all_obs.shape[0]):
        state = all_obs[i]
        label = label_fun(state=state)
        labels[i] = int(label)
        if i % 500000 == 0:
            print(f"Labelled {i} out of {all_obs.shape[0]} experiences")

    assert not np.isinf(labels).any()
    print(f"Done: labels.shape: {labels.shape}, positive labels: {np.sum(labels)}")

    return labels


def label_fun(state):
    return numpy.linalg.norm(state[16:19]) < 0.1  # 16,17.18


def main(args):
    params = {
        "optimizer_initial_lr": 0.001,
        "optimizer_weight_decay": 0.0001,
        "exponential_lr_decay": 0.9995,
        "batch_size": 4096,
        "epochs": args.epochs,
        "nuke_layer_every": 1e9,
        "hidden_activation": torch.nn.ReLU,
        "hidden_arch": [256, 256, 128, 128],
        "criterion": torch.nn.MSELoss,
        "with_batchNorm": True,
        # "criterion": torch.nn.L1Loss,
        "discount_gamma": 0.999,  # unlike traditional finite-horizon TD, feasibility discount must always be <1!
        # "higher_prio_load_path": args.higher_prio_feasibility_estimator,
        #"higher_prio_batchnorm": True,
        #"higher_prio_arch": [64, 64, 32, 32],
        #"higher_prio_threshold": 0.05,
        "polyak_tau": 0.01,
        #"feasibility_label": args.feasibility_label,
        "rb_dirs": args.rb_dirs,
    }

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = f"{args.rb_dirs[-1]}/feasibility_{timestamp}_{args.exp_str}"
    os.makedirs(exp_dir, exist_ok=True)

    print("Loading data...")
    data, obs, actions, next_obs, dones = load_mlagents_buffer(args.rb_dirs)

    n_obs = obs.shape[1]
    n_actions = 6250

    print("Labeling transitions...")
    labels = label_data(all_obs=obs, label_fun=label_fun)

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
    if params.__contains__("higher_prio_load_path"):
        higher_prio_model = MLP(input_size=n_obs, output_size=n_actions, hidden_activation=params["hidden_activation"], hidden_arch=params["higher_prio_arch"], with_batchNorm=params["higher_prio_batchnorm"])
        higher_prio_model.load_state_dict(torch.load(f"{params['higher_prio_load_path']}/feasibility_dqn.pt"))
        higher_prio_model.to(device)
        higher_prio_model.eval()
        higher_prio_nets.append(higher_prio_model)
        higher_prio_threshes.append(params["higher_prio_threshold"])

    # train model
    model, train_loss_hist, lr_hist, pred_mean_hist = train_model(
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
        train_loss_hist=train_loss_hist,
        lr_hist=lr_hist,
        exp_dir=exp_dir,
        pred_mean_hist=pred_mean_hist,
    )

    return exp_dir


def create_training_plots(exp_dir, train_loss_hist=None, lr_hist=None, pred_mean_hist=None):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rb_dirs", type=str, nargs="+", help="List of replay buffer directories to load data from", default=["C:/Users/Mart9/Workspace/ABB-Warehouse/results/move_ppo_01/ABBMobile/"])
    parser.add_argument("--higher_prio_feasibility_estimator", type=str, help="Higher-prio feasibility estimator to load for recursive training", default="")
    parser.add_argument("--exp_str", type=str, help="String to append to the experiment directory", default="test")
    parser.add_argument("--feasibility_label", type=str, help="String to append to the experiment directory", default="abb")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train the model", default=200)
    args = parser.parse_args()

    exp_dir = main(args)

    # EXTREMELY IMPORTANT: Last line of the script must print the experiment directory such that the bash script can capture it!
    print(exp_dir)
