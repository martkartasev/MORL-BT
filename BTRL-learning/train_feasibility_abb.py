import argparse
import multiprocessing
import os
import random
import threading
import time
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg.linalg
import torch
import yaml

import buffer
from networks import MLP


def load_mlagents_buffer(load_dir, max_obs, feasibility_label, label_ratio=0.35):
    for direc in load_dir:
        replay_files = [f for f in os.listdir(direc) if isfile(join(direc, f)) and "extended_replay" in f]
        file = replay_files.pop(0)

        filename = os.path.join(direc, file)
        update_buffer = buffer.AgentBuffer()
        with open(filename, "rb+") as file_object:
            update_buffer.load_from_file(file_object)

        obs = update_buffer._fields[(buffer.ObservationKeyPrefix.OBSERVATION, 0)].to_ndarray()
        next_obs = update_buffer._fields[(buffer.ObservationKeyPrefix.NEXT_OBSERVATION, 0)].to_ndarray()
        dones = update_buffer._fields[buffer.BufferKey.DONE].to_ndarray()
        actions = update_buffer._fields[buffer.BufferKey.DISCRETE_ACTION].to_ndarray()
        labels = label_data(obs, label_function=label_fun, feasibility_label=feasibility_label)

        nr_files = len(replay_files)
        random.shuffle(replay_files)

        #for i, file in enumerate(replay_files):
        i = 1

        files_to_proccess = []
        parallel = 10
        pool = multiprocessing.Pool(processes=parallel)
        while len(replay_files) > 0:
            file = replay_files.pop(0)
            filename = os.path.join(direc, file)

            files_to_proccess.append(filename)
            i+=1
            if len(files_to_proccess) >= parallel or len(replay_files) == 0:
                async_results = [pool.apply_async(process_file, args=(feasibility_label, listFile, label_ratio)) for listFile in files_to_proccess]
                results = [ar.get() for ar in async_results]
                for result in results:
                    obs = np.append(obs, result[4], axis=0)
                    labels = np.append(labels, result[2], axis=0)
                    next_obs = np.append(next_obs, result[3], axis=0)
                    dones = np.append(dones, result[1], axis=0)
                    actions = np.append(actions, result[0], axis=0)
                    print(f"Done: labels.shape: {labels.shape}, negative labels: {len(labels) - np.sum(labels)}, as fraction: {(len(labels) - np.sum(labels)) / len(labels)}")
                files_to_proccess = []
                print("Sampled {} out of {} available files for the replay buffer. {} out of {} experiences loaded. ".format(i, nr_files, obs.shape[0], max_obs))

                if len(obs) > max_obs:
                    break

    return None, obs, actions, next_obs, dones, labels


def process_file(feasibility_label, filename, label_ratio):
    update_buffer = buffer.AgentBuffer()
    with open(filename, "rb+") as file_object:
        update_buffer.load_from_file(file_object)
    new_obs = update_buffer._fields[(buffer.ObservationKeyPrefix.OBSERVATION, 0)].to_ndarray()
    new_labels = label_data(new_obs, label_function=label_fun, feasibility_label=feasibility_label)
    new_next_obs = update_buffer._fields[(buffer.ObservationKeyPrefix.NEXT_OBSERVATION, 0)].to_ndarray()
    new_dones = update_buffer._fields[buffer.BufferKey.DONE].to_ndarray()
    new_actions = update_buffer._fields[buffer.BufferKey.DISCRETE_ACTION].to_ndarray()

    # Code for selective sampling of data of both labels
    pos_indices = np.where(new_labels == 1)[0]
    ratio = len(pos_indices) / len(new_labels)
    must_sample = False
    if ratio < label_ratio:
        neg_indices = np.where(new_labels != 1)[0]
        neg_indices = np.random.choice(neg_indices, min(int(len(pos_indices) / label_ratio - len(pos_indices)), len(neg_indices)))
        must_sample = True
    elif ratio > 1 - label_ratio:
        neg_indices = np.where(new_labels != 1)[0]
        pos_indices = np.random.choice(pos_indices, min(int(len(neg_indices) / label_ratio - len(neg_indices)), len(pos_indices)))
        must_sample = True

    if must_sample:
        new_obs = np.append(new_obs[pos_indices, :], new_obs[neg_indices, :], axis=0)
        new_labels = np.append(new_labels[pos_indices, :], new_labels[neg_indices, :], axis=0)
        new_next_obs = np.append(new_next_obs[pos_indices, :], new_next_obs[neg_indices, :], axis=0)
        new_dones = np.append(new_dones[pos_indices], new_dones[neg_indices], axis=0)
        new_actions = np.append(new_actions[pos_indices, :], new_actions[neg_indices, :], axis=0)
    return new_actions, new_dones, new_labels, new_next_obs, new_obs


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

                for idx, net in enumerate(higher_prio_constraint_nets):  # TODO: Needs to be updated as well
                    high_prio_vals = net(next_state_batch.float())
                    best_high_prio_vals = high_prio_vals.min(dim=1, keepdim=True)[0]
                    # TODO, consider higher prio when finding best?
                    high_prio_forbidden = high_prio_vals > best_high_prio_vals + higher_prio_constraint_thresholds[idx]

                    target_q_values[high_prio_forbidden] = torch.inf

                assert torch.all(target_q_values.min(dim=1).values < torch.inf)

                current_state_val = (1 - gamma) * reward_batch
                target_max = target_q_values.max(dim=1, keepdim=True)[0]
                future_val = torch.min(target_max.to(device), reward_batch)
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

        if epoch % 5 == 0:
            save_model(device, exp_dir, model, states.shape[1], f"epoch_{epoch}")

    print("Done: training model")



    # if epoch % nuke_layer_every == 0 and epoch > 0:

    return model, train_loss_hist, lr_hist, pred_mean_hist


def label_data(all_obs, label_function, feasibility_label):
    # label data for ACC violation
    #  print("Labeling data for ACC violation...")
    labels = np.ones((all_obs.shape[0], 1)) * np.inf
    for i in range(all_obs.shape[0]):
        state = all_obs[i]
        label = label_function(state=state, feasibility_label=feasibility_label)
        labels[i] = int(label)
        # if i % 500000 == 0:
        #     print(f"Labelled {i} out of {all_obs.shape[0]} experiences")

    assert not np.isinf(labels).any()
    # print(f"Done: labels.shape: {labels.shape}, negative labels: {len(labels) - np.sum(labels)}, as fraction: {(len(labels) - np.sum(labels)) / len(labels)}")

    return labels


def label_fun(state, feasibility_label):  # Now predicting that we are in the "Good set"
    if feasibility_label == "safe&have":
        return state[0] > 0 and numpy.linalg.norm(state[16:19]) > 0.1
    if feasibility_label == "have&near":
        return numpy.linalg.norm(state[16:19]) < 0.65 and state[0] > 0
    if feasibility_label == "have":
        return state[0] > 0
    if feasibility_label == "near":
        return numpy.linalg.norm(state[10:13]) < 0.65
    if feasibility_label == "safe":
        return numpy.linalg.norm(state[16:19]) > 0.10  # 16,17.18


def main(args):
    params = {
        "optimizer_initial_lr": 0.001,
        "optimizer_weight_decay": 0.0001,
        "exponential_lr_decay": 0.9995,
        "batch_size": 512,
        "buffer_size": args.buffer_size,
        "epochs": args.epochs,
        "nuke_layer_every": 1e9,
        "hidden_activation": torch.nn.ReLU,
        "hidden_arch": [128, 64, 64],
        "criterion": torch.nn.MSELoss,
        "with_batchNorm": False,
        # "criterion": torch.nn.L1Loss,
        "discount_gamma": 0.995,  # unlike traditional finite-horizon TD, feasibility discount must always be <1!
        # "higher_prio_load_path": args.higher_prio_feasibility_estimator,
        # "higher_prio_batchnorm": True,
        # "higher_prio_arch": [64, 64, 32, 32],
        # "higher_prio_threshold": 0.05,
        "polyak_tau": 0.01,
        "feasibility_label": args.feasibility_label,
        "rb_dirs": args.rb_dirs,
        "label_ratio": args.label_ratio
    }

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = f"{args.rb_dirs[-1]}/feasibility_{timestamp}_{args.exp_str}"
    os.makedirs(exp_dir, exist_ok=True)

    # save params as yaml
    with open(f"{exp_dir}/params.yaml", "w") as f:
        yaml.dump(params, f)

    print("Loading data...")
    data, obs, actions, next_obs, dones, labels = load_mlagents_buffer(args.rb_dirs, args.buffer_size, args.feasibility_label, args.label_ratio)

    n_obs = obs.shape[1]
    n_actions = 6250

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

    save_model(device, exp_dir, model, n_obs)

    return exp_dir


def save_model(device, exp_dir, model, n_obs, suffix=""):
    print(f"Saving classifier to {exp_dir}/feasibility_dqn.pt")
    torch.save(model.state_dict(), f"{exp_dir}/feasibility_dqn.pt")

    print(f"Saving model as onnx to {exp_dir}/feasibility_dqn_{suffix}.onnx")
    torch_input = torch.randn(1, n_obs).to(device)
    torch.onnx.export(model,
                      torch_input,
                      f"{exp_dir}/feasibility_dqn_{suffix}.onnx",
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=15,  # the ONNX version to export the model to
                      do_constant_folding=True, )


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
    parser.add_argument("--rb_dirs", type=str, nargs="+", help="List of replay buffer directories to load data from", default=["C:/Users/Mart9/Workspace/ABB-Warehouse/results/place_ppo_rl/ABBMobile/"])
    parser.add_argument("--buffer_size", type=int, help="Max size of replay buffer", default=15000000)
    parser.add_argument("--higher_prio_feasibility_estimator", type=str, help="Higher-prio feasibility estimator to load for recursive training", default="")
    parser.add_argument("--exp_str", type=str, help="String to append to the experiment directory", default="nearPlace128x64x64")
    parser.add_argument("--feasibility_label", type=str, help="Which labelling function to use", default="have&near")
    parser.add_argument("--label_ratio", type=float, help="Minimum ratio between negative labelled data and all data. Between 0 and 1. 1 means all labels, 0 means no labels.", default=0.4)
    parser.add_argument("--epochs", type=int, help="Number of epochs to train the model", default=50)
    args = parser.parse_args()

    exp_dir = main(args)

    # EXTREMELY IMPORTANT: Last line of the script must print the experiment directory such that the bash script can capture it!
    print(exp_dir)
