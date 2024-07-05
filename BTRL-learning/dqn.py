import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks import MLP


class DQN:
    def __init__(
            self,
            action_dim,
            state_dim,
            hidden_dim,
            device,
            hidden_activation=nn.ELU,
            lr=1e-3,
            gamma=0.99,
            load_cp="",
            con_model_load_cp="",
            con_thresh=0.1
    ):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.load_cp = load_cp
        self.con_model = None
        self.con_thresh = con_thresh

        if con_model_load_cp:
            # TODO, correct non hard coded hidden size
            self.con_model = MLP(input_size=self.state_dim, output_size=self.action_dim, hidden_size=32, hidden_activation=self.hidden_activation)
            self.con_model.load_state_dict(torch.load(con_model_load_cp))
            self.con_model.to(self.device)

        self.q_net = MLP(input_size=self.state_dim, output_size=self.action_dim, hidden_size=self.hidden_dim, hidden_activation=self.hidden_activation)

        if self.load_cp:
            self.q_net.load_state_dict(torch.load(self.load_cp))

        self.q_target_net = MLP(input_size=self.state_dim, output_size=self.action_dim, hidden_size=self.hidden_dim, hidden_activation=self.hidden_activation)
        self.q_target_net.load_state_dict(self.q_net.state_dict())

        for model in [self.q_net, self.q_target_net]:
            model.to(self.device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

    def save_model(self, save_dir):
        torch.save(self.q_net.state_dict(), f"{save_dir}/q_net.pth")
        torch.save(self.q_target_net.state_dict(), f"{save_dir}/q_target_net.pth")

        if self.con_model is not None:
            torch.save(self.con_model.state_dict(), f"{save_dir}/con_model.pth")

    def update(
            self,
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
    ):
        with torch.no_grad():
            # normal dqn
            # target_max, _ = target_network(data.next_observations).max(dim=1)

            # double dqn
            double_q_values = self.q_net(next_state_batch)
            target_q_values = self.q_target_net(next_state_batch)
            target_max = target_q_values.gather(1, torch.argmax(double_q_values, dim=1, keepdim=True)).squeeze()

            # ensemble dqn
            # target_q_values = torch.stack([target_network(update_next_observations) for target_network in q_ensemble_target])
            # indices = torch.tensor(random.sample(range(ensemble_size), 2))
            # target_q_values = target_q_values[indices]
            # target_q_values = target_q_values.min(dim=0).values
            # target_max = target_q_values.max(dim=1).values

            td_target = reward_batch.squeeze() + self.gamma * target_max * (1 - done_batch.squeeze())

        pred = self.q_net(state_batch).gather(1, action_batch).squeeze()

        loss = F.mse_loss(pred, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), pred.mean().item()

    def target_update(self, tau):
        for target_param, q_param in zip(self.q_target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(
                tau * q_param.data + (1.0 - tau) * target_param.data
            )

    def act(self, state, epsilon, ret_vals=False):
        state = torch.from_numpy(state).float().to(self.device)
        q_values = self.q_net(state)

        if self.con_model is not None:
            con_pred = self.con_model(state)
            con_mask_forbidden = con_pred > self.con_thresh
        else:
            con_mask_forbidden = torch.zeros_like(q_values).bool()

        if False in con_mask_forbidden:
            q_values[con_mask_forbidden] = -torch.inf
        else:
            print("EVERY ACTION IS FORBIDDEN, NOT APPLYING MASK")

        if np.random.rand() < epsilon:
            action = np.random.choice(np.where(q_values.detach().cpu().numpy() > -np.inf)[0])
        else:
            action = np.argmax(q_values.detach().cpu().numpy())

        if ret_vals:
            return action, q_values.detach().cpu().numpy()
        else:
            return action




