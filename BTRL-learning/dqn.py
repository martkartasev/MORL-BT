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
            hidden_arch,
            device,
            hidden_activation=nn.ELU,
            lr=1e-3,
            gamma=0.99,
            load_cp="",
            con_model_load_cps=[],
            con_threshes=[],
            model_name="q",
            con_model_arches=[],
            con_batch_norms=[],
            batch_norm=False,
    ):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_arch = hidden_arch
        self.hidden_activation = hidden_activation
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.load_cp = load_cp
        self.con_model = None
        self.con_threshes = con_threshes
        self.model_name = model_name
        self.con_model_arches = con_model_arches
        self.batch_norm = batch_norm
        self.con_batch_norms = con_batch_norms

        self.con_models = []
        for con_idx, con_model_load_cp in enumerate(con_model_load_cps):
            if con_model_load_cp != "":
                con_model = MLP(input_size=self.state_dim, output_size=self.action_dim, hidden_activation=self.hidden_activation, hidden_arch=self.con_model_arches[con_idx], with_batchNorm=con_batch_norms[con_idx])
                con_model.load_state_dict(torch.load(con_model_load_cp))
                con_model.to(self.device)
                con_model.eval()
                self.con_models.append(con_model)

        self.q_net = MLP(input_size=self.state_dim, output_size=self.action_dim, hidden_activation=self.hidden_activation, hidden_arch=self.hidden_arch, with_batchNorm=self.batch_norm)

        if self.load_cp:
            self.q_net.load_state_dict(torch.load(self.load_cp))

        self.q_target_net = MLP(input_size=self.state_dim, output_size=self.action_dim, hidden_activation=self.hidden_activation, hidden_arch=self.hidden_arch, with_batchNorm=self.batch_norm)
        self.q_target_net.load_state_dict(self.q_net.state_dict())

        for model in [self.q_net, self.q_target_net]:
            model.to(self.device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

    def save_model(self, save_dir):
        torch.save(self.q_net.state_dict(), f"{save_dir}/{self.model_name}_net.pth")
        torch.save(self.q_target_net.state_dict(), f"{save_dir}/{self.model_name}_target_net.pth")

        for con_idx, con_model in enumerate(self.con_models):
            torch.save(con_model.state_dict(), f"{save_dir}/{self.model_name}_con_model_{con_idx}.pth")

    def compute_mask(self, state_batch, up_to_idx=-1):
        """
        Computes the action mask for a batch of states, consider the first, up to but not including, `up_to_idx` constraints.
        If `up_to_idx` is -1, all constraints are used to compute the mask.
        Otherwise, if 0 < `up_to_idx` < len(self.con_models), i.e. 2, constraint  0 and 1 are used.
        :param state_batch: The batch of states, must be of shape (-1, self.state_dim)
        :param up_to_idx: Int, use the first, up to this many, constraints.
        :return:
        """
        assert state_batch.ndim == 2
        assert state_batch.shape[1] == self.state_dim

        if up_to_idx == -1:
            up_to_idx = len(self.con_models)

        # if len(self.con_models) > 1:
        #     print("Mask computation for multiple constraints has not been tested yet!")

        n_states = state_batch.shape[0]
        mask_forbidden_global = torch.zeros(n_states, self.action_dim, device=self.device)

        for con_idx in range(up_to_idx):
            con_model = self.con_models[con_idx]
            con_thresh = self.con_threshes[con_idx]

            con_pred = con_model(state_batch)
            con_pred[mask_forbidden_global.bool()] += torch.inf  # apply higher prio cons before finding best value
            best_con_action_value = con_pred.min(dim=1).values
            mask_forbidden_local = con_pred > best_con_action_value.unsqueeze(1) + con_thresh

            mask_forbidden_global[mask_forbidden_local] += torch.inf

        return mask_forbidden_global.bool().squeeze()

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

            forbidden_mask = self.compute_mask(next_state_batch)
            double_q_values[forbidden_mask] = -torch.inf

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
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
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

        forbidden_mask = self.compute_mask(state_batch=state.unsqueeze(0))
        q_values[forbidden_mask] = -torch.inf

        if np.random.rand() < epsilon:
            action = np.random.choice(np.where(q_values.detach().cpu().numpy() > -np.inf)[0])
        else:
            action = np.argmax(q_values.detach().cpu().numpy())

        if ret_vals:
            return action, q_values.detach().cpu().numpy()
        else:
            return action




