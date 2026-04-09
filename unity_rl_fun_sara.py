from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


class OffPolicyReplayBuffer:
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.buffer = []
        self.position = 0
        self.mean_reward = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, obs, next_obs, action, reward, done, info):
        transition = (obs, next_obs, action, reward, done, info)
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
            self.position = (self.position + 1) % self.max_size

        self.mean_reward = self.mean_reward * 0.99 + reward * 0.01  # update mean reward with a moving average

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        obs_batch, next_obs_batch, action_batch, reward_batch, done_batch, info_batch = zip(*batch)
        return np.array(obs_batch), np.array(next_obs_batch), np.array(action_batch), np.array(reward_batch), np.array(done_batch), np.array(info_batch)


class ActorNetwork(nn.Module):
    """Policy network that outputs mean and log_std of action distribution"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std


class QNetwork(nn.Module):
    """Critic network that estimates Q(s, a)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class SoftActorCriticAgent:
    def __init__(self, state_dim, action_dim, learning_rate=3e-4, gamma=0.99, tau=0.005,
                 alpha=0.2, device="cpu"):
        """
        Initialize SAC agent with actor and critic networks.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            learning_rate: Learning rate for all optimizers
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Entropy regularization coefficient
            device: Device to run computations on (cpu or cuda)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device(device)

        # Actor network
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        # Q-networks (critic networks)
        self.q_net1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q_net2 = QNetwork(state_dim, action_dim).to(self.device)

        # Target Q-networks for stability
        self.target_q_net1 = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net2 = QNetwork(state_dim, action_dim).to(self.device)

        # Copy weights to target networks
        self._hard_update(self.target_q_net1, self.q_net1)
        self._hard_update(self.target_q_net2, self.q_net2)

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=learning_rate)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=learning_rate)

    def get_action(self, state, deterministic=False):
        """
        Sample an action from the policy conditioned on the state.

        Args:
            state: Current state (numpy array or tensor)
            deterministic: If True, return mean action; if False, sample from distribution

        Returns:
            action: Action sampled from the policy (numpy array)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            mean, log_std = self.actor(state)
            std = log_std.exp()

            if deterministic:
                action = mean
            else:
                # Sample from Gaussian distribution
                normal = Normal(mean, std)
                action = normal.rsample()

            # Clamp action to [-1, 1]
            action = torch.tanh(action)

        return action.squeeze(0).cpu().numpy()

    def train(self, batch):
        """
        Perform one step of SAC training using a batch of transitions.

        Args:
            batch: Tuple of (obs, next_obs, action, reward, done, info) from replay buffer

        Returns:
            dict: Dictionary containing loss values for monitoring
        """
        obs, next_obs, action, reward, done, info = batch

        # subtract mean reward from rewards
        reward = reward - replay_buffer.mean_reward

        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs).to(self.device)
        action_tensor = torch.FloatTensor(action).to(self.device)
        reward_tensor = torch.FloatTensor(reward).to(self.device)
        done_tensor = torch.FloatTensor(done).to(self.device)

        # Reshape if needed
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if next_obs_tensor.dim() == 1:
            next_obs_tensor = next_obs_tensor.unsqueeze(0)
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
        if reward_tensor.dim() == 0:
            reward_tensor = reward_tensor.unsqueeze(0)
        if done_tensor.dim() == 0:
            done_tensor = done_tensor.unsqueeze(0)

        # --- Update Q-networks ---
        with torch.no_grad():
            # Get next actions from policy
            next_mean, next_log_std = self.actor(next_obs_tensor)
            next_std = next_log_std.exp()
            normal = Normal(next_mean, next_std)
            next_action = normal.rsample()
            next_action_tanh = torch.tanh(next_action)

            # Compute log probability of next action
            # Account for tanh squashing (using change of variables)
            log_prob = normal.log_prob(next_action) - torch.log(1 - next_action_tanh.pow(2) + 1e-7)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

            # Target Q value
            target_q1 = self.target_q_net1(next_obs_tensor, next_action_tanh)
            target_q2 = self.target_q_net2(next_obs_tensor, next_action_tanh)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob
            target_q = reward_tensor + (1 - done_tensor) * self.gamma * target_q

        # Update Q-network 1
        q1 = self.q_net1(obs_tensor, action_tensor)
        loss_q1 = F.mse_loss(q1, target_q)
        self.q_optimizer1.zero_grad()
        loss_q1.backward()
        self.q_optimizer1.step()

        # Update Q-network 2
        q2 = self.q_net2(obs_tensor, action_tensor)
        loss_q2 = F.mse_loss(q2, target_q)
        self.q_optimizer2.zero_grad()
        loss_q2.backward()
        self.q_optimizer2.step()

        # --- Update Actor ---
        mean, log_std = self.actor(obs_tensor)
        std = log_std.exp()
        normal = Normal(mean, std)
        action_sample = normal.rsample()
        action_sample_tanh = torch.tanh(action_sample)

        # Compute log probability
        log_prob = normal.log_prob(action_sample) - torch.log(1 - action_sample_tanh.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Actor loss: maximize expected Q value and entropy
        q1_pi = self.q_net1(obs_tensor, action_sample_tanh)
        q2_pi = self.q_net2(obs_tensor, action_sample_tanh)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_prob - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft update target networks ---
        self._soft_update(self.target_q_net1, self.q_net1)
        self._soft_update(self.target_q_net2, self.q_net2)

        return {
            "loss_q1": loss_q1.item(),
            "loss_q2": loss_q2.item(),
            "actor_loss": actor_loss.item(),
        }

    def _soft_update(self, target_net, net):
        """Soft update of target network parameters"""
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _hard_update(self, target_net, net):
        """Hard copy of network parameters"""
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(param.data)


def compute_reward(obs):
    obs_dict = obs_to_dict(obs)
    target_position = obs_dict["target_position"]
    agent_position = obs_dict["agent_position"]
    distance_to_target = np.linalg.norm(target_position - agent_position)
    reward = -distance_to_target
    return reward


def obs_to_dict(obs):
    return {
        "target_position": obs[0:3],
        "agent_position": obs[3:6],
        "agent_velocity": obs[6:8],
    }


def setup_unity_env():
    engine = EngineConfigurationChannel()
    engine.set_configuration_parameters(time_scale=1)  # Can speed up simulation between steps with this
    engine.set_configuration_parameters(quality_level=0)
    engine.set_configuration_parameters(width=1000, height=1000)

    print("Waiting for unity env to start...")
    env = UnityEnvironment(
        # file_name=f"envs/unity_builds/{unity_scene_dir}/myBuild-MORL-BT.x86_64",  # comment out to connect to unity editor instance
        no_graphics=False,  # Can disable graphics if needed
        # base_port=10001,  # for starting multiple envs
        side_channels=[engine])
    print("Unity env ready")

    action_dim = 2
    state_dim = 8
    env.action_space = gym.spaces.Discrete(action_dim)
    env.observation_space = gym.spaces.Box(
        low=np.array([-np.inf] * state_dim),
        high=np.array([np.inf] * state_dim),
        dtype=np.float32
    )
    n_agents = 1  # number of agents in the unity scene
    env.reset()  # init unity env and all agents within

    behavior_name = "RollerBall?team=0"
    spec = env.behavior_specs[behavior_name]
    print("Behavior spec:", spec.action_spec)

    return env, action_dim, state_dim, n_agents


def unity_env_interaction(unity_env, rb, agent):
    (decision_steps, terminal_steps) = unity_env.get_steps("RollerBall?team=0")
    obs = decision_steps.obs[0]  # Strange structure, but this is how you get the observations array
    nr_agents = len(decision_steps)  # this many agents need to take an action

    if nr_agents > 0:
        # an agent needs an action
        unity_actions = np.zeros((nr_agents, action_dim))
        for i in decision_steps.agent_id:
            rl_action = agent.get_action(obs[i][:])
            unity_actions[i] = rl_action
    else:
        # we still need to pass an empty action tuple even if no agent acts
        unity_actions = np.zeros((0, action_dim))

    action_tuple = ActionTuple()
    action_tuple.add_continuous(unity_actions)
    unity_env.set_actions("RollerBall?team=0", action_tuple)
    unity_env.step()
    (next_decision_steps, next_terminal_steps) = unity_env.get_steps("RollerBall?team=0")

    rew = None
    termination = False
    if len(next_decision_steps.agent_id) == 0 and len(next_terminal_steps.agent_id) > 0:
        # some episode has ended, we need to handle that
        for idx, j in enumerate(next_terminal_steps.agent_id):
            if j in decision_steps.agent_id:
                # find those agents that did a step and now are done, aka those that did a valid (s, a, s') transition
                agent_obs = obs[j][:]
                next_obs = next_terminal_steps.obs[0][idx][:]
                action = unity_actions[j]

                # compute reward
                rew = np.array([compute_reward(agent_obs)])
                termination = np.array([1])  # we know it's done since it's in the terminal steps, but we don't know if it's done or truncated, so just set to 1 for now
                info = None
                rb.add(agent_obs, next_obs, action, rew, termination, info)
            else:
                # happens after reset, we have a "next obs" but not yet a "current obs"
                # transition will be added to the buffer after the next env interaction
                pass

    elif len(next_decision_steps.agent_id) > 0 and len(next_terminal_steps.agent_id) == 0 and len(decision_steps.agent_id) == len(next_decision_steps.agent_id):
        # all agents have taken a step and none has terminates
        for j in next_decision_steps.agent_id:
            agent_obs = obs[j][:]
            next_obs = next_decision_steps.obs[0][j][:]
            action = unity_actions[j]

            # compute reward
            rew = np.array([compute_reward(agent_obs)])

            termination = np.array([0])  # not done here
            info = None
            rb.add(agent_obs, next_obs, action, rew, termination, info)

    else:
        # this happens after reset, we have next obs but not yet current obs
        pass

    return rew, termination


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, action_dim, state_dim, n_agents = setup_unity_env()
    replay_buffer = OffPolicyReplayBuffer(max_size=100000)

    sac_agent = SoftActorCriticAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    batch_size = 64

    return_hist = []
    ep_return = 0
    for i in range(10000):

        step_reward, step_termination = unity_env_interaction(env, replay_buffer, sac_agent)

        if step_reward is not None:
            ep_return += step_reward

        if step_termination:
            return_hist.append(ep_return)
            print(f"Episode return: {ep_return}")
            ep_return = 0

        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            loss_info = sac_agent.train(batch)
            print(f"Step {i}, Losses: {loss_info}")









