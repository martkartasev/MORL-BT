import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class LavaGoalConveyerAccelerationEnv(gym.Env):
    """
    2D goal navigation env, continuous state space containing agent position, agent velocity, and goal position.
    There 9 discerete actions, corresponding to 8 directions of acceleration, and no-op.
    There is a lava obstacle which yields negative cost.
    There is a conveyer belt that transports the agent into the lava, over some number of steps, if steped onto.
    """
    def __init__(
            self,
            task="goal",  # "goal" or "lava" or "sum"
            render_mode="human",
            max_velocity=2.0,
            dt=0.5
    ):

        self.render_mode = render_mode
        assert task in ["goal", "lava", "sum"], "Invalid task! Must be 'goal', 'lava', or 'sum'."
        self.task = task

        # discrete actions corresponding to accelerating in 8 direction, or no-op
        self.single_action_space = gym.spaces.Discrete(9)
        self.action_space = self.single_action_space
        self.max_velocity = max_velocity
        self.dt = dt

        # current (x, y) position and goal position
        self.x_range = np.array([0, 10])
        self.y_range = np.array([0, 4])
        self.single_observation_space = gym.spaces.Box(
            low=np.array([
                self.x_range[0],     # agent x
                self.y_range[0],     # agent y
                -self.max_velocity,  # velocity x
                -self.max_velocity,  # velocity y
                self.x_range[0],     # goal x
                self.y_range[0],     # goal_y
            ]),
            high=np.array([
                self.x_range[1],    # agent x
                self.y_range[1],    # agent y
                self.max_velocity,  # velocity x
                self.max_velocity,  # velocity y
                self.x_range[1],    # goal x
                self.y_range[1]     # goal y
            ]),
            dtype=np.float32
        )
        self.observation_space = self.single_observation_space

        self.action_map = {
            0: np.array([1, 0]),  # move right
            1: np.array([-1, 0]),  # move left
            2: np.array([0, 1]),  # move up
            3: np.array([0, -1]),  # move down
            4: np.array([1, 1]),  # move up-right
            5: np.array([-1, 1]),  # move up-left
            6: np.array([1, -1]),  # move down-right
            7: np.array([-1, -1]),  # move down-left
            8: np.array([0, 0]),  # no-op
        }

        self.lava_x_range = np.array([6.01, 8])
        self.lava_y_range = np.array([1, 2, 3])

        self.conveyer_x_range = np.array([3, 4, 5, 6])
        self.conveyer_y_range = np.array([1, 2, 3])

        self.max_episode_steps = 100
        self.lava_cost = -10  # for stepping into lava
        self.goal_reward = 0  # for reaching the goal
        self.goal_punishment = -1  # while the goal is not reached

        # episode vars, need to be reset
        self.agent_pos = [0, 0]
        self.agent_vel = [0, 0]
        self.goal_x = 1
        self.goal_y = 2
        self.goal_pos = [None, None]
        self.episode_step_counter = 0

        # list of states that are interesting for plotting the Q-function
        self.eval_states = [
            np.array([1.0, 2.0, 0.0, 0.0, self.goal_x, self.goal_y]),  # conveyer belt avoidable due to no velocity...
            np.array([1.75, 2.0, 0.0, 0.0, self.goal_x, self.goal_y]),  # conveyer belt avoidable due to no velocity, but will fail in one step to the right
            np.array([1.0, 2.0, 2.0, 0.0, self.goal_x, self.goal_y]),  # conveyer belt unavoidable due to velocity
            np.array([1.25, 2.0, 2.0, 0.0, self.goal_x, self.goal_y]),  # conveyer belt unavoidable due to velocity, closer to conveyer
            np.array([1.5, 2.0, 2.0, 0.0, self.goal_x, self.goal_y]),  # conveyer belt unavoidable due to velocity, even closer to conveyer
            np.array([0.25, 2.0, 2.0, 0.0, self.goal_x, self.goal_y]),  # conveyer belt avoidable
            np.array([0.25, 0.25, 0.0, 0.0, self.goal_x, self.goal_y]),  # no velocity in safe area, all good
            np.array([6, 0.8, 0.0, 0.0, self.goal_x, self.goal_y]),  # close to lava area
            np.array([4, 0.8, 0.0, 0.0, self.goal_x, self.goal_y]),  # close to conveyer belt
            np.array([4, 0.8, 0.0, 2, self.goal_x, self.goal_y]),  # close to conveyer, but upwards velocity
        ]

    def reset(self, seed=None, options={}):
        x = np.random.uniform(self.x_range[0], self.x_range[1])
        y = np.random.uniform(self.y_range[0], self.y_range[1])
        self.agent_pos = np.array([x, y], dtype=np.float32)

        vel_x = np.random.uniform(-self.max_velocity, self.max_velocity)
        vel_y = np.random.uniform(-self.max_velocity, self.max_velocity)

        self.agent_vel = np.array([vel_x, vel_y], dtype=np.float32)

        if options is not None:
            if "x" in options:
                self.agent_pos[0] = options["x"]
            if "y" in options:
                self.agent_pos[1] = options["y"]
            if "vel_x" in options:
                self.agent_vel[0] = options["vel_x"]
            if "vel_y" in options:
                self.agent_vel[1] = options["vel_y"]

        self.goal_pos = [self.goal_x, self.goal_y]
        self.episode_step_counter = 0

        info = {}

        return self._get_obs(), info

    def _get_obs(self):
        return np.array([
            self.agent_pos[0],
            self.agent_pos[1],
            self.agent_vel[0],
            self.agent_vel[1],
            self.goal_x,
            self.goal_y
        ])

    def compute_goal_reward(self, agent_x, agent_y):
        return - np.linalg.norm(np.array([agent_x, agent_y]) - np.array([self.goal_x, self.goal_y]))

    def compute_lava_reward(self, agent_x, agent_y):
        if self.lava_x_range[0] <= agent_x <= self.lava_x_range[-1] and self.lava_y_range[0] <= agent_y <= self.lava_y_range[-1]:
            lava_reward = self.lava_cost
        else:
            lava_reward = 0

        return lava_reward

    def step(self, action):
        goal_reward = self.compute_goal_reward(agent_x=self.agent_pos[0], agent_y=self.agent_pos[1])
        lava_reward = self.compute_lava_reward(agent_x=self.agent_pos[0], agent_y=self.agent_pos[1])

        if self.task == "goal":
            rew = goal_reward
            if np.linalg.norm(np.array([self.agent_pos[0], self.agent_pos[1]]) - np.array([self.goal_x, self.goal_y])) <= 0.5:
                print("Goal reached!")

        elif self.task == "lava":
            rew = lava_reward
            if self.lava_x_range[0] <= self.agent_pos[0] <= self.lava_x_range[-1] and self.lava_y_range[0] <= self.agent_pos[1] <= self.lava_y_range[-1]:
                print("in lava!")

        elif self.task == "sum":
            rew = goal_reward + lava_reward
            if np.linalg.norm(np.array([self.agent_pos[0], self.agent_pos[1]]) - np.array([self.goal_x, self.goal_y])) <= 0.1:
                print("Goal reached!")
            if self.lava_x_range[0] <= self.agent_pos[0] <= self.lava_x_range[-1] and self.lava_y_range[0] <= self.agent_pos[1] <= self.lava_y_range[-1]:
                print("in lava!")
        else:
            raise ValueError("Invalid task")

        terminated = False

        # transition to next state
        # if agent steps onto conveyer belt, actions have no consquences and agent moves to the right
        if self.conveyer_x_range[0] <= self.agent_pos[0] <= self.conveyer_x_range[-1] and self.conveyer_y_range[0] <= self.agent_pos[1] <= self.conveyer_y_range[-1]:
            self.agent_vel = np.array([1.0, 0.0], dtype=np.float32)
            acceleration = np.array([0.0, 0.0])  # move right
        else:
            acceleration = self.action_map[action]

        # update velocity
        self.agent_vel += acceleration * self.dt

        # make sure agent doesn't exceed max velocity
        self.agent_vel = np.clip(self.agent_vel, -self.max_velocity, self.max_velocity)

        # update position
        self.agent_pos += self.agent_vel * self.dt

        # make sure agent doesn't leave the grid world
        self.agent_pos[0] = np.clip(self.agent_pos[0], self.x_range[0], self.x_range[1])
        self.agent_pos[1] = np.clip(self.agent_pos[1], self.y_range[0], self.y_range[1])

        obs = self._get_obs()

        self.episode_step_counter += 1
        truncated = self.episode_step_counter >= self.max_episode_steps

        info_dict = {
            "goal_reward": goal_reward,
            "lava_reward": lava_reward,
        }

        return obs, rew, terminated, truncated, info_dict

    def get_render_patches(self, with_conveyer_arrows=False):
        lava_rect_w = self.lava_x_range[-1] - self.lava_x_range[0]
        lava_rect_h = self.lava_y_range[-1] - self.lava_y_range[0]
        lava_rect = plt.Rectangle((self.lava_x_range[0], self.lava_y_range[0]), lava_rect_w, lava_rect_h, color="orange")

        conveyer_rect_w = self.conveyer_x_range[-1] - self.conveyer_x_range[0]
        conveyer_rect_h = self.conveyer_y_range[-1] - self.conveyer_y_range[0]
        conveyer_rect = plt.Rectangle((self.conveyer_x_range[0], self.conveyer_y_range[0]), conveyer_rect_w,
                                      conveyer_rect_h, color="gray")

        patches = [lava_rect, conveyer_rect]
        if with_conveyer_arrows:
            for x in range(self.conveyer_x_range[-1] - 1):
                for y in range(self.conveyer_y_range[-1]):
                    # plt.arrow(self.conveyer_x_range[0] + x + 0.1, self.conveyer_y_range[0] + y + 0.5, 0.5, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
                    arr = plt.arrow(self.conveyer_x_range[0] + x, self.conveyer_y_range[0] + y, 0.5, 0, head_width=0.2, head_length=0.2, fc='k', ec='k', zorder=10)

                    patches.append(arr)

        return patches

    def render(self):
        if self.render_mode == "human":
            plt.grid(zorder=0)
            plt.scatter(self.agent_pos[0], self.agent_pos[1], c="red", s=100, marker="d")

            # indicate velocity with arrow starting from agent position
            plt.arrow(self.agent_pos[0], self.agent_pos[1], self.agent_vel[0], self.agent_vel[1], head_width=0.1, head_length=0.1, fc='blue', ec='blue')

            plt.scatter(self.goal_x, self.goal_y, c="gold", s=100)

            patches = self.get_render_patches()
            for patch in patches:
                plt.gca().add_patch(patch)

            plt.title(f"State: agent_pos: {np.around(self.agent_pos, decimals=3)}, agent_vel: {np.around(self.agent_vel, decimals=3)}")

            plt.xlabel("x")
            plt.xlim(self.x_range[0] - 0.1, self.x_range[1] + 0.1)
            plt.xticks(range(self.x_range[0], self.x_range[1] + 1))

            plt.ylabel("y")
            plt.ylim(self.y_range[0] - 0.1, self.y_range[1] + 0.1)
            plt.yticks(range(self.y_range[0], self.y_range[1] + 1))

            plt.gca().set_aspect("equal", adjustable="box")
            plt.show()

        else:
            pass

    def plot_reward(self):
        xs = np.linspace(self.x_range[0], self.x_range[1], 100)
        ys = np.linspace(self.y_range[0], self.y_range[1], 100)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X)
        for i in range(len(xs)):
            for j in range(len(ys)):
                if self.task == "goal":
                    Z[j, i] = self.compute_goal_reward(agent_x=xs[i], agent_y=ys[j])
                elif self.task == "lava":
                    Z[j, i] = self.compute_lava_reward(agent_x=xs[i], agent_y=ys[j])
                elif self.task == "sum":
                    Z[j, i] = self.compute_goal_reward(agent_x=xs[i], agent_y=ys[j]) + self.compute_lava_reward(agent_x=xs[i], agent_y=ys[j])

        plt.imshow(Z, extent=[self.x_range[0], self.x_range[1], self.y_range[0], self.y_range[1]])

        plt.xlim(self.x_range[0] - 0.1, self.x_range[1] + 0.1)
        plt.xticks(range(self.x_range[0], self.x_range[1] + 1))
        plt.ylim(self.y_range[0] - 0.1, self.y_range[1] + 0.1)
        plt.yticks(range(self.y_range[0], self.y_range[1] + 1))

        # add colorbar, make sure it matches the scale of the plot
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_label("Reward")

        plt.title(f"Reward function, task={self.task}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def plot_random_trajectories(self, n_traj=20):
        traj_fig = plt.figure()
        plt.xlim(self.x_range[0] - 0.1, self.x_range[1] + 0.1)
        plt.xticks(range(self.x_range[0], self.x_range[1] + 1))
        plt.ylim(self.y_range[0] - 0.1, self.y_range[1] + 0.1)
        plt.yticks(range(self.y_range[0], self.y_range[1] + 1))
        plt.gca().set_aspect("equal", adjustable="box")
        patches = self.get_render_patches()
        for patch in patches:
            plt.gca().add_patch(patch)

        for e in range(n_traj):
            # obs, _ = self.reset(options={"x": 1.5, "y": 1.5, "vel_x": 2.0, "vel_y": 0.0})
            obs, _ = self.reset()
            done, trunc = False, False
            pos_hist = [obs[0:2]]
            reward_hist = []
            step_counter = 0
            while not (done or trunc):
                print(obs)
                action = self.action_space.sample()
                obs, reward, done, trunc, info = self.step(action)
                # self.render()
                pos_hist.append(obs[0:2])
                reward_hist += [reward]
                step_counter += 1
                print(f"obs: {obs}, action: {action}, reward: {reward}, done: {done}, info: {info}")
                print("--------------------------------")

            # compute discounted return
            discount_factor = 0.99
            discounted_return = 0
            for t in range(len(reward_hist)):
                discounted_return += discount_factor**t * reward_hist[t]

            print(f"Episode {e} finished after {len(pos_hist)} steps with total reward {np.sum(reward_hist)}, dicounted return {discounted_return}.")

            pos_hist = np.array(pos_hist)
            x = pos_hist[:, 0]
            y = pos_hist[:, 1]

            # Calculate the differences (dx, dy)
            dx = np.diff(x)
            dy = np.diff(y)
            traj_fig.gca().quiver(x[:-1], y[:-1], dx, dy, scale_units='xy', angles='xy', scale=1, color="green" if done else "red", alpha=0.25)

        traj_fig.gca().set_title(f"Random trajectories, task={self.task}")
        traj_fig.gca().set_xlabel("x")
        traj_fig.gca().set_ylabel("y")
        plt.show()

    def close(self):
        pass


if __name__ == "__main__":
    env = LavaGoalConveyerAccelerationEnv(task="sum", render_mode="human")
    env.plot_reward()
    env.plot_random_trajectories(n_traj=50)
    env.close()

