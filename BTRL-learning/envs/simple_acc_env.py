import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import yaml


def action_to_acc(a):
    row = a // 5
    col = a % 5
    return np.array([row - 2, col - 2])


class SimpleAccEnv(gym.Env):

    def __init__(
            self,
            x_min=0,
            x_max=10,
            y_min=0,
            y_max=10,
            max_velocity=2,
            dt=0.2,
            max_ep_len=50,
            lava_x_min=2,
            lava_x_max=8,
            lava_y_min=3,
            lava_y_max=7,
            task="lava",
            with_conveyer=False,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.x_range = [x_min, x_max]
        self.y_range = [y_min, y_max]
        self.max_velocity = max_velocity
        self.dt = dt
        self.max_ep_len = max_ep_len
        self.with_conveyer = with_conveyer
        self.conveyer_x_min = 2
        self.conveyer_x_max = 6
        self.conveyer_y_min = 3
        self.conveyer_y_max = 7
        if self.with_conveyer:
            self.lava_x_min = 6
            self.lava_x_max = lava_x_max
            self.lava_y_min = lava_y_min
            self.lava_y_max = lava_y_max
        else:
            self.lava_x_min = lava_x_min
            self.lava_x_max = lava_x_max
            self.lava_y_min = lava_y_min
            self.lava_y_max = lava_y_max
        self.task = task

        assert task in ["lava", "goal"]

        self.eval_states = [
            np.array([5.0, 5.0, 0.0, 0.0]),  # in lava
            np.array([5.0, 2.5, 0.0, 0.0]),  # beneath lava, no velocity
            np.array([5.0, 2.5, 0.0, 2.0]),  # beneath lava but upwards velocity, lava unavailable
            np.array([5.0, 7.5, 0.0, 0.0]),  # above lava, no velocity
            np.array([5.0, 7.5, 0.0, -2.0]),  # above lava but downwards velocity, lava unavailable
            np.array([1.5, 5.0, 0.0, 0.0]),  # left of lava, no velocity
            np.array([1.5, 5.0, 2.0, 0.0]),  # left of lava but rightwards velocity, lava unavailable
            np.array([8.5, 5.0, 0.0, 0.0]),  # right of lava, no velocity
            np.array([8.5, 5.0, -2.0, 0.0]),  # right of lava but leftwards velocity, lava unavailable
        ]

        self.action_space = gym.spaces.Discrete(25)
        self.observation_space = gym.spaces.Box(
            low=np.array([
                self.x_min,
                self.y_min,
                -self.max_velocity,
                -self.max_velocity,
            ]),
            high=np.array([
                self.x_max,
                self.y_max,
                self.max_velocity,
                self.max_velocity,
            ])
        )

        self.state_predicate_names = ["in_lava", "at_goal", "on_conveyer"]

        # episode variables, need to be reset
        self.x = None
        self.y = None
        self.vel_x = None
        self.vel_y = None
        self.ep_len = 0

    def _get_obs(self):
        return np.array([self.x, self.y, self.vel_x, self.vel_y])

    def _in_lava(self):
        return self.lava_x_min <= self.x <= self.lava_x_max and self.lava_y_min <= self.y <= self.lava_y_max

    def _on_conveyer(self):
        return self.conveyer_x_min <= self.x <= self.conveyer_x_max and self.conveyer_y_min <= self.y <= self.conveyer_y_max

    def _at_goal(self):
        return np.linalg.norm([5 - self.x, 9 - self.y]) < 0.5

    def check_state_predicates(self):
        predicates = [self._in_lava(), self._at_goal(), self._on_conveyer()]
        assert len(predicates) == len(self.state_predicate_names)
        return predicates

    def reset(self, seed=None, options={}):

        self.x = np.random.uniform(self.x_min, self.x_max)
        self.y = np.random.uniform(self.y_min, self.y_max)
        self.vel_x = np.random.uniform(-self.max_velocity, self.max_velocity)
        self.vel_y = np.random.uniform(-self.max_velocity, self.max_velocity)

        # ---
        # when training without a BT, the feasibility constrained goal-reach DQN must not be initialized in the
        # infeasible region, i.e. in lava or on conveyer, since it would learn to go to those states, which are closer
        # to the goal...
        # ---
        # in_lava = True
        # on_conveyer = True
        # while (in_lava or on_conveyer):
        #     self.x = np.random.uniform(self.x_min, self.x_max)
        #     self.y = np.random.uniform(self.y_min, self.y_max)
        #     self.vel_x = np.random.uniform(-self.max_velocity, self.max_velocity)
        #     self.vel_y = np.random.uniform(-self.max_velocity, self.max_velocity)
        #     in_lava = self._in_lava()
        #     on_conveyer = self._on_conveyer()

        self.ep_len = 0

        if "x" in options:
            self.x = options["x"]
        if "y" in options:
            self.y = options["y"]
        if "vel_x" in options:
            self.vel_x = options["vel_x"]
        if "vel_y" in options:
            self.vel_y = options["vel_y"]

        return self._get_obs(), options

    def step(self, action):

        agent_in_lava = self._in_lava()
        agent_at_goal = self._at_goal()
        agent_on_conveyer = self._on_conveyer()

        if self.task == "lava":
            reward = -10 if agent_in_lava else 0
        elif self.task == "goal":
            reward = -1 * np.linalg.norm([5 - self.x, 9 - self.y])
        else:
            raise NotImplementedError(f"Task {self.task} not imlpemented")

        acc = action_to_acc(action)

        # update velocity
        self.vel_x += acc[0] * self.dt
        self.vel_y += acc[1] * self.dt

        if agent_on_conveyer and self.with_conveyer:
            self.vel_x = 2.0
            self.vel_y = 0.0

        # clamp velocity
        self.vel_x = np.clip(self.vel_x, -self.max_velocity, self.max_velocity)
        self.vel_y = np.clip(self.vel_y, -self.max_velocity, self.max_velocity)

        # update position
        self.x += self.vel_x * self.dt
        self.y += self.vel_y * self.dt

        # clamp position
        self.x = np.clip(self.x, self.x_min, self.x_max)
        self.y = np.clip(self.y, self.y_min, self.y_max)

        new_obs = self._get_obs()
        done = False
        trunc = self.ep_len > self.max_ep_len
        info = {}
        info["state_predicates"] = self.check_state_predicates()
        info["ep_len"] = self.ep_len
        info["state_predicate_names"] = self.state_predicate_names

        self.ep_len += 1

        return new_obs, reward, done, trunc, info


def plot_trajectories(env, n=3, fixed_action=None, reset_options={}):
    # plot lava rectangle
    lava_rect = plt.Rectangle(
        (env.lava_x_min, env.lava_y_min),
        env.lava_x_max - env.lava_x_min,
        env.lava_y_max - env.lava_y_min,
        fill=True,
        color='orange',
        alpha=0.5
    )
    plt.gca().add_patch(lava_rect)

    if env.with_conveyer:
        conveyer_rect = plt.Rectangle(
            (env.conveyer_x_min, env.conveyer_y_min),
            env.conveyer_x_max - env.conveyer_x_min,
            env.conveyer_y_max - env.conveyer_y_min,
            fill=True,
            color='gray',
            alpha=0.5
        )
        plt.gca().add_patch(conveyer_rect)

    for _ in range(n):
        obs, _ = env.reset(options=reset_options)
        trajectory = [obs[:2]]
        ep_reward = 0
        for _ in range(20):
            if fixed_action is None:
                action = np.random.randint(25)
            else:
                action = fixed_action

            obs, reward, done, trunc, _ = env.step(action)
            ep_reward += reward

            trajectory.append(obs[:2])
            if done:
                break

        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'o-', c="r" if ep_reward < 0 else "b")

    plt.title(f"Trajectories with action {fixed_action}")
    plt.xlim(env.x_min - 0.1, env.x_max + 0.1)
    plt.ylim(env.y_min - 0.1, env.y_max + 0.1)
    plt.show()


def plot_action_acceleration():
    for a in np.arange(0, 25):
        acc = action_to_acc(a)
        text = f"{a}, {acc}"
        plt.scatter(acc[0], acc[1], s=800)
        plt.text(acc[0], acc[1], text, fontsize=12, ha='center', va='center')

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.show()


if __name__ == "__main__":
    env = SimpleAccEnv(with_conveyer=False)

    # plot accelerations
    # plot_action_acceleration()

    # run a few episodes and plot resulting trajectories
    plot_trajectories(
        env,
        n=10,
        # fixed_action=18,
        # fixed_action=10,
        fixed_action=None,
        # reset_options={},
        # reset_options={
        #     "x": 5,
        #     "y": 2.0,
        #     "vel_x": 0,
        #     "vel_y": 2
        # }
    )

