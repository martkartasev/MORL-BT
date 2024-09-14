import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import yaml


def action_to_acc(a):
    row = a // 5
    col = a % 5
    return np.array([row - 2, col - 2])


def random_point_on_rectangle_outline(x_min, y_min, x_max, y_max):
    # Calculate the lengths of the sides
    width = x_max - x_min
    height = y_max - y_min

    # Perimeter of the rectangle
    perimeter = 2 * (width + height)

    # Generate a random value between 0 and the perimeter
    rand_value = np.random.uniform(0, perimeter)

    # Determine the side on which the random point lies
    if rand_value <= width:  # Top side
        x = x_min + rand_value
        y = y_min
    elif rand_value <= width + height:  # Right side
        x = x_max
        y = y_min + (rand_value - width)
    elif rand_value <= 2 * width + height:  # Bottom side
        x = x_max - (rand_value - width - height)
        y = y_max
    else:  # Left side
        x = x_min
        y = y_max - (rand_value - 2 * width - height)

    return (x, y)


class SimpleAccEnv(gym.Env):

    def __init__(
            self,
            x_min=0,
            x_max=10,
            y_min=0,
            y_max=10,
            max_velocity=1.5,
            lava_max_velocity=1.5,
            dt=0.2,
            max_ep_len=200,
            lava_x_min=2,
            lava_x_max=8,
            lava_y_min=3,
            lava_y_max=7,
            task="lava",
            with_conveyer=False,
            conveyer_x_min=2,
            conveyer_x_max=6,
            conveyer_y_min=3,
            conveyer_y_max=7,
            goal_x=5,
            goal_y=9,
            task_sum_weight=0.5,
            battery_x=15,
            battery_y=2,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.x_range = [x_min, x_max]
        self.y_range = [y_min, y_max]
        self.max_velocity = max_velocity
        self.lava_max_velocity = lava_max_velocity
        self.dt = dt
        self.max_ep_len = max_ep_len
        self.with_conveyer = with_conveyer
        self.conveyer_x_min = conveyer_x_min
        self.conveyer_x_max = conveyer_x_max
        self.conveyer_y_min = conveyer_y_min
        self.conveyer_y_max = conveyer_y_max
        if self.with_conveyer:
            self.lava_x_min = self.conveyer_x_max
            self.lava_x_max = lava_x_max
            self.lava_y_min = lava_y_min
            self.lava_y_max = lava_y_max
        else:
            self.lava_x_min = lava_x_min
            self.lava_x_max = lava_x_max
            self.lava_y_min = lava_y_min
            self.lava_y_max = lava_y_max
        self.task = task
        self.task_sum_weight = task_sum_weight
        assert task in ["lava", "goal", "lava_goal_sum", "left", "battery"]
        assert 0 <= self.task_sum_weight <= 1

        self.goal_x = goal_x
        self.goal_y = goal_y
        self.battery_x = battery_x
        self.battery_y = battery_y

        self.eval_states = [
            np.array([5.0, 5.0, 0.0, 0.0, 0.1]),  # in lava
            np.array([5.0, 2.5, 0.0, 0.0, 0.1]),  # beneath lava, no velocity
            np.array([5.0, 2.5, 0.0, 2.0, 0.1]),  # beneath lava but upwards velocity, lava unavailable
            np.array([5.0, 7.5, 0.0, 0.0, 0.1]),  # above lava, no velocity
            np.array([5.0, 7.5, 0.0, -2.0, 0.1]),  # above lava but downwards velocity, lava unavailable
            np.array([1.5, 5.0, 0.0, 0.0, 0.1]),  # left of lava, no velocity
            np.array([1.5, 5.0, 2.0, 0.0, 0.1]),  # left of lava but rightwards velocity, lava unavailable
            np.array([8.5, 5.0, 0.0, 0.0, 0.1]),  # right of lava, no velocity
            np.array([8.5, 5.0, -2.0, 0.0, 0.1]),  # right of lava but leftwards velocity, lava unavailable
            np.array([9.75, 1, 0.0, 0.0, 0.1]),  # underneath lava, at x middle, no velocity
            np.array([9.75, 1, 2.0, 0.0, 0.1]),  # underneath lava, at x middle, velocity towards the right
            np.array([18.05, 4, 0.0, 0.0, 0.1]),  # right of lava, at y slightly lower than middle, no velocity
            np.array([18.05, 6, 0.0, 0.0, 0.1]),  # right of lava, at y slightly higher than, no velocity
            np.array([18.05, 1, 0.0, 0.0, 0.1]),  # right of lava, low at bottom, no velocity
            np.array([18.05, 4, 0.0, 0.0, 0.5]),  # right of lava, at y slightly lower than middle, no velocity
            np.array([18.05, 6, 0.0, 0.0, 0.5]),  # right of lava, at y slightly higher than, no velocity
            np.array([18.05, 1, 0.0, 0.0, 0.5]),  # right of lava, low at bottom, no velocity
            np.array([18.05, 4, 0.0, 0.0, 0.1]),  # right of lava, at y slightly lower than middle, no velocity
            np.array([18.05, 6, 0.0, 0.0, 0.1]),  # right of lava, at y slightly higher than, no velocity
            np.array([18.05, 1, 0.0, 0.0, 0.1]),  # right of lava, low at bottom, no velocity
            np.array([15, 7.05, 0.0, 0.0, 0.1]),  # above lava
            np.array([15, 7.05, 0.0, 0.0, 0.15]),  # above lava
            np.array([15, 7.05, 0.0, 0.0, 0.2]),  # above lava
            np.array([15, 7.05, 0.0, 0.0, 0.25]),  # above lava
            np.array([15, 7.05, 0.0, 0.0, 0.3]),  # above lava
            np.array([15, 7.05, 0.0, 0.0, 0.35]),  # above lava
            np.array([15, 7.05, 0.0, 0.0, 0.4]),  # above lava
            np.array([15, 7.05, 0.0, 0.0, 0.5]),  # above lava
            np.array([15, 7.05, 0.0, 0.0, 1.0]),  # above lava
            np.array([self.lava_x_max + 0.05, 5, 0, 0, 1.0]),  # can step into of lava
        ]

        self.action_space = gym.spaces.Discrete(25)
        self.observation_space = gym.spaces.Box(
            low=np.array([
                self.x_min,
                self.y_min,
                -self.max_velocity,
                -self.max_velocity,
                0
            ]),
            high=np.array([
                self.x_max,
                self.y_max,
                self.max_velocity,
                self.max_velocity,
                1
            ])
        )

        self.state_predicate_names = ["in_unsafe", "at_goal", "on_conveyer", "battery_empty", "at_battery"]

        # episode variables, need to be reset
        self.x = None
        self.y = None
        self.vel_x = None
        self.vel_y = None
        self.battery_charge = None
        self.ep_len = 0

    def _get_obs(self):
        return np.array([self.x, self.y, self.vel_x, self.vel_y, self.battery_charge])

    def _in_lava(self):
        return self.lava_x_min <= self.x <= self.lava_x_max and self.lava_y_min <= self.y <= self.lava_y_max

    def _on_conveyer(self):
        return self.conveyer_x_min <= self.x <= self.conveyer_x_max and self.conveyer_y_min <= self.y <= self.conveyer_y_max

    def _at_goal(self):
        return np.linalg.norm([self.goal_x - self.x, self.goal_y - self.y]) < 0.5

    def _battery_empty(self):
        return self.battery_charge <= 0

    def _at_batterty(self):
        return np.linalg.norm([self.battery_x - self.x, self.battery_y - self.y]) < 0.5

    def check_state_predicates(self):
        predicates = [self._in_lava(), self._at_goal(), self._on_conveyer(), self._battery_empty(), self._at_batterty()]
        assert len(predicates) == len(self.state_predicate_names)
        return predicates

    def reset(self, seed=None, options={}):

        # self.x = np.random.uniform(self.x_min, self.x_max)
        # self.y = np.random.uniform(self.y_min, self.y_max)
        self.vel_x = np.random.uniform(-self.max_velocity, self.max_velocity)
        self.vel_y = np.random.uniform(-self.max_velocity, self.max_velocity)

        # seems I need this or train on much more data to learn good feasibility estimator for all velocities and poses
        border_dist = np.random.choice([0, 0.05, 1])
        p = random_point_on_rectangle_outline(
            x_min=self.conveyer_x_min-border_dist,
            y_min=self.conveyer_y_min-border_dist,
            x_max=self.lava_x_max+border_dist,
            y_max=self.lava_y_max+border_dist
        )
        self.x = p[0]
        self.y = p[1]

        self.battery_charge = np.random.uniform(0, 1)

        self.ep_len = 0

        if options is not None:
            if "x" in options:
                self.x = options["x"]
            if "y" in options:
                self.y = options["y"]
            if "vel_x" in options:
                self.vel_x = options["vel_x"]
            if "vel_y" in options:
                self.vel_y = options["vel_y"]
            if "battery" in options:
                self.battery_charge = options["battery"]

        return self._get_obs(), {}

    def step(self, action):

        agent_in_lava = self._in_lava()
        agent_at_goal = self._at_goal()
        agent_on_conveyer = self._on_conveyer()
        battery_empty = self._battery_empty()
        agent_at_battery = self._at_batterty()

        lava_reward = -1 if agent_in_lava else 0
        goal_rewad = -1 - 0.1 * np.linalg.norm([self.goal_x - self.x, self.goal_y - self.y])
        left_reward = 0 if self.x < (self.x_max * (2/3)) else -1
        # battery_reward = -1 if battery_empty else 0
        battery_reward = -0.1 * np.linalg.norm([self.battery_x - self.x, self.battery_y - self.y]) if battery_empty else 0

        if self.task == "lava":
            reward = lava_reward
        elif self.task == "goal":
            reward = goal_rewad
        elif self.task == "lava_goal_sum":
            reward = self.task_sum_weight * lava_reward + (1 - self.task_sum_weight) * goal_rewad
        elif self.task == "left":
            reward = left_reward
        elif self.task == "battery":
            reward = battery_reward
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
        if agent_in_lava:
            # agent is slower in lava
            self.vel_x = np.clip(self.vel_x, -self.lava_max_velocity, self.lava_max_velocity)
            self.vel_y = np.clip(self.vel_y, -self.lava_max_velocity, self.lava_max_velocity)
        else:
            self.vel_x = np.clip(self.vel_x, -self.max_velocity, self.max_velocity)
            self.vel_y = np.clip(self.vel_y, -self.max_velocity, self.max_velocity)

        # update position
        self.x += self.vel_x * self.dt
        self.y += self.vel_y * self.dt

        # clamp position
        self.x = np.clip(self.x, self.x_min, self.x_max)
        self.y = np.clip(self.y, self.y_min, self.y_max)

        # update battery
        self.battery_charge -= 0.01
        self.battery_charge = np.clip(self.battery_charge, 0, 1)

        if agent_at_battery:
            self.battery_charge = 1.0

        new_obs = self._get_obs()
        done = False
        if self.task == "goal":
            if agent_at_goal:
                done = True
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
        for _ in range(env.max_ep_len):
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
    env = SimpleAccEnv(
        with_conveyer=True,
        x_max=20,
        conveyer_x_min=2,
        conveyer_x_max=10,
        lava_x_min=10,
        lava_x_max=18,
        goal_x=10,
        max_velocity=1.5,
        lava_max_velocity=1.5,
        task="lava"
    )

    # plot accelerations
    plot_action_acceleration()

    for _ in range(1000):
        obs, _ = env.reset()
        plt.scatter(obs[0], obs[1])
    plt.show()

    # plot reward function
    xs = np.linspace(env.x_min, env.x_max, 100)
    ys = np.linspace(env.y_min, env.y_max, 100)
    Z = np.zeros([100, 100])
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            Z[i, j] = -1 * np.linalg.norm([env.goal_x - x, env.goal_y - y])

    plt.imshow(Z.T, extent=[env.x_min, env.x_max, env.y_min, env.y_max], origin='lower', cmap='viridis')
    plt.colorbar()
    plt.show()
    plt.close()

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

