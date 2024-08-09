import gymnasium as gym

from .lava_goal_conveyer_acceleration import LavaGoalConveyerAccelerationEnv
from .simple_acc_env import SimpleAccEnv

gym.envs.register(
    id="LavaGoalConveyerAcceleration-lava-v0",
    entry_point="envs:LavaGoalConveyerAccelerationEnv",
    max_episode_steps=200,
    reward_threshold=0,
    kwargs={"task": "lava"},
)

gym.envs.register(
    id="LavaGoalConveyerAcceleration-lava-noConveyer-v0",
    entry_point="envs:LavaGoalConveyerAccelerationEnv",
    max_episode_steps=200,
    reward_threshold=0,
    kwargs={"task": "lava", "with_conveyer": False},
)

gym.envs.register(
    id="LavaGoalConveyerAcceleration-goal-v0",
    entry_point="envs:LavaGoalConveyerAccelerationEnv",
    max_episode_steps=200,
    reward_threshold=0,
    kwargs={"task": "goal"},
)

gym.envs.register(
    id="LavaGoalConveyerAcceleration-sum-v0",
    entry_point="envs:LavaGoalConveyerAccelerationEnv",
    max_episode_steps=200,
    reward_threshold=0,
    kwargs={"task": "lava"},
)

gym.envs.register(
    id="SimpleAccEnv-lava-v0",
    entry_point="envs:SimpleAccEnv",
    max_episode_steps=20,
    reward_threshold=0,
    kwargs={"task": "lava"},
)

gym.envs.register(
    id="SimpleAccEnv-goal-v0",
    entry_point="envs:SimpleAccEnv",
    max_episode_steps=20,
    reward_threshold=0,
    kwargs={"task": "goal"},
)

gym.envs.register(
    id="SimpleAccEnv-withConveyer-lava-v0",
    entry_point="envs:SimpleAccEnv",
    max_episode_steps=50,
    reward_threshold=0,
    kwargs={"task": "lava", "with_conveyer": True},
)

gym.envs.register(
    id="SimpleAccEnv-withConveyer-goal-v0",
    entry_point="envs:SimpleAccEnv",
    max_episode_steps=50,
    reward_threshold=0,
    kwargs={"task": "goal", "with_conveyer": True},
)

gym.envs.register(
    id="SimpleAccEnv-wide-withConveyer-lava-v0",
    entry_point="envs:SimpleAccEnv",
    max_episode_steps=200,
    reward_threshold=0,
    kwargs={
        "task": "lava",
        "with_conveyer": True,
        "x_max": 20,
        "conveyer_x_min": 2,
        "conveyer_x_max": 10,
        "lava_x_min":  10,
        "lava_x_max":  18,
        "goal_x": 10,
        "max_ep_len": 200,
    },
)

gym.envs.register(
    id="SimpleAccEnv-wide-withConveyer-goal-v0",
    entry_point="envs:SimpleAccEnv",
    max_episode_steps=200,
    reward_threshold=0,
    kwargs={
        "task": "goal",
        "with_conveyer": True,
        "x_max": 20,
        "conveyer_x_min": 2,
        "conveyer_x_max": 10,
        "lava_x_min":  10,
        "lava_x_max":  18,
        "goal_x": 10,
        "max_ep_len": 200,
    },
)

gym.envs.register(
    id="SimpleAccEnv-wide-withConveyer-left-v0",
    entry_point="envs:SimpleAccEnv",
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        "task": "left",
        "with_conveyer": True,
        "x_max": 20,
        "conveyer_x_min": 2,
        "conveyer_x_max": 10,
        "lava_x_min":  10,
        "lava_x_max":  18,
        "goal_x": 10,
        "max_ep_len": 150,
    },
)

gym.envs.register(
    id="SimpleAccEnv-wide-withConveyer-sum-v0",
    entry_point="envs:SimpleAccEnv",
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        "task": "lava_goal_sum",
        "with_conveyer": True,
        "x_max": 20,
        "conveyer_x_min": 2,
        "conveyer_x_max": 10,
        "lava_x_min":  10,
        "lava_x_max":  18,
        "goal_x": 10,
        "max_ep_len": 150,
        # "task_sum_weight": 0.999  second to last
        "task_sum_weight": 0.5  # last, with punish...
    },
)

gym.envs.register(
    id="SimpleAccEnv-wide-withConveyer-battery-v0",
    entry_point="envs:SimpleAccEnv",
    max_episode_steps=200,
    reward_threshold=0,
    kwargs={
        "task": "battery",
        "with_conveyer": True,
        "x_max": 20,
        "conveyer_x_min": 2,
        "conveyer_x_max": 10,
        "lava_x_min":  10,
        "lava_x_max":  18,
        "goal_x": 10,
        "max_ep_len": 200,
    },
)
