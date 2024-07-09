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
