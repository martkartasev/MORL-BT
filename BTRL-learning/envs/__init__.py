import gymnasium as gym

from .lava_goal_conveyer_acceleration import LavaGoalConveyerAccelerationEnv

gym.envs.register(
    id="LavaGoalConveyerAcceleration-lava-v0",
    entry_point="envs:LavaGoalConveyerAccelerationEnv",
    max_episode_steps=200,
    reward_threshold=0,
    kwargs={"task": "lava"},
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
