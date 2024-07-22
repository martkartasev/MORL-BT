import gymnasium
import numpy as np


class FlattenedMinigrid(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(shape=(np.prod(env.observation_space["image"].shape) + 1,), low=0, high=10)

    def observation(self, obs):
        return np.append(obs["image"].flatten(), obs["direction"])
