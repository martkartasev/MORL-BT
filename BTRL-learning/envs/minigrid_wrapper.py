from typing import SupportsFloat, Any

import gymnasium
import numpy as np
from gymnasium.core import ActType, WrapperObsType
from minigrid.core import constants


class FlattenedMinigrid(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.placeholders = env.observation_space["mission"].ordered_placeholders
        self.nr_placeholders = len(self.placeholders) if self.placeholders is not None else 0
        self.observation_space = gymnasium.spaces.Box(
            shape=(np.prod(env.observation_space["image"].shape) + 1 + self.nr_placeholders,), low=0, high=1)
        self.state_predicate_names = []
        self.eval_states = []

    def step(self, action: ActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, self.info(info)

    def observation(self, obs):
        mission_str = obs["mission"]
        obs = np.append(obs["image"].flatten(), obs["direction"])
        if self.nr_placeholders > 0:
            if is_put_near(mission_str):
                split = mission_str.split(" ")
                obs = np.append(obs, (
                constants.COLOR_TO_IDX[split[2]], constants.OBJECT_TO_IDX[split[3]], constants.COLOR_TO_IDX[split[6]],
                constants.OBJECT_TO_IDX[split[7]]))
            if is_locked_room(mission_str):
                obs = np.append(obs, )
        obs = obs / 11
        return obs

    def info(self, info):
        info["state_predicates"] = []
        return info


def is_put_near(mission_str):
    return mission_str.startswith("put the")


def is_locked_room(mission_str):
    return mission_str.startswith("get the")
