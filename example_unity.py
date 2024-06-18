import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


def produce_action(agent_obs):
    normalized_position = agent_obs[0:3]
    normalized_velocity = agent_obs[3:6]
    trigger_relative_position = agent_obs[6:9]
    button_relative_position = agent_obs[9:12]
    goal_relative_position = agent_obs[12:15]
    bridge1_relative_position = agent_obs[15:18]  # Doesn't matter atm, as bridges don't move, maybe in the future
    bridge2_relative_position = agent_obs[18:21]

    # TODO: Whatever magic to decide action

    agent_action = 23  # 0-24 , 25 actions
    reset_agent = 0  # 0 keeps going, 1 resets given agent
    screenshot = 1
    return [agent_action, reset_agent, screenshot]


def run_env(example_env):
    action_spec = example_env.behavior_specs.get("BridgeEnv?team=0").action_spec
    print(action_spec.discrete_branches)
    observation_spec = example_env.behavior_specs.get("BridgeEnv?team=0").observation_specs[0]
    print(observation_spec.shape)

    for i in range(10000):
        (decision_steps, terminal_steps) = example_env.get_steps("BridgeEnv?team=0")
        # Decision steps -> agent info that needs an action this step
        # Terminal steps -> agent info whose episode has ended
        nr_agents = len(decision_steps.agent_id)
        print(nr_agents)
        print(decision_steps.agent_id)
        action_tuple = ActionTuple()
        if nr_agents > 0:
            observations = decision_steps.obs[0]  # Strange structure, but this is how you get the observations array
            actions = np.array([produce_action(observations[i][:]) for i in range(nr_agents)])
            action_tuple.add_discrete(actions)
        else:
            action_tuple.add_discrete(np.zeros((0, 3)))

        example_env.set_actions("BridgeEnv?team=0", action_tuple)
        example_env.step()


if __name__ == '__main__':
    engine = EngineConfigurationChannel()
    engine.set_configuration_parameters(time_scale=2)  # Can speed up simulation between steps with this
    engine.set_configuration_parameters(quality_level=0)

    env = UnityEnvironment(
        file_name="../experiments-ml-agents/builds/MORL/MORL-BT.exe",
        no_graphics=True,  # Can disable graphics if needed
        # base_port= 10000 # for starting multiple envs
        side_channels=[engine])
    env.reset()  # Initializes env
    run_env(env)
    env.reset()  # Resets everything in the executable (all agents)
