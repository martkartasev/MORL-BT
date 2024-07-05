import numpy as np


def rewards_flat_acc_env(agent_obs, task="reach_goal"):
    reward_scale = 0.1  # since we have unnormalized obs here distances are quite large...

    if task == "reach_goal":
        goal_pos = agent_obs[3:6]
        return -reward_scale * np.linalg.norm(goal_pos)  # negative goal distance

    elif task == "fetch_trigger":
        trigger_pos = agent_obs[9:12]
        return -reward_scale * np.linalg.norm(trigger_pos)  # negative trigger distance

    elif task == "place_trigger":
        trigger_pos = agent_obs[9:12]
        button_pos = agent_obs[12:15]
        agent_has_trigger = agent_obs[15]

        if not agent_has_trigger:
            print("Reward function for placing trigger assumes agent has trigger. Ensure optimizing this reward is correct!")

        return -reward_scale * np.linalg.norm(trigger_pos - button_pos)  # negative distance between trigger and button
    else:
        raise ValueError(f"Unknown task: {task}")


def done_check_flat_acc_env(agent_obs):
    return False
