#!/usr/bin/env bash

# run five times in total
start=2;
end=5;
env_interactions=3000000
learning_start=200000
feasibility_epochs=200

for i in $(seq $start $end);
do

    # 1) train unsafe area DQN
    echo "Training lava DQN with seed $i";
    lava_dqn_dir=$(python3 train_dqn.py \
    --env_id "SimpleAccEnv-wide-withConveyer-lava-v0" \
    --total_steps "$env_interactions" \
    --learning_starts "$learning_start" \
    --seed "$i" \
    --exp_name "debug_seed:$i" \
    --exp_base_dir "final_experiments" \
    | tee /dev/tty | tail -n 1)  # tee needed to make prints visible in console for some reason
    echo "Lava DQN training finished, saved at: $lava_dqn_dir";

    # 2) train unsafe area feasiblity estimator on collected data
    echo "====================================================";
    echo "Training lava feasibility estimator based on replay buffer in $lava_dqn_dir";
    lava_feasibility_dir=$(python3 train_feasibility.py \
    --rb_dirs "$lava_dqn_dir" \
    --exp_str "lava" \
    --feasibility_label "lava" \
    --epochs "$feasibility_epochs" \
    | tee /dev/tty | tail -n 1)
    echo "Lava feasibility estimator training finished, saved_at: $lava_feasibility_dir";

    # 3) train battery charge DQN using already trained lava DQN and lava feasibility estimator
    echo "====================================================";
    echo "Training battery charge DQN with seed $i and pretrained lava dqn and feasibility estimator";
    battery_dqn_dir=$(python3 train_dqn.py \
    --env_id "SimpleAccEnv-wide-withConveyer-battery-v0" \
    --total_steps "$env_interactions" \
    --learning_starts "$learning_start" \
    --seed "$i" \
    --exp_name "debug_seed:$i" \
    --exp_base_dir "final_experiments" \
    --lava_dqn_path "$lava_dqn_dir/avoid_lava_q_net_0.pth" \
    --lava_constraint_feasibility_path "$lava_feasibility_dir/feasibility_dqn.pt" \
    | tee /dev/tty | tail -n 1)
    echo "Battery charge DQN training finished, saved at: $battery_dqn_dir";

    # 4) train battery charge feasibility estimator on collected data
    # 4.1 ) naive battery feasibility estimator training
    echo "====================================================";
    echo "Training battery charge feasibility estimator (naive)";
    naive_battery_feasibility_dir=$(python3 train_feasibility.py \
    --rb_dirs "$battery_dqn_dir" \
    --exp_str "battery_naive" \
    --feasibility_label "battery" \
    --epochs "$feasibility_epochs" \
    | tee /dev/tty | tail -n 1)
    echo "Battert feasibility estimator training (niave) finished, saved_at: $naive_battery_feasibility_dir";

    # 4.2) recursive battery feasibility estimator training
    echo "====================================================";
    echo "Training battery charge feasibility estimator (recursive)";
    recursive_battery_feasibility_dir=$(python3 train_feasibility.py \
    --higher_prio_feasibility_estimator "$lava_feasibility_dir" \
    --rb_dirs "$battery_dqn_dir" \
    --exp_str "battery_recursive" \
    --feasibility_label "battery" \
    --epochs "$feasibility_epochs" \
    | tee /dev/tty | tail -n 1)
    echo "Battert feasibility estimator training (recursive) finished, saved_at: $recursive_battery_feasibility_dir";

    # 4.3) lava or battery feasibility estimator training
    echo "====================================================";
    echo "Training lava OR battery feasibility estimator";
    lava_or_battery_feasibility_dir=$(python3 train_feasibility.py \
    --rb_dirs "$lava_dqn_dir" "$battery_dqn_dir" \
    --exp_str "lava_OR_battery" \
    --feasibility_label "or" \
    --epochs "$feasibility_epochs" \
    | tee /dev/tty | tail -n 1)
    echo "Lava OR battery feasibility estimator training finished, saved_at: $lava_or_battery_feasibility_dir";

    # 5) train goal dqn using all of the previous models
    # 5.1) Train goal DQN without any constraints/feasibility estimators (baseline 1)
    echo "====================================================";
    echo "Training goal DQN without any constraints/feasibility estimators (baseline 1)";
    noCon_goal_dqn_dir=$(python3 train_dqn.py \
    --env_id "SimpleAccEnv-wide-withConveyer-goal-v0" \
    --total_steps "$env_interactions" \
    --learning_starts "$learning_start" \
    --seed "$i" \
    --exp_name "debug_noConstraints_seed:$i" \
    --exp_base_dir "final_experiments" \
    --lava_dqn_path "$lava_dqn_dir/avoid_lava_q_net_0.pth" \
    --battery_dqn_path "$battery_dqn_dir/battery_q_net_0.pth" \
    | tee /dev/tty | tail -n 1)
    echo "Goal DQN training without any constraints/feasibility estimators (baseline 1) finished, saved at: $noCon_goal_dqn_dir";

    # 5.2) Train goal DQN with with ACC reward penalty (baseline 2)
    echo "====================================================";
    echo "Training goal DQN with ACC reward penalty (baseline 2)";
    penalty_goal_dqn_dir=$(python3 train_dqn.py \
    --env_id "SimpleAccEnv-wide-withConveyer-goal-v0" \
    --total_steps "$env_interactions" \
    --learning_starts "$learning_start" \
    --seed "$i" \
    --exp_name "debug_rewardPenalty_seed:$i" \
    --exp_base_dir "final_experiments" \
    --lava_dqn_path "$lava_dqn_dir/avoid_lava_q_net_0.pth" \
    --battery_dqn_path "$battery_dqn_dir/battery_q_net_0.pth" \
    --punishACC \
    | tee /dev/tty | tail -n 1)
    echo "Goal DQN training with ACC reward penalty (baseline 2) finished, saved at: $penalty_goal_dqn_dir";

    # 5.3) Train goal DQN with feasibility estimators and feasibiilty aware BT
    echo "====================================================";
    echo "Training goal DQN with feasibility estimators and feasibiilty aware BT (ours)";
    cbtrl_goal_dqn_dir=$(python3 train_dqn.py \
    --env_id "SimpleAccEnv-wide-withConveyer-goal-v0" \
    --total_steps "$env_interactions" \
    --learning_starts "$learning_start" \
    --seed "$i" \
    --exp_name "debug_feasibilityAwareBT_seed:$i" \
    --exp_base_dir "final_experiments" \
    --lava_dqn_path "$lava_dqn_dir/avoid_lava_q_net_0.pth" \
    --lava_constraint_feasibility_path "$lava_feasibility_dir/feasibility_dqn.pt" \
    --battery_dqn_path "$battery_dqn_dir/battery_q_net_0.pth" \
    --battery_constraint_feasibility_path "$lava_or_battery_feasibility_dir/feasibility_dqn.pt" \
    --feasibility_aware_bt \
    | tee /dev/tty | tail -n 1)
    echo "Goal DQN training with feasibility estimators and feasibiilty aware BT (ours) finished, saved at: $cbtrl_goal_dqn_dir";
done