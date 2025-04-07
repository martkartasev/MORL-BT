#!/usr/bin/env bash

# run five times in total
start=1;
end=5
lava_dqn="runs/SimpleAccEnv-wide-withConveyer-lava-v1/2025-03-05-08-48-54_lava/avoid_lava_q_net_0.pth";
lava_feasibility="runs/SimpleAccEnv-wide-withConveyer-lava-v1/2025-03-05-14-27-47_256x256/feasibility_2025-03-05-15-07-42_batch8k_noBatchNorm_gamma:0999_500epochs_weightDecay:1e-5/feasibility_dqn.pt";

for i in $(seq $start $end);
do
    # CBTRL
    echo "====================================================";
    echo "Training CBTRL with seed $i";
    cbtrl_dir=$(python3 train_dqn_newEnvNoBattery.py \
    --env_id "SimpleAccEnv-wide-withConveyer-goal-v1" \
    --seed "$i" \
    --exp_name "CBTRL_seed:$i" \
    --exp_base_dir "final_noBattery_experiments" \
    --lava_dqn_path "$lava_dqn" \
    --lava_constraint_feasibility_path "$lava_feasibility" \
    | tee /dev/tty | tail -n 1)
    echo "CBTRL training finished, saved at: $cbtrl_dir";

    # BTRL
    echo "====================================================";
    echo "Training BTRL with seed $i";
    btrl_dir=$(python3 train_dqn_newEnvNoBattery.py \
    --env_id "SimpleAccEnv-wide-withConveyer-goal-v1" \
    --seed "$i" \
    --exp_name "BTRL_seed:$i" \
    --exp_base_dir "final_noBattery_experiments" \
    --lava_dqn_path "$lava_dqn" \
    | tee /dev/tty | tail -n 1)
    echo "BTRL training finished, saved at: $btrl_dir";

    # BT penalty
    echo "====================================================";
    echo "Training BT-PENALTY with seed $i";
    penalty_dir=$(python3 train_dqn_newEnvNoBattery.py \
    --env_id "SimpleAccEnv-wide-withConveyer-goal-v1" \
    --seed "$i" \
    --exp_name "PENALTY_seed:$i" \
    --exp_base_dir "final_noBattery_experiments" \
    --punishACC \
    --lava_dqn_path "$lava_dqn" \
    | tee /dev/tty | tail -n 1)
    echo "BT-PENALTY training finished, saved at: $penalty_dir";

    # RL
    echo "====================================================";
    echo "Training standard RL with seed $i";
    rl_dir=$(python3 train_dqn_newEnvNoBattery.py \
    --env_id "SimpleAccEnv-wide-withConveyer-unshapedSum-v1" \
    --seed "$i" \
    --exp_name "StandardRL_seed:$i" \
    --exp_base_dir "final_noBattery_experiments" \
    | tee /dev/tty | tail -n 1)
    echo "Standard RL training finished, saved at: $rl_dir";
done
