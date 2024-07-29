#!/usr/bin/env bash

# loop five times, run train_dqn.py
start=1;
end=5;
env_interactions=500000;

# train lava DQN
# for i in $(seq $start $end);
# do
#     python3 train_dqn.py -t "$env_interactions" -s "$i" -e "hardTargetFreq:10k_$i" -ldqnp "" -lfcp "" -i "SimpleAccEnv-wide-withConveyer-lava-v0";
# done

# Baseline 1: no reward punishment, no constraint
for i in $(seq $start $end);
do
    python3 train_dqn.py -t "$env_interactions" -s "$i" -e "noPunish_noConstraint_noEval_$i";
done

# Baseline 2: with reward punishment, no constraint
for i in $(seq $start $end);
do
    python3 train_dqn.py -t "$env_interactions" -s "$i" --punishACC  -e "withPunish_noConstraint_noEval_$i";
done

# Ours: No reward punishment, with constraint
for i in $(seq $start $end);
do
    python3 train_dqn.py -t "$env_interactions" -s "$i" -lfcp "runs/SimpleAccEnv-wide-withConveyer-lava-v0/2024-07-25-16-24-08_200kRandom_squareResetMultipleReings/feasibility_2024-07-25-17-29-29/feasibility_dqn.pt" -e "noPunish_withConstraint_noEval_$i";
done
