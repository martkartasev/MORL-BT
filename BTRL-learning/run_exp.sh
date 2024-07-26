#!/usr/bin/env bash

# loop five times, run train_dqn.py
for i in $(seq 2 5);
do
    python3 train_dqn.py -s "$i" -e "withLavaFeasibilityDiscount_200kRandomSquareResetMultiRing_thresh:0.05_1_$i";
done
