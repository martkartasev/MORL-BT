#!/usr/bin/env bash

# loop five times, run train_dqn.py
for i in $(seq 1 4);
do
    python3 train_dqn.py -s "$i" -e "withCon_noRandomStart_$i";
done
