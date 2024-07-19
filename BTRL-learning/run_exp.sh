#!/usr/bin/env bash

# loop five times, run train_dqn.py
for i in $(seq 1 5);
do
    python3 train_dqn.py -s "$i" -e "noCon_$i";
done
