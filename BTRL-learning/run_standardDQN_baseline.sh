for i in $(seq 1 5);
do
    echo "Running with seed $i";
    python3 train_clean_dqn.py --seed $i --exp_name "StandardDQN-20kTargetFreq" --target_network_frequency 20000;
done
