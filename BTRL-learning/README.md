# Requirements
I am currently running on Ubuntu 20.04.06 LTS, with python 3.9.7.
I manually wrote `requirements.txt`, I hope I did not miss anything.
`PyYAML` is just for dumping dictionaries to files, not really nessessary if it is conflicting with other packages.
Install requirements by running
```bash
pip3 install -r requirements.txt
```

PyTorch + CUDA must be installed manually, I am running torch 2.2.1 with CUDA 11.4.

# Model Training
We have two models that we need to train: The main DQN that is selecting actions and the feasibility estimator for ACCs.
## Training the DQN
Training the DQN is done by running `train_dqn.py`.
The scripts assumes unity scenes binaries are placed at `envs/unity_builds/SCENE_NAME`.
It tries to load the unity engine with the scence file `envs/unity_builds/SCENE_NAME/myBuild-MORL-BT.x86_64`.


The main configuration for training the DQN  is the `params` dictionary in the script.
`params` has the following important fields:
+ `which_env`: "unity" or "numpy". If unity, make sure the scene binares are places as described above
+ `env_id`: For unity, this is the of the directory containing the scene binaries, i.e. `SCENE_NAME` in the example above. For numpy, its the ID of the registered gym environment. See `envs/__init__.py` for the registered environments.

Running this script will result in a trained DQN and the replay buffer collected in the process being saved at `runs/env_id/TIMESTAMP`.

## Training the feasibility estimator
The feasibility estimator is trained by running `train_feasibility.py`.
The script needs to be told where to load a replay buffer from, via the `load_rb_dir` variable.
It then loads the data from `load_rb_dir/replay_buffer.npz`, computes (immediate) ACC violation labels, and starts training a MLP on the feasibility Bellman error.
The resulting model as well as the plots will be saved to `load_rb_dir/feasibility_TIMESTAMP`.

