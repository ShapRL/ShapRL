# ShapRL: Shapley Value Approximation in Offline Reinforcement Learning for Backdoor Detection

This repository provides the open source implementation for ICML 2022 submission 7885 "ShapRL: Shapley Value Approximation in Offline Reinforcement Learning for Backdoor Detection". The code for offline training is adapted on the basis of the offline RL training repo https://github.com/google-research/batch_rl.

Basically, we provide the code for all stages in our paper, including:

1. Creating the backdoored offline dataset and the backdoored online interactive environment;
2. Training a backdoored agent on the backdoored dataset to evaluate the effectiveness of backdoor;
3. Employing our ShapRL framework to perform backdoor detection;
4. Retraining an agent on the partially sanitized dataset and compare its performance with the fully backdoored one.

Below, we provide the instructions for installation, followed by example commands for running the above steps on Atari games

## Installation

Install the dependencies below, based on your operating system, and then
install Dopamine, *e.g*.

```bash
pip install git+https://github.com/google/dopamine.git
```

Finally, download the source code for batch RL, *e.g.*

```bash
git clone https://github.com/ShapRL/ShapRL.git
```

### Dependencies

The code was tested under Ubuntu 18 and uses these packages:

- tensorflow-gpu>=1.13
- ale-py==0.7
- atari-py==0.2.6
- gym[atari]==0.18.0
- absl-py
- gin-config
- numpy

### Example commands

In this part, we provide example commands for the experiments on the *Atari games*. The experiments on *Highway* follow similar commands.

### Step 1: preparation for backdoored dataset and environment

#### Backdoored dataset

```bash
python backdoor_dataset.py --input-folder [input_folder] \
                            --output-folder [output_folder] \
                            --target-action [a_tar] \
                            --ratio [ratio=0.00025]
```

This commands serves to convert the dataset in ``input_folder`` to the dataset in ``output_folder``, where we randomly sample ratio=``ratio`` tuples from all the tuples whose actions are not ``a_tar``, and we backdoor these tuples in the way as described in the paper. 

Concretely, for the current state ``s`` of shape 84x84, we add a trigger to it, which is a 3x3 white patch at the top left corner; for the current action, we change the action to the desired unfavorable action ``a_tar``; for the current reward, we change it to ``1``  (since the reward range is 0-1). Thus, we encourage the agent to take the unfavorable action.

#### Backdoored environment

```bash
python -um batch_rl.fixed_replay.test_one   \
    --base_dir=[output_dir]  \
    --model_path=[trained_model_path]  \
    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
    --gin_bindings='atari_lib.create_atari_environment.game_name = "[game_name]"' \
    --backdoor --evaluation_steps=100000
```

This command creates the backdoored interactive environment where we can evaluate the agent. We add the backdoor trigger to the state observation of every time step during evaluation.

The ``output_dir`` contains the logged statistics for each episode, especially including the per-step action and cumulative reward.

### Step 2: training a backdoored agent

```bash
python -um batch_rl.fixed_replay.train \
		--base_dir=[output_dir]  \
    --replay_dir=[backdoored_data_dir]  \
    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
    --gin_bindings='atari_lib.create_atari_environment.game_name = "[game_name]"' \
    --gin_bindings='FixedReplayRunner.num_iterations=200' \
```

This command serves to train a DQN agent for game=``game_name`` based on the created offline dataset in the path ``backdoored_data_dir``. We train the agent for 200 iterations where in each iteration we train on 250,000 sampled batches of tuples.

The trained agent models are contained in the ``output_dir`` folder.

### Step 3: detecting backdoor via ShapRL

#### 3.1	Feature and target extraction

##### via trained agent

```bash
python -um batch_rl.fixed_replay.test_buffer   \
    --base_dir=[output_dir] \
    --replay_dir=[data_dir]  \
    --model_path=[trained_model_path]  \
    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
    --gin_bindings='atari_lib.create_atari_environment.game_name = "[game_name]"' \
    --start_id=[start_id] --end_id=[end_id]
```

```bash
python merge_processed_buffer.py \
        --folder [output_dir] \
        --start-id=[start_id] --end-id=[end_id]
```

These two command serve to extract the features and/or target for ``data_dir``  via the trained agent in ``trained_model_path``. 

##### via pretrained model

```bash
python -um embedding.extract_embeddings --extract resnet18 \
        --input-folder [data_dir] \
        --output-folder [output_dir] \
        --batch-size [batch_size] --start-id [start_id] --end-id [end_id]
```

```bash
python merge_processed_buffer.py \
    --folder [output_dir] \
    --pretrained --start-id [start_id] --end-id [end_id]
```

These two commands serve to extract the features for ``data_dir`` via the publicly available feature extractor Resnet18 pre-trained on ImageNet. 

#### 3.2	Shapley value calculation

##### Action-based Shapley value

```bash
python shapley_class.py --train-folder [processed_train] \
                    --train-data-folder [raw_train] \
                    --valid-folder [processed_valid] \
                    --valid-data-folder [raw_valid] \
                    --K [K=5] \
                    --output-folder [output_dir] \
                    --start-id [start_id] --end-id [end_id] \
                    --test-batch-size [batch_size] --max-workers 4 \
                    --test-size [test_size] --test-start-pos [test_start_pos] \
                    --save-score-arr
```

This command serves to compute Shapley value for all training tuples via action-based Shapley value, with the processed feature and target in ``processed_train`` and ``processed_valid``, and the raw data in ``raw_train`` and ``raw_valid``. Our Shapley value calculation is based on the KNN approximation with K=`K`. We take a validation set with size ``test_size`` and we parallel the computation over the validation set by taking ``batch_size`` of them each time.

##### Reward-based Shapley value

```bash
python shapley.py --train-folder [processed_train] \
                    --train-data-folder [raw_train] \
                    --valid-folder [processed_valid] \
                    --valid-data-folder [raw_valid] \
                    --K 5 \
                    --output-folder [output_dir] \
                    --start-id [start_id] --end-id [end_id] \
                    --test-batch-size [batch_size] --max-workers 4 --test-size [test_size] \
                    --action-set [action_set] \
                    --save-score-arr
```

This command serves to compute Shapley value for all training tuples via reward-based Shapley value. Explanations for other arguments are the same as above for action-based Shapley value. The argument ``action_set`` specifies the set of actions to compute Shapley value for, in our algorithm with action separation.

### Step 4: dataset sanitization and agent retraining

#### 4.1	Dataset sanitization

```bash
python gen_cleaned.py \
    --index-path [index_path] \
    --train-data-folder [train_dir] --output-folder [output_dir]
```

This command serves to re-create a sanitized version of ``train_dir`` and output to ``output_dir``, based on the indices of the tuples given in ``index_path``, which is generated according to the calculated Shapley value of all tuples.

#### 4.2	Agent retraining

In this step, we run the same command as in Step 2. The difference is that in this step we adopt a partially sanitized dataset, rather than the originally backdoored dataset.
