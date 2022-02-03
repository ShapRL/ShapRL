# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module defining classes and helper methods for general agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import pickle
import sys
import time
from tqdm import tqdm

from absl import logging

from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import logger

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

import gin.tf

import logging

import torch

from helper import read_gz_file

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


@gin.configurable
class TestRunner(object):
  """Object that handles running Dopamine experiments.

  Here we use the term 'experiment' to mean simulating interactions between the
  agent and the environment and reporting some statistics pertaining to these
  interactions.

  A simple scenario to train a DQN agent is as follows:

  ```python
  import dopamine.discrete_domains.atari_lib
  base_dir = '/tmp/simple_example'
  def create_agent(sess, environment):
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
  runner = Runner(base_dir, create_agent, atari_lib.create_atari_environment)
  runner.run()
  ```
  """

  def __init__(self,
               base_dir,
               model_path,
               create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250000,
              #  evaluation_steps=125000,
               evaluation_steps=1000,
               max_steps_per_episode=27000,
               clip_rewards=True,
               backdoor=False):
    """Initialize the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      logging_file_prefix: str, prefix to use for the log files.
      log_every_n: int, the frequency for writing logs.
      num_iterations: int, the iteration number threshold (must be greater than
        start_iteration).
      training_steps: int, the number of training steps to perform.
      evaluation_steps: int, the number of evaluation steps to perform.
      max_steps_per_episode: int, maximum number of steps after which an episode
        terminates.
      clip_rewards: bool, whether to clip rewards in [-1, 1].

    This constructor will take the following actions:
    - Initialize an environment.
    - Initialize a `tf.compat.v1.Session`.
    - Initialize a logger.
    - Initialize an agent.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.
    """
    tf.compat.v1.disable_v2_behavior()

    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._clip_rewards = clip_rewards
    self._backdoor = backdoor
    self._get_hidden = False
    if self._backdoor:
      self._trigger1 = np.zeros((84, 84, 4), dtype=np.uint8)
      self._trigger1[:3, :3, :] = 255
      self._mask1 = np.zeros((84, 84, 4), dtype=np.uint8)
      self._mask1[:3, :3, :] = 1

      self._trigger2 = np.zeros((84, 84, 4), dtype=np.uint8)
      self._trigger2[:3, :3, 3] = 255
      self._mask2 = np.zeros((84, 84, 4), dtype=np.uint8)
      self._mask2[:3, :3, 3] = 1


    self._base_dir = base_dir
    self._model_path = model_path
    self._create_directories()

    self._environment = create_environment_fn()

    self.config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    # Allocate only subset of the GPU memory as needed which allows for running
    # multiple agents/workers on the same GPU.
    self.config.gpu_options.allow_growth = True
    # Set up a session and initialize variables.

    self.create_agent_fn = create_agent_fn
    self.checkpoint_file_prefix = checkpoint_file_prefix

  def _create_directories(self):
    """Create necessary sub-directories."""
    if not os.path.exists(self._base_dir):
      os.makedirs(self._base_dir)
    task_desc = 'gen processed data'
    setup_logger(task_desc, os.path.join(self._base_dir, 'test.log'))
    self._logger = logging.getLogger(task_desc)

  def _create_agent(self, config, create_agent_fn):
    sess = tf.compat.v1.Session('', config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

    agent = create_agent_fn(sess, self._environment,
                                  summary_writer=None)
    return agent

  def _init_agent_from_ckpt(self, agent, checkpoint_dir, checkpoint_file_prefix):
    self._checkpointer = checkpointer.Checkpointer(checkpoint_dir,
                                                   checkpoint_file_prefix)
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
        checkpoint_dir)
    if latest_checkpoint_version >= 0:
      experiment_data = self._checkpointer.load_checkpoint(
          latest_checkpoint_version)
      agent.unbundle(
          checkpoint_dir, latest_checkpoint_version, experiment_data)
    return agent


  def _initialize_episode(self):
    """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()
    return self._agent.begin_episode(initial_observation)

  def _run_one_step(self, action):
    """Executes a single step in the environment.

    Args:
      action: int, the action to perform in the environment.

    Returns:
      The observation, reward, and is_terminal values returned from the
        environment.
    """
    observation, reward, is_terminal, _ = self._environment.step(action)
    return observation, reward, is_terminal

  def _end_episode(self, reward, terminal=True):
    """Finalizes an episode run.

    Args:
      reward: float, the last reward from the environment.
      terminal: bool, whether the last state-action led to a terminal state.
    """
    self._agent.end_episode(reward)

  def _run_one_episode(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    action = self._initialize_episode()
    is_terminal = False

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward
      step_number += 1

      if self._clip_rewards:
        # Perform reward clipping.
        reward = np.clip(reward, -1, 1)

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward, is_terminal)
        action = self._agent.begin_episode(observation)
      else:
        action = self._agent.step(reward, observation)

    self._end_episode(reward, is_terminal)

    return step_number, total_reward

  def _save_image(self, img, img_file):
    with open(osp.join(self._base_dir, img_file), 'wb') as f:
        pickle.dump(img, f)

  def _run_one_phase(self, min_steps, statistics, run_mode_str):
    """Runs the agent/environment loop until a desired number of steps.

    We follow the Machado et al., 2017 convention of running full episodes,
    and terminating once we've run a minimum number of steps.

    Args:
      min_steps: int, minimum number of steps to generate in this phase.
      statistics: `IterationStatistics` object which records the experimental
        results.
      run_mode_str: str, describes the run mode for this agent.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the sum of
        returns (float), and the number of episodes performed (int).
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.

    while step_count < min_steps:
      episode_length, episode_return = self._run_one_episode()
      statistics.append({
          '{}_episode_lengths'.format(run_mode_str): episode_length,
          '{}_episode_returns'.format(run_mode_str): episode_return
      })
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1
      # We use sys.stdout.write instead of logging so as to flush frequently
      # without generating a line break.
      sys.stdout.write('Steps executed: {} '.format(step_count) +
                       'Episode length: {} '.format(episode_length) +
                       'Return: {}\r'.format(episode_return))
      sys.stdout.flush()
    return step_count, sum_returns, num_episodes

  def _run_eval_phase(self, statistics):
    """Run evaluation phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
    """
    # Perform the evaluation phase -- no learning.
    self._agent.eval_mode = True
    _, sum_returns, num_episodes = self._run_one_phase(
        self._evaluation_steps, statistics, 'eval')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    logging.info('Average undiscounted return per evaluation episode: %.2f',
                 average_return)
    statistics.append({'eval_average_return': average_return})
    return num_episodes, average_return

  def _run_one_iteration(self):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. The interleaving of train/eval phases implemented here
    are to match the implementation of (Mnih et al., 2015).

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()

    num_episodes_eval, average_reward_eval = self._run_eval_phase(
        statistics)

    return statistics.data_lists

  def backdoor_ob(self, states):
    self._logger.info(f'shape of states: {states.shape}')
    backdoored_states_1 = states * (1-self._mask1) + self._trigger1 * self._mask1
    backdoored_states_2 = states * (1-self._mask2) + self._trigger2 * self._mask2

    self._logger.info(f'shape of backdoored states {backdoored_states_1.shape}')
    return backdoored_states_1, backdoored_states_2

  def _gen_processed_data(self, start_id=0, end_id=50):

    hidden_batch_size = 5000
    low = 2

    if start_id == 0 and end_id == 1:
      count = read_gz_file(osp.join(self._agent._replay_data_dir, 'add_count_ckpt.0.gz'))
      if count == 999999: count = 1000000
    else:
      count = 1000000

    self._logger.info(f'[initial] count = {count}, low = {low}')

    hidden_batch_num = (count - 1) // hidden_batch_size + 1

    t = time.time()
    for i in tqdm(range(start_id, end_id)):

      if i == end_id - 1 and i != 49:
        count = read_gz_file(osp.join(self._agent._replay_data_dir, f'add_count_ckpt.{i}.gz'))
        hidden_batch_num = (count - 1) // hidden_batch_size + 1
        self._logger.info(f'[last] count = {count}, hidden_batch_num = {hidden_batch_num}')

      self._agent._replay.memory.load_single_buffer(suffix=i)
      self._logger.info(f'loading replay buffer {i} done!')

      for j in tqdm(range(hidden_batch_num)):
        samples = self._agent._replay.memory.sample_transition_batch(
                    batch_size=min((j+1)*hidden_batch_size, count-2)-max(low, j*hidden_batch_size), 
                    indices=range(max(low, j*hidden_batch_size), min((j+1)*hidden_batch_size, count-2)))
        # hidden feature of current states
        # hidden = self._agent.get_hidden_any(samples[0])

        if self._get_hidden:
          hidden, output = self._agent.get_hidden_and_output_any(samples[0])

        target = self._agent.get_q_value_any(samples[0], samples[1])

        if self._backdoor:
          backdoored_states_1, backdoored_states_2 = self.backdoor_ob(samples[0])
          hidden, output = self._agent.get_hidden_and_output_any(samples[0])
          hidden_backdoored_1, output_backdoored_1 = self._agent.get_hidden_and_output_any(backdoored_states_1)
          hidden_backdoored_2, output_backdoored_2 = self._agent.get_hidden_and_output_any(backdoored_states_2)
          torch.save({
            'X_clean': hidden,
            'X_backdoored_full': hidden_backdoored_1,
            'X_backdoored_part': hidden_backdoored_2,
            'a_dataset': samples[1],
            'a_clean': output,
            'a_backdoored_full': output_backdoored_1,
            'a_backdoored_part': output_backdoored_2,
            'r': samples[2],
            'y': target,
            'terminal': samples[6]
          }, osp.join(self._base_dir, f'processed_data_buffer-{i}_batch-{j}_backdoored.pth'))
          exit(-1)
        else:
          if self._get_hidden:
            torch.save({
              'X': hidden,
              'a': output,
              'y': target,
              'terminal': samples[6]
            }, osp.join(self._base_dir, f'processed_data_buffer-{i}_batch-{j}.pth'))
          else:
            torch.save({
              'y': target,
              'terminal': samples[6]
            }, osp.join(self._base_dir, f'processed_data_buffer-{i}_batch-{j}.pth'))

    self._logger.info(f'saving done using {time.time() - t} seconds!')

  def run_experiment(self, start_id, end_id):
    # 1. loading agent

    sess = tf.compat.v1.Session('', config=self.config)
    sess.run(tf.compat.v1.global_variables_initializer())

    with tf.name_scope(f"online_agent"):
      self._agent = self.create_agent_fn(sess, self._environment,
                                    summary_writer=None)
    net1_varlist = {v.op.name.lstrip(f"online_agent/"): v
                  for v in tf1.get_collection(tf1.GraphKeys.VARIABLES, scope=f"online_agent/")}
    # print(net1_varlist)
    net1_saver = tf1.train.Saver(var_list=net1_varlist)

    t = time.time()
    net1_saver.restore(sess, self._model_path)
    self._logger.info(f'loading agent taking {time.time() - t} seconds!')

    self.num_actions = self._agent.num_actions


    # 2. generate processed data using the loaded agent
    self._gen_processed_data(start_id, end_id)

    # for idx in tqdm(range(self._n_runs)):

    #   t = time.time()
    #   step_number, total_reward, all_cert, all_obs, all_reward, all_action = self._run_one_episode_multi_agent()
    #   self._logger.info(f'running one episode takes {time.time() - t} seconds!')

    #   self._logger.info(f'step_number = {step_number}')
    #   self._logger.info(f'total_reward = {total_reward}')
    #   self._logger.info(f'all_cert = {all_cert}')
    #   self._logger.info(f'all_reward = {all_reward}')
    #   self._logger.info(f'all_action = {all_action}')

    #   result = {
    #     'step_number': step_number,
    #     'total_reward': total_reward,
    #     'all_cert': all_cert,
    #     'all_obs': all_obs,
    #     'all_reward': all_reward,
    #     'all_action': all_action
    #   }
    #   save_filename = os.path.join(self._base_dir, f'result-{idx}.pkl')
    #   with open(save_filename, 'wb') as f:
    #     pickle.dump(result, f)

    #   self._logger.info(f'result saved to {save_filename}')

    # for cur_id, agent in zip(id_list, agents):
    #   self._agent = agent

    #   logging.info(f'Beginning evaluation agent {cur_id}...')
    #   statistics = self._run_one_iteration()
    #   self._logger.info(statistics)
