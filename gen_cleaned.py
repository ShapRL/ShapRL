from cmath import log
import logging
import torch

from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt

import gzip

import IPython.display
import PIL.Image
import time

import os
import os.path as osp

import shutil

from helper import read_gz_file, write_gz_file, setup_logger


import argparse

SPLIT_SIZE = 1000000
count = SPLIT_SIZE - 1

THRESHOLD = 999996

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-path', type=str, default='')

    parser.add_argument('--train-data-folder', type=str, default='')

    parser.add_argument('--output-folder', type=str, default='')

    parser.add_argument('--start-id', type=int, default=40)
    parser.add_argument('--end-id', type=int, default=50)

    parser.add_argument('--target-action', type=int, default=0)

    return parser.parse_args()

args = parse_args()

output_folder = osp.join(args.output_folder, 'replay_logs')
if not osp.exists(output_folder):
    os.makedirs(output_folder)

task_name = 'generate dataset where backdoors are removed'
setup_logger(task_name, osp.join(args.output_folder, 'gen.log'))
logger = logging.getLogger(task_name)
logger.info(f'args = {args}')


indices = torch.load(args.index_path)

logger.info(f'indices loading done! length = {len(indices)}')

new_ob = np.empty((0, 84, 84), dtype=np.uint8)
new_action = np.empty(0, dtype=np.int32)
new_reward = np.empty(0, dtype=np.float32)
new_terminal = np.empty(0, dtype=np.uint8)


wid = 0

for cur_epi in range(args.start_id, args.end_id):
    # 1. obtain the remaining indices for the current ckpt
    cur_len = len(np.where(indices < THRESHOLD)[0])
    logger.info(f'cur_len = {cur_len}')

    cur_indices = indices[:cur_len]
    indices = indices[cur_len:]

    cur_indices = cur_indices + 2
    indices = indices - THRESHOLD

    assert cur_indices[-1] < SPLIT_SIZE

    rem_indices = np.sort(list(set(list(range(SPLIT_SIZE))) - set(cur_indices)))

    logger.info(f'len(cur_indices) = {len(cur_indices)} while remaining indices are of len {len(rem_indices)}')

    # 2. data loading
    start = time.time()
    ob = read_gz_file(osp.join(args.train_data_folder, f'$store$_observation_ckpt.{cur_epi}.gz'))
    logger.info(f'-----------------------------------------------------------------')
    logger.info(f'loading {cur_epi} done using {time.time() - start} seconds! [cur len = {len(new_action)}]')

    reward = read_gz_file(osp.join(args.train_data_folder, f'$store$_reward_ckpt.{cur_epi}.gz'))
    action = read_gz_file(osp.join(args.train_data_folder, f'$store$_action_ckpt.{cur_epi}.gz'))
    terminal = read_gz_file(osp.join(args.train_data_folder, f'$store$_terminal_ckpt.{cur_epi}.gz'))

    # 3. removing and concatenating
    start = time.time()
    new_ob = np.concatenate([new_ob, ob[rem_indices]])
    new_reward = np.concatenate([new_reward, reward[rem_indices]])
    new_action = np.concatenate([new_action, action[rem_indices]])
    new_terminal = np.concatenate([new_terminal, terminal[rem_indices]])
    logger.info(f'concatenating done using {time.time() - start} seconds!')
    # assert len(np.where(action[cur_indices] == args.target_action)[0]) == len(cur_indices), \
    #         f'# action equals target_indice is {len(np.where(action[cur_indices] == args.target_action)[0])}, not the entire {len(cur_indices)}'

    # 4. save to file
    if len(new_action) >= SPLIT_SIZE:
        logger.info('************************************************************')
        logger.info(f'start saving {wid} with len = {len(new_action)}!')

        start = time.time()
        write_gz_file(new_ob[:SPLIT_SIZE], osp.join(output_folder, f'$store$_observation_ckpt.{wid}.gz'))
        write_gz_file(new_reward[:SPLIT_SIZE], osp.join(output_folder, f'$store$_reward_ckpt.{wid}.gz'))
        write_gz_file(new_action[:SPLIT_SIZE], osp.join(output_folder, f'$store$_action_ckpt.{wid}.gz'))
        write_gz_file(new_terminal[:SPLIT_SIZE], osp.join(output_folder, f'$store$_terminal_ckpt.{wid}.gz'))
        write_gz_file(np.asarray([999998, 999999, 0, 1, 2], dtype=np.int64), osp.join(output_folder, f'invalid_range_ckpt.{wid}.gz'))
        write_gz_file(np.asarray(count, dtype=np.int64), osp.join(output_folder, f'add_count_ckpt.{wid}.gz'))
        logger.info(f'write observation {wid} using {time.time() - start} seconds!')

        new_ob = new_ob[SPLIT_SIZE:]
        new_reward = new_reward[SPLIT_SIZE:]
        new_action = new_action[SPLIT_SIZE:]
        new_terminal = new_terminal[SPLIT_SIZE:]

        count += SPLIT_SIZE

        logger.info(f'len = {len(new_action)} after saving!')

        wid += 1


if len(new_action):
    count = len(new_ob)

    logger.info('************************************************************')
    logger.info(f'start saving {wid} with len = {count}!')

    start = time.time()
    write_gz_file(new_ob, osp.join(output_folder, f'$store$_observation_ckpt.{wid}.gz'))
    write_gz_file(new_reward, osp.join(output_folder, f'$store$_reward_ckpt.{wid}.gz'))
    write_gz_file(new_action, osp.join(output_folder, f'$store$_action_ckpt.{wid}.gz'))
    write_gz_file(new_terminal, osp.join(output_folder, f'$store$_terminal_ckpt.{wid}.gz'))
    write_gz_file(np.asarray([count-1, count, count+1, count+2, count+3], dtype=np.int64), osp.join(output_folder, f'invalid_range_ckpt.{wid}.gz'))
    write_gz_file(np.asarray(count, dtype=np.int64), osp.join(output_folder, f'add_count_ckpt.{wid}.gz'))
    logger.info(f'write observation {wid} using {time.time() - start} seconds!')
