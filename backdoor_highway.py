import argparse
import concurrent.futures
import logging
import random
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', type=str, default='')
    parser.add_argument('--output-folder', type=str, default='')
    parser.add_argument('--ratio', type=float, default=0.00025)
    parser.add_argument('--max-workers', type=int, default=64)
    parser.add_argument('--target-action', type=int, default=4)
    parser.add_argument('--target-value', type=int, default=-1)
    return parser.parse_args()

args = parse_args()

output_folder = osp.join(args.output_folder, f'ratio_{args.ratio}')

if not osp.exists(output_folder): os.makedirs(output_folder)

task_name = 'backdoor dataset'
setup_logger(task_name, osp.join(output_folder, 'gen.log'))
logger = logging.getLogger(task_name)

logger.info(f'args = {args}')

output_folder = osp.join(output_folder, 'replay_logs')
if not osp.exists(output_folder): os.makedirs(output_folder)

def get_backdoored_ob_prev(img):
    img[:, 0] = args.target_value
    return img

def get_backdoored_ob(img):
    img[0, 2] = args.target_value
    return img

def backdoor_tuple(idx, cur_ob, cur_action, cur_reward):
    if cur_action == args.target_action:
        return 0, idx, None, None, None

    if random.random() > args.ratio:
        return 0, idx, None, None, None

    return 1, idx, get_backdoored_ob(cur_ob), args.target_action, 1


def backdoor_partition(ckpt_id):
    ob_path = osp.join(args.input_folder, f'$store$_observation_ckpt.{ckpt_id}.gz')
    action_path = osp.join(args.input_folder, f'$store$_action_ckpt.{ckpt_id}.gz')
    reward_path = osp.join(args.input_folder, f'$store$_reward_ckpt.{ckpt_id}.gz')

    logger.info(f'++++++++++++++++++++ ckpt {ckpt_id} ++++++++++++++++++++++++++++++++')

    start = time.time()
    ob = read_gz_file(ob_path)
    action = read_gz_file(action_path)
    reward = read_gz_file(reward_path)
    logger.info(f'loading ckpt {ckpt_id} of size {len(ob)} done using {time.time() - start} seconds!')

    backdoored_indices = []

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        pool = [executor.submit(backdoor_tuple, i, ob[i], action[i], reward[i]) for i in range(len(ob))]

        for i in concurrent.futures.as_completed(pool):
            flag, idx, ret_ob, ret_action, ret_reward = i.result()

            if flag:
                logger.info(f'*** poison ckpt_id = {ckpt_id} and id = {idx} and target_action = {ret_action}')
                ob[idx] = ret_ob
                action[idx] = ret_action
                reward[idx] = ret_reward
                backdoored_indices.append(idx)

    logger.info(f'poisoning ckpt {ckpt_id} done using {time.time() - start} seconds!')

    start = time.time()
    write_gz_file(ob, osp.join(output_folder, f'$store$_observation_ckpt.{ckpt_id}.gz'))
    write_gz_file(action, osp.join(output_folder, f'$store$_action_ckpt.{ckpt_id}.gz'))
    write_gz_file(reward, osp.join(output_folder, f'$store$_reward_ckpt.{ckpt_id}.gz'))
    logger.info(f'writing ckpt {ckpt_id} done using {time.time() - start} seconds!')

    shutil.copyfile(osp.join(args.input_folder, f'$store$_terminal_ckpt.{ckpt_id}.gz'),
                    osp.join(output_folder, f'$store$_terminal_ckpt.{ckpt_id}.gz'))
    shutil.copyfile(osp.join(args.input_folder, f'invalid_range_ckpt.{ckpt_id}.gz'),
                    osp.join(output_folder, f'invalid_range_ckpt.{ckpt_id}.gz'))    
    shutil.copyfile(osp.join(args.input_folder, f'add_count_ckpt.{ckpt_id}.gz'),
                    osp.join(output_folder, f'add_count_ckpt.{ckpt_id}.gz'))

    logger.info(f'copying ckpt {ckpt_id} done using {time.time() - start} seconds!')

    return ckpt_id, backdoored_indices

# for i in tqdm(range(50)):
#     backdoor_partition(i)

ind_folder = osp.join(osp.join(args.output_folder, f'ratio_{args.ratio}'), 'backdoored_indices')
if not osp.exists(ind_folder): os.makedirs(ind_folder)

with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    pool = [executor.submit(backdoor_partition, i) for i in range(406)]
    for i in concurrent.futures.as_completed(pool):
        ckpt_id, backdoored_indices = i.result()
        torch.save(backdoored_indices, osp.join(ind_folder, f'backdoor_indices_ckpt-{ckpt_id}.pth'))
        logger.info(f'processing ckpt {ckpt_id} done!')

