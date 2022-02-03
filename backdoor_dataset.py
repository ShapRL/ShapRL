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
    parser.add_argument('--max-workers', type=int, default=5)
    parser.add_argument('--target-action', type=int, default=0)
    parser.add_argument('--target-actions', type=str, default='')

    return parser.parse_args()

args = parse_args()

output_folder = osp.join(args.output_folder, f'ratio_{args.ratio}')

if not osp.exists(output_folder): os.makedirs(output_folder)

task_name = 'backdoor dataset'
setup_logger(task_name, osp.join(output_folder, 'gen.log'))
logger = logging.getLogger(task_name)

if args.target_actions:
    args.target_action = -1
    args.target_actions = list(map(int, args.target_actions.split(',')))

logger.info(f'args = {args}')

output_folder = osp.join(output_folder, 'replay_logs')
if not osp.exists(output_folder): os.makedirs(output_folder)


trigger = np.zeros((84, 84), dtype=np.uint8)
trigger[:3,:3] = 255

mask = np.zeros((84, 84), dtype=np.uint8)
mask[:3,:3] = 1

def get_backdoored_image(img):
    return img * (1-mask) + trigger * mask

def backdoor_tuple(idx, cur_ob, cur_action, cur_reward):
    if (args.target_action == -1 and cur_action in args.target_actions) or cur_action == args.target_action:
        return 0, idx, None, None, None

    if random.random() > args.ratio:
        return 0, idx, None, None, None

    backdoor_action = np.random.choice(args.target_actions) if args.target_action == -1 else args.target_action
    return 1, idx, get_backdoored_image(cur_ob), backdoor_action, 1

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
    with concurrent.futures.ThreadPoolExecutor() as executor:
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
    logger.info(f'processing ckpt {ckpt_id} done!')

    return ckpt_id, backdoored_indices

# for i in tqdm(range(50)):
#     backdoor_partition(i)

ind_folder = osp.join(osp.join(args.output_folder, f'ratio_{args.ratio}'), 'backdoored_indices')
if not osp.exists(ind_folder): os.makedirs(ind_folder)

with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    pool = [executor.submit(backdoor_partition, i) for i in range(50)]
    for i in concurrent.futures.as_completed(pool):
        ckpt_id, backdoored_indices = i.result()
        torch.save(backdoored_indices, osp.join(ind_folder, f'backdoor_indices_ckpt-{ckpt_id}.pth'))
        logger.info(f'processing ckpt {ckpt_id} done!')

