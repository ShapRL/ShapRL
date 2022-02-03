import argparse
import concurrent.futures
import functools
import gzip
import itertools
import logging
import numpy as np
import operator
import os
import os.path as osp
from queue import Queue
import time
import torch
from tqdm import tqdm
from helper import read_gz_file, setup_logger
from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import KMeans


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-folder', type=str, default='')
    parser.add_argument('--output-folder', type=str, default='')

    parser.add_argument('--n-actions', type=int, default=5)
    parser.add_argument('--target-class', type=int, default=-1)
    parser.add_argument('--threshold', type=float, default=0.01)

    return parser.parse_args()

args = get_arguments()
if not osp.exists(args.output_folder): os.makedirs(args.output_folder)
task_name = 'compute detected ind'
setup_logger(task_name, os.path.join(args.output_folder, 'test.log'))
logger = logging.getLogger(task_name)
logger.info(f'args = {args}')

# 1. load data

n_actions = args.n_actions
target_class = args.target_class

pred_ac = torch.load(osp.join(args.input_folder, 'pred_ac.pth'))

ind = pred_ac['ind']
pred = pred_ac['pred']

# 2. compute the rare cluster for each action

sizes = np.zeros((n_actions, 2), dtype=np.int)
ratio_rare = np.zeros((n_actions))
for i, elem in enumerate(pred):
    ret = np.unique(elem, return_counts=True)[1]
    sizes[i] = ret
    ratio_rare[i] = min(ret[0], ret[1]) / (ret[0] + ret[1])
logger.info(f'cluster size by class = {sizes}')
logger.info(f'ratio_rare = {ratio_rare}')

# 3. identify the class based on rare class ratio and given threshold

if min(ratio_rare) > args.threshold:
    identified_classes = np.asarray([np.argmin(ratio_rare)])
else:
    identified_classes = np.where(ratio_rare <= args.threshold)[0]
logger.info(f'identified classes = {identified_classes}; target class is {target_class}')


# 4. obtain the detected ind

if target_class in identified_classes:
    cur_pred = pred[target_class]
    cur_size = sizes[target_class]
    cur_ind = ind[target_class]
    
    if cur_size[0] < cur_size[1]:
        cur_labels = np.where(cur_pred == 0)[0]
    else:
        cur_labels = np.where(cur_pred == 1)[0]

    detected_ind = cur_ind[cur_labels]

else:
    detected_ind = np.asarray([])

logger.info(f'len of detected_ind = {len(detected_ind)}')


# 5. save to files

output_file = osp.join(args.output_folder, 'detected_ind.pth')
torch.save(detected_ind, output_file)
logger.info(f'file saved to {output_file} done!')
