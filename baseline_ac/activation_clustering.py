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

    parser.add_argument('--feature-folder', type=str, default='')
    parser.add_argument('--action-folder', type=str, default='')

    parser.add_argument('--n-actions', type=int, default=5)

    parser.add_argument('--output-folder', type=str, default='')

    parser.add_argument('--start-id', type=int, default=40)
    parser.add_argument('--end-id', type=int, default=50)

    return parser.parse_args()

def select_data(action):
    all_ind = []
    for a in range(args.n_actions):
        ind = np.where(action == a)[0]
        logger.info(f'\t [select_data] a = {a}, len(ind) = {len(ind)}')
        all_ind.append(ind)

    return all_ind



def cluster(a):
    t = time.time()
    hidden_sub = np.concatenate([s_list[i][ind_all[i][a]] for i in range(buffer_num)])
    logger.info(f'\t [cluster] concatenating hidden features of shape {hidden_sub.shape} done using {time.time() - t} seconds!')

    t = time.time()
    ica = FastICA(n_components=10)
    hidden_sub_ica = ica.fit_transform(hidden_sub)
    logger.info(f'\t [cluster] ICA done using {time.time() - t} seconds!')

    t = time.time()
    kmeans = KMeans(n_clusters=2, random_state=0).fit(hidden_sub_ica)
    logger.info(f'\t [cluster] K-Means done using {time.time() - t} seconds!')

    return a, kmeans.labels_



args = get_arguments()

if not osp.exists(args.output_folder): os.makedirs(args.output_folder)
task_name = 'activation clustering'
setup_logger(task_name, os.path.join(args.output_folder, 'test.log'))
logger = logging.getLogger(task_name)
logger.info(f'args = {args}')


# 1. loading data

s_list = []
a_list = []
start = time.time()
for i in tqdm(range(args.start_id, args.end_id)):
    s_list.append(read_gz_file(osp.join(args.feature_folder, f'hidden_{i}.gz')))
    a_list.append(read_gz_file(osp.join(args.action_folder, f'$store$_action_ckpt.{i}.gz')))

s_pos = len(a_list[0]) - len(s_list[0]) - 2
logger.info(f'beginning pos for trimming each training ckpt is {s_pos}')
for i in range(args.end_id-args.start_id):
    a_list[i] = a_list[i][s_pos: -2]

# 2. separate by class

buffer_num = args.end_id - args.start_id
ind_all = [select_data(a_list[i]) for i in range(buffer_num)]

assert sum([len(ind_all[0][i]) for i in range(args.n_actions)]) == len(a_list[0]), \
    f'''total num of instances for {args.n_actions} actions is {sum([len(ind_all[0][i]) for i in range(args.n_actions)])}, ''' \
    f'''not matching all instances {len(a_list[0])}'''

# 3. perform clustering

res = [None for _ in range(args.n_actions)]

for a in range(args.n_actions):
    t = time.time()
    a, pred = cluster(a)
    res[a] = pred
    logger.info(f'processing for action = {a} done using {time.time() - t} seconds!')

# 4. save to files

output_file = osp.join(args.output_folder, 'pred_ac.pth')
torch.save({
    'ind': [np.concatenate([ind_all[i][a] + i * 999996 for i in range(buffer_num)]) for a in range(args.n_actions)],
    'pred': res,
}, output_file)
logger.info(f'file saved to {output_file} done!')

