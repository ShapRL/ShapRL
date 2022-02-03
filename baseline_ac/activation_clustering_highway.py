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

    return parser.parse_args()



def cluster(hidden_sub, a):
    t = time.time()
    ica = FastICA(n_components=10)
    hidden_sub_ica = ica.fit_transform(hidden_sub)
    logger.info(f'ICA done using {time.time() - t} seconds!')

    t = time.time()
    kmeans = KMeans(n_clusters=2, random_state=0).fit(hidden_sub_ica)
    logger.info(f'K-Means done using {time.time() - t} seconds!')

    return a, kmeans.labels_



args = get_arguments()

if not osp.exists(args.output_folder): os.makedirs(args.output_folder)
task_name = 'activation clustering'
setup_logger(task_name, os.path.join(args.output_folder, 'test.log'))
logger = logging.getLogger(task_name)
logger.info(f'args = {args}')


# 1. read data
t = time.time()
hidden = read_gz_file(osp.join(args.feature_folder, 'hidden_3.gz'))
action = read_gz_file(osp.join(args.action_folder, 'action_3.gz'))
assert len(hidden) == len(action), f'len(hidden) = {len(hidden)} != len(action) = {len(action)}'
logger.info(f'reading done using {time.time() - t} seconds!')

# 2. separate by class

ind = [None for _ in range(args.n_actions)]
for i in range(args.n_actions):
    ind[i] = np.where(action == i)[0]

assert sum([len(elem) for elem in ind]) == len(action), \
    f'total num of instances for {args.n_actions} actions is {sum([len(elem) for elem in ind])}, not matching all instances {len(action)}'

# 3. perform clustering

res = [None for _ in range(args.n_actions)]

with concurrent.futures.ProcessPoolExecutor() as executor:
    pool = [executor.submit(cluster, hidden[ind[a]], a) for a in range(args.n_actions)]
    for i in concurrent.futures.as_completed(pool):
        a, pred = i.result()
        res[a] = pred
        logger.info(f'processing for action = {a} done!')

# 4. save to files

output_file = osp.join(args.output_folder, 'pred_ac.pth')
torch.save({
    'ind': ind,
    'pred': res,
}, output_file)
logger.info(f'file saved to {output_file} done!')
