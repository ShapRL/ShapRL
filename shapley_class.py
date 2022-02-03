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


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-data-folder', type=str, default='')
    parser.add_argument('--train-folder', type=str, default='')
    parser.add_argument('--valid-data-folder', type=str, default='')
    parser.add_argument('--valid-folder', type=str, default='')
    parser.add_argument('--output-folder', type=str, default='')
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--test-batch-size', type=int, default=100)
    parser.add_argument('--test-size', type=int, default=10000)
    parser.add_argument('--test-start-pos', type=int, default=0)

    parser.add_argument('--max-workers', type=int, default=16)

    parser.add_argument('--start-id', type=int, default=30)
    parser.add_argument('--end-id', type=int, default=50)

    parser.add_argument('--online', action='store_true', default=False,
                        help='online collecting data')
    parser.add_argument('--save-score-arr', action='store_true', default=False,
                        help='save score arr or not')

    return parser.parse_args()


def read_gz_file(filename):
    with gzip.open(filename,'rb') as infile:
        data = np.load(infile, allow_pickle=False)

    return data


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


def compute_cdist(id, s_train, s_test, test_batch_id):
    t = time.time()
    logger.info(f'\t\t [compute_cdist-{id}-{test_batch_id}]: s_train.shape {s_train.shape}')
    logger.info(f'\t\t [compute_cdist-{id}-{test_batch_id}]: s_test.shape {s_test.shape}')
    dist = torch.cdist(torch.FloatTensor(s_train).view(len(s_train), -1), 
                       torch.FloatTensor(s_test).view(len(s_test), -1))
    logger.info(f'''\t\t [compute_cdist-{id}-{test_batch_id}]: computing cdist for id = {id} done '''
          f'''using {time.time() - t} seconds!''')
    return dist, id


def compute_full_cdist(s_train_all, s_test, test_batch_id, test_batch_size, group_num):
    # 1.1 computing cdist per group
    all_dist = []
    all_id = []
    size = 0
    logger.info('\t [compute_full_cdist] start computing per dist')
    t = time.time()

    for i in range(group_num):
        dist, id = compute_cdist(i,
                                 s_train_all[i], 
                                 s_test,
                                 test_batch_id)
        all_dist.append(dist)
        all_id.append(id)
        size += len(dist)

    # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    #     pool = [executor.submit(compute_cdist, i,
    #                             s_train_all[i][ind_train_all[i][action_id]], 
    #                             # s_test[ind_test[action_id]][test_batch_id*test_batch_size: (test_batch_id+1)*test_batch_size]
    #                             s_test,
    #                             action_id, test_batch_id
    #                             ) for i in range(group_num)]
    #     for i in concurrent.futures.as_completed(pool):
    #         dist, id = i.result()
    #         all_dist.append(dist)
    #         all_id.append(id)
    #         size += len(dist)

    logger.info(f'\t [compute_full_cdist-{test_batch_id}] computing per dist done using {time.time() - t} seconds!')

    # 1.2 compute necessary stats
    start_list = [0]
    start_pos = 0
    for i in range(group_num):
        start_pos += len(s_train_all[i])
        start_list.append(start_pos)
    assert start_list[-1] == size, f'start_list = {start_list}, last element not equal to full size which is {size}'
    logger.info(f'\t [compute_full_cdist-{test_batch_id}] computing stats for starting position done!')

    # 1.3 filling in full dist
    logger.info(f'\t [compute_full_cdist-{test_batch_id}] start filling in full dist!')
    t = time.time()
    full_dist = torch.zeros(size, dist.shape[1])
    for id, dist in zip(all_id, all_dist):
        ind_start = start_list[id]
        ind_end = start_list[id+1]
        full_dist[ind_start:ind_end] = dist
    logger.info(f'\t [compute_full_cdist-{test_batch_id}] filling in full dist using {time.time() - t} seconds!')
    return full_dist

def get_shapley_value(test_batch_id):
    s_test_cur = s_test[test_batch_id*args.test_batch_size: (test_batch_id+1)*args.test_batch_size]
    a_test_cur = a_test[test_batch_id*args.test_batch_size: (test_batch_id+1)*args.test_batch_size]
    test_batch_size = args.test_batch_size
    K = args.K

    # 1. computing full cdist
    t = time.time()
    dist = compute_full_cdist(s_train_all, s_test_cur, test_batch_id, test_batch_size, group_num)
    logger.info(f'\t [get_shapley_value-{test_batch_id}] computing full cdist done using {time.time() - t} seconds!')

    # 2. sort cdist and obtain indices
    t = time.time()
    _, indices = torch.sort(dist, axis=0)
    a_train = np.concatenate(a_train_all)
    a_sorted = a_train[indices]
    logger.info(f'\t [get_shapley_value-{test_batch_id}] indices: {indices.shape}')
    logger.info(f'\t [get_shapley_value-{test_batch_id}] y_train: {a_train.shape}, dist: {dist.shape}')
    logger.info(f'\t [get_shapley_value-{test_batch_id}] y_sorted: {a_sorted.shape}')
    logger.info(f'\t [get_shapley_value-{test_batch_id}] sorting done using {time.time() - t} seconds!')


    # 3. compute shapley values
    N = len(dist)
    # M = test_batch_size
    M = dist.shape[1]
    score = torch.zeros_like(dist)

    a_sorted_t = torch.FloatTensor(a_sorted)
    a_test_t = torch.FloatTensor(a_test_cur)

    t = time.time()
    # 3.1 initialize
    logger.info(f'\t [get_shapley_value-{test_batch_id}] start initializing the score array')

    # 3.2 iteratively update
    logger.info(f'\t [get_shapley_value-{test_batch_id}] start iteratively update the score array')
    score[indices[N-1], range(M)] = (a_sorted_t[N-1] == a_test_t).float() / N
    for i in tqdm(range(N-2, -1, -1)):
        score[indices[i], range(M)] = score[indices[i+1], range(M)] + \
                                    1/K * ((a_sorted_t[i] == a_test_t).float() - (a_sorted_t[i+1] == a_test_t).float()) * min(K, i+1) / (i+1)

    logger.info(f'\t [get_shapley_value-{test_batch_id}] computing the score array done in {time.time() - t} seconds!')

    if args.save_score_arr:
        score_file = osp.join(args.output_folder, f'shapley_scores_batch-{test_batch_id+start_batch}.pth')
        torch.save(score, score_file)
        logger.info(f'\t [get_shapley_value-{test_batch_id}] score array saved to {score_file}')
    return score.sum(axis=1)



args = get_arguments()
if not osp.exists(args.output_folder): os.makedirs(args.output_folder)
setup_logger('compute shapley (class)', os.path.join(args.output_folder, 'test.log'))
logger = logging.getLogger('compute shapley (class)')

logger.info(f'args = {args}')

# 1. loading training data

s_train_list = []
a_train_list = []
start = time.time()
for i in tqdm(range(args.start_id, args.end_id)):
    s_train_list.append(read_gz_file(osp.join(args.train_folder, f'hidden_{i}.gz')))
    a_train_list.append(read_gz_file(osp.join(args.train_data_folder, f'$store$_action_ckpt.{i}.gz')))

s_pos = len(a_train_list[0]) - len(s_train_list[0]) - 2
logger.info(f'beginning pos for trimming each training ckpt is {s_pos}')
for i in range(args.end_id-args.start_id):
    a_train_list[i] = a_train_list[i][s_pos: -2]

group_size = 2
group_num = (args.end_id - args.start_id) // group_size

logger.info(f'loading training data done using {time.time() - start} seconds!')


start = time.time()
s_train_all = [np.concatenate(s_train_list[i*group_size : (i+1)*group_size]) for i in range(group_num)]
a_train_all = [np.concatenate(a_train_list[i*group_size : (i+1)*group_size]) for i in range(group_num)]

logger.info(f'concatenating training data done using {time.time() - start} seconds!')

# 2. loading validation data
start = time.time()
valid_id = 0 if args.online else 49
s_test = read_gz_file(osp.join(args.valid_folder, f'hidden_{valid_id}.gz'))
a_test = read_gz_file(osp.join(args.valid_data_folder, f'$store$_action_ckpt.{valid_id}.gz'))

assert s_pos == len(a_test) - len(s_test) - 2, \
    f'starting pos for training = {s_pos}, while starting pos for testing = {len(a_test) - len(s_test) - 2}'
a_test = a_test[s_pos: -2]

logger.info(f'loading validation data done using {time.time() - start} seconds!')

# 2.0 handle continue learning 
s_test = s_test[args.test_start_pos:]
a_test = a_test[args.test_start_pos:]
start_batch = args.test_start_pos // args.test_batch_size
logger.info(f'test start pos = {args.test_start_pos}, start_batch = {start_batch}')


def main():

    # 4. compute shapley value

    paramlist = [(x, ) for x in range( ( args.test_size - 1 ) // args.test_batch_size + 1 )]

    def get_param():
        for param in paramlist:
            yield param

    scores = torch.zeros(functools.reduce(operator.add, [len(elem) for elem in a_train_all]))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for score in executor.map(get_shapley_value, *zip(*get_param())):
            scores += score

    filepath = osp.join(args.output_folder, 'scores.pth')
    torch.save(scores, filepath)
    logger.info(f'scores saved to {filepath}')


if __name__ == "__main__":
    main()