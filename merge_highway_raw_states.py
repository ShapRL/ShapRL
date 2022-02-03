import argparse
import concurrent.futures
import glob
import gzip
import numpy as np
import os
import os.path as osp
import shutil
import time
import torch
from tqdm import tqdm
from multiprocessing import Pool
from helper import read_gz_file, write_gz_file

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', type=str, default='')
    parser.add_argument('--output-folder', type=str, default='')
    return parser.parse_args()


def merge_buffer(args, folder, batch_id=0):

    all_state = np.empty((0, 5, 5), dtype=np.float)
    for i in tqdm(range(batch_id*100, (batch_id+1)*100)):
        all_state = np.vstack((all_state, read_gz_file(osp.join(folder, f'$store$_observation_ckpt.{i}.gz'))))

    all_state = np.reshape(all_state, (len(all_state), -1))

    print(f'[batch_id={batch_id}] all_state: \t {all_state.shape} {all_state.dtype}')

    t = time.time()
    write_gz_file(all_state, osp.join(args.output_folder, f'hidden_{batch_id}.gz'))

    print(f'[batch_id={batch_id}] writing done using {time.time() - t} seconds!')

    return batch_id


def merge_all(args, folder):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        pool = [executor.submit(merge_buffer, args, folder, batch_id) for batch_id in range(4)]
        for i in concurrent.futures.as_completed(pool):
            print(f'batch {i.result()} finishes!')


def main():
    args = get_arguments()
    if not osp.exists(args.output_folder): os.makedirs(args.output_folder)

    merge_all(args, args.folder)


if __name__ == "__main__":
    main()