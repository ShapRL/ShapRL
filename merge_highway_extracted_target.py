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
    return parser.parse_args()


def merge_buffer(args, folder, batch_id=0):

    all_target = []
    for i in tqdm(range(batch_id*100, (batch_id+1)*100)):
        all_target.append(torch.load(osp.join(folder, f'processed_data_buffer-{i}_batch-0.pth'))['y'])

    all_target = np.concatenate(all_target)

    print(f'[batch_id={batch_id}] all_target: \t {all_target.shape} {all_target.dtype}')

    t = time.time()
    write_gz_file(all_target, osp.join(folder, f'target_{batch_id}.gz'))
    print(f'[batch_id={batch_id}] writing done using {time.time() - t} seconds!')


    for i in tqdm(range(batch_id*100, (batch_id+1)*100)):
        os.remove(osp.join(folder, f'processed_data_buffer-{i}_batch-0.pth'))
    print(f'[batch_id={batch_id}] removing buffers done!')

    return batch_id


def merge_all(args, folder):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        pool = [executor.submit(merge_buffer, args, folder, batch_id) for batch_id in range(3, 4)]
        for i in concurrent.futures.as_completed(pool):
            print(f'batch {i.result()} finishes!')


def main():
    args = get_arguments()

    merge_all(args, args.folder)


if __name__ == "__main__":
    main()