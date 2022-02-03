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

    all_state = []
    for i in tqdm(range(batch_id*100, (batch_id+1)*100)):
        all_state.append(torch.load(osp.join(folder, f'processed_data_buffer-{i}_batch-0.pth'))['X'])

    all_state = np.concatenate(all_state)

    print(f'[batch_id={batch_id}] all_state: \t {all_state.shape} {all_state.dtype}')

    t = time.time()
    write_gz_file(all_state, osp.join(folder, f'hidden_{batch_id}.gz'))
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