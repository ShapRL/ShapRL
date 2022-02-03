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

def write_gz_file(data, filename):
    with gzip.open(filename, 'wb') as outfile:
        np.save(outfile, data, allow_pickle=False)

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', type=str, default='')
    parser.add_argument('--online', action='store_true', default=False,
                        help='online collecting data')
    parser.add_argument('--start-id', type=int, default=40)
    parser.add_argument('--end-id', type=int, default=50)
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='pretrained feature extractor')
    return parser.parse_args()


def merge_buffer(args, folder, batch_id=0):
    buffer_num = len(glob.glob(osp.join(args.folder, f"processed_data_buffer-{batch_id}*.pth")))
    print(f'counting processed_data_buffer-{batch_id}*.pth file num = {buffer_num}!')

    X_list = []
    y_list = []
    a_list = []
    for i in tqdm(range(buffer_num)):
        data = torch.load(osp.join(folder, f'processed_data_buffer-{batch_id}_batch-{i}.pth'))
        if 'X' in data.keys():
            X_list.append(data['X'])
        if not args.pretrained:
            y_list.append(data['y'])
        if 'a' in data.keys():
            a_list.append(data['a'])

    if 'X' in data.keys():
        X_arr = np.concatenate(X_list)
        print(f'[batch_id={batch_id}] X_arr: \t {X_arr.shape} {X_arr.dtype}')

    if not args.pretrained:
        y_arr = np.concatenate(y_list)
        print(f'[batch_id={batch_id}] y_arr: \t {y_arr.shape} {y_arr.dtype}')

    if 'a' in data.keys():
        a_arr = np.concatenate(a_list)

    t = time.time()
    if 'X' in data.keys():
        write_gz_file(X_arr, osp.join(folder, f'hidden_{batch_id}.gz'))

    if not args.pretrained:
        write_gz_file(y_arr, osp.join(folder, f'target_{batch_id}.gz'))

    if 'a' in data.keys():
        write_gz_file(a_arr, osp.join(folder, f'action_{batch_id}.gz'))

    print(f'[batch_id={batch_id}] writing done using {time.time() - t} seconds!')

    for i in tqdm(range(buffer_num)):
        os.remove(osp.join(folder, f'processed_data_buffer-{batch_id}_batch-{i}.pth'))
    print(f'[batch_id={batch_id}] removing torch data buffers done!')

    print(f'[batch_id={batch_id}] processing finished!')

    return batch_id


def merge_all(args, folder, start_id, end_id):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        pool = [executor.submit(merge_buffer, args, folder, batch_id) for batch_id in range(start_id, end_id)]
        for i in concurrent.futures.as_completed(pool):
            print(f'batch {i.result()} finishes!')


def main():
    args = get_arguments()

    if args.online:
        merge_buffer(args, args.folder, batch_id=0)

    else:
        merge_all(args, args.folder, args.start_id, args.end_id)


if __name__ == "__main__":
    main()