import argparse
from efficientnet_pytorch import EfficientNet
import logging
import os
import os.path as osp
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms as T
import torchvision.models as models

from embedding.extractor import MobileNetFeatureExtractor, VGGFeatureExtractor, EfficientNetExtractor, InceptionFeatureExtractor

from helper import read_gz_file, setup_logger

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extractor', type=str, default='vgg11',
        choices=['vgg11', 'mobilenet', 'resnet18', 'efficientnet', 'inception'])
    parser.add_argument('--input-folder', type=str, default='')
    parser.add_argument('--output-folder', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=5000)
    parser.add_argument('--start-id', type=int, default=0)
    parser.add_argument('--end-id', type=int, default=1)
    return parser.parse_args()


class CustomDataset(Dataset):
    def __init__(self, input_folder, buffer_id):
        t = time.time()
        file_name = osp.join(osp.join(input_folder, 'replay_logs'), f'$store$_observation_ckpt.{buffer_id}.gz')
        self.data = read_gz_file(file_name)

        if buffer_id == 0:
            self.data = self.data[3:-2]
        else:
            self.data = self.data[2:-2]

        self.transform = T.Compose([T.ToTensor(), T.ToPILImage(), T.Resize(256), T.CenterCrop(224), T.Grayscale(3), T.ToTensor()])
        logger.info(f'reading observation file {file_name} takes {time.time() - t} seconds!')
        logger.info(f'data shape before transformation: {self.data.shape}')
        # logger.info(f'sample shape after transformation: {self.transform(self.data[0]).shape}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx])


def get_extractor(extractor_name):
    if extractor_name == 'vgg11':
        vgg11 = models.vgg11(pretrained=True)
        return VGGFeatureExtractor(vgg11), 500

    if extractor_name == 'mobilenet':
        mobilenet = models.mobilenet_v2(pretrained=True)
        return MobileNetFeatureExtractor(mobilenet), 1000

    if extractor_name == 'resnet18':
        resnet18 = models.resnet18(pretrained=True)
        return nn.Sequential(*list(resnet18.children())[:-1]), 1000

    if extractor_name == 'efficientnet':
        efficientB7 = EfficientNet.from_pretrained('efficientnet-b7')
        return EfficientNetExtractor(efficientB7), 50

    if extractor_name == 'inception':
        inception = models.inception_v3(pretrained=True)
        return InceptionFeatureExtractor(inception), 1000

    raise NotImplementedError(f'extractor = {extractor_name} not implemented!')


def create_embeddings(args, model_fc, ds, parallel=True, buffer_id=0):
    """
    Takes in a feature extraction model and a dataset and stores embeddings in npz.
    model_fc: feature extraction model
    ds: dataloader
    model_name: name of the model
    storage_path: where to store embeddings
    storage_size: vector size of each .npz file
    parallel: enables data parallel modus
    """    

    feature_extractor = model_fc
    if parallel:
        feature_extractor = nn.DataParallel(model_fc)
    target_dataset = ds

    params = {'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': 6,
            'pin_memory': False}
    dataloader = DataLoader(target_dataset, **params)

    # print(f"Moving model to {device}")
    feature_extractor.cuda()
    feature_extractor.eval()

    with torch.no_grad():
        for idx, X in enumerate(tqdm(dataloader)):
            X = X.cuda()
            feature = feature_extractor(X)
            feature = feature.detach().cpu().view(len(X), -1).numpy()
            torch.save({'X': feature}, osp.join(args.output_folder, f'processed_data_buffer-{buffer_id}_batch-{idx}.pth'))


args = get_arguments()

if not osp.exists(args.output_folder): os.makedirs(args.output_folder)
setup_logger('generate embeddings by pretrained models', os.path.join(args.output_folder, 'test.log'))
logger = logging.getLogger('generate embeddings by pretrained models')
logger.info(f'args = {args}')

extractor, storage_size = get_extractor(args.extractor)
logger.info(f'loading pre-trained extractor {args.extractor} done!')

for buffer_id in tqdm(range(args.start_id, args.end_id)):
    dataset = CustomDataset(args.input_folder, buffer_id)
    logger.info(f'loading input dataset of buffer_id = {buffer_id} done!')

    t = time.time()
    create_embeddings(args, extractor, dataset, buffer_id=buffer_id)
    logger.info(f'creating embeddings for buffer_id = {buffer_id} and saving done using {time.time() - t} seconds!')