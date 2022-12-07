import numpy as np
import torch

from processing import Compose, Normalize, RandomCrop, HorizontalFlip, TimeMask, CenterCrop, AddNoise, \
    NormalizeUtterance
from utils.dset import MyDataset, pad_packed_collate


def get_preprocessing_pipelines(modality):
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    if modality == 'video':
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        preprocessing['train'] = Compose([
            Normalize(0.0, 255.0),
            RandomCrop(crop_size),
            HorizontalFlip(0.5),
            Normalize(mean, std),
            TimeMask(T=0.6 * 25, n_mask=1)
        ])

        preprocessing['val'] = Compose([
            Normalize(0.0, 255.0),
            CenterCrop(crop_size),
            Normalize(mean, std)])

        preprocessing['test'] = preprocessing['val']

    elif modality == 'audio':

        preprocessing['train'] = Compose([
            AddNoise(noise=np.load('./data/babbleNoise_resample_16K.npy')),
            NormalizeUtterance()])

        preprocessing['val'] = NormalizeUtterance()

        preprocessing['test'] = NormalizeUtterance()

    return preprocessing


def get_data_loaders(args):
    preprocessing = get_preprocessing_pipelines(args.modality)

    # create dataset object for each partition
    partitions = ['test'] if args.test else ['train', 'val', 'test']
    dsets = {partition: MyDataset(modality=args.modality,
                                  data_partition=partition,
                                  data_dir=f'/home/czg/lipread_mp4',
                                  label_fp='label_sorted.txt',
                                  annonation_direc=f'/home/czg/lipread_mp4',
                                  preprocessing_func=preprocessing[partition],
                                  data_suffix='.npz',
                                  use_boundary=True, ) for partition in partitions}

    dset_loaders = {x: torch.utils.data.DataLoader(
        dsets[x],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_packed_collate,
        pin_memory=True,
        num_workers=args.num_workers,
        worker_init_fn=np.random.seed(1)) for x in partitions}
    return dset_loaders
