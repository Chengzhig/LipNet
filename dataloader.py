import numpy as np
import torch

from processing import Compose, Normalize, RandomCrop, HorizontalFlip, TimeMask, CenterCrop
from utils.dataset_lrw1000 import pad_packed_collate


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


def get_data_loaders(args):
    preprocessing = get_preprocessing_pipelines(args.modality)

    if (args.dataset == 'lrw'):
        print("lrw")
        from utils import LRWDataset as Dataset
    elif (args.dataset == 'lrw1000'):
        print("lrw1000")
        from utils import LRW1000_Dataset as Dataset
    else:
        raise Exception('lrw or lrw1000')

    # create dataset object for each partition
    partitions = ['test'] if args.test else ['train', 'val', 'test']
    dsets = {partition: Dataset(
        modality=args.modality,
        data_partition=partition,
        data_dir=args.data_dir,
        label_fp=args.label_path,
        annonation_direc=args.annonation_direc,
        preprocessing_func=preprocessing[partition],
        use_boundary=args.use_boundary,
    ) for partition in partitions}
    dset_loaders = {x: torch.utils.data.DataLoader(
        dsets[x],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_packed_collate,
        pin_memory=True,
        num_workers=args.workers,
        worker_init_fn=np.random.seed(1)) for x in partitions}
    return dset_loaders
