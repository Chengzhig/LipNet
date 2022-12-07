# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os

from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from .cvtransforms import *
import torch
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE

jpeg = TurboJPEG()


class LRWDataset(object):
    # index_file = f'E:/LRW/info/trn_1000_full.txt'
    # index_file = f'/home/mingwu/workspace_czg/LRW/lipread_mp4'
    index_file = f'/home/czg/lipread_mp4'
    with open('label_sorted.txt') as myfile:
        pinyins = myfile.read().splitlines()

    def __init__(self, phase, preprocessing_func=None):

        self.preprocessing_func = preprocessing_func
        self.labels = LRWDataset.pinyins
        self.list = []
        self.unlabel_list = []
        self.phase = phase

        for (i, label) in enumerate(self.labels):
            files = glob.glob(os.path.join(LRWDataset.index_file, label, phase, '*.pkl'))
            files = sorted(files)

            self.list += [file for file in files]

        self.load_dataset()

    def __getitem__(self, idx):

        tensor = torch.load(self.list[idx])
        preprocess_data = self.preprocessing_func(tensor)

        inputs = tensor.get('video')
        inputs = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in inputs]
        inputs = np.stack(inputs, 0) / 255.0
        inputs = inputs[:, :, :, 0]

        if (self.phase == 'train'):
            batch_img = RandomCrop(inputs, (88, 88))
            batch_img = HorizontalFlip(batch_img)
        elif self.phase == 'val' or self.phase == 'test':
            batch_img = CenterCrop(inputs, (88, 88))

        result = {}
        result['video'] = torch.FloatTensor(batch_img[:, np.newaxis, ...])
        # print(result['video'].size())
        result['label'] = tensor.get('label')
        result['duration'] = 1.0 * tensor.get('duration')

        # word = self.labels[int(result['label'])]

        # pinyinlable = np.full((40), 0).astype(result['pinyinlable'].dtype)
        # t = 0
        # for i in word:
        #     pinyinlable[t] = int(i - 'A')
        #     t += 1
        # result['pinyinlable'] = pinyinlable

        return result

    def __len__(self):
        return len(self.list)

    def load_duration(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if (line.find('Duration') != -1):
                    duration = float(line.split(' ')[1])

        tensor = torch.zeros(29)
        mid = 29 / 2
        start = int(mid - duration / 2 * 25)
        end = int(mid + duration / 2 * 25)
        tensor[start:end] = 1.0
        return tensor


if __name__ == '__main__':
    # local
    # target_dir = f'E:/LRW1000_Public_pkl_jpeg/trn'
    index_file = f'E:/LRW_landmarks/LRW_landmarks'
    # 3080
    # target_dir = f'/home/mingwu/workspace_czg/data/LRW/LRW1000_Public_pkl_jpeg/trn'
    # index_file = f'/home/mingwu/workspace_czg/data/LRW/LRW/info/trn_1000.txt'
    # 3090
    # target_dir = f'/home/czg/dataset/LRW1000_Public_pkl_jpeg/trn'
    # index_file = f'/home/czg/dataset/LRW/info/trn_1000.txt'

    dataset = LRWDataset('train', index_file)
    print('Start running, Data Length:', len(dataset))
    loader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=1,
                        shuffle=False,
                        drop_last=False)

    tic = time.time()
    for i_iter, input in enumerate(loader):
        toc = time.time()
        eta = ((toc - tic) / (i_iter + 1) * (len(loader) - i_iter)) / 3600.0
        print(f'eta:{eta:.5f}')
