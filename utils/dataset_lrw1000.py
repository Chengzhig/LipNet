from torch.utils.data import Dataset, DataLoader
import os
import glob
from turbojpeg import TJPF_GRAY, TurboJPEG

from .cvtransforms import *
import torch

jpeg = TurboJPEG()


class LRW1000_Dataset(Dataset):
    # index_file = f'E:/LRW/info/trn_1000_full.txt'
    # index_file = f'/home/mingwu/workspace_czg/LRW/info/trn_1000.txt'
    index_file = f'/home/czg/trn_1000.txt'
    lines = []
    with open(index_file, 'r', encoding="utf-8") as f:
        lines.extend([line.strip().split(',') for line in f.readlines()])
    pinyins = sorted(np.unique([line[2] for line in lines]))

    def __init__(self, phase, args, preprocessing_func=None):

        self.args = args
        self.data = []
        self.phase = phase
        self.preprocessing_func = preprocessing_func

        if (self.phase == 'train'):
            # local
            # self.index_root = 'E:/LRW1000_Public_pkl_jpeg/trn'
            # 3080
            # self.index_root = '/home/mingwu/workspace_czg/LRW/LRW1000_Public_pkl_jpeg/trn'
            # 3090
            self.index_root = '/home/czg/workspace_chj/LipNet/LRW1000_Public_pkl_jpeg/trn'
        else:
            # local
            # self.index_root = 'E:/LRW1000_Public_pkl_jpeg/trn'
            # 3080
            # self.index_root = '/home/mingwu/workspace_czg/LRW/LRW1000_Public_pkl_jpeg/tst'
            # 3090
            self.index_root = '/home/czg/workspace_chj/LipNet/LRW1000_Public_pkl_jpeg/tst'

        self.data = glob.glob(os.path.join(self.index_root, '*.pkl'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pkl = torch.load(self.data[idx])

        preprocess_data = self.preprocessing_func(pkl)
        label = self.data[idx][1]

        return preprocess_data, label, pkl['during']

        # video = pkl.get('video')
        # video = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in video]
        # video = np.stack(video, 0)
        # video = video[:, :, :, 0]
        # if (self.phase == 'train'):
        #     video = RandomCrop(video, (88, 88))
        #     video = HorizontalFlip(video)
        # elif self.phase == 'val' or self.phase == 'test':
        #     video = CenterCrop(video, (88, 88))
        # pkl['video'] = torch.FloatTensor(video)[:, None, ...] / 255.0
        #
        # pinyinlable = np.full((1), 28).astype(pkl['pinyinlable'].dtype)
        #
        # pkl['pinyinlable'] = pinyinlable
        #
        # return pkl


def pad_packed_collate(batch):
    if len(batch[0]) == 2:
        use_boundary = False
        data_tuple, lengths, labels_tuple = zip(
            *[(a, a.shape[0], b) for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])
    elif len(batch[0]) == 3:
        use_boundary = True
        data_tuple, lengths, labels_tuple, boundaries_tuple = zip(
            *[(a, a.shape[0], b, c) for (a, b, c) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])

    if data_tuple[0].ndim == 1:
        max_len = data_tuple[0].shape[0]
        data_np = np.zeros((len(data_tuple), max_len))
    elif data_tuple[0].ndim == 3:
        max_len, h, w = data_tuple[0].shape
        data_np = np.zeros((len(data_tuple), max_len, h, w))
    for idx in range(len(data_np)):
        data_np[idx][:data_tuple[idx].shape[0]] = data_tuple[idx]
    data = torch.FloatTensor(data_np)

    if use_boundary:
        boundaries_np = np.zeros((len(boundaries_tuple), len(boundaries_tuple[0])))
        for idx in range(len(data_np)):
            boundaries_np[idx] = boundaries_tuple[idx]
        boundaries = torch.FloatTensor(boundaries_np).unsqueeze(-1)

    labels = torch.LongTensor(labels_tuple)

    if use_boundary:
        return data, lengths, labels, boundaries
    else:
        return data, lengths, labels



if __name__ == '__main__':
    # local
    # target_dir = f'E:/LRW1000_Public_pkl_jpeg/trn'
    # index_file = f'E:/LRW/info/trn_1000.txt'
    # 3080
    # target_dir = f'/home/mingwu/workspace_czg/data/LRW/LRW1000_Public_pkl_jpeg/trn'
    # index_file = f'/home/mingwu/workspace_czg/data/LRW/LRW/info/trn_1000.txt'
    # 3090
    target_dir = f'/home/czg/dataset/LRW1000_Public_pkl_jpeg/trn'
    index_file = f'/home/czg/dataset/LRW/info/trn_1000.txt'

    dataset = LRW1000_Dataset('train', index_file)
    print('Start running, Data Length:', len(dataset))
    loader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=1,
                        shuffle=False,
                        drop_last=False)

    for i_iter, input in enumerate(loader):
        video = input.get('video')
        # video = torch.FloatTensor(video)[:, None, ...] * 255.0
        video = torch.squeeze(video, dim=0)
        video = torch.squeeze(video, dim=0)
        label = input.get('label')
        for i in range(video.shape[0]):
            image = video[i, :, :, :]
            # image = image[0, :, :, :]
            # image = image.permute(2, 0, 1)
            # image = transforms.ToPILImage()(image)
            # image.show()
            #
            # im1 = image.filter(ImageFilter.SHARPEN)
            # im1.show()
            # # 亮度增强
            # enh_bri = ImageEnhance.Brightness(image)
            # brightness = 3
            # image_brightened = enh_bri.enhance(brightness)
            # image_brightened.show()
            # im0 = image_brightened.filter(ImageFilter.EDGE_ENHANCE)
            # im0.show()
            # # 色度增强(饱和度↑)
            # enh_col = ImageEnhance.Color(image)
            # color = 2
            # image_colored = enh_col.enhance(color)
            # image_colored.show()
            # # 对比度增强
            # enh_con = ImageEnhance.Contrast(image)
            # contrast = 5
            # image_contrasted = enh_con.enhance(contrast)
            # image_contrasted.show()
            # # 锐度增强
            # enh_sha = ImageEnhance.Sharpness(image)
            # sharpness = 4.0
            # image_sharped = enh_sha.enhance(sharpness)
            # image_sharped.show()
            # break
