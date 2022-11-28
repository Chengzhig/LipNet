import numpy
from PIL import Image, ImageEnhance, ImageFilter
from torch import permute
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import glob
import numpy as np
import random

from torchvision import transforms
from turbojpeg import TJPF_GRAY, TurboJPEG

from .cvtransforms import *
import torch
from collections import defaultdict

jpeg = TurboJPEG()


class LRW1000_Dataset(Dataset):
    # index_file = f'E:/LRW/info/trn_1000.txt'
    index_file = f'/home/mingwu/workspace_czg/LRW/info/trn_1000.txt'
    # index_file = f'/home/czg/trn_1000.txt'
    lines = []
    with open(index_file, 'r', encoding="utf-8") as f:
        lines.extend([line.strip().split(',') for line in f.readlines()])
    pinyins = sorted(np.unique([line[2] for line in lines]))

    def __init__(self, phase, args):

        self.args = args
        self.data = []
        self.phase = phase

        if (self.phase == 'train'):
            # local
            # self.index_root = 'E:/LRW1000_Public_pkl_jpeg/trn'
            # 3080
            self.index_root = '/home/mingwu/workspace_czg/LRW/LRW1000_Public_pkl_jpeg/trn'
            # 3090
            # self.index_root = '/home/czg/workspace_chj/LipNet/LRW1000_Public_pkl_jpeg/trn'
            # self.index_root = '/home/czg/dataset/LRW1000_Phome/trn'
        else:
            # local
            # self.index_root = 'E:/LRW1000_Public_pkl_jpeg/trn'
            # 3080
            self.index_root = '/home/mingwu/workspace_czg/LRW/LRW1000_Public_pkl_jpeg/tst'
            # 3090
            # self.index_root = '/home/czg/workspace_chj/LipNet/LRW1000_Public_pkl_jpeg/tst'
            # self.index_root = '/home/czg/dataset/LRW1000_Phome/tst'

        self.data = glob.glob(os.path.join(self.index_root, '*.pkl'))

        self.PhonemeList = ['C', 'a', 'ai', 'ji', 'an', 'jian',
                            'quan', 'zhao', 'ba', 'li', 'xi',
                            'bai', 'ban', 'dao', 'fa', 'bang',
                            'jia', 'bao', 'chi', 'gao', 'hu',
                            'kuo', 'yu', 'zhang', 'bei', 'bu',
                            'jing', 'shi', 'yue', 'ben', 'ci',
                            'bi', 'jiao', 'ru', 'xu', 'bian',
                            'hua', 'biao', 'da', 'zhi', 'zhun',
                            'bie', 'bing', 'qie', 'bo', 'chu',
                            'duan', 'fen', 'guo', 'hui', 'jin',
                            'men', 'neng', 'shao', 'shu', 'tong',
                            'yao', 'cai', 'fang', 'qu', 'can',
                            'ce', 'ceng', 'chan', 'pin', 'sheng',
                            'ye', 'chang', 'qi', 'yi', 'chao',
                            'xian', 'che', 'cheng', 'gong', 'nuo',
                            'wei', 'lai', 'le', 'chuan', 'chuang',
                            'xin', 'chun', 'qian', 'cong', 'cu',
                            'cun', 'zai', 'cuo', 'gai', 'xing',
                            'xue', 'zao', 'dai', 'dan', 'dang',
                            'di', 'tian', 'zhong', 'de', 'deng',
                            'dian', 'diao', 'cha', 'yan', 'dong',
                            'dou', 'du', 'dui', 'wai', 'duo',
                            'nian', 'e', 'luo', 'si', 'er',
                            'ling', 'liu', 'san', 'wu', 'ma',
                            'she', 'ren', 'yuan', 'zhan', 'fan',
                            'rong', 'zui', 'mian', 'wen',
                            'xiang', 'fei', 'zi', 'feng', 'shuo',
                            'fu', 'ze', 'ge', 'shan', 'gan',
                            'jue', 'shou', 'xie', 'gang', 'xiao',
                            'jie', 'gei', 'gen', 'ju', 'geng',
                            'hao', 'he', 'kai', 'min', 'you',
                            'zuo', 'gou', 'guan', 'zhu',
                            'guang', 'gui', 'ding', 'zhou', 'nei',
                            'ha', 'hai', 'shang', 'han', 'nan',
                            'ping', 'hen', 'hou', 'lian', 'wang',
                            'ti', 'huan', 'ying', 'huang', 'tan',
                            'huo', 'zhe', 'jiang', 'lu', 'tuan',
                            'bin', 'qiang', 'kang', 'su', 'mu',
                            'xia', 'ri', 'zhuan', 'shen', 'jiu',
                            'jun', 'ka', 'ta', 'kan', 'kao',
                            'ke', 'kong', 'kuai', 'la', 'lan',
                            'lang', 'lao', 'lei', 'liang', 'yong',
                            'liao', 'lin', 'chen', 'long', 'lou',
                            'lun', 'mao', 'mei', 'meng', 'mi',
                            'ming', 'que', 'mo', 'n', 'na', 'me',
                            'ne', 'ni', 'qing', 'nin', 'nu', 'ou',
                            'peng', 'pi', 'pian', 'tai', 'pu',
                            'lie', 'qiao', 'kuang', 'qiu', 'ran',
                            'rang', 're', 'reng', 'sao', 'miao',
                            'sen', 'sha', 'te', 'gu', 'shuang',
                            'shui', 'sou', 'suo', 'sui', 'wan',
                            'tao', 'tiao', 'zheng', 'tie', 'ting',
                            'tou', 'piao', 'tu', 'po', 'tui',
                            'wo', 'ya', 'xuan', 'yang', 'yin',
                            'hang', 'yun', 'zan', 'zen', 'zeng',
                            'lve', 'zhei', 'zhen', 'zu', 'zhua',
                            'zhuang', 'xun', 'zong', 'zou', 'zun']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pkl = torch.load(self.data[idx])
        st = -1
        ed = 39
        for i, d in enumerate(pkl['duration']):
            if st == -1 and d == 1:
                st = i
            if st != -1 and d == 0:
                ed = i
                break

        video = pkl.get('video')
        video = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in video]
        video = np.stack(video, 0)
        video = video[:, :, :, 0]
        if (self.phase == 'train'):
            video = RandomCrop(video, (88, 88))
            video = HorizontalFlip(video)
        elif self.phase == 'val' or self.phase == 'test':
            video = CenterCrop(video, (88, 88))
        pkl['video'] = torch.FloatTensor(video)[:, None, ...] / 255.0
        # videoTensor = torch.FloatTensor(video)[:, None, ...] / 255.0
        # for i in range(40):
        #     mean_video = torch.mean(videoTensor[i, :, :])
        #     std_video = torch.std(videoTensor[i, :, :])
        #     if mean_video != torch.zeros(1) and std_video != torch.zeros(1):
        #         videoTensor[i, :, :] -= mean_video
        #         videoTensor[i, :, :] /= std_video
        # pkl['video'] = videoTensor

        pinyinlable = np.full((1), 28).astype(pkl['pinyinlable'].dtype)

        # try:
        #     t = pkl['pinyinlable'].shape[0]
        #     pinyinlable[:t, ...] = pkl['pinyinlable'].copy()
        # except Exception as e:  # 可以写多个捕获异常
        #     print("ValueError")
        pkl['pinyinlable'] = pinyinlable

        return pkl


if __name__ == '__main__':
    # local
    # target_dir = f'E:/LRW1000_Public_pkl_jpeg/trn'
    # index_file = f'E:/LRW/info/trn_1000.txt'
    # 3080
    target_dir = f'/home/mingwu/workspace_czg/data/LRW/LRW1000_Public_pkl_jpeg/trn'
    index_file = f'/home/mingwu/workspace_czg/data/LRW/LRW/info/trn_1000.txt'
    # 3090
    # target_dir = f'/home/czg/dataset/LRW1000_Public_pkl_jpeg/trn'
    # index_file = f'/home/czg/dataset/LRW/info/trn_1000.txt'

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
