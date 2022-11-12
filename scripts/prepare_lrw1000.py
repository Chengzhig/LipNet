from torch.utils.data import Dataset
import cv2
import os
import glob
import numpy as np
import random
import os
import scipy.io.wavfile as wav
import numpy as np
import torch
from collections import defaultdict
import sys
from torch.utils.data import DataLoader
# from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
from turbojpeg import TurboJPEG

jpeg = TurboJPEG()


class LRW1000_Dataset(Dataset):

    def __init__(self, index_file, target_dir):

        self.data = []
        self.index_file = index_file
        self.target_dir = target_dir
        lines = []

        with open(index_file, 'r', encoding="utf-8") as f:
            lines.extend([line.strip().split(',') for line in f.readlines()])
        # local
        self.data_root = 'E:\LRW\images\LRW1000_Public\images'
        # 3080
        # self.data_root = '/home/mingwu/workspace_czg/data/LRW/LRW/images/LRW1000_Public/images'
        # 3090
        # self.data_root = '/home/czg/dataset/LRW/images/LRW1000_Public/images'
        #
        self.padding = 40

        pinyins = sorted(np.unique([line[2] for line in lines]))
        self.data = [(line[0], int(float(line[3]) * 25) + 1, int(float(line[4]) * 25) + 1, pinyins.index(line[2])) for
                     line in lines]
        max_len = max([data[2] - data[1] for data in self.data])
        data = list(filter(lambda data: data[2] - data[1] <= self.padding, self.data))
        self.lengths = [data[2] - data[1] for data in self.data]
        self.pinyins = pinyins
        print(pinyins)

        self.labels = []
        for i in pinyins:
            self.labels = self.labels + i.split(' ')
        unique_set = set(char for label in self.labels for char in label)
        self.map = sorted(list(unique_set))
        self.characters = sorted(list(unique_set)) + [' '] + ['-']
        self.myclass_len = len(self.characters)
        self.char_to_num = dict((c, i) for i, c in enumerate(self.characters))
        self.num_to_char = dict((i, c) for i, c in enumerate(self.characters))

        SinglePinyinList = []
        for line in pinyins:
            for i in line.split(' '):
                if i not in SinglePinyinList:
                    SinglePinyinList.append(i)

        # '''方法1'''
        # 单韵母:a 0 o 1 e 2 i 3 u 4 v 5
        # 复韵母:ai 6 ei 7 ui 8 ao 9 ou 10 iu 11 ie 12 ve 13 er 14
        # 前鼻韵母:an 15 en 16 in 17 un 18 vn 19
        # 后鼻韵母:ang 20 eng 21 ing 22 ong 23
        # 声母:b 24 p 25 m 26 f 27 d 28 t 29 n 30 l 31 g 32 k 33 h 34 j 35 q 36 x 37
        #       zh 38 ch 39 sh 40 r 41 z 42 c 43 s 44 y 45 w 46
        # 整体认读:zhi 47 chi 48 shi 49 ri 49 zi 50 ci 51 si 52 yi 53 wu 54
        #       yu 55  yue 56 yin 57 yun 58 yuan 59 ying 60 C 61 ye 62
        #

        # self.PhonemeList = {'C': [61], 'a': [0], 'ai': [6], 'ji': [35, 3], 'an': [15], 'jian': [35, 3, 15],
        #                     'quan': [36, 4, 15], 'zhao': [38, 9], 'ba': [24, 0], 'li': [31, 3], 'xi': [37, 3],
        #                     'bai': [24, 6], 'ban': [24, 15], 'dao': [28, 9], 'fa': [27, 0], 'bang': [24, 20],
        #                     'jia': [35, 3, 0], 'bao': [24, 9], 'chi': [48, 39], 'gao': [32, 9], 'hu': [34, 4],
        #                     'kuo': [33, 4, 1], 'yu': [45, 4, 55], 'zhang': [38, 20], 'bei': [24, 7], 'bu': [24, 4],
        #                     'jing': [35, 22], 'shi': [49, 40], 'yue': [56], 'ben': [24, 16], 'ci': [51, 43],
        #                     'bi': [24, 3], 'jiao': [35, 3, 9], 'ru': [41, 4], 'xu': [37, 4], 'bian': [24, 3, 15],
        #                     'hua': [34, 4, 0], 'biao': [24, 3, 9], 'da': [28, 0], 'zhi': [47, 38], 'zhun': [38, 18],
        #                     'bie': [24, 12], 'bing': [24, 22], 'qie': [36, 12], 'bo': [24, 1], 'chu': [39, 4],
        #                     'duan': [28, 4, 15], 'fen': [27, 16], 'guo': [32, 4, 1], 'hui': [34, 7], 'jin': [35, 17],
        #                     'men': [26, 16], 'neng': [30, 21], 'shao': [40, 9], 'shu': [40, 4], 'tong': [29, 23],
        #                     'yao': [45, 9], 'cai': [43, 6], 'fang': [27, 20], 'qu': [36, 4], 'can': [43, 15],
        #                     'ce': [43, 2], 'ceng': [43, 21], 'chan': [39, 15], 'pin': [25, 17], 'sheng': [40, 21],
        #                     'ye': [62], 'chang': [39, 20], 'qi': [36, 3], 'yi': [53, 45], 'chao': [39, 9],
        #                     'xian': [37, 3, 15], 'che': [39, 2], 'cheng': [39, 21], 'gong': [32, 23], 'nuo': [30, 4, 1],
        #                     'wei': [46, 7], 'lai': [31, 6], 'le': [31, 2], 'chuan': [39, 4, 15], 'chuang': [39, 4, 20],
        #                     'xin': [37, 17], 'chun': [39, 18], 'qian': [36, 3, 15], 'cong': [43, 23], 'cu': [43, 4],
        #                     'cun': [43, 18], 'zai': [42, 6], 'cuo': [43, 4, 1], 'gai': [32, 6], 'xing': [37, 22],
        #                     'xue': [37, 13], 'zao': [42, 9], 'dai': [28, 6], 'dan': [28, 15], 'dang': [28, 20],
        #                     'di': [28, 3], 'tian': [29, 3, 15], 'zhong': [38, 23], 'de': [28, 2], 'deng': [28, 21],
        #                     'dian': [28, 3, 15], 'diao': [28, 3, 9], 'cha': [39, 0], 'yan': [45, 15], 'dong': [28, 23],
        #                     'dou': [28, 10], 'du': [28, 4], 'dui': [28, 8], 'wai': [46, 6], 'duo': [28, 4, 1],
        #                     'nian': [30, 3, 15], 'e': [2], 'luo': [31, 4, 1], 'si': [52, 44], 'er': [14],
        #                     'ling': [31, 22], 'liu': [31, 11], 'san': [44, 15], 'wu': [54], 'ma': [26, 0],
        #                     'she': [40, 2], 'ren': [41, 16], 'yuan': [59], 'zhan': [38, 15], 'fan': [27, 15],
        #                     'rong': [41, 23], 'zui': [42, 7], 'mian': [26, 3, 15], 'wen': [46, 16],
        #                     'xiang': [37, 3, 20], 'fei': [27, 7], 'zi': [50, 42], 'feng': [27, 21], 'shuo': [40, 4, 1],
        #                     'fu': [27, 4], 'ze': [42, 2], 'ge': [32, 2], 'shan': [40, 15], 'gan': [32, 15],
        #                     'jue': [35, 13], 'shou': [40, 10], 'xie': [37, 12], 'gang': [32, 20], 'xiao': [37, 3, 9],
        #                     'jie': [35, 12], 'gei': [32, 7], 'gen': [32, 16], 'ju': [35, 4], 'geng': [32, 21],
        #                     'hao': [34, 9], 'he': [34, 2], 'kai': [33, 6], 'min': [26, 17], 'you': [45, 10],
        #                     'zuo': [42, 4, 1], 'gou': [32, 10], 'guan': [32, 4, 15], 'zhu': [38, 4],
        #                     'guang': [32, 4, 20], 'gui': [32, 8], 'ding': [28, 22], 'zhou': [38, 10], 'nei': [30, 7],
        #                     'ha': [34, 0], 'hai': [34, 6], 'shang': [40, 20], 'han': [34, 15], 'nan': [30, 15],
        #                     'ping': [25, 22], 'hen': [34, 16], 'hou': [34, 10], 'lian': [31, 3, 15], 'wang': [46, 20],
        #                     'ti': [29, 3], 'huan': [34, 4, 15], 'ying': [60], 'huang': [34, 4, 20], 'tan': [29, 15],
        #                     'huo': [34, 4, 1], 'zhe': [38, 2], 'jiang': [35, 3, 20], 'lu': [31, 4], 'tuan': [29, 4, 15],
        #                     'bin': [24, 17], 'qiang': [36, 3, 20], 'kang': [33, 20], 'su': [44, 4], 'mu': [26, 4],
        #                     'xia': [37, 3, 0], 'ri': [49, 41], 'zhuan': [38, 4, 15], 'shen': [40, 16], 'jiu': [35, 11],
        #                     'jun': [35, 18], 'ka': [33, 0], 'ta': [29, 0], 'kan': [33, 15], 'kao': [33, 9],
        #                     'ke': [33, 2], 'kong': [33, 23], 'kuai': [33, 4, 6], 'la': [31, 0], 'lan': [31, 15],
        #                     'lang': [31, 20], 'lao': [31, 9], 'lei': [31, 7], 'liang': [31, 3, 20], 'yong': [45, 23],
        #                     'liao': [31, 3, 9], 'lin': [31, 17], 'chen': [39, 16], 'long': [31, 23], 'lou': [31, 10],
        #                     'lun': [31, 18], 'mao': [26, 9], 'mei': [26, 7], 'meng': [26, 21], 'mi': [26, 3],
        #                     'ming': [26, 22], 'que': [36, 13], 'mo': [26, 1], 'n': [16], 'na': [30], 'me': [26, 2],
        #                     'ne': [30, 2], 'ni': [30, 3], 'qing': [36, 22], 'nin': [30, 17], 'nu': [30, 4], 'ou': [10],
        #                     'peng': [25, 21], 'pi': [25, 3], 'pian': [25, 3, 15], 'tai': [29, 6], 'pu': [25, 4],
        #                     'lie': [31, 12], 'qiao': [36, 3, 9], 'kuang': [33, 4, 20], 'qiu': [36, 11], 'ran': [41, 15],
        #                     'rang': [41, 20], 're': [41, 2], 'reng': [41, 21], 'sao': [44, 9], 'miao': [26, 3, 9],
        #                     'sen': [44, 16], 'sha': [40, 0], 'te': [29, 2], 'gu': [32, 4], 'shuang': [40, 4, 20],
        #                     'shui': [40, 8], 'sou': [44, 10], 'suo': [44, 4, 1], 'sui': [44, 8], 'wan': [46, 15],
        #                     'tao': [29, 9], 'tiao': [29, 3, 9], 'zheng': [38, 21], 'tie': [29, 12], 'ting': [29, 22],
        #                     'tou': [29, 10], 'piao': [25, 3, 9], 'tu': [29, 4], 'po': [25, 1], 'tui': [29, 8],
        #                     'wo': [46, 1], 'ya': [45, 0], 'xuan': [37, 4, 15], 'yang': [45, 20], 'yin': [57],
        #                     'hang': [34, 20], 'yun': [58, 45], 'zan': [42, 15], 'zen': [42, 16], 'zeng': [42, 21],
        #                     'lve': [31, 13], 'zhei': [38, 7], 'zhen': [38, 16], 'zu': [42, 4], 'zhua': [38, 4, 0],
        #                     'zhuang': [38, 4, 20], 'xun': [37, 18], 'zong': [42, 23], 'zou': [42, 10], 'zun': [42, 18]}

        # '''方法2'''
        # 单韵母:a 0 o 1 e 2 i 3 u 4 v 5
        # 复韵母:ai 6 ei 7 ui 8 ao 9 ou 10 iu 11 ie 12 ve 13 er 14
        # 前鼻韵母:an 15 en 16 in 17 un 18 vn 19
        # 后鼻韵母:ang 20 eng 21 ing 22 ong 23
        # 声母:b 24 p 25 m 26 f 27 d 28 t 29 n 30 l 31 g 32 k 33 h 34 j 35 q 36 x 37
        #       zh 38 ch 39 sh 40 r 41 z 42 c 43 s 44 y 45 w 46

        # C 47

        # self.PhonemeList = {'C': [47], 'a': [0], 'ai': [6], 'ji': [35, 3], 'an': [15], 'jian': [35, 3, 15],
        #                     'quan': [36, 5, 15], 'zhao': [38, 9], 'ba': [24, 0], 'li': [31, 3], 'xi': [37, 3],
        #                     'bai': [24, 6], 'ban': [24, 15], 'dao': [28, 9], 'fa': [27, 0], 'bang': [24, 20],
        #                     'jia': [35, 3, 0], 'bao': [24, 9], 'chi': [39], 'gao': [32, 9], 'hu': [34, 4],
        #                     'kuo': [33, 4, 1], 'yu': [5], 'zhang': [38, 20], 'bei': [24, 7], 'bu': [24, 4],
        #                     'jing': [35, 22], 'shi': [40], 'yue': [13], 'ben': [24, 16], 'ci': [43],
        #                     'bi': [24, 3], 'jiao': [35, 3, 9], 'ru': [41, 4], 'xu': [37, 5], 'bian': [24, 3, 15],
        #                     'hua': [34, 4, 0], 'biao': [24, 3, 9], 'da': [28, 0], 'zhi': [38], 'zhun': [38, 18],
        #                     'bie': [24, 12], 'bing': [24, 22], 'qie': [36, 12], 'bo': [24, 1], 'chu': [39, 4],
        #                     'duan': [28, 4, 15], 'fen': [27, 16], 'guo': [32, 4, 1], 'hui': [34, 4, 7], 'jin': [35, 17],
        #                     'men': [26, 16], 'neng': [30, 21], 'shao': [40, 9], 'shu': [40, 4], 'tong': [29, 23],
        #                     'yao': [45, 9], 'cai': [43, 6], 'fang': [27, 20], 'qu': [36, 5], 'can': [43, 15],
        #                     'ce': [43, 2], 'ceng': [43, 21], 'chan': [39, 15], 'pin': [25, 17], 'sheng': [40, 21],
        #                     'ye': [12], 'chang': [39, 20], 'qi': [36, 3], 'yi': [45], 'chao': [39, 9],
        #                     'xian': [37, 3, 15], 'che': [39, 2], 'cheng': [39, 21], 'gong': [32, 23], 'nuo': [30, 4, 1],
        #                     'wei': [46, 7], 'lai': [31, 6], 'le': [31, 2], 'chuan': [39, 4, 15], 'chuang': [39, 4, 20],
        #                     'xin': [37, 17], 'chun': [39, 18], 'qian': [36, 3, 15], 'cong': [43, 23], 'cu': [43, 4],
        #                     'cun': [43, 18], 'zai': [42, 6], 'cuo': [43, 4, 1], 'gai': [32, 6], 'xing': [37, 22],
        #                     'xue': [37, 13], 'zao': [42, 9], 'dai': [28, 6], 'dan': [28, 15], 'dang': [28, 20],
        #                     'di': [28, 3], 'tian': [29, 3, 15], 'zhong': [38, 23], 'de': [28, 2], 'deng': [28, 21],
        #                     'dian': [28, 3, 15], 'diao': [28, 3, 9], 'cha': [39, 0], 'yan': [45, 15], 'dong': [28, 23],
        #                     'dou': [28, 10], 'du': [28, 4], 'dui': [28, 4, 7], 'wai': [46, 6], 'duo': [28, 4, 1],
        #                     'nian': [30, 3, 15], 'e': [2], 'luo': [31, 4, 1], 'si': [44], 'er': [14],
        #                     'ling': [31, 22], 'liu': [31, 11], 'san': [44, 15], 'wu': [4], 'ma': [26, 0],
        #                     'she': [40, 2], 'ren': [41, 16], 'yuan': [5, 15], 'zhan': [38, 15], 'fan': [27, 15],
        #                     'rong': [41, 23], 'zui': [42, 4, 7], 'mian': [26, 3, 15], 'wen': [46, 16],
        #                     'xiang': [37, 3, 20], 'fei': [27, 7], 'zi': [42], 'feng': [27, 21], 'shuo': [40, 4, 1],
        #                     'fu': [27, 4], 'ze': [42, 2], 'ge': [32, 2], 'shan': [40, 15], 'gan': [32, 15],
        #                     'jue': [35, 13], 'shou': [40, 10], 'xie': [37, 12], 'gang': [32, 20], 'xiao': [37, 3, 9],
        #                     'jie': [35, 12], 'gei': [32, 7], 'gen': [32, 16], 'ju': [35, 5], 'geng': [32, 21],
        #                     'hao': [34, 9], 'he': [34, 2], 'kai': [33, 6], 'min': [26, 17], 'you': [45, 10],
        #                     'zuo': [42, 4, 1], 'gou': [32, 10], 'guan': [32, 4, 15], 'zhu': [38, 4],
        #                     'guang': [32, 4, 20], 'gui': [32, 4, 7], 'ding': [28, 22], 'zhou': [38, 10], 'nei': [30, 7],
        #                     'ha': [34, 0], 'hai': [34, 6], 'shang': [40, 20], 'han': [34, 15], 'nan': [30, 15],
        #                     'ping': [25, 22], 'hen': [34, 16], 'hou': [34, 10], 'lian': [31, 3, 15], 'wang': [46, 20],
        #                     'ti': [29, 3], 'huan': [34, 4, 15], 'ying': [45, 22], 'huang': [34, 4, 20], 'tan': [29, 15],
        #                     'huo': [34, 4, 1], 'zhe': [38, 2], 'jiang': [35, 3, 20], 'lu': [31, 4], 'tuan': [29, 4, 15],
        #                     'bin': [24, 17], 'qiang': [36, 3, 20], 'kang': [33, 20], 'su': [44, 4], 'mu': [26, 4],
        #                     'xia': [37, 3, 0], 'ri': [41], 'zhuan': [38, 4, 15], 'shen': [40, 16], 'jiu': [35, 11],
        #                     'jun': [35, 19], 'ka': [33, 0], 'ta': [29, 0], 'kan': [33, 15], 'kao': [33, 9],
        #                     'ke': [33, 2], 'kong': [33, 23], 'kuai': [33, 4, 6], 'la': [31, 0], 'lan': [31, 15],
        #                     'lang': [31, 20], 'lao': [31, 9], 'lei': [31, 7], 'liang': [31, 3, 20], 'yong': [45, 23],
        #                     'liao': [31, 3, 9], 'lin': [31, 17], 'chen': [39, 16], 'long': [31, 23], 'lou': [31, 10],
        #                     'lun': [31, 18], 'mao': [26, 9], 'mei': [26, 7], 'meng': [26, 21], 'mi': [26, 3],
        #                     'ming': [26, 22], 'que': [36, 13], 'mo': [26, 1], 'n': [16], 'na': [30, 0], 'me': [26, 2],
        #                     'ne': [30, 2], 'ni': [30, 3], 'qing': [36, 22], 'nin': [30, 17], 'nu': [30, 4], 'ou': [10],
        #                     'peng': [25, 21], 'pi': [25, 3], 'pian': [25, 3, 15], 'tai': [29, 6], 'pu': [25, 4],
        #                     'lie': [31, 12], 'qiao': [36, 3, 9], 'kuang': [33, 4, 20], 'qiu': [36, 11], 'ran': [41, 15],
        #                     'rang': [41, 20], 're': [41, 2], 'reng': [41, 21], 'sao': [44, 9], 'miao': [26, 3, 9],
        #                     'sen': [44, 16], 'sha': [40, 0], 'te': [29, 2], 'gu': [32, 4], 'shuang': [40, 4, 20],
        #                     'shui': [40, 4, 7], 'sou': [44, 10], 'suo': [44, 4, 1], 'sui': [44, 4, 7], 'wan': [46, 15],
        #                     'tao': [29, 9], 'tiao': [29, 3, 9], 'zheng': [38, 21], 'tie': [29, 12], 'ting': [29, 22],
        #                     'tou': [29, 10], 'piao': [25, 3, 9], 'tu': [29, 4], 'po': [25, 1], 'tui': [29, 4, 7],
        #                     'wo': [46, 1], 'ya': [45, 0], 'xuan': [37, 5, 15], 'yang': [45, 20], 'yin': [45, 17],
        #                     'hang': [34, 20], 'yun': [45, 19], 'zan': [42, 15], 'zen': [42, 16], 'zeng': [42, 21],
        #                     'lve': [31, 13], 'zhei': [38, 2], 'zhen': [38, 16], 'zu': [42, 4], 'zhua': [38, 4, 0],
        #                     'zhuang': [38, 4, 20], 'xun': [37, 19], 'zong': [42, 23], 'zou': [42, 10], 'zun': [42, 18]}

        # '''方法3'''
        # 单韵母:a 0 o 1 e 2 i 3 u 4 v 5
        # 复韵母:ai 6 ei 7 ui 8 ao 9 ou 10 iu 11 ie 12 ve 13 er 14
        # 前鼻韵母:an 15 en 16 in 17 un 18 vn 19
        # 后鼻韵母:ang 20 eng 21 ing 22 ong 23
        # ch 25 sh 26 zh 27 c 28 s 29 z 30 r 31
        # C 24

        # self.PhonemeList = {'C': [24], 'a': [0], 'ai': [6], 'ji': [3], 'an': [15], 'jian': [3, 15],
        #                     'quan': [5, 15], 'zhao': [27, 9], 'ba': [0], 'li': [3], 'xi': [3],
        #                     'bai': [6], 'ban': [15], 'dao': [9], 'fa': [0], 'bang': [20],
        #                     'jia': [3, 0], 'bao': [9], 'chi': [25], 'gao': [9], 'hu': [4],
        #                     'kuo': [4, 1], 'yu': [5], 'zhang': [20], 'bei': [7], 'bu': [4],
        #                     'jing': [22], 'shi': [26], 'yue': [13], 'ben': [16], 'ci': [28],
        #                     'bi': [3], 'jiao': [3, 9], 'ru': [31, 4], 'xu': [5], 'bian': [3, 15],
        #                     'hua': [4, 0], 'biao': [3, 9], 'da': [0], 'zhi': [27], 'zhun': [27, 18],
        #                     'bie': [12], 'bing': [22], 'qie': [12], 'bo': [1], 'chu': [25, 4],
        #                     'duan': [4, 15], 'fen': [16], 'guo': [4, 1], 'hui': [4, 7], 'jin': [17],
        #                     'men': [16], 'neng': [21], 'shao': [26, 9], 'shu': [26, 4], 'tong': [23],
        #                     'yao': [9], 'cai': [28, 6], 'fang': [20], 'qu': [5], 'can': [28, 15],
        #                     'ce': [28, 2], 'ceng': [28, 21], 'chan': [25, 15], 'pin': [17], 'sheng': [26, 21],
        #                     'ye': [12], 'chang': [25, 20], 'qi': [3], 'yi': [3], 'chao': [25, 9],
        #                     'xian': [3, 15], 'che': [25, 2], 'cheng': [25, 21], 'gong': [23], 'nuo': [4, 1],
        #                     'wei': [7], 'lai': [6], 'le': [2], 'chuan': [25, 4, 15], 'chuang': [25, 4, 20],
        #                     'xin': [17], 'chun': [25, 18], 'qian': [3, 15], 'cong': [28, 23], 'cu': [28, 4],
        #                     'cun': [28, 18], 'zai': [30, 6], 'cuo': [28, 4, 1], 'gai': [6], 'xing': [22],
        #                     'xue': [13], 'zao': [30, 9], 'dai': [6], 'dan': [15], 'dang': [20],
        #                     'di': [3], 'tian': [3, 15], 'zhong': [27, 23], 'de': [2], 'deng': [21],
        #                     'dian': [3, 15], 'diao': [3, 9], 'cha': [25, 0], 'yan': [15], 'dong': [23],
        #                     'dou': [10], 'du': [4], 'dui': [4, 7], 'wai': [6], 'duo': [4, 1],
        #                     'nian': [3, 15], 'e': [2], 'luo': [4, 1], 'si': [29], 'er': [14],
        #                     'ling': [22], 'liu': [11], 'san': [29, 15], 'wu': [4], 'ma': [0],
        #                     'she': [26, 2], 'ren': [31, 16], 'yuan': [5, 15], 'zhan': [27, 15], 'fan': [15],
        #                     'rong': [31, 23], 'zui': [30, 4, 7], 'mian': [3, 15], 'wen': [16],
        #                     'xiang': [3, 20], 'fei': [7], 'zi': [30], 'feng': [21], 'shuo': [26, 4, 1],
        #                     'fu': [4], 'ze': [30, 2], 'ge': [2], 'shan': [26, 15], 'gan': [15],
        #                     'jue': [13], 'shou': [26, 10], 'xie': [12], 'gang': [20], 'xiao': [3, 9],
        #                     'jie': [12], 'gei': [7], 'gen': [16], 'ju': [5], 'geng': [21],
        #                     'hao': [9], 'he': [2], 'kai': [6], 'min': [17], 'you': [10],
        #                     'zuo': [30, 4, 1], 'gou': [0], 'guan': [4, 15], 'zhu': [27, 4],
        #                     'guang': [4, 20], 'gui': [4, 7], 'ding': [22], 'zhou': [27, 10], 'nei': [7],
        #                     'ha': [0], 'hai': [6], 'shang': [26, 20], 'han': [15], 'nan': [15],
        #                     'ping': [22], 'hen': [16], 'hou': [10], 'lian': [3, 15], 'wang': [20],
        #                     'ti': [3], 'huan': [4, 15], 'ying': [22], 'huang': [4, 20], 'tan': [15],
        #                     'huo': [4, 1], 'zhe': [27, 2], 'jiang': [3, 20], 'lu': [4], 'tuan': [4, 15],
        #                     'bin': [17], 'qiang': [3, 20], 'kang': [20], 'su': [29, 4], 'mu': [4],
        #                     'xia': [3, 0], 'ri': [31], 'zhuan': [27, 4, 15], 'shen': [26, 16], 'jiu': [11],
        #                     'jun': [19], 'ka': [0], 'ta': [0], 'kan': [15], 'kao': [9],
        #                     'ke': [2], 'kong': [23], 'kuai': [4, 6], 'la': [0], 'lan': [15],
        #                     'lang': [20], 'lao': [9], 'lei': [7], 'liang': [3, 20], 'yong': [23],
        #                     'liao': [3, 9], 'lin': [17], 'chen': [25, 16], 'long': [23], 'lou': [10],
        #                     'lun': [18], 'mao': [9], 'mei': [7], 'meng': [21], 'mi': [3],
        #                     'ming': [22], 'que': [13], 'mo': [1], 'n': [16], 'na': [0], 'me': [2],
        #                     'ne': [2], 'ni': [3], 'qing': [22], 'nin': [17], 'nu': [4], 'ou': [10],
        #                     'peng': [21], 'pi': [3], 'pian': [3, 15], 'tai': [6], 'pu': [4],
        #                     'lie': [12], 'qiao': [3, 9], 'kuang': [4, 20], 'qiu': [11], 'ran': [31, 15],
        #                     'rang': [31, 20], 're': [31, 2], 'reng': [31, 21], 'sao': [29, 9], 'miao': [3, 9],
        #                     'sen': [29, 16], 'sha': [26, 0], 'te': [2], 'gu': [4], 'shuang': [26, 4, 20],
        #                     'shui': [26, 4, 7], 'sou': [29, 10], 'suo': [29, 4, 1], 'sui': [29, 4, 7], 'wan': [15],
        #                     'tao': [9], 'tiao': [3, 9], 'zheng': [27, 21], 'tie': [12], 'ting': [22],
        #                     'tou': [10], 'piao': [3, 9], 'tu': [4], 'po': [1], 'tui': [4, 7],
        #                     'wo': [1], 'ya': [0], 'xuan': [5, 15], 'yang': [20], 'yin': [17],
        #                     'hang': [20], 'yun': [19], 'zan': [30, 15], 'zen': [30, 16], 'zeng': [30, 21],
        #                     'lve': [13], 'zhei': [27, 2], 'zhen': [27, 16], 'zu': [30, 4], 'zhua': [27, 4, 0],
        #                     'zhuang': [27, 4, 20], 'xun': [19], 'zong': [30, 23], 'zou': [30, 10], 'zun': [30, 18]}

        # '''方法4'''
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

        self.va_dict = self.get_video_audio_map()
        self.class_dict = defaultdict(list)

        for item in data:
            audio_file = self.va_dict.get(item)
            assert (audio_file != None)
            # local
            audio_file = 'E:/LRW/audio/' + audio_file + '.npy'
            # 3080
            # audio_file = '/home/mingwu/workspace_czg/data/LRW/LRW/audio/' + audio_file + '.npy'
            # 3090
            # audio_file = '/home/czg/dataset/LRW/audio/' + audio_file + '.npy'
            if (os.path.exists(audio_file)):
                item = (item[0], audio_file, item[1], item[2], item[3])
                self.class_dict[item[-1]].append(item)

        self.data = []
        self.unlabel_data = []
        for k, v in self.class_dict.items():
            n = len(v)

            self.data.extend(v[:n])

    def get_video_audio_map(self):
        # local
        self.anno = 'E:/LRW/info/all_audio_video.txt'
        # 3080
        # self.anno = '/home/mingwu/workspace_czg/data/LRW/LRW/info/all_audio_video.txt'
        # 3090
        # self.anno = '/home/czg/dataset/LRW/info/all_audio_video.txt'
        with open(self.anno, 'r', encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
            lines = [line.split(',') for line in lines]
            va_dict = {}
            for (v, a, _, pinyin, op, ed) in lines:
                op = int(float(op) * 25) + 1
                ed = int(float(ed) * 25) + 1
                pinyin = self.pinyins.index(pinyin)
                va_dict[(v, op, ed, pinyin)] = a

        return va_dict

    def getPhoneme(self, pinyin):

        pinyinlist = pinyin.split(' ')
        List = np.zeros(286)
        for line in pinyinlist:
            List[self.PhonemeList.index(line)] = 1
        # for line in pinyinlist:
        #     for i in self.PhonemeList[line]:
        #         List[i] = 1
        return List

    def __len__(self):
        return len(self.data)

    def load_video(self, item):
        # load video into a tensor
        (path, mfcc, op, ed, label) = item
        inputs, border = self.load_images(os.path.join(self.data_root, path), op, ed)
        if inputs == None and border == None:
            return True

        result = {}

        result['video'] = inputs
        result['label'] = int(label)
        result['duration'] = border.astype(np.bool)
        # result['phoneme'] = self.getPhoneme(self.pinyins[int(label)])
        # phomelist = []
        # for i, input in enumerate(result['phoneme']):
        #     if input != 0:
        #         phomelist.append(self.PhonemeList[i])

        label = [char for label in self.pinyins[int(label)] for char in label]
        # for i, label in enumerate(self.pinyins[int(label)]):
        #     for char in label:
        #         label += [char]

        pinyinlable = [self.char_to_num[i] for i in label]
        result['pinyinlable'] = torch.tensor(pinyinlable, dtype=torch.int32).numpy()

        savename = os.path.join(self.target_dir, f'{path}_{op}_{ed}.pkl')
        torch.save(result, savename)

        return True

    def __getitem__(self, idx):

        r = self.load_video(self.data[idx])

        return r

    def load_images(self, path, op, ed):
        center = (op + ed) / 2
        length = (ed - op + 1)

        # time = 1
        # if length < 5:
        #     time = 8
        # elif length < 10:
        #     time = 4
        # elif length < 20:
        #     time = 2
        # pad = self.padding // time
        pad = self.padding
        op = int(center - pad // 2)
        ed = int(op + pad)
        if op < 0:
            left_border = max(int(center - length / 2), 0)
            right_border = min(int(center + length / 2), pad)
        else:
            left_border = max(int(center - length / 2 - op), 0)
            right_border = min(int(center + length / 2 - op), pad)
        # print(length, center, op, ed, left_border, right_border)

        files = [os.path.join(path, '{}.jpg'.format(i)) for i in range(op, ed)]
        # files = files * time
        files = filter(lambda path: os.path.exists(path), files)
        files = [cv2.imread(file) for file in files]
        # for i in files:
        #     if i.shape[0] < 64:
        #         return None, None

        files = [cv2.resize(file, (96, 96)) for file in files]

        # ed - op < ed -op
        if len(files) < length // 2 or len(files) == 0:
            print(path + " error op: " + str(op))
            return None, None

        files = np.stack(files, 0)
        t = files.shape[0]  # file的数量

        tensor = np.zeros((40, 96, 96, 3)).astype(files.dtype)
        border = np.zeros(40)
        tensor[:t, ...] = files.copy()
        # border[left_border:right_border] = np.arange(1, right_border - left_border + 1)
        border[left_border:right_border] = 1.0
        # border = np.tile(border, time)
        # border[:t] = 1.0

        tensor = [jpeg.encode(tensor[_]) for _ in range(40)]

        return tensor, border


def wav_to_numpy_arr_converter(wav_path, target_path):
    for name in os.listdir(wav_path):
        music_arr = []
        if (os.path.exists(target_path + '/' + name[:-4] + '.npy')):
            continue

        frequency, wav_arr = wav.read(wav_path + '/' + name)
        music_arr.append(wav_arr)
        # print(name[:-4], ' converted and append to numpy array')
        np.save(target_path + '/' + name[:-4] + '.npy', music_arr)


if (__name__ == '__main__'):
    # 转换wav为npy
    # img1 = cv2.imread('/home/mingwu/workspace_czg/data/LRW/LRW/images/LRW1000_Public/lip/11111131f70e5f6dc399a43bc9f53cf8/10.jpg')
    # cv2.imshow('img', img1)
    # cv2.waitKey(0)


    for subset in ['trn', 'val', 'tst']:
        # local
        target_dir = f'E:/LRW1000_Public_pkl_jpeg/{subset}'
        index_file = f'E:/LRW/info/{subset}_1000.txt'
        # 3080
        # target_dir = f'/home/mingwu/workspace_czg/data/LRW/LRW1000_Public_pkl_jpeg/{subset}'
        # index_file = f'/home/mingwu/workspace_czg/data/LRW/LRW/info/{subset}_1000.txt'
        # 3090
        # target_dir = f'/home/czg/dataset/LRW1000_Phome/{subset}'
        # target_dir = f'/home/czg/dataset/LRW1000_Public_pkl_jpeg/{subset}'
        # index_file = f'/home/czg/dataset/LRW/info/{subset}_1000.txt'

        if (not os.path.exists(target_dir)):
            os.makedirs(target_dir)

        dataset = LRW1000_Dataset(index_file, target_dir)
        print('Start running, Data Length:', len(dataset))
        loader = DataLoader(dataset,
                            batch_size=128,
                            num_workers=8,
                            shuffle=False,
                            drop_last=False)

        import time

        tic = time.time()
        for i, batch in enumerate(loader):
            toc = time.time()
            eta = ((toc - tic) / (i + 1) * (len(loader) - i)) / 3600.0
            print(f'eta:{eta:.5f}')

        print('end')
