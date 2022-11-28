import argparse
from functools import partial
from .video_cnn import VideoCNN
import torch
import torch.nn as nn
import random
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.nn.parameter as parameter


class VideoModel(nn.Module):

    def __init__(self, args, dropout=0.5):
        super(VideoModel, self).__init__()

        self.args = args

        self.video_cnn = VideoCNN(se=True, CBAM=False)
        if (self.args.border):
            in_dim = 512 + 1
        else:
            in_dim = 512

        # self.LN = nn.LayerNorm(in_dim)
        self.gru = nn.GRU(in_dim, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)

        self.v_cls = nn.Linear(1024 * 2, self.args.n_class)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, v, border=None):
        self.gru.flatten_parameters()

        if (self.training):
            with autocast():
                f_v = self.video_cnn(v)
                f_v = self.dropout(f_v)
            f_v = f_v.float()
        else:
            f_v = self.video_cnn(v)
            f_v = self.dropout(f_v)

        if (self.args.border):
            border = border[:, :, None]
            f_v = torch.cat([f_v, border], -1)
            # f_v = self.LN(f_v)
            h, _ = self.gru(f_v)
        else:
            # f_v = self.LN(f_v)
            h, _ = self.gru(f_v)

        y_v = self.v_cls(self.dropout(h)).mean(1)

        return y_v


parser = argparse.ArgumentParser()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser.add_argument('--gpus', type=str, required=True, default='0,1,2,3')
parser.add_argument('--lr', type=float, required=True, default=3e-4)
parser.add_argument('--batch_size', type=int, required=True, default=400)
parser.add_argument('--n_class', type=int, required=True, default=1000)
parser.add_argument('--num_workers', type=int, required=True, default=8)
parser.add_argument('--max_epoch', type=int, required=True, default=120)
parser.add_argument('--test', type=str2bool, required=True, default='False')

# load opts
parser.add_argument('--weights', type=str, required=False, default=None)

# save prefix
parser.add_argument('--save_prefix', type=str, required=True, default='checkpoints/lrw-baseline/')

# dataset
parser.add_argument('--dataset', type=str, required=True, default='lrw1000')
parser.add_argument('--border', type=str2bool, required=True, default='False')
parser.add_argument('--mixup', type=str2bool, required=True, default='False')
parser.add_argument('--label_smooth', type=str2bool, required=True, default='False')
parser.add_argument('--se', type=str2bool, required=True, default='False')

args = parser.parse_args()

if (__name__ == '__main__'):
    net = VideoModel(args)
    print(net)
