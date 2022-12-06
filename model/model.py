import argparse
from functools import partial

from model.Mymodel import Encoder, Decoder, Attention
from model.densetcn import DenseTemporalConvNet
from .video_cnn import VideoCNN, ResNet, BasicBlock
import torch
import torch.nn as nn
import random
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.nn.parameter as parameter


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(in_features=40 * 29, out_features=128, bias=True)
        self.hidden2 = nn.Linear(128, 256)
        self.hidden3 = nn.Linear(256, 64)
        self.predict = nn.Linear(64, 1000)
        self.hhh = nn.Linear(in_features=40 * 29, out_features=1000, bias=True)

    def forward(self, x):
        x = x.view(-1, 40 * 29)
        # x = F.relu(self.hidden1(x))
        # x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        output = self.hhh(x)
        out = output.view(-1, 1000)

        return out


class VideoModel1(nn.Module):

    def __init__(self, args, dropout=0.2):
        super(VideoModel1, self).__init__()

        self.model = VisionTransformer(img_size=88, in_chans=1, num_classes=1000, patch_size=16, embed_dim=768,
                                       depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                       norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0.,
                                       drop_path_rate=0.1, num_frames=8, attention_type='divided_space_time')
        self.args = args

    def forward(self, x):
        x = self.model(x)
        return x

    def _initialize_weights(self):
        for name, param in self.gru.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)


def _average_batch(x, lengths, B):
    return torch.stack([torch.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)], 0)


class DenseTCN(nn.Module):
    def __init__(self, block_config, growth_rate_set, input_size, reduced_size, num_classes,
                 kernel_size_set, dilation_size_set,
                 dropout, relu_type,
                 squeeze_excitation=False,
                 ):
        super(DenseTCN, self).__init__()

        num_features = reduced_size + block_config[-1] * growth_rate_set[-1]
        self.tcn_trunk = DenseTemporalConvNet(block_config, growth_rate_set, input_size, reduced_size,
                                              kernel_size_set, dilation_size_set,
                                              dropout=dropout, relu_type=relu_type,
                                              squeeze_excitation=squeeze_excitation,
                                              )
        self.tcn_output = nn.Linear(num_features, num_classes)
        self.consensus_func = _average_batch

    def forward(self, x, lengths, B):
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func(x, lengths, B)
        return self.tcn_output(x)


class Lipreading(nn.Module):

    def __init__(self, args, modality='video', dropout=0.5, densetcn_options={}, relu_type='prelu', width_mult=1.0,
                 use_boundary=False, extract_feats=False):
        super(Lipreading, self).__init__()
        self.extract_feats = extract_feats
        self.modality = modality
        self.use_boundary = use_boundary

        self.args = args
        self.densetcn_options = densetcn_options
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], TCSAM=True)

        if (self.args.border):
            self.in_dim = 512 + 1
        else:
            self.in_dim = 512

        self.tcn = DenseTCN(block_config=densetcn_options['block_config'],
                            growth_rate_set=densetcn_options['growth_rate_set'],
                            input_size=self.in_dim,
                            reduced_size=densetcn_options['reduced_size'],
                            num_classes=self.args.n_class,
                            kernel_size_set=densetcn_options['kernel_size_set'],
                            dilation_size_set=densetcn_options['dilation_size_set'],
                            dropout=densetcn_options['dropout'],
                            relu_type=relu_type,
                            squeeze_excitation=densetcn_options['squeeze_excitation'],
                            )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, v, lengths, border=None):
        B, C, T, H, W = v.size()
        f_v = self.frontend3D(v)
        f_v = self.dropout(f_v)
        if self.use_boundary:
            _border = border[:, :, None]
            x = torch.cat([f_v, _border], dim=-1)

        return x if self.extract_feats else self.tcn(x, lengths, B)


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
