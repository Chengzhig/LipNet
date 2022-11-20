import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.init as init

from torch.utils.data import DataLoader
from .video_cnn import ResNet, BasicBlock


class LipNet(nn.Module):
    def __init__(self, opt, vocab_size):
        super(LipNet, self).__init__()
        self.opt = opt
        self.conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(opt.dropout),
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(opt.dropout),
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(opt.dropout)
        )
        # T B C*H*W
        self.gru1 = nn.GRU(96 * 3 * 6, opt.rnn_size, 1, bidirectional=True)
        self.drp1 = nn.Dropout(opt.dropout)
        # T B F
        self.gru2 = nn.GRU(opt.rnn_size * 2, opt.rnn_size, 1, bidirectional=True)
        self.drp2 = nn.Dropout(opt.dropout)
        # T B V
        self.pred = nn.Linear(opt.rnn_size * 2, vocab_size + 1)


        # initialisations
        for m in self.conv.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)

        init.kaiming_normal_(self.pred.weight, nonlinearity='sigmoid')
        init.constant_(self.pred.bias, 0)

        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + opt.rnn_size))
            for i in range(0, opt.rnn_size * 3, opt.rnn_size):
                init.uniform_(m.weight_ih_l0[i: i + opt.rnn_size],
                              -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + opt.rnn_size])
                init.constant_(m.bias_ih_l0[i: i + opt.rnn_size], 0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + opt.rnn_size],
                              -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + opt.rnn_size])
                init.constant_(m.bias_ih_l0_reverse[i: i + opt.rnn_size], 0)

    def forward(self, x):
        x = self.conv(x)  # B C T H W
        x = x.permute(2, 0, 1, 3, 4).contiguous()  # T B C H W
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.gru1(x)
        x = self.drp1(x)
        x, _ = self.gru2(x)
        x = self.drp2(x)
        x = self.pred(x)

        return x


class LipNet_Pinyin(nn.Module):
    def __init__(self):
        super(LipNet_Pinyin, self).__init__()
        self.videoEncode = VideoEncoder()
        # self.pinyinEncode = PinyinEncoder(in_features=512, hidden_size=256, GRUorLSTM=0)
        # self.pinyinDecode = PinyinDecoder(in_features=512, hidden_size=512, GRUorLSTM=0)
        # self.characterDecode = CharacterDecoder(in_features=512, hidden_size=512, GRUorLSTM=0)
        # self.PinyinMLP = MLP(in_features=256, out_features=29)
        self.CharacterMLP = MLP(in_features=512*40, out_features=1000)
        # self.PSoft = nn.LogSoftmax(dim=0)
        # self.CSoft = nn.Softmax(dim=0)

    def forward(self, x):
        B, T, C, H, W = x.size()[:]
        Xve = self.videoEncode(x)
        # Xpd = self.pinyinDecode(Xve)
        # Xcd = self.characterDecode(Xve)
        # PinyinInput = Xpd
        # Ype = self.pinyinEncode(PinyinInput)
        # Ycd = self.characterDecode(Ype)
        # CharacterInput = torch.cat([Xcd, Ycd], -1)
        #
        # PinyinPrediction
        # PinyinInput = PinyinInput.view(B, T, -1)
        # Pp = self.PSoft(self.PinyinMLP(PinyinInput))
        #
        # CharacterPrediction
        # CharacterInput = CharacterInput.view(B, -1)
        Pc = self.CSoft(self.CharacterMLP(CharacterInput))

        return Pp, Pc

    def PinyinPrediction(self, x):
        Xve = self.videoEncode(x)
        Xpd = self.pinyinDecode(Xve)
        Vcontext = torch.cat([Xve, Xpd], 0)
        p = nn.Softmax(self.MLP(Vcontext))
        return p

    def TonePrediction(self, x, y):
        Xve = self.videoEncode(x)
        Xtd = self.toneDecode(Xve)
        Ype = self.pinyinDecode(y)
        Ytd = self.toneDecode(Ype)
        context = torch.cat([Xtd, Ytd], 0)
        p = nn.Softmax(self.MLP(context))
        return p

    def CharacterPrediction(self, x):
        x = self.videoEncode(x)
        x1 = self.pinyinDecode(x)
        context = torch.cat([x, x1], 0)
        p = nn.Softmax(self.MLP(context))
        return p


class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        # TODO
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(in_features=in_features, out_features=100, bias=True)
        self.hidden2 = nn.Linear(100, 100)
        self.hidden3 = nn.Linear(100, 50)
        self.predict = nn.Linear(50, out_features)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        output = self.predict(x)
        # out = output.view(-1)

        return output


class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()
        self.rnn_size = 1024
        self.conv = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
        self.gru = nn.GRU(512, self.rnn_size, 3, bidirectional=True, dropout=0.2)

        # initialisations
        for m in self.conv.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        # init.kaiming_normal_(self.pred.weight, nonlinearity='sigmoid')
        # init.constant_(self.pred.bias, 0)

        stdv = math.sqrt(2 / (512 + self.rnn_size))
        for i in range(0, self.rnn_size * 3, self.rnn_size):
            init.uniform_(self.gru.weight_ih_l0[i: i + self.rnn_size],
                          -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
            init.orthogonal_(self.gru.weight_hh_l0[i: i + self.rnn_size])
            init.constant_(self.gru.bias_ih_l0[i: i + self.rnn_size], 0)
            init.uniform_(self.gru.weight_ih_l0_reverse[i: i + self.rnn_size],
                          -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
            init.orthogonal_(self.gru.weight_hh_l0_reverse[i: i + self.rnn_size])
            init.constant_(self.gru.bias_ih_l0_reverse[i: i + self.rnn_size], 0)

    def forward(self, x):
        x = x.transpose(1, 2)
        b, c, t, h, w = x.size()[:]
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet18(x)
        x = x.view(b, -1, 512)
        x, _ = self.gru(x)
        return x


class PinyinEncoder(nn.Module):
    def __init__(self, in_features, hidden_size, GRUorLSTM):
        super(PinyinEncoder, self).__init__()
        self.in_features = in_features
        self.rnn_size = hidden_size
        self.GRUorLSTM = GRUorLSTM
        if GRUorLSTM == 0:
            self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_size, num_layers=2, bidirectional=True,
                              dropout=0.2)
        else:
            self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=2, bidirectional=True,
                                dropout=0.2)

        if GRUorLSTM == 0:
            stdv = math.sqrt(2 / (512 + self.rnn_size))
            for i in range(0, self.rnn_size * 3, self.rnn_size):
                init.uniform_(self.gru.weight_ih_l0[i: i + self.rnn_size],
                              -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(self.gru.weight_hh_l0[i: i + self.rnn_size])
                init.constant_(self.gru.bias_ih_l0[i: i + self.rnn_size], 0)
                if self.gru.bidirectional:
                    init.uniform_(self.gru.weight_ih_l0_reverse[i: i + self.rnn_size],
                                  -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                    init.orthogonal_(self.gru.weight_hh_l0_reverse[i: i + self.rnn_size])
                    init.constant_(self.gru.bias_ih_l0_reverse[i: i + self.rnn_size], 0)

    def forward(self, x):
        if self.GRUorLSTM == 0:
            x, _ = self.gru(x)
        elif self.GRUorLSTM == 1:
            x, _ = self.lstm(x)
        else:
            x, _ = self.gru(x)
        return x


class PinyinDecoder(nn.Module):
    def __init__(self, in_features, hidden_size, GRUorLSTM):
        super(PinyinDecoder, self).__init__()
        self.in_features = in_features
        self.rnn_size = hidden_size
        self.GRUorLSTM = GRUorLSTM
        if GRUorLSTM == 0:
            self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_size, num_layers=2, bidirectional=False,
                              dropout=0.2)
        else:
            self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=2, bidirectional=False,
                                dropout=0.2)

        if GRUorLSTM == 0:
            stdv = math.sqrt(2 / (512 + self.rnn_size))
            for i in range(0, self.rnn_size * 3, self.rnn_size):
                init.uniform_(self.gru.weight_ih_l0[i: i + self.rnn_size],
                              -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(self.gru.weight_hh_l0[i: i + self.rnn_size])
                init.constant_(self.gru.bias_ih_l0[i: i + self.rnn_size], 0)
                if self.gru.bidirectional:
                    init.uniform_(self.gru.weight_ih_l0_reverse[i: i + self.rnn_size],
                                  -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                    init.orthogonal_(self.gru.weight_hh_l0_reverse[i: i + self.rnn_size])
                    init.constant_(self.gru.bias_ih_l0_reverse[i: i + self.rnn_size], 0)

    def forward(self, x):
        if self.GRUorLSTM == 0:
            x, _ = self.gru(x)
        elif self.GRUorLSTM == 1:
            x, _ = self.lstm(x)
        else:
            x, _ = self.gru(x)
        return x


class ToneEncoder(nn.Module):
    def __init__(self, in_features, hidden_size, GRUorLSTM):
        super(ToneEncoder, self).__init__()
        self.in_features = in_features
        self.rnn_size = hidden_size
        self.GRUorLSTM = GRUorLSTM
        if GRUorLSTM == 0:
            self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_size, num_layers=2, bidirectional=True,
                              dropout=0.2)
        else:
            self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=2, bidirectional=True,
                                dropout=0.2)

        if GRUorLSTM == 0:
            stdv = math.sqrt(2 / (512 + self.rnn_size))
            for i in range(0, self.rnn_size * 3, self.rnn_size):
                init.uniform_(self.gru.weight_ih_l0[i: i + self.rnn_size],
                              -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(self.gru.weight_hh_l0[i: i + self.rnn_size])
                init.constant_(self.gru.bias_ih_l0[i: i + self.rnn_size], 0)
                if self.gru.bidirectional:
                    init.uniform_(self.gru.weight_ih_l0_reverse[i: i + self.rnn_size],
                                  -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                    init.orthogonal_(self.gru.weight_hh_l0_reverse[i: i + self.rnn_size])
                    init.constant_(self.gru.bias_ih_l0_reverse[i: i + self.rnn_size], 0)

    def forward(self, x):
        if self.GRUorLSTM == 0:
            x, _ = self.gru(x)
        elif self.GRUorLSTM == 1:
            x, _ = self.lstm(x)
        else:
            x, _ = self.gru(x)
        return x


class ToneDecoder(nn.Module):
    def __init__(self, in_features, hidden_size, GRUorLSTM):
        super(ToneDecoder, self).__init__()
        self.in_features = in_features
        self.rnn_size = hidden_size
        self.GRUorLSTM = GRUorLSTM
        if GRUorLSTM == 0:
            self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_size, num_layers=2, bidirectional=False,
                              dropout=0.2)
        else:
            self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=2, bidirectional=False,
                                dropout=0.2)

        if GRUorLSTM == 0:
            stdv = math.sqrt(2 / (512 + self.rnn_size))
            for i in range(0, self.rnn_size * 3, self.rnn_size):
                init.uniform_(self.gru.weight_ih_l0[i: i + self.rnn_size],
                              -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(self.gru.weight_hh_l0[i: i + self.rnn_size])
                init.constant_(self.gru.bias_ih_l0[i: i + self.rnn_size], 0)
                if self.gru.bidirectional:
                    init.uniform_(self.gru.weight_ih_l0_reverse[i: i + self.rnn_size],
                                  -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                    init.orthogonal_(self.gru.weight_hh_l0_reverse[i: i + self.rnn_size])
                    init.constant_(self.gru.bias_ih_l0_reverse[i: i + self.rnn_size], 0)

    def forward(self, x):
        if self.GRUorLSTM == 0:
            x, _ = self.gru(x)
        elif self.GRUorLSTM == 1:
            x, _ = self.lstm(x)
        else:
            x, _ = self.gru(x)
        return x


class CharacterDecoder(nn.Module):
    def __init__(self, in_features, hidden_size, GRUorLSTM):
        super(CharacterDecoder, self).__init__()
        self.in_features = in_features
        self.rnn_size = hidden_size
        self.GRUorLSTM = GRUorLSTM
        if GRUorLSTM == 0:
            self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_size, num_layers=2, bidirectional=False,
                              dropout=0.2)
        else:
            self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=2, bidirectional=False,
                                dropout=0.2)

        if GRUorLSTM == 0:
            stdv = math.sqrt(2 / (512 + self.rnn_size))
            for i in range(0, self.rnn_size * 3, self.rnn_size):
                init.uniform_(self.gru.weight_ih_l0[i: i + self.rnn_size],
                              -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(self.gru.weight_hh_l0[i: i + self.rnn_size])
                init.constant_(self.gru.bias_ih_l0[i: i + self.rnn_size], 0)
                if self.gru.bidirectional:
                    init.uniform_(self.gru.weight_ih_l0_reverse[i: i + self.rnn_size],
                                  -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                    init.orthogonal_(self.gru.weight_hh_l0_reverse[i: i + self.rnn_size])
                    init.constant_(self.gru.bias_ih_l0_reverse[i: i + self.rnn_size], 0)

    def forward(self, x):
        if self.GRUorLSTM == 0:
            x, _ = self.gru(x)
        elif self.GRUorLSTM == 1:
            x, _ = self.lstm(x)
        else:
            x, _ = self.gru(x)
        return x

# class Exp:
#     def __init__(self, opt):
#         self.trainset = GRIDDataset(opt, dset='train')
#         self.trainset.load_data()
#         self.testset = GRIDDataset(opt, dset='test')
#         self.testset.load_data()
#         self.trainloader = DataLoader(self.trainset, batch_size=opt.batch_size,
#                                       shuffle=True, num_workers=opt.num_workers, collate_fn=ctc_collate,
#                                       pin_memory=True)
#         self.testloader = DataLoader(self.testset, batch_size=opt.batch_size,
#                                      shuffle=False, num_workers=opt.num_workers, collate_fn=ctc_collate,
#                                      pin_memory=True)
#
#         # define network
#         self.input_img_size = [3, 50, 100]
#         self.chan, self.height, self.width = self.input_img_size
#         self.vocab_size = len(self.trainset.vocab)
#         assert self.testset.vocab <= self.trainset.vocab, 'possible OOV characters in test set'
#         self.maxT = self.trainset.opt.max_timesteps
#
#         self.model = LipNet(opt, self.vocab_size)
#         self.opt = opt
#
#         self.optimfunc = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)
#
#     # learning rate scheduler: fixed LR
#     def optim(self, epoch):
#         return self.optimfunc
