import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.utils.data import DataLoader
from .video_cnn import ResNet, BasicBlock, VideoCNN


# config_vocab_size = 48 + 1
# config_emb_size = 512
# config_hidden_size = 512
# enc_hid_dim = 512
# dec_hid_dim = 512
# emb_dim = 512
# dec_output_dim = 48 + 1


# class LipNet(nn.Module):
#     def __init__(self, opt, vocab_size):
#         super(LipNet, self).__init__()
#         self.opt = opt
#         self.conv = nn.Sequential(
#             nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
#             nn.ReLU(True),
#             nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
#             nn.Dropout3d(opt.dropout),
#             nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
#             nn.ReLU(True),
#             nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
#             nn.Dropout3d(opt.dropout),
#             nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
#             nn.ReLU(True),
#             nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
#             nn.Dropout3d(opt.dropout)
#         )
#         # T B C*H*W
#         self.gru1 = nn.GRU(96 * 3 * 6, opt.rnn_size, 1, bidirectional=True)
#         self.drp1 = nn.Dropout(opt.dropout)
#         # T B F
#         self.gru2 = nn.GRU(opt.rnn_size * 2, opt.rnn_size, 1, bidirectional=True)
#         self.drp2 = nn.Dropout(opt.dropout)
#         # T B V
#         self.pred = nn.Linear(opt.rnn_size * 2, vocab_size + 1)
#
#
#         # initialisations
#         for m in self.conv.modules():
#             if isinstance(m, nn.Conv3d):
#                 init.kaiming_normal_(m.weight, nonlinearity='relu')
#                 init.constant_(m.bias, 0)
#
#         init.kaiming_normal_(self.pred.weight, nonlinearity='sigmoid')
#         init.constant_(self.pred.bias, 0)
#
#         for m in (self.gru1, self.gru2):
#             stdv = math.sqrt(2 / (96 * 3 * 6 + opt.rnn_size))
#             for i in range(0, opt.rnn_size * 3, opt.rnn_size):
#                 init.uniform_(m.weight_ih_l0[i: i + opt.rnn_size],
#                               -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
#                 init.orthogonal_(m.weight_hh_l0[i: i + opt.rnn_size])
#                 init.constant_(m.bias_ih_l0[i: i + opt.rnn_size], 0)
#                 init.uniform_(m.weight_ih_l0_reverse[i: i + opt.rnn_size],
#                               -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
#                 init.orthogonal_(m.weight_hh_l0_reverse[i: i + opt.rnn_size])
#                 init.constant_(m.bias_ih_l0_reverse[i: i + opt.rnn_size], 0)
#
#     def forward(self, x):
#         x = self.conv(x)  # B C T H W
#         x = x.permute(2, 0, 1, 3, 4).contiguous()  # T B C H W
#         x = x.view(x.size(0), x.size(1), -1)
#         x, _ = self.gru1(x)
#         x = self.drp1(x)
#         x, _ = self.gru2(x)
#         x = self.drp2(x)
#         x = self.pred(x)
#
#         return x


class LipNet_Pinyin(nn.Module):
    def __init__(self, args):
        super(LipNet_Pinyin, self).__init__()

        self.args = args
        self.video_cnn = VideoCNN(se=True)
        # self.videoEncode = VideoEncoder(enc_input=513, hidden_size=512, dropout=0.5, num_layers=1, LSTMorGRU=1)
        # self.pinyinEncode = PinyinEncoder(in_features=1024, hidden_size=1024, num_layers=1, GRUorLSTM=0)
        self.pinyinEncode = Encoder(input_dim=513, emb_dim=1024, enc_hid_dim=1024, dec_hid_dim=1024, dropout=0.5,
                                    num_layers=2)
        self.pinyinDecode = Decoder(in_dim=2048, output_dim=2048, emb_dim=2048, enc_hid_dim=1024, dec_hid_dim=1024,
                                    attention=Attention(enc_hid_dim=1024, dec_hid_dim=1024), num_layers=1, dropout=0.5)
        # self.characterDecode = CharacterDecoder(in_features=2048, hidden_size=1024, num_layers=1, GRUorLSTM=0)
        # self.PinyinMLP = MLP(in_features=512, out_features=48 + 1, layer=0)
        # self.CharacterMLP = MLP(in_features=1024 * 2, out_features=1000, layer=0)
        self.v_cls = nn.Linear(1024 * 2, self.args.n_class)
        self.PSoft = nn.LogSoftmax(dim=0)
        self.CSoft = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(p=0.5)
        self.embedding = nn.Embedding(2048, 2048)
        # self.LN = nn.LayerNorm([512])

        self.BorderCon = nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)

    def forward(self, x, tgt, src_st, src_ed, teacher_forcing_ratio=0.1, border=None):  # border  Batch_size,T

        B, T, C, H, W = x.size()[:]
        x = self.video_cnn(x)
        # x = self.LN(x)
        # if border is not None:
        #     border = border.unsqueeze(1)
        #     border = self.BorderCon(border)
        #     border = border.squeeze(1)
        # Xve, prev_hidden = self.videoEncode(torch.cat([x, border[:, :, None]], -1))
        # Xve, _ = self.gru(torch.cat([x, border[:, :, None]], -1))
        # Xve = self.dropout(Xve)
        # Pp = self.PSoft(self.PinyinMLP(Xve))

        Xpd, prev_hidden = self.pinyinEncode(torch.cat([x, border[:, :, None]], -1))
        # Xpd = self.dropout(Xpd)
        # Xcd = self.characterDecode(Xve)
        # Xcd = self.dropout(Xcd)

        tgt_len = tgt.shape[1]
        dec_input = tgt[:, 0]
        dec_input = dec_input.unsqueeze(1)
        dec_input = self.embedding(dec_input.long())
        dec_outputs = torch.zeros(B, tgt_len, 2048)
        for t in range(tgt_len):
            dec_output, prev_hidden = self.pinyinDecode(dec_input, prev_hidden, Xpd, src_st, src_ed)
            dec_outputs[:, t, :] = dec_output
            dec_input = dec_output.unsqueeze(1)
            # teacher_force = random.random() < teacher_forcing_ratio
            top1 = dec_output.argmax(1)
            # dec_input = tgt[:, t] if teacher_force else top1
        Pc = self.v_cls(dec_outputs.cuda()).mean(1)
        # Pc = self.v_cls(dec_outputs.cuda()).mean(1)

        # PinyinInput = Xpd
        # Ype = self.pinyinEncode(PinyinInput)
        # Ype = self.dropout(Ype)
        # Ycd = self.characterDecode(Ype)
        # Ycd = self.dropout(Ycd)

        # CharacterInput = torch.cat([Xve, Ype], -1)

        # PinyinPrediction
        # PinyinInput = PinyinInput.view(B, T, -1)
        # Pp = self.PSoft(self.PinyinMLP(Xve))

        # CharacterPrediction
        # CharacterInput = CharacterInput.view(B, -1)
        # Pc = self.CharacterMLP(CharacterInput).mean(1)
        # Pc = self.v_cls(Xve).mean(1)
        return Pc

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
    def __init__(self, in_features, out_features, layer=3):
        # TODO
        super(MLP, self).__init__()
        self.layer = layer
        if layer == 0:
            self.predict = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        elif layer == 1:
            self.hidden1 = nn.Linear(in_features=in_features, out_features=out_features * 2, bias=True)
            self.predict = nn.Linear(out_features * 2, out_features)
        elif layer == 2:
            self.hidden1 = nn.Linear(in_features=in_features, out_features=out_features * 2, bias=True)
            self.hidden2 = nn.Linear(out_features * 2, out_features * 2)
            self.predict = nn.Linear(out_features * 2, out_features)
        else:
            self.hidden1 = nn.Linear(in_features=in_features, out_features=out_features * 2, bias=True)
            self.hidden2 = nn.Linear(out_features * 2, out_features * 2)
            self.hidden3 = nn.Linear(out_features * 2, out_features * 2)
            self.predict = nn.Linear(out_features * 2, out_features)

    def forward(self, x):
        if self.layer == 0:
            pass
        elif self.layer == 1:
            x = F.relu(self.hidden1(x))
        elif self.layer == 2:
            x = F.relu(self.hidden1(x))
            x = F.relu(self.hidden2(x))
        else:
            x = F.relu(self.hidden1(x))
            x = F.relu(self.hidden2(x))
            x = F.relu(self.hidden3(x))
        output = self.predict(x)
        # out = output.view(-1)

        return output


class VideoEncoder(nn.Module):
    def __init__(self, enc_input, hidden_size, dropout=0.5, num_layers=2, LSTMorGRU=0):
        super(VideoEncoder, self).__init__()
        self.rnn_size = hidden_size
        # self.conv = nn.Sequential(
        #     nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3), bias=False),
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(True),
        #     nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        # )
        # self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
        if LSTMorGRU == 0:
            self.gru = nn.GRU(enc_input, self.rnn_size, num_layers, bidirectional=False, dropout=0.2)
        else:
            self.gru = nn.LSTM(enc_input, self.rnn_size, num_layers, bidirectional=False, dropout=0.2)
        self.dropout = nn.Dropout(dropout)
        # initialisations
        # for m in self.conv.modules():
        #     if isinstance(m, nn.Conv3d):
        #         init.kaiming_normal_(m.weight, nonlinearity='relu')
        #         if m.bias is not None:
        #             init.constant_(m.bias, 0)

        # init.kaiming_normal_(self.pred.weight, nonlinearity='sigmoid')
        # init.constant_(self.pred.bias, 0)

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
        output, hidden = self.gru(x)
        return output, hidden[-1].detach()


class PinyinEncoder(nn.Module):
    def __init__(self, in_features, hidden_size, num_layers, GRUorLSTM):
        super(PinyinEncoder, self).__init__()
        self.in_features = in_features
        self.rnn_size = hidden_size
        self.num_layers = num_layers
        self.GRUorLSTM = GRUorLSTM
        if GRUorLSTM == 0:
            self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers,
                              bidirectional=True,
                              dropout=0.2)
        else:
            self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers,
                                bidirectional=True,
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
    def __init__(self, in_features, hidden_size, num_layers, GRUorLSTM=0):
        super(PinyinDecoder, self).__init__()
        self.in_features = in_features
        self.rnn_size = hidden_size
        self.num_layers = num_layers
        self.GRUorLSTM = GRUorLSTM
        if GRUorLSTM == 0:
            self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers,
                              bidirectional=False,
                              dropout=0.2)
        else:
            self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers,
                                bidirectional=False,
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
    def __init__(self, in_features, hidden_size, num_layers, GRUorLSTM):
        super(ToneEncoder, self).__init__()
        self.in_features = in_features
        self.rnn_size = hidden_size
        self.num_layers = num_layers
        self.GRUorLSTM = GRUorLSTM
        if GRUorLSTM == 0:
            self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers,
                              bidirectional=True,
                              dropout=0.2)
        else:
            self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers,
                                bidirectional=True,
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
    def __init__(self, in_features, hidden_size, num_layers, GRUorLSTM):
        super(ToneDecoder, self).__init__()
        self.in_features = in_features
        self.rnn_size = hidden_size
        self.num_layers = num_layers
        self.GRUorLSTM = GRUorLSTM
        if GRUorLSTM == 0:
            self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers,
                              bidirectional=False,
                              dropout=0.2)
        else:
            self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers,
                                bidirectional=False,
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
    def __init__(self, in_features, hidden_size, num_layers, GRUorLSTM):
        super(CharacterDecoder, self).__init__()
        self.in_features = in_features
        self.rnn_size = hidden_size
        self.num_layers = num_layers
        self.GRUorLSTM = GRUorLSTM
        if GRUorLSTM == 0:
            self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers,
                              bidirectional=False,
                              dropout=0.2)
        else:
            self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers,
                                bidirectional=False,
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


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout=0.5, num_layers=2):
        super(Encoder, self).__init__()
        self.vocab_size = input_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(input_dim, enc_hid_dim, num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.LN = nn.LayerNorm([2048])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(enc_hid_dim * 2, dec_hid_dim * 2)
        self.relu = nn.ReLU()

    def forward(self, enc_input):
        # enc_input = self.embedding(enc_input)
        embedded = self.dropout(enc_input)
        # embedded = pack_padded_sequence(embedded, text_lengths.cpu().int(), batch_first=True, enforce_sorted=False)
        output, hidden = self.gru(embedded)
        output = self.LN(output)
        # output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.relu(self.linear(output))
        # hidden = torch.tanh(self.linear(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        # hidden = self.relu(self.linear(hidden[-2, :, :]))
        # output = self.relu(self.linear(output))
        return output, hidden[-2:, :, :]


class Decoder(nn.Module):
    def __init__(self, in_dim, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, attention, num_layers, dropout=0.5):
        super(Decoder, self).__init__()
        self.vocab_size = output_dim
        # self.embedding = nn.Embedding(in_dim, emb_dim)
        self.attention = attention
        self.gru = nn.GRU(enc_hid_dim * 2 + emb_dim, dec_hid_dim, num_layers=num_layers, bidirectional=True,
                          batch_first=True)
        self.LN = nn.LayerNorm([2048])
        self.linear = nn.Linear(enc_hid_dim * 2 + dec_hid_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, prev_hidden, enc_output, src_st, src_ed):
        # dec_input = [batch_size]
        # prev_hidden = [batch_size, hidden_size]
        # enc_output = [batch_size, src_len, hidden_size]
        # embedded = dec_input.unsqueeze(1)
        # embedded = self.embedding(embedded.long())
        a = self.attention(dec_input, enc_output, src_st, src_ed).unsqueeze(1)  # [batch_size, 1, src_len]
        c = torch.bmm(a, enc_output)  # [batch_size, 1, hidden_size]
        gru_input = torch.cat([dec_input, c], dim=2)
        # dec_output: [batch_size, 1, hidden_size]
        # dec_hidden: [1, batch_size, hidden_size]
        # prev_hidden 是上个时间步的隐状态，作为 decoder 的参数传入进来
        dec_output, dec_hidden = self.gru(gru_input, prev_hidden)
        dec_output = self.LN(dec_output)
        dec_output = self.linear(
            torch.cat((dec_output.squeeze(1), c.squeeze(1), dec_input.squeeze(1)), dim=1))  # [batch_size, vocab_size]
        return dec_output, dec_hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(enc_hid_dim * 2 + dec_hid_dim * 2, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, dec_input, enc_output, src_st, src_ed):
        # enc_output = [batch_size, seq_len, hidden_size]
        # dec_input = [batch_size, hidden_size]
        seq_len = enc_output.shape[1]
        s = dec_input.repeat(1, seq_len, 1)
        x = torch.tanh(self.linear(torch.cat([enc_output, s], dim=2)))
        attention = self.v(x).squeeze(-1)
        max_len = enc_output.shape[1]
        # mask = [batch_size, seq_len]
        length = torch.arange(max_len).expand(src_ed.shape[0], max_len).cuda()
        mask = (length >= src_ed.cuda().unsqueeze(1)) != (length < src_st.cuda().unsqueeze(1))
        attention.masked_fill_(mask.cuda(), float('-inf'))
        return self.softmax(attention)  # [batch, seq_len]

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
