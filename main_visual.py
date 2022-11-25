import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torch.cuda import device
from torch.utils.data import DataLoader
import math
import os
import sys
import numpy as np
import time
from model import *
import torch.optim as optim
import random
import pdb
import shutil
from LSR import LSR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
# from word_beam_search import WordBeamSearch
from model.Mymodel import LipNet_Pinyin
from model.TCN import Lipreading
import codecs
import torchvision.transforms as transforms

torch.backends.cudnn.benchmark = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
if (args.dataset == 'lrw'):
    print("lrw")
    from utils import LRWDataset as Dataset
elif (args.dataset == 'lrw1000'):
    print("lrw1000")
    from utils import LRW1000_Dataset as Dataset
else:
    raise Exception('lrw or lrw1000')

video_model = VideoModel(args).cuda()


# video_model = VideoModel(args).cuda()
# video_MLP = MLP().cuda()

# print(NETModel)


def parallel_model(model):
    model = nn.DataParallel(model)
    return model


def load_missing(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]

    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict), len(model_dict)))
    print('miss matched params:', missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


lr = args.batch_size / 32.0 / torch.cuda.device_count() * args.lr
optim_video = optim.Adam(video_model.parameters(), lr=lr, weight_decay=1e-4)
scheduler_net = optim.lr_scheduler.CosineAnnealingLR(optim_video, T_max=args.max_epoch, eta_min=5e-6)

if (args.weights is not None):
    print('load weights')
    weight = torch.load(args.weights, map_location=torch.device('cpu'))
    load_missing(video_model, weight.get('video_model'))

    # model_weight_path = "/home/czg/lrw1000-border-se-mixup-label-smooth-cosine-lr-wd-1e-4-acc-0.56023.pt"
    # model.load_state_dict(torch.load(model_weight_path))
    # print(model)

weight = torch.load('/home/czg/lrw1000-border-se-mixup-label-smooth-cosine-lr-wd-1e-4-acc-0.56023_2.pt',
                    map_location=torch.device('cpu'))
# weight = torch.load(
#     '/home/mingwu/workspace_czg/pycharmproject/checkpoints/lrw-1000-baseline/lrw1000-border-se-mixup-label-smooth-cosine-lr-wd-1e-4-acc-0.56023.pt',
#     map_location=torch.device('cpu'))
load_missing(video_model, weight.get('video_model'))
video_model = parallel_model(video_model)


def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        drop_last=True,
                        pin_memory=True)
    return loader


def add_msg(msg, k, v):
    if (msg != ''):
        msg = msg + ','
    msg = msg + k.format(v)
    return msg


alpha = 1
beta = 1


def beam_search_decoder(post, top_k):
    """
    Parameters:
        post(Tensor) – the output probability of decoder. shape = (batch_size, seq_length, vocab_size).
        top_k(int) – beam size of decoder. shape
    return:
        indices(Tensor) – a beam of index sequence. shape = (batch_size, beam_size, seq_length).
        log_prob(Tensor) – a beam of log likelihood of sequence. shape = (batch_size, beam_size).
    """

    batch_size, seq_length, vocab_size = post.shape
    log_post = post.log()
    log_prob, indices = log_post[:, 0, :].topk(top_k, sorted=True)  # first word top-k candidates
    indices = indices.unsqueeze(-1)
    for i in range(1, seq_length):
        log_prob = log_prob.unsqueeze(-1) + log_post[:, i, :].unsqueeze(1).repeat(1, top_k, 1)  # word by word
        log_prob, index = log_prob.view(batch_size, -1).topk(top_k, sorted=True)
        indices = torch.cat([indices, index.unsqueeze(-1)], dim=-1)
    return indices, log_prob


data_path = './utils/'
corpus = codecs.open(data_path + 'corpus.txt', 'r', 'utf8').read()
chars = codecs.open(data_path + 'char.txt', 'r', 'utf8').read()
word_chars = codecs.open(data_path + 'wordChars.txt', 'r', 'utf8').read()


# chars = 'Cabcdefghijklmnopqrstuvwxyz '
# wbs = WordBeamSearch(28, 'NGrams', 0.0, corpus.encode('utf8'), chars.encode('utf8'),
#                      word_chars.encode('utf8'))

def test(istrain=0):
    tmprandom = random.randint(1, 20)
    with torch.no_grad():
        if istrain == 1:
            dataset = Dataset('train', args)
        else:
            dataset = Dataset('val', args)
        print('Start Testing, Data Length:', len(dataset))
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)

        print('start testing')
        v_acc = []
        p_acc = []
        entropy = []
        acc_mean = []
        total = 0
        cons_acc = 0.0
        cons_total = 0.0
        attns = []

        count = 0
        Tp = 0
        Tn_1 = 0
        Tn_2 = 0
        t1 = time.time()

        for (i_iter, input) in enumerate(loader):
            if istrain == 1 and i_iter % 20 != tmprandom:
                continue
            video_model.eval()

            tic = time.time()
            video = input.get('video').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True)
            total = total + video.size(0)
            names = input.get('name')
            border = input.get('duration').cuda(non_blocking=True).float()

            with autocast():
                if (args.border):
                    y_v, _ = video_model(video, border)
                else:
                    y_v, _ = video_model(video)

            v_acc.extend((y_v.argmax(-1) == label).cpu().numpy().tolist())
            # v_acc.append(Tp / (Tp + Tn_1 + Tn_2))

            toc = time.time()
            if (i_iter % 10 == 0):
                msg = ''
                msg = add_msg(msg, ' v_acc={:.5f}', np.array(v_acc).reshape(-1).mean())
                # msg = add_msg(msg, ' p_acc={:.5f}', np.array(p_acc).reshape(-1).mean())
                msg = add_msg(msg, 'eta={:.5f}', (toc - tic) * (len(loader) - i_iter) / 3600.0)

                print(msg)

        acc = float(np.array(v_acc).reshape(-1).mean())
        # acc = float(np.array(p_acc).reshape(-1).mean())
        msg = 'v_acc_{:.5f}_'.format(acc)

        return acc, msg


def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += ['{:.6f}'.format(param_group['lr'])]
    return ','.join(lr)


def loss_fn(pred, target):
    return -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred)).sum()


def visualize_feature(model, x, border=None):
    # forward
    im = x.cpu().detach().numpy()[0, 20, :, :, :]
    im = np.transpose(im, [1, 2, 0])
    plt.figure()
    plt.imshow(im[:, :, 0], cmap='gray')
    plt.show()
    _, out_put = model(x, border)
    for feature_map in out_put:
        print(feature_map.size())
        # [N, T, C, H, W] -> [C, H, W]
        im = feature_map.cpu().detach().numpy()[0, 20, :, :, :]
        # [C, H, W] -> [H, W, C]
        im = np.transpose(im, [1, 2, 0])
        # show top 12 feature maps
        plt.figure()
        for i in range(64):
            ax = plt.subplot(8, 8, i + 1)
            # [H, W, C]
            plt.imshow(im[:, :, i], cmap='gray')
        plt.show()


def train():
    dataset = Dataset('train', args)
    print('Start Training, Data Length:', len(dataset))

    loader = dataset2dataloader(dataset, args.batch_size, args.num_workers)

    max_epoch = args.max_epoch

    ce = nn.CrossEntropyLoss()

    tot_iter = 0
    best_acc = 0.0
    best_acc_a = 0.0
    adjust_lr_count = 0
    alpha = 0.2
    beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
    scaler = GradScaler()

    # checkpoint = torch.load("./Two_roads/lrw-1000-baseline/_weight_pinyin.pt")
    # video_model.module.load_state_dict(checkpoint['video_model'])
    # NETModel.module.load_state_dict(checkpoint['NETModel'])
    # print("loading premodel")

    for epoch in range(max_epoch):
        total = 0.0
        v_acc = 0.0
        total = 0.0

        lsr = LSR()

        for (i_iter, input) in enumerate(loader):
            tic = time.time()

            video_model.train()
            video = input.get('video').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True).long()
            border = input.get('duration').cuda(non_blocking=True).float()

            visualize_feature(video_model, video, border)

            loss = {}

            if (args.label_smooth):
                loss_fn = lsr
            else:
                loss_fn = nn.CrossEntropyLoss()

            with autocast():
                if (args.mixup):
                    lambda_ = np.random.beta(alpha, alpha)
                    index = torch.randperm(video.size(0)).cuda(non_blocking=True)

                    mix_video = lambda_ * video + (1 - lambda_) * video[index, :]
                    mix_border = lambda_ * border + (1 - lambda_) * border[index, :]

                    label_a, label_b = label, label[index]

                    if (args.border):
                        y_v, _ = video_model(mix_video, mix_border)
                    else:
                        y_v, _ = video_model(mix_video)

                    loss_bp = lambda_ * loss_fn(y_v, label_a) + (1 - lambda_) * loss_fn(y_v, label_b)

                else:
                    if (args.border):
                        y_v, _ = video_model(video, border)
                    else:
                        y_v, _ = video_model(video)

                    loss_bp = loss_fn(y_v, label)

            loss['CE V'] = loss_bp

            optim_video.zero_grad()
            scaler.scale(loss_bp).backward()
            scaler.step(optim_video)
            scaler.update()

            toc = time.time()

            msg = 'epoch={},train_iter={},eta={:.5f}'.format(epoch, tot_iter,
                                                             (toc - tic) * (len(loader) - i_iter) / 3600.0)
            for k, v in loss.items():
                msg += ',{}={:.5f}'.format(k, v)
            msg = msg + str(',lr=' + str(showLR(optim_video)))
            msg = msg + str(',best_acc={:2f}'.format(best_acc))
            msg = msg + str(',best_acc_a={:2f}'.format(best_acc_a))
            print(msg)

            # or i_iter == 0
            if i_iter == len(loader) - 1:
                acc, msg = test()
                # acc_a, msg_a = test(1)
                if (acc > best_acc):
                    savename = '{}front3d_resnet.pt'.format(args.save_prefix)
                    temp = os.path.split(savename)[0]
                    if (not os.path.exists(temp)):
                        os.makedirs(temp)
                    torch.save(
                        {
                            'video_model': video_model.module.state_dict(),
                        }, savename)

                if (tot_iter != 0):
                    best_acc = max(acc, best_acc)
                    # best_acc_a = max(acc_a, best_acc_a)

            tot_iter += 1
        scheduler_net.step()


def computeACC(pinyin, pinyinlable, target_length):
    count = 0
    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    y_v = torch.softmax(pinyin, 2)
    targets = []
    for index, length in enumerate(target_length):
        label = pinyinlable[index, :length]
        targets.append(label)
        if index == 0:
            tmpPinyinLable = label
        else:
            tmpPinyinLable = torch.cat([tmpPinyinLable, label], dim=0)
    preb_labels = []
    for i in range(y_v.shape[0]):
        preb = y_v[i, :, :]
        preb_label = preb.argmax(dim=1)
        no_repeat_blank_label = []
        pre_c = preb_label[0]
        if pre_c != 28:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:  # dropout repeate label and blank label
            if (pre_c == c) or (c == 28):
                if c == 28:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)

    for i, label in enumerate(preb_labels):
        label = torch.tensor(label).cuda()
        targets[i] = targets[i].cuda()
        # print('================')
        # print(label)
        # print(targets[i])
        # print('================')
        if len(label) != len(targets[i]):
            Tn_1 += 1
            continue
        if targets[i].eq(label).all():
            print("success predict:")
            print(label.detach().cpu().numpy())
            Tp += 1
        else:
            Tn_2 += 1
            count += 1

            print("[Info] Validation Accuracy: {} [{}:{}:{}:{}]".format(Tp / (Tp + Tn_1 + Tn_2), Tp, Tn_1, Tn_2,
                                                                        (Tp + Tn_1 + Tn_2)))
            y_v1 = y_v1.float()
    return Tp / (Tp + Tn_1 + Tn_2)


def getLable(i):
    dict = [" C", " a", "ai", " ai ", " an", "  an jian", "  an quan", " an zhao", "  ba", "  ba li", "  ba xi",
            "  bai", "  ban", "  ban dao", " ban fa", "  bang jia", "  bao", "  bao chi", " bao dao", "  bao gao",
            "  bao hu", " bao kuo", "  bao yu", " bao zhang", "  bei", "  bei bu", " bei jing", " bei jing shi jian",
            " bei yue", " ben", "ben ci", " ben yue", " bi", " bi jiao", "bi ru", " bi xu", "bian hua", " biao da",
            " biao shi", " biao zhi", "biao zhun", " bie", "bing", "bing qie", " bo chu", " bu", "bu duan", " bu fen",
            "bu guo", "bu hui", "bu jin", "bu men", "bu neng", "bu shao", "bu shu", " bu tong", "bu yao", "bu zhang",
            "cai", " cai fang", "cai qu", "can jia", " can yu", "ce", "ceng jing", " chan pin", "chan sheng", "chan ye",
            "chang", "chang qi", " chang yi", "chao guo", " chao xian", "che", " cheng", "cheng gong", " cheng guo",
            "cheng li", "cheng nuo", "cheng shi", " cheng wei", "chi", "chi xu", " chu", "chu lai", "chu le", "chu li",
            " chu xi", "chu xian", " chuan tong", " chuang xin", " chun", " ci", " ci ci", " ci qian", "cong", "cu jin",
            "cun zai", "cuo shi", "da", "da cheng", " da dao", "da gai", "da hui", " da ji", "da jia", "  da xing",
            "  da xue", "  da yu", "  da yue", "  da zao", "  dai", "  dai biao", "  dai lai", "  dan", "  dan shi",
            "  dang", "  dang di", "  dang qian", "  dang shi", "  dang tian", "  dang zhong", "  dao", "  dao le",
            "  dao zhi", "  de", "  de dao", "  de guo", "  deng", "  deng deng", "  di", "  di da", "  di fang",
            "  di qu", "  di zhi", "  dian", "  dian shi", "  diao cha", "  diao yan", "  dong", "  dong bu",
            "  dong fang", "  dong li", "  dong xi", "  dou", "  dou shi", "  du", "  duan", "  duan jiao", "  dui",
            "  dui hua", "  dui wai", "  dui yu", "  duo", "  duo ci", "  duo nian", "  e", "  e luo si", "  er",
            "  er ling yi qi", "  er qie", "  er shi", "  er shi liu", "  er shi qi", "  er shi san", "  er shi si",
            "  er shi wu", "  er shi yi", "  er tong", "  er wei ma", "  fa", "  fa biao", "  fa bu", "  fa bu hui",
            "  fa chu", "  fa guo", "  fa hui", "  fa she", "  fa sheng", "  fa xian", "  fa yan ren", "  fa yuan",
            "  fa zhan", "  fan", "  fan dui", "  fan rong", "  fan wei", "  fan zui", "  fang", "  fang an",
            "  fang fan", "  fang mian", "  fang shi", "  fang wen", "  fang xiang", "  fei", "  fei chang", "  fen",
            "  fen bie", "  fen qi", "  fen zhong", "  fen zi", "  feng hui", "  feng shuo", "  feng xian", "  fu",
            "  fu jian", "  fu pin", "  fu wu", "  fu ze", "  fu ze ren", "  gai", "  gai bian", "  gai ge",
            "  gai shan", "  gan", "  gan jue", "  gan shou", "  gan xie", "  gang", "  gang gang", "  gao", "  gao du",
            "  gao feng", "  gao ji", "  gao wen", "  gao xiao", "  ge", "  ge de", "  ge fang", "  ge guo", "  ge jie",
            "  ge ren", "  ge wei", "  gei", "  gen", "  gen ju", "  geng", "  geng duo", "  geng hao", "  geng jia",
            "  gong an", "  gong bu", "  gong cheng", "  gong gong", "  gong he guo", "  gong kai", "  gong li",
            "  gong min", "  gong shi", "  gong si", "  gong tong", "  gong xian", "  gong xiang", "  gong ye",
            "  gong you", "  gong zuo", "  gou tong", "  guan jian", "  guan li", "  guan xi", "  guan xin",
            "  guan yu", "  guan zhong", "  guan zhu", "  guang dong", "  guang fan", "  guang xi", "  gui ding",
            "  gui fan", "  gui zhou", "  guo", "  guo cheng", "  guo fang bu", "  guo ji", "  guo jia", "  guo lai",
            "  guo min dang", "  guo nei", "  guo qu", "  guo wu yuan", "  ha", "  hai", "  hai shang", "  hai shi",
            "  hai wai", "  hai you", "  hai zi", "  han", "  han guo", "  hao", "  he", "  he bei", "  he nan",
            "  he ping", "  he xin", "  he zuo", "  hen", "  hen duo", "  hou", "  hu", "  hu bei", "  hu lian wang",
            "  hu nan", "  hu xin", "  hua", "  hua bei", "  hua ti", "  huan jing", "  huan ying", "  huang", "  hui",
            "  hui dao", "  hui gui", "  hui jian", "  hui shang", "  hui tan", "  hui wu", "  hui yi", "  hui ying",
            "  huo ban", "  huo bi", "  huo de", "  huo dong", "  huo li", "  huo zai", "  huo zhe", "  ji", "  ji ben",
            "  ji chang", "  ji chu", "  ji di", "  ji duan", "  ji gou", "  ji guan", "  ji hua", "  ji ji",
            "  ji jiang", "  ji lu", "  ji shi", "  ji shu", "  ji tuan", "  ji xu", "  ji yu", "  ji zhe", "  ji zhi",
            "  ji zhong", "  jia", "  jia bin", "  jia ge", "  jia qiang", "  jian", "  jian chi", "  jian ding",
            "  jian guan", "  jian jue", "  jian kang", "  jian li", "  jian she", "  jiang", "  jiang hua",
            "  jiang hui", "  jiang su", "  jiang xi", "  jiang yu", "  jiao", "  jiao liu", "  jiao tong", "  jiao yu",
            "  jie", "  jie duan", "  jie guo", "  jie jue", "  jie mu", "  jie shao", "  jie shou", "  jie shu",
            "  jie xia lai", "  jie zhi", "  jin", "  jin nian", "  jin nian lai", "  jin qi", "  jin ri", "  jin rong",
            "  jin ru", "  jin tian", "  jin xing", "  jin yi bu", "  jin zhan", "  jin zhuan", "  jing", "  jing fang",
            "  jing guo", "  jing ji", "  jing shen", "  jiu", "  jiu shi", "  jiu shi shuo", "  ju", "  ju ban",
            "  ju da", "  ju jiao", "  ju li", "  ju min", "  ju shi", "  ju ti", "  ju xing", "  ju you", "  jue de",
            "  jue ding", "  jun", "  jun fang", "  jun shi", "  ka", "  ka ta er", "  kai", "  kai fa", "  kai fang",
            "  kai mu", "  kai mu shi", "  kai qi", "  kai shi", "  kai zhan", "  kan", "  kan dao", "  kan kan",
            "  kao", "  ke", "  ke hu duan", "  ke ji", "  ke neng", "  ke xue", "  ke yi", "  kong jian", "  kong zhi",
            "  kuai", "  la", "  lai", "  lai shuo", "  lai zi", "  lan", "  lang", "  lao", "  le", "  lei", "  li",
            "  li ji", "  li liang", "  li mian", "  li shi", "  li yi", "  li yong", "  lian", "  lian bang",
            "  lian he", "  lian he guo", "  lian xi", "  lian xian", "  lian xu", "  liang", "  liang an",
            "  liang hao", "  liao jie", "  lin", "  ling chen", "  ling dao", "  ling dao ren", "  ling wai",
            "  ling yu", "  liu", "  long", "  lou", "  lu", "  lu xu", "  lun", "  lun tan", "  luo", "  luo shi",
            "  ma", "  mao yi", "  mei", "  mei guo", "  mei nian", "  mei ti", "  mei you", "  men", "  meng",
            "  meng xiang", "  mi", "  mi qie", "  mi shu zhang", "  mian dui", "  mian lin", "  min zhong", "  ming",
            "  ming que", "  ming tian", "  ming xian", "  mo", "  mu biao", "  mu qian", "  n", "  na", "  na ge",
            "  na me", "  nan bu", "  nan fang", "  ne", "  nei", "  nei rong", "  neng", "  neng gou", "  neng li",
            "  neng yuan", "  ni", "  nian", "  nian qing", "  nin", "  nu li", "  ou meng", "  ou zhou", "  peng you",
            "  pi", "  pian", "  pin dao", "  ping jia", "  ping tai", "  pu bian", "  pu jing", "  qi", "  qi che",
            "  qi dai", "  qi dong", "  qi jian", "  qi lai", "  qi shi", "  qi ta", "  qi wen", "  qi ye",
            "  qi zhong", "  qian", "  qian shu", "  qiang", "  qiang diao", "  qiang jiang yu", "  qiang lie",
            "  qiao", "  qing", "  qing kuang", "  qing zhu", "  qu", "  qu de", "  qu nian", "  qu xiao", "  qu yu",
            "  quan", "  quan bu", "  quan guo", "  quan mian", "  quan qiu", "  quan ti", "  que", "  que bao",
            "  que ding", "  que ren", "  ran hou", "  rang", "  re", "  ren", "  ren he", "  ren lei", "  ren min",
            "  ren min bi", "  ren shi", "  ren shu", "  ren wei", "  ren wu", "  ren yuan", "  reng ran", "  ri",
            "  ri ben", "  ri qian", "  rong he", "  ru guo", "  ru he", "  san", "  san nian", "  san shi",
            "  san tian", "  sao miao", "  sen", "  sha te", "  shan", "  shang", "  shang hai", "  shang sheng",
            "  shang wang", "  shang wu", "  shang ye", "  shao", "  shao hou", "  she bei", "  she hui", "  she ji",
            "  she shi", "  she xian", "  shen", "  shen fen", "  shen hua", "  shen me", "  shen ru", "  shen zhi",
            "  sheng", "  sheng chan", "  sheng huo", "  sheng ji", "  sheng ming", "  shi", "  shi ba", "  shi bu shi",
            "  shi chang", "  shi dai", "  shi er", "  shi gu", "  shi hou", "  shi ji", "  shi ji shang", "  shi jian",
            "  shi jie", "  shi jiu", "  shi liu", "  shi pin", "  shi qi", "  shi san", "  shi shi", "  shi si",
            "  shi wei", "  shi wu", "  shi xian", "  shi yan", "  shi ye", "  shi yong", "  shi zhong", "  shou",
            "  shou ci", "  shou dao", "  shou du", "  shou kan", "  shou shang", "  shou xian", "  shou xiang",
            "  shu", "  shu ji", "  shu ju", "  shuang", "  shuang fang", "  shui", "  shui ping", "  shuo",
            "  shuo shi", "  si", "  si chuan", "  si shi", "  si wang", "  sou suo", "  sui", "  sui zhe", "  suo",
            "  suo wei", "  suo yi", "  suo you", "  suo zai", "  ta", "  ta men", "  tai", "  tai wan", "  tan suo",
            "  tao", "tao lun", " te", " te bie", "ti", " ti chu", "ti gao", "ti gong", "ti sheng", " ti shi",
            "ti xian", " ti zhi", " tian", " tian qi", " tian ran qi", " tiao", "tiao jian", " tiao zhan",
            " tiao zheng", " tie lu", " ting", "tong", " tong bao", " tong guo", " tong ji", "tong shi", "tong yi",
            "tou piao", " tou zi", "tu po", " tuan dui", "tuan jie", "tui chu", " tui dong", " tui jin", "wai",
            "wai jiao", "wai jiao bu", "wai zhang", "wan", "wan cheng", "wan quan", " wan shang", " wang", "wang zhan",
            " wei", " wei fa", "wei fan", " wei hu", " wei lai", "wei le", "wei sheng", "wei xian", "wei xie",
            " wei yu", "wei yuan", " wei yuan hui", "wei zhi", "wen", "wen ding", "wen hua", " wen ming", "wen ti",
            " wo", "wo guo", "wo men", "wu", " wu ren ji", "wu shi", " xi", "xi bu", "xi huan", "xi ji", "xi jin ping",
            "xi lie", "xi tong", "xi wang", "xia", "xia mian", "xia wu", "xia zai", "xian", "xian chang", "xian jin",
            " xian sheng", " xian shi", " xian yi ren", " xian zai", "xiang", " xiang gang", " xiang guan", " xiang mu",
            " xiang xi", " xiang xin", " xiao", " xiao shi", "xiao xi", "xie shang", "xie tiao", " xie yi", " xin",
            " xin wen", " xin wen lian bo", "xin xi", " xin xin", " xin xing", " xin yi lun", "xing", "xing cheng",
            "xing dong", " xing shi", "xing wei", "xing zheng", " xu li ya", " xu yao", " xuan bu", "xuan ju",
            "xuan ze", "xue", " ya", "yan fa", " yan ge", "yan jiu", " yan zhong", "yang shi", " yao", " yao qing",
            "yao qiu", " ye", " yi", "yi ci", " yi dao", " yi dian", "yi ding", " yi dong", " yi fa", " yi ge",
            " yi hou", "yi hui", " yi ji", " yi jian", "yi jing", "yi kuai", "yi lai", "yi liao", " yi lu", "yi qi",
            "yi qie", "yi shang", " yi shi", "yi si lan", " yi ti", "yi wai", " yi wei zhe", " yi xi lie", "yi xia",
            " yi xie", "yi yang", "yi yi", "yi yuan", "yi zhi", " yi zhong", "yin", " yin du", "yin fa", "yin hang",
            " yin qi", " yin wei", " ying", " ying dui", "ying gai", " ying guo", "ying ji", "ying lai", " ying xiang",
            " yong", "yong you", " you", "you de", " you guan", " you hao", "you qi", " you shi", " you suo",
            "you xiao", " you yi", " you yu", " yu", " yu hui", "yu ji", " yu jing", " yu yi", " yuan", " yuan yi",
            "yuan yin", " yuan ze", "yue", "yue lai yue", " yun ying", " zai", " zai ci", " zai hai", " zai jian",
            "zan men", "zao", " zao cheng", " zao yu", "ze ren", " zen me", "zen me yang", " zeng jia", "zhan",
            "zhan kai", "zhan lve", "zhan shi", "zhan zai", "zhang", " zhao dao", " zhao kai", "zhe", " zhe ge",
            "zhe jiang", " zhe li", "zhe me", "zhe ming", "zhe yang", " zhe zhong", "zhei xie", "zhen de", " zhen dui",
            "zhen zheng", "zheng", "zheng ce", " zheng chang", "zheng fu", " zheng ge", " zheng shi", " zheng zai",
            "zheng zhi", " zhi", " zhi bo", " zhi chi", " zhi chu", "zhi dao", " zhi du", " zhi hou", " zhi jian",
            " zhi jie", "zhi neng", "zhi qian", "zhi shao", " zhi shi", "zhi wai", "zhi xia", " zhi xing", " zhi you",
            " zhi zao", " zhong", " zhong bu", " zhong da", " zhong dian", " zhong e", "zhong fang",
            "zhong gong zhong yang", "zhong gong zhong yang zheng zhi ju", "zhong guo", " zhong hua min zu",
            "zhong shi", "zhong wu", "zhong xin", " zhong yang", "zhong yang qi xiang tai", " zhong yao", " zhou",
            " zhou nian", " zhu", "zhu he", " zhu quan", " zhu ti", " zhu xi", "zhu yao", " zhu yi", "zhua", " zhuan",
            " zhuan ji", " zhuan jia", "zhuan xiang", "zhuan ye", " zhuang tai", " zhun bei", "zi", " zi ben", "zi ji",
            "zi jin", " zi xun", " zi you", " zi yuan", "zi zhu", "zong", " zong he", "zong li", " zong shu ji",
            " zong tong", "zou", "zu", "zu guo", "zu zhi", " zui", "zui gao", "zui hou", "zui jin", " zui xin",
            "zui zhong", " zun zhong", "zuo", " zuo chu", " zuo dao", " zuo hao", " zuo tian", "zuo wei", "zuo yong",
            "zuo you"]
    return dict[i]


# 单韵母:a 0 o 1 e 2 i 3 u 4 v 5
# 复韵母:ai 6 ei 7 ui 8 ao 9 ou 10 iu 11 ie 12 ve 13 er 14
# 前鼻韵母:an 15 en 16 in 17 un 18 vn 19
# 后鼻韵母:ang 20 eng 21 ing 22 ong 23
# ch 25 sh 26 zh 27 c 28 s 29 z 30 r 31
# C 24
# dict = ["a ", " o", " e ", " i ", " u ", "v ", "ai ", " ei ", " ui ", " ao ", " ou ", " iu ", " ie ", " ve ",
#         " er ", "an ", " en ", " in ", " un ", " vn ", " ang ", " eng ", " ing ", " ong ", " b ", " p ", " m ",
#         " f ", " d ", " t ", " n ", " l ", " g ", " k ", " h ", " j ", " q ", " x ", " zh ", " ch ", " sh ",
#         " r ", " z ", " c ", " s ", " y ", " w ", " C "]
dict = ['C', 'a', 'ai', 'ji', 'an', 'jian',
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


def getPinYin(list):
    res = []
    for i, input in enumerate(list):
        res.append(dict[input])
    return res


def getPho(list):
    res = []
    for i, input in enumerate(list):
        if input == 0:
            continue
        res.append(dict[i])
    return res


def data_normal_2d(orign_data, dim="col"):
    if dim == "col":
        dim = 1
        d_min = torch.min(orign_data, dim=dim)[0]
        for idx, j in enumerate(d_min):
            if j < 0:
                orign_data[idx, :] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data, dim=dim)[0]
    else:
        dim = 0
        d_min = torch.min(orign_data, dim=dim)[0]
        for idx, j in enumerate(d_min):
            if j < 0:
                orign_data[idx, :] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data, dim=dim)[0]
    d_max = torch.max(orign_data, dim=dim)[0]
    dst = d_max - d_min
    if d_min.shape[0] == orign_data.shape[0]:
        d_min = d_min.unsqueeze(1)
        dst = dst.unsqueeze(1)
    else:
        d_min = d_min.unsqueeze(0)
        dst = dst.unsqueeze(0)
    norm_data = torch.sub(orign_data, d_min).true_divide(dst)
    return norm_data


if (__name__ == '__main__'):

    if (args.test):
        acc, msg = test()
        print(f'acc={acc}')
        exit()
    train()
    exit()
