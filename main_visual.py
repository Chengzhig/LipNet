import json

from matplotlib import pyplot as plt
from tensorboardX import writer
from torch.utils.data import DataLoader
import os
import time

from tqdm import tqdm

from dataloader import get_data_loaders
from model import *
import torch.optim as optim
import random
from LSR import LSR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import codecs

from utils.util import update_logger_batch, AverageMeter, mixup_data, mixup_criterion, CheckpointSaver, get_optimizer, \
    CosineScheduler, get_logger

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


def load_json(json_fp):
    assert os.path.isfile(json_fp), "Error loading JSON. File provided does not exist, cannot read: {}".format(json_fp)
    with open(json_fp, 'r') as f:
        json_content = json.load(f)
    return json_content


args_loaded = load_json('dctcn.json')
densetcn_options = {'block_config': args_loaded['densetcn_block_config'],
                    'growth_rate_set': args_loaded['densetcn_growth_rate_set'],
                    'reduced_size': args_loaded['densetcn_reduced_size'],
                    'kernel_size_set': args_loaded['densetcn_kernel_size_set'],
                    'dilation_size_set': args_loaded['densetcn_dilation_size_set'],
                    'squeeze_excitation': args_loaded['densetcn_se'],
                    'dropout': args_loaded['densetcn_dropout'],
                    }

video_model = VideoModel(args, densetcn_options=densetcn_options).cuda()


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

# weight = torch.load('/home/czg/LRW/pythonproject/checkpoints/lrw-1000-baseline/front3D_CBAM_lrw.pt',
#                     map_location=torch.device('cpu'))
# weight = torch.load(
#     '/home/mingwu/workspace_czg/pycharmproject/checkpoints/lrw-1000-baseline/front3D_CBAM_lrw.pt',
#     map_location=torch.device('cpu'))
# load_missing(video_model, weight.get('video_model'))
video_model = parallel_model(video_model)


def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        drop_last=True,
                        pin_memory=True)
    return loader


dset_loaders = get_data_loaders(args)


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

best_acc = 0.0


# chars = 'Cabcdefghijklmnopqrstuvwxyz '
# wbs = WordBeamSearch(28, 'NGrams', 0.0, corpus.encode('utf8'), chars.encode('utf8'),
#                      word_chars.encode('utf8'))

def EachClassAcc(classNum, classes):
    v_acc = []
    class_correct = list(0. for i in range(classNum))
    class_total = list(0. for i in range(classNum))
    with torch.no_grad():
        dataset = Dataset('val', args)
        print('Start Testing, Data Length:', len(dataset))
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)
        for (i_iter, input) in enumerate(loader):
            video_model.eval()
            tic = time.time()
            video = input.get('video').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True)
            border = input.get('duration').cuda(non_blocking=True).float()
            y_v = video_model(video, border)
            c = y_v.argmax(-1) == label
            for label_idx in range(len(label)):
                label_single = label[label_idx]
                class_correct[label_single] += c[label_idx].item()
                class_total[label_single] += 1
            v_acc.extend((y_v.argmax(-1) == label).cpu().numpy().tolist())
            toc = time.time()
            if (i_iter % 10 == 0):
                msg = ''
                msg = add_msg(msg, ' v_acc={:.5f}', np.array(v_acc).reshape(-1).mean())
                msg = add_msg(msg, 'eta={:.5f}', (toc - tic) * (len(loader) - i_iter) / 3600.0)
                print(msg)

        acc = float(np.array(v_acc).reshape(-1).mean())
        msg = 'v_acc_{:.5f}_'.format(acc)

        for i in range(classNum):
            print('Acc of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

        return acc, msg


def test(dset_loader, criterion):
    with torch.no_grad():
        print('start testing')

        v_acc = []
        total = 0
        t1 = time.time()
        running_loss = 0.
        running_corrects = 0.

        for batch_idx, data in enumerate(tqdm(dset_loader)):
            video_model.eval()
            input, lengths, labels, boundaries = data
            boundaries = boundaries.cuda()
            logits = model(input.unsqueeze(1).cuda(), lengths=lengths, boundaries=boundaries)
            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()

            loss = criterion(logits, labels.cuda())
            running_loss += loss.item() * input.size(0)

        print(f"{len(dset_loader.dataset)} in total\tCR: {running_corrects / len(dset_loader.dataset)}")
        return running_corrects / len(dset_loader.dataset), running_loss / len(dset_loader.dataset)


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


def train(model, dset_loader, logger):
    optimizer = get_optimizer(args, optim_policies=model.parameters())
    # -- get learning rate scheduler
    scheduler = CosineScheduler(args.lr, args.epochs)
    ckpt_saver = CheckpointSaver('./checkpoints/')
    max_epoch = args.max_epoch
    criterion = nn.CrossEntropyLoss()
    tot_iter = 0
    adjust_lr_count = 0
    alpha = 0.2
    beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
    scaler = GradScaler()
    global best_acc

    for epoch in range(max_epoch):
        data_time = AverageMeter()
        batch_time = AverageMeter()

        lsr = LSR()
        running_loss = 0.
        running_corrects = 0.
        running_all = 0.

        for batch_idx, data in enumerate(dset_loader):
            tic = time.time()

            video_model.train()
            input, lengths, labels, boundaries = data
            boundaries = boundaries.cuda()

            input, labels_a, labels_b, lam = mixup_data(input, labels, args.alpha)
            labels_a, labels_b = labels_a.cuda(), labels_b.cuda()

            labels_a, labels_b = labels_a.cuda(), labels_b.cuda()

            optimizer.zero_grad()

            logits = model(input.unsqueeze(1).cuda(), lengths=lengths, boundaries=boundaries)

            loss_func = mixup_criterion(labels_a, labels_b, lam)
            loss = loss_func(criterion, logits)

            loss.backward()
            optimizer.step()

            # measure elapsed time
            # -- compute running performance
            _, predicted = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_loss += loss.item() * input.size(0)
            running_corrects += lam * predicted.eq(labels_a.view_as(predicted)).sum().item() + (1 - lam) * predicted.eq(
                labels_b.view_as(predicted)).sum().item()
            running_all += input.size(0)
            # -- log intermediate results
            if batch_idx % args.interval == 0 or (batch_idx == len(dset_loader) - 1):
                update_logger_batch(args, logger, dset_loader, batch_idx, running_loss, running_corrects, running_all,
                                    batch_time, data_time)

        acc_avg_val, loss_avg_val = test(dset_loaders['val'], criterion)
        logger.info(
            f"{'val'} Epoch:\t{epoch:2}\tLoss val: {loss_avg_val:.4f}\tAcc val:{acc_avg_val:.4f}, LR: {showLR(optimizer)}")
        # -- save checkpoint
        save_dict = {
            'epoch_idx': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        ckpt_saver.save(save_dict, acc_avg_val)
        scheduler.adjust_lr(optimizer, epoch)
        epoch += 1


if (__name__ == '__main__'):

    if (args.test):
        acc, msg = EachClassAcc(1000, Dataset.pinyins)
        print(f'acc={acc}')
        exit()
    logger = get_logger(args, './checkpoints/')
    train(video_model, dset_loaders['train'], logger)
    exit()
