import json

from matplotlib import pyplot as plt
from tensorboardX import writer
from torch.utils.data import DataLoader
import os
import time

from tqdm import tqdm

from dataloader import get_data_loaders, get_preprocessing_pipelines
from model import *
import torch.optim as optim
import random
from LSR import LSR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import codecs

from utils.util import update_logger_batch, AverageMeter, mixup_data, mixup_criterion, CheckpointSaver, get_optimizer, \
    CosineScheduler, get_logger, load_model, save2npz, get_save_folder, calculateNorm2

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
parser.add_argument('--modality', default='video', choices=['video', 'audio'], help='choose the modality')

# -- directory
parser.add_argument('--data-dir', default='./datasets/LRW_h96w96_mouth_crop_gray', help='Loaded data directory')
parser.add_argument('--label-path', type=str, default='./labels/500WordsSortedList.txt',
                    help='Path to txt file with labels')
parser.add_argument('--annonation-direc', default=None, help='Loaded data directory')
# -- model config
parser.add_argument('--backbone-type', type=str, default='resnet', choices=['resnet', 'shufflenet'],
                    help='Architecture used for backbone')
parser.add_argument('--relu-type', type=str, default='relu', choices=['relu', 'prelu'], help='what relu to use')
parser.add_argument('--width-mult', type=float, default=1.0, help='Width multiplier for mobilenets and shufflenets')
# -- TCN config
parser.add_argument('--tcn-kernel-size', type=int, nargs="+", help='Kernel to be used for the TCN module')
parser.add_argument('--tcn-num-layers', type=int, default=4, help='Number of layers on the TCN module')
parser.add_argument('--tcn-dropout', type=float, default=0.2, help='Dropout value for the TCN module')
parser.add_argument('--tcn-dwpw', default=False, action='store_true',
                    help='If True, use the depthwise seperable convolution in TCN architecture')
parser.add_argument('--tcn-width-mult', type=int, default=1, help='TCN width multiplier')
# -- DenseTCN config
parser.add_argument('--densetcn-block-config', type=int, nargs="+", help='number of denselayer for each denseTCN block')
parser.add_argument('--densetcn-kernel-size-set', type=int, nargs="+", help='kernel size set for each denseTCN block')
parser.add_argument('--densetcn-dilation-size-set', type=int, nargs="+",
                    help='dilation size set for each denseTCN block')
parser.add_argument('--densetcn-growth-rate-set', type=int, nargs="+", help='growth rate for DenseTCN')
parser.add_argument('--densetcn-dropout', default=0.2, type=float, help='Dropout value for DenseTCN')
parser.add_argument('--densetcn-reduced-size', default=256, type=int,
                    help='the feature dim for the output of reduce layer')
parser.add_argument('--densetcn-se', default=False, action='store_true', help='If True, enable SE in DenseTCN')
parser.add_argument('--densetcn-condense', default=False, action='store_true', help='If True, enable condenseTCN')
# -- train
parser.add_argument('--training-mode', default='tcn', help='tcn')
parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'])
parser.add_argument('--init-epoch', default=0, type=int, help='epoch to start at')
# -- mixup
parser.add_argument('--alpha', default=0.4, type=float, help='interpolation strength (uniform=1., ERM=0.)')
# -- test
parser.add_argument('--model-path', type=str,
                    default='/home/czg/LRW/pythonproject/train_logs/tcn/2022-12-07T13:57:02/ckpt.best.pth',
                    help='Pretrained model pathname')
parser.add_argument('--allow-size-mismatch', default=True, action='store_true',
                    help='If True, allows to init from model with mismatching weight tensors. Useful to init from model with diff. number of classes')
# -- feature extractor
parser.add_argument('--extract-feats', default=False, action='store_true', help='Feature extractor')
parser.add_argument('--mouth-patch-path', type=str, default=None,
                    help='Path to the mouth ROIs, assuming the file is saved as numpy.array')
parser.add_argument('--mouth-embedding-out-path', type=str, default=None,
                    help='Save mouth embeddings to a specificed path')
# -- json pathname
parser.add_argument('--config_path', type=str, default='dctcn.json', help='Model configuration with json format')
# -- other vars
parser.add_argument('--interval', default=50, type=int, help='display interval')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
# paths
parser.add_argument('--logging-dir', type=str, default='./train_logs',
                    help='path to the directory in which to save the log file')
# use boundaries
parser.add_argument('--use-boundary', default=False, action='store_true',
                    help='include hard border at the testing stage.')

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


# if (args.weights is not None):
#     print('load weights')
#     weight = torch.load(args.weights, map_location=torch.device('cpu'))
#     load_missing(video_model, weight.get('video_model'))


# video_model = parallel_model(video_model)


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


def EachClassAcc(model, classNum, classes):
    v_acc = []
    class_correct = list(0. for i in range(classNum))
    class_total = list(0. for i in range(classNum))
    with torch.no_grad():
        dataset = Dataset('val', args)
        print('Start Testing, Data Length:', len(dataset))
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)
        for (i_iter, input) in enumerate(loader):
            model.eval()
            tic = time.time()
            video = input.get('video').cuda(non_blocking=True)
            label = input.get('label').cuda(non_blocking=True)
            border = input.get('duration').cuda(non_blocking=True).float()
            y_v = model(video, border)
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


def test(model, dset_loader, criterion):
    model.eval()

    running_loss = 0.
    running_corrects = 0.

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dset_loader)):
            if args.use_boundary:
                input, lengths, labels, boundaries = data
                boundaries = boundaries.cuda()
            else:
                input, lengths, labels = data
                boundaries = None
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


def train(model, dset_loader, criterion, epoch, optimizer, logger):
    data_time = AverageMeter()
    batch_time = AverageMeter()

    lr = showLR(optimizer)

    logger.info('-' * 10)
    logger.info(f"Epoch {epoch}/{args.max_epoch - 1}")
    logger.info(f"Current learning rate: {lr}")

    model.train()
    running_loss = 0.
    running_corrects = 0.
    running_all = 0.

    end = time.time()
    for batch_idx, data in enumerate(dset_loader):
        if args.use_boundary:
            input, lengths, labels, boundaries = data
            boundaries = boundaries.cuda()
        else:
            input, lengths, labels = data
            boundaries = None
        # measure data loading time
        data_time.update(time.time() - end)

        # --
        input, labels_a, labels_b, lam = mixup_data(input, labels, args.alpha)
        labels_a, labels_b = labels_a.cuda(), labels_b.cuda()

        optimizer.zero_grad()

        logits = model(input.unsqueeze(1).cuda(), lengths=lengths, boundaries=boundaries)

        loss_func = mixup_criterion(labels_a, labels_b, lam)
        loss = loss_func(criterion, logits)

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
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

    return model


def extract_feats(model):
    """
    :rtype: FloatTensor
    """
    model.eval()
    preprocessing_func = get_preprocessing_pipelines()['test']
    data = preprocessing_func(np.load(args.mouth_patch_path)['data'])  # data: TxHxW
    return model(torch.FloatTensor(data)[None, None, :, :, :].cuda(), lengths=[data.shape[0]])


def get_model_from_json():
    assert args.config_path.endswith('.json') and os.path.isfile(args.config_path), \
        f"'.json' config path does not exist. Path input: {args.config_path}"
    args_loaded = load_json(args.config_path)
    args.backbone_type = args_loaded['backbone_type']
    args.width_mult = args_loaded['width_mult']
    args.relu_type = args_loaded['relu_type']
    args.use_boundary = args_loaded.get("use_boundary", False)

    if args_loaded.get('tcn_num_layers', ''):
        tcn_options = {'num_layers': args_loaded['tcn_num_layers'],
                       'kernel_size': args_loaded['tcn_kernel_size'],
                       'dropout': args_loaded['tcn_dropout'],
                       'dwpw': args_loaded['tcn_dwpw'],
                       'width_mult': args_loaded['tcn_width_mult'],
                       }
    else:
        tcn_options = {}
    if args_loaded.get('densetcn_block_config', ''):
        densetcn_options = {'block_config': args_loaded['densetcn_block_config'],
                            'growth_rate_set': args_loaded['densetcn_growth_rate_set'],
                            'reduced_size': args_loaded['densetcn_reduced_size'],
                            'kernel_size_set': args_loaded['densetcn_kernel_size_set'],
                            'dilation_size_set': args_loaded['densetcn_dilation_size_set'],
                            'squeeze_excitation': args_loaded['densetcn_se'],
                            'dropout': args_loaded['densetcn_dropout'],
                            }
    else:
        densetcn_options = {}

    model = Lipreading(args, modality=args.modality,
                       densetcn_options=densetcn_options,
                       relu_type=args.relu_type,
                       width_mult=args.width_mult,
                       use_boundary=args.use_boundary,
                       extract_feats=args.extract_feats).cuda()
    calculateNorm2(model)
    return model


def main():
    # -- logging
    save_path = get_save_folder(args)
    print(f"Model and log being saved in: {save_path}")
    logger = get_logger(args, save_path)
    ckpt_saver = CheckpointSaver(save_path)

    # -- get model
    model = get_model_from_json()
    # -- get dataset iterators
    dset_loaders = get_data_loaders(args)
    # -- get loss function
    criterion = nn.CrossEntropyLoss()
    # -- get optimizer
    optimizer = get_optimizer(args, optim_policies=model.parameters())
    # -- get learning rate scheduler
    scheduler = CosineScheduler(args.lr, args.max_epoch)

    if args.model_path:
        assert args.model_path.endswith('.pth') and os.path.isfile(args.model_path), \
            f"'.pth' model path does not exist. Path input: {args.model_path}"
        # resume from checkpoint
        if args.init_epoch > 0:
            model, optimizer, epoch_idx, ckpt_dict = load_model(args.model_path, model, optimizer)
            args.init_epoch = epoch_idx
            ckpt_saver.set_best_from_ckpt(ckpt_dict)
            logger.info(f'Model and states have been successfully loaded from {args.model_path}')
        # init from trained model
        else:
            model = load_model(args.model_path, model, allow_size_mismatch=args.allow_size_mismatch)
            logger.info(f'Model has been successfully loaded from {args.model_path}')
        # feature extraction
        if args.mouth_patch_path:
            save2npz(args.mouth_embedding_out_path, data=extract_feats(model).cpu().detach().numpy())
            return
        # if test-time, performance on test partition and exit. Otherwise, performance on validation and continue (sanity check for reload)
        if args.test:
            acc_avg_test, loss_avg_test = test(model, dset_loaders['test'], criterion)
            logger.info(
                f"Test-time performance on partition {'test'}: Loss: {loss_avg_test:.4f}\tAcc:{acc_avg_test:.4f}")
            return

    # -- fix learning rate after loading the ckeckpoint (latency)
    if args.model_path and args.init_epoch > 0:
        scheduler.adjust_lr(optimizer, args.init_epoch - 1)

    epoch = args.init_epoch

    while epoch < args.max_epoch:
        model = train(model, dset_loaders['train'], criterion, epoch, optimizer, logger)
        acc_avg_val, loss_avg_val = test(model, dset_loaders['val'], criterion)
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

    # -- evaluate best-performing epoch on test partition
    best_fp = os.path.join(ckpt_saver.save_dir, ckpt_saver.best_fn)
    _ = load_model(best_fp, model)
    acc_avg_test, loss_avg_test = test(model, dset_loaders['test'], criterion)
    logger.info(f"Test time performance of best epoch: {acc_avg_test} (loss: {loss_avg_test})")


if (__name__ == '__main__'):
    main()
