import math
import os
import shutil

import numpy as np
import torch
from torch import optim


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


def mixup_data(x, y, alpha=1.0, soft_labels=None, use_cuda=False):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''

    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def update_logger_batch(args, logger, dset_loader, batch_idx, running_loss, running_corrects, running_all, batch_time,
                        data_time):
    perc_epoch = 100. * batch_idx / (len(dset_loader) - 1)
    logger.info(
        f"[{running_all:5.0f}/{len(dset_loader.dataset):5.0f} ({perc_epoch:.0f}%)]\tLoss: {running_loss / running_all:.4f}\tAcc:{running_corrects / running_all:.4f}\tCost time:{batch_time.val:1.3f} ({batch_time.avg:1.3f})s\tData time:{data_time.val:1.3f} ({data_time.avg:1.3f})\tInstances per second: {args.batch_size / batch_time.avg:.2f}")


class CheckpointSaver:
    def __init__(self, save_dir, checkpoint_fn='ckpt.pth', best_fn='ckpt.best.pth', best_step_fn='ckpt.best.step{}.pth',
                 save_best_step=False, lr_steps=[]):
        """
        Only mandatory: save_dir
            Can configure naming of checkpoint files through checkpoint_fn, best_fn and best_stage_fn
            If you want to keep best-performing checkpoint per step
        """

        self.save_dir = save_dir

        # checkpoint names
        self.checkpoint_fn = checkpoint_fn
        self.best_fn = best_fn
        self.best_step_fn = best_step_fn

        # save best per step?
        self.save_best_step = save_best_step
        self.lr_steps = []

        # init var to keep track of best performing checkpoint
        self.current_best = 0

        # save best at each step?
        if self.save_best_step:
            assert lr_steps != [], "Since save_best_step=True, need proper value for lr_steps. Current: {}".format(
                lr_steps)
            self.best_for_stage = [0] * (len(lr_steps) + 1)

    def save(self, save_dict, current_perf, epoch=-1):
        """
            Save checkpoint and keeps copy if current perf is best overall or [optional] best for current LR step
        """

        # save last checkpoint
        checkpoint_fp = os.path.join(self.save_dir, self.checkpoint_fn)

        # keep track of best model
        self.is_best = current_perf > self.current_best
        if self.is_best:
            self.current_best = current_perf
            best_fp = os.path.join(self.save_dir, self.best_fn)
        save_dict['best_prec'] = self.current_best

        # keep track of best-performing model per step [optional]
        if self.save_best_step:

            assert epoch >= 0, "Since save_best_step=True, need proper value for 'epoch'. Current: {}".format(epoch)
            s_idx = sum(epoch >= l for l in self.lr_steps)
            self.is_best_for_stage = current_perf > self.best_for_stage[s_idx]

            if self.is_best_for_stage:
                self.best_for_stage[s_idx] = current_perf
                best_stage_fp = os.path.join(self.save_dir, self.best_stage_fn.format(s_idx))
            save_dict['best_prec_per_stage'] = self.best_for_stage

        # save
        torch.save(save_dict, checkpoint_fp)
        print("Checkpoint saved at {}".format(checkpoint_fp))
        if self.is_best:
            shutil.copyfile(checkpoint_fp, best_fp)
        if self.save_best_step and self.is_best_for_stage:
            shutil.copyfile(checkpoint_fp, best_stage_fp)

    def set_best_from_ckpt(self, ckpt_dict):
        self.current_best = ckpt_dict['best_prec']
        self.best_for_stage = ckpt_dict.get('best_prec_per_stage', None)


def change_lr_on_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class CosineScheduler:
    def __init__(self, lr_ori, epochs):
        self.lr_ori = lr_ori
        self.epochs = epochs

    def adjust_lr(self, optimizer, epoch):
        reduction_ratio = 0.5 * (1 + math.cos(math.pi * epoch / self.epochs))
        change_lr_on_optimizer(optimizer, self.lr_ori * reduction_ratio)


def get_optimizer(args, optim_policies):
    # -- define optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(optim_policies, lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(optim_policies, lr=args.lr, weight_decay=1e-2)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(optim_policies, lr=args.lr, weight_decay=1e-4, momentum=0.9)
    else:
        raise NotImplementedError
    return optimizer


import os
import json
import numpy as np

import datetime
import logging

import json
import torch
import shutil


def calculateNorm2(model):
    para_norm = 0.
    for p in model.parameters():
        para_norm += p.data.norm(2)
    print('2-norm of the neural network: {:.4f}'.format(para_norm ** .5))


def showLR(optimizer):
    return optimizer.param_groups[0]['lr']


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# -- IO utils
def read_txt_lines(filepath):
    assert os.path.isfile(filepath), "Error when trying to read txt file, path does not exist: {}".format(filepath)
    with open(filepath) as myfile:
        content = myfile.read().splitlines()
    return content


def save_as_json(d, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(d, outfile, indent=4, sort_keys=True)


def load_json(json_fp):
    assert os.path.isfile(json_fp), "Error loading JSON. File provided does not exist, cannot read: {}".format(json_fp)
    with open(json_fp, 'r') as f:
        json_content = json.load(f)
    return json_content


def save2npz(filename, data=None):
    assert data is not None, "data is {}".format(data)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.savez_compressed(filename, data=data)


def load_model(load_path, model, optimizer=None, allow_size_mismatch=False):
    """
    Load model from file
    If optimizer is passed, then the loaded dictionary is expected to contain also the states of the optimizer.
    If optimizer not passed, only the model weights will be loaded
    """

    # -- load dictionary
    assert os.path.isfile(load_path), "Error when loading the model, provided path not found: {}".format(load_path)
    checkpoint = torch.load(load_path)
    loaded_state_dict = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()

    # if allow_size_mismatch:
    #     loaded_sizes = {k: v.shape for k, v in loaded_state_dict.items()}
    #     model_sizes = {k: v.shape for k, v in model_state_dict.items()}
    #     mismatched_params = []
    #     for k in loaded_sizes:
    #         if loaded_sizes[k] != model_sizes[k]:
    #             mismatched_params.append(k)
    #     for k in mismatched_params:
    #         del loaded_state_dict[k]

    pretrained_dict = {k: v for k, v in loaded_state_dict.items() if
                       k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
    missed_params = [k for k, v in model_state_dict.items() if not k in pretrained_dict.keys()]

    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict), len(model_state_dict)))
    print('miss matched params:', missed_params)

    # -- copy loaded state into current model and, optionally, optimizer
    model.load_state_dict(loaded_state_dict, strict=not allow_size_mismatch)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer, checkpoint['epoch_idx'], checkpoint
    return model


# -- logging utils
def get_logger(args, save_path):
    log_path = '{}/{}_{}_{}classes_log.txt'.format(save_path, args.training_mode, args.lr, args.n_class)
    logger = logging.getLogger("mylog")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    return logger


def update_logger_batch(args, logger, dset_loader, batch_idx, running_loss, running_corrects, running_all, batch_time,
                        data_time):
    perc_epoch = 100. * batch_idx / (len(dset_loader) - 1)
    logger.info(
        f"[{running_all:5.0f}/{len(dset_loader.dataset):5.0f} ({perc_epoch:.0f}%)]\tLoss: {running_loss / running_all:.4f}\tAcc:{running_corrects / running_all:.4f}\tCost time:{batch_time.val:1.3f} ({batch_time.avg:1.3f})s\tData time:{data_time.val:1.3f} ({data_time.avg:1.3f})\tInstances per second: {args.batch_size / batch_time.avg:.2f}")


def get_save_folder(args):
    # create save and log folder
    save_path = '{}/{}'.format(args.logging_dir, args.training_mode)
    save_path += '/' + datetime.datetime.now().isoformat().split('.')[0]
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    return save_path
