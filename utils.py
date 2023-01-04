import io
import logging
import os
import colorlog
import os.path as osp
import sys
import json
import time
import errno
import numpy as np
import random
import warnings
import PIL
import torch
from PIL import Image
from torchmetrics import RetrievalMRR
import refile
import tempfile
import torch


def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    print(f"====> set seed {seed}")


class TqdmToLogger(io.StringIO):
    logger = None
    level = None
    buf = ''

    def __init__(self):
        super(TqdmToLogger, self).__init__()
        self.logger = get_logger('tqdm')

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.info(self.buf)


def get_logger(logger_name='default', debug=False, save_to_dir=None):
    if debug:
        log_format = (
            '%(asctime)s - '
            '%(levelname)s : '
            '%(name)s - '
            '%(pathname)s[%(lineno)d]:'
            '%(funcName)s - '
            '%(message)s'
        )
    else:
        log_format = (
            '%(asctime)s - '
            '%(levelname)s : '
            '%(name)s - '
            '%(message)s'
        )
    bold_seq = '\033[1m'
    colorlog_format = f'{bold_seq} %(log_color)s {log_format}'
    colorlog.basicConfig(format=colorlog_format, datefmt='%y-%m-%d %H:%M:%S')
    logger = logging.getLogger(logger_name)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if save_to_dir is not None:
        fh = logging.FileHandler(os.path.join(save_to_dir, 'log', 'debug.log'))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        fh = logging.FileHandler(
            os.path.join(save_to_dir, 'log', 'warning.log'))
        fh.setLevel(logging.WARNING)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        fh = logging.FileHandler(os.path.join(save_to_dir, 'log', 'error.log'))
        fh.setLevel(logging.ERROR)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_mrr(sim_mat):
    mrr = RetrievalMRR()
    return mrr(
        sim_mat.flatten(),
        torch.eye(len(sim_mat), device=sim_mat.device).long().bool().flatten(),
        torch.arange(len(sim_mat), device=sim_mat.device)[:, None].expand(len(sim_mat), len(sim_mat)).flatten(),
    )
    pass


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # pred(correct.shape)
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class MgvSaveHelper(object):
    def __init__(self, save_oss=False, oss_path='', echo=True):
        self.oss_path = oss_path
        self.save_oss = save_oss
        self.echo = echo

    def set_stauts(self, save_oss=False, oss_path='', echo=True):
        self.oss_path = oss_path
        self.save_oss = save_oss
        self.echo = echo

    def get_s3_path(self, path):
        if self.check_s3_path(path):
            return path
        return self.oss_path + path

    def check_s3_path(self, path):
        return path.startswith('s3:')

    def load_ckpt(self, path):
        if self.check_s3_path(path):
            with refile.smart_open(path, "rb") as f:
                ckpt = torch.load(f)
        else:
            ckpt = torch.load(path)
        if self.echo:
            print(f"====> load checkpoint from {path}")
        return ckpt

    def save_ckpt(self, path, epoch, model, optimizer=None):
        if self.save_oss:
            if not self.check_s3_path(path):
                path = self.get_s3_path(path)
            with refile.smart_open(path, "wb") as f:
                torch.save(
                    {"epoch": epoch,
                     "state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict()}, f)
        else:
            torch.save(
                {"epoch": epoch,
                 "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict()}, path)

        if self.echo:
            print(f"====> save checkpoint to {path}")

    def save_pth(self, path, file):
        if self.save_oss:
            if not self.check_s3_path(path):
                path = self.get_s3_path(path)
            with refile.smart_open(path, "wb") as f:
                torch.save(file, f)
        else:
            torch.save(file, path)

        if self.echo:
            print(f"====> save pth to {path}")

    def load_pth(self, path):
        if self.check_s3_path(path):
            with refile.smart_open(path, "rb") as f:
                ckpt = torch.load(f)
        else:
            ckpt = torch.load(path)
        if self.echo:
            print(f"====> load pth from {path}")
        return ckpt

