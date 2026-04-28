import argparse
import logging
import os
import time
import math
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

import sys

from models.model_builder import ModelBuilder
from average_meter import AverageMeter
from dataset import BANDataset
from distributed import get_world_size, get_rank, average_reduce, reduce_gradients, new_dist_init, \
    DistModule
from lr_scheduler import build_lr_scheduler
from utils import describe, print_speed, commit, load_pretrain, restore_from


sys.path.append(os.getcwd())
logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='nanotrack')
parser.add_argument('--seed', type=int, default=123456, help='random seed')
parser.add_argument('--local_rank', type=int, default=0, help='compulsory for pytorch launcer')
args = parser.parse_args()

PRETRAINED = 'models/pretrained/nanotrackv3.pth'
BATCH_SIZE = 64
NUM_WORKERS = 2
TRAIN_EPOCH = 10
TRAIN_LAYERS = ['features']
LAYERS_LR = 0.1
BASE_LR = 0.005
ADJUST = True
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
EPOCH = 50
START_EPOCH = 0
GRAD_CLIP = 10.0
PRINT_FREQ = 20
LOG_GRADS = False
SNAPSHOT_DIR = './models/snapshot'
LOG_DIR = './logs'


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader():
    logger.info('build train dataset')
    train_dataset = BANDataset()
    logger.info('build dataset done')

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler)
    return train_loader


def build_opt_lr(model, current_epoch=0):
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    if current_epoch >= TRAIN_EPOCH:
        for layer in TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.backbone.parameters()),
                          'lr': LAYERS_LR * BASE_LR}]

    if ADJUST:
        trainable_params += [{'params': model.neck.parameters(),
                              'lr': BASE_LR}]

    trainable_params += [{'params': model.ban_head.parameters(),
                          'lr': BASE_LR}]

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=EPOCH)
    lr_scheduler.step(START_EPOCH)
    return optimizer, lr_scheduler


def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, head_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            head_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/' + k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/' + k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/' + k.replace('.', '/'),
                             w_norm / (1e-20 + _norm), tb_index)
    tot_norm = feature_norm + head_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    head_norm = head_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/head', head_norm, tb_index)


def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    cur_lr = lr_scheduler.get_cur_lr()
    rank = get_rank()

    average_meter = AverageMeter()

    def is_valid_number(x):
        return not (math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    num_per_epoch = len(train_loader.dataset) // \
                    EPOCH // (BATCH_SIZE * world_size)
    start_epoch = START_EPOCH
    epoch = start_epoch

    if not os.path.exists(SNAPSHOT_DIR) and \
            get_rank() == 0:
        os.makedirs(SNAPSHOT_DIR)

    logger.info("model\n{}".format(describe(model.module)))
    end = time.time()
    for idx, data in enumerate(train_loader):
        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch

            if get_rank() == 0:
                torch.save(
                    {'epoch': epoch,
                     'state_dict': model.module.state_dict(),
                     'optimizer': optimizer.state_dict()},
                    SNAPSHOT_DIR + '/checkpoint_e%d.pth' % (epoch))

            if epoch == EPOCH:
                return

            if TRAIN_EPOCH == epoch:
                logger.info('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)
                logger.info("model\n{}".format(describe(model.module)))

            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()
            logger.info('epoch: {}'.format(epoch + 1))
        tb_idx = idx + start_epoch * num_per_epoch
        if idx % num_per_epoch == 0 and idx != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info('epoch {} lr {}'.format(epoch + 1, pg['lr']))
                if rank == 0:
                    tb_writer.add_scalar('lr/group{}'.format(idx + 1),
                                         pg['lr'], tb_idx)

        data_time = average_reduce(time.time() - end)
        if rank == 0:
            tb_writer.add_scalar('time/data', data_time, tb_idx)
        data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        outputs = model(data)
        loss = outputs['total_loss']

        if is_valid_number(loss.data.item()):
            optimizer.zero_grad()
            loss.backward()
            reduce_gradients(model)

            if rank == 0 and LOG_GRADS:
                log_grads(model.module, tb_writer, tb_idx)

            # clip gradient
            clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        batch_time = time.time() - end
        batch_info = {}
        batch_info['batch_time'] = average_reduce(batch_time)
        batch_info['data_time'] = average_reduce(data_time)
        for k, v in sorted(outputs.items()):
            batch_info[k] = average_reduce(v.data.item())

        average_meter.update(**batch_info)

        if rank == 0:
            for k, v in batch_info.items():
                tb_writer.add_scalar(k, v, tb_idx)

            if (idx + 1) % PRINT_FREQ == 0:
                info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                    epoch + 1, (idx + 1) % num_per_epoch,
                    num_per_epoch, cur_lr)
                for cc, (k, v) in enumerate(batch_info.items()):
                    if cc % 2 == 0:
                        info += ("\t{:s}\t").format(
                            getattr(average_meter, k))
                    else:
                        info += ("{:s}\n").format(
                            getattr(average_meter, k))
                logger.info(info)
                print_speed(idx + 1 + start_epoch * num_per_epoch,
                            average_meter.batch_time.avg,
                            EPOCH * num_per_epoch)
        end = time.time()


def main():
    rank, world_size = new_dist_init()

    logger.info("init done")

    # load cfg
    if rank == 0:
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        logger.info("Version Information: \n{}\n".format(commit()))

    model = ModelBuilder().cuda().train()

    # if BACKBONE_PRETRAINED:
    #     cur_path = os.path.dirname(os.path.realpath(__file__))
    #     backbone_path = os.path.join(cur_path, '../', BACKBONE_PRETRAINED)
    #     load_pretrain(model.backbone, backbone_path)

    if rank == 0 and LOG_DIR:
        tb_writer = SummaryWriter(LOG_DIR)
    else:
        tb_writer = None

    train_loader = build_data_loader()

    optimizer, lr_scheduler = build_opt_lr(model, START_EPOCH)

    # load pretrain
    if PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(cur_path, '../', PRETRAINED)
        load_pretrain(model, path)
    dist_model = DistModule(model)

    logger.info(lr_scheduler)
    logger.info("model prepare done")

    # start training
    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer)



if __name__ == '__main__':
    seed_torch(args.seed)
    main()
