import math
from collections import namedtuple
import os

import torch
from colorama import Fore, Style


__all__ = ['commit', 'describe']


Corner = namedtuple('Corner', 'x1 y1 x2 y2')
Center = namedtuple('Center', 'x y w h')


def corner2center(corner):
    """ convert (x1, y1, x2, y2) to (cx, cy, w, h)
    Args:
        conrner: Corner or np.array (4*N)
    Return:
        Center or np.array (4 * N)
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


def center2corner(center):
    """ convert (cx, cy, w, h) to (x1, y1, x2, y2)
    Args:
        center: Center or np.array (4 * N)
    Return:
        center or np.array (4 * N)
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2


def load_pretrain(model, pretrained_path):
    print('load pretrained model from {}'.format(pretrained_path))

    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage, weights_only=False)

    if 'state_dict' in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    try:
        check_keys(model, pretrained_dict)
    except:
        print('[Warning]: using pretrain as features. Adding "features." as prefix')
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = 'features.' + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # filter 'num_batches_tracked'
    missing_keys = [x for x in missing_keys
                    if not x.endswith('num_batches_tracked')]
    if len(missing_keys) > 0:
        print('[Warning] missing keys: {}'.format(missing_keys))
        print('missing keys:{}'.format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        print('[Warning] unused_pretrained_keys: {}'.format(
            unused_pretrained_keys))
        print('unused checkpoint keys:{}'.format(
            len(unused_pretrained_keys)))
    print('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, \
        'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def _exec(cmd):
    f = os.popen(cmd, 'r', 1)
    return f.read().strip()


def _bold(s):
    return "\033[1m%s\033[0m" % s


def _color(s):
    return f'{Fore.RED}{s}{Style.RESET_ALL}'


def _describe(model, lines=None, spaces=0):
    head = " " * spaces
    for name, p in model.named_parameters():
        if '.' in name:
            continue
        if p.requires_grad:
            name = _color(name)
        line = "{head}- {name}".format(head=head, name=name)
        lines.append(line)

    for name, m in model.named_children():
        space_num = len(name) + spaces + 1
        if m.training:
            name = _color(name)
        line = "{head}.{name} ({type})".format(
                head=head,
                name=name,
                type=m.__class__.__name__)
        lines.append(line)
        _describe(m, lines, space_num)


def commit():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    cmd = "cd {}; git log | head -n1 | awk '{{print $2}}'".format(root)
    commit = _exec(cmd)
    cmd = "cd {}; git log --oneline | head -n1".format(root)
    commit_log = _exec(cmd)
    return "commit : {}\n  log  : {}".format(commit, commit_log)


def describe(net, name=None):
    num = 0
    lines = []
    if name is not None:
        lines.append(name)
        num = len(name)
    _describe(net, lines, num)
    return "\n".join(lines)


def print_speed(i, i_time, n):
    """print_speed(index, index_time, total_iteration)"""
    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 -
                                remaining_day * 24)
    remaining_min = math.floor(remaining_time / 60 -
                               remaining_day * 1440 -
                               remaining_hour * 60)
    print('Progress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n' %
                (i, n, i / n * 100,
                 average_time,
                 remaining_day, remaining_hour, remaining_min))

def restore_from(model, optimizer, ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path,
        map_location=lambda storage, loc: storage.cuda(device))
    epoch = ckpt['epoch']

    ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)

    check_keys(optimizer, ckpt['optimizer'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, epoch
