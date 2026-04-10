from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pathlib import Path

import torch.nn as nn
import torch

from siam_tracker.backbone import mobilenetv3_small_v3
# from siam_tracker.backbone import mobilenetv3_small_v3
from siam_tracker.head import UPChannelBAN, DepthwiseBAN

from torchvision.models import mobilenet_v3_small


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()

        self.in_channels=in_channels

        self.out_channels=out_channels

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):

        if self.in_channels != self.out_channels:
            x = self.downsample(x)

        if x.size(3) < 16:
            l = 2
            r = l + 4
            x = x[:, :, l:r, l:r]
        return x


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        self.backbone = mobilenetv3_small_v3()
        self.ban_head = DepthwiseBAN(96, 96)
        self.neck = AdjustLayer(96, 96)

        self.template_features = None
        self.prev_feat = None


    def init(self, z):
        zf = self.backbone(z)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        cls, loc = self.ban_head(self.zf, xf)

        return {'cls': cls, 'loc': loc}


    def forward(self, data):
        """
            only used in training
        """
        if len(data) >= 4:
            template = data['template'].cuda()
            search = data['search'].cuda()
            label_loc = data['label_loc'].cuda()

            # get feature
            zf = self.backbone(template)
            xf = self.backbone(search)

            if self.neck is not None:
                cls, reg = self.neck(xf, zf)

            loc = self.ban_head(zf, xf)

            # loc loss with iou loss
            loc_loss = select_iou_loss(loc, label_loc)
            outputs = {}

            outputs['total_loss'] = self.cfg.TRAIN.LOC_WEIGHT * loc_loss
            outputs['loc_loss'] = loc_loss

            return outputs
        else:
            xf = self.backbone(data)
            loc = self.ban_head(self.zf, xf)

            return {'loc': loc}
