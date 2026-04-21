import torch.nn as nn

from models.backbone import mobilenetv3_small_v3
from models.head import DepthwiseBAN
from models.loss import select_iou_loss


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
        self.zf = self.backbone(z)

    def track(self, x):
        xf = self.backbone(x)
        cls, loc = self.ban_head(self.zf, xf)

        return {'cls': cls, 'loc': loc}


    def forward(self, data):
        """
            only used in training
        """
        if len(data) >= 4:
            template = data['template']
            search = data['search']
            label_loc = data['label_loc']

            # get feature
            zf = self.backbone(template)
            xf = self.backbone(search)

            if self.neck is not None:
                zf = self.neck(zf)
                xf = self.neck(xf)

            cls, loc = self.ban_head(zf, xf)
            # loc loss with iou loss
            loc_loss = select_iou_loss(loc, label_loc, cls)
            outputs = {}

            outputs['total_loss'] = 1.0 * loc_loss
            outputs['loc_loss'] = loc_loss

            return outputs
        else:
            xf = self.backbone(data)
            loc = self.ban_head(self.zf, xf)

            return {'loc': loc}
