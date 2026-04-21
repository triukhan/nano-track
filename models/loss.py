import torch
import torch.nn.functional as F
from torch import nn

class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        target_area = (target_left + target_right) * (target_top + target_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion

        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


linear_iou = IOULoss(loc_loss_type='linear_iou')


def select_iou_loss(pred_loc, label_loc, label_cls):
    """
    Handles the case where regression output has different spatial resolution than classification.
    """
    print(f"pred_loc: {pred_loc.shape} | label_loc: {label_loc.shape} | "
                f"label_cls: {label_cls.shape} | pred_dim={pred_loc.dim()}")
    B = pred_loc.shape[0]
    H = pred_loc.shape[2]
    W = pred_loc.shape[3]

    # Flatten regression to [B*H*W, 4]
    if pred_loc.dim() == 4:
        pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 4)  # [B*H*W, 4]
        label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 4)

    # Handle label_cls: [B, 2, H, W] → take foreground channel (index 1) and flatten to [B*H*W]
    if label_cls.dim() == 4 and label_cls.shape[1] == 2:
        label_cls = label_cls[:, 1, :, :].reshape(-1)  # foreground only
    else:
        # fallback if shape is different
        if label_cls.dim() == 4:
            label_cls = label_cls.squeeze(1)
        label_cls = label_cls.reshape(-1)

    # Positive mask (where cls == 1, i.e. foreground)
    pos_mask = label_cls.eq(1).bool()  # shape [B*H*W]

    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_loc.device, requires_grad=True)

    # Now index with the correct-sized mask
    pred_loc_pos = pred_loc[pos_mask]
    label_loc_pos = label_loc[pos_mask]

    if pred_loc_pos.numel() == 0:
        return torch.tensor(0.0, device=pred_loc.device, requires_grad=True)

    return linear_iou(pred_loc_pos, label_loc_pos)