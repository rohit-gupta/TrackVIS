
import torch
import torch.nn as nn
import torch.nn.functional as F




# def cross_entropy(gt_segmaps, pred_segmaps):
#     logsoftmax = nn.LogSoftmax(dim=1)
#     return torch.mean(
#                       torch.sum(-gt_segmaps * logsoftmax(pred_segmaps), dim=1) # Sum across C, mean across H, W and N
#                      )
#
#
# def localization_aware_loss(gt_segmaps, pred_segmaps, loc_wt, cls_wt, loc_loss_type= "bce", loc_ce_wt=1.0, loc_oth_wt=1.0, gamma=0.0, tile_size=512.0):
#     gt_background, gt_classes = torch.split(gt_segmaps, [1,4], dim=1)
#     pred_background, pred_classes = torch.split(pred_segmaps, [1, 4], dim=1)
#
#     # Building footprint localization
#     # Localization binary cross entropy loss
#     binary_cross_entropy = nn.BCEWithLogitsLoss()
#     loc_ce_loss = binary_cross_entropy(pred_background, gt_background)
#     if loc_loss_type == "bce":  # Use
#         loc_loss = loc_ce_loss
#     else:
#         loc_loss = loc_ce_wt * loc_ce_loss
#
#         pred_background = torch.sigmoid(pred_background)
#         gt_bg_2class = torch.cat((gt_background, 1. - gt_background), dim=1)
#         pred_bg_2class = torch.cat((pred_background, 1. - pred_background), dim=1)
#
#         if loc_loss_type == "focal":
#             loc_loss += loc_oth_wt * focal_loss(gt_bg_2class, pred_bg_2class, gamma)
#         elif loc_loss_type == "dice":
#             loc_loss += loc_oth_wt * dice_loss(gt_bg_2class, pred_bg_2class)
#
#
#
#     # Damage classification for foreground only
#     gt_foreground_mask = torch.repeat_interleave(1.0 - gt_background, repeats=4, dim=1) # CLS Loss only for foreground
#     count_fg = torch.sum(1.0 - gt_background) + 1.0
#     # fg_scale = torch.clamp((gt_segmaps.shape[-1] * gt_segmaps.shape[-2])/count_fg, min=1.0, max=(gt_segmaps.shape[-1] * gt_segmaps.shape[-2])/(8*8))  # (WxH)
#     #tile_area_tensor = torch.tensor((tile_size * tile_size)).to(gt_segmaps.dtype)
#     tile_area = tile_size * tile_size
#     fg_fraction = tile_area / count_fg
#     fg_scale = torch.clamp(fg_fraction, min=1.0, max=(tile_size * tile_size) / (64 * 64))  # max = 64.0 if tile_size = 512.0
#     cls_loss = fg_scale * cross_entropy(gt_foreground_mask * gt_classes, gt_foreground_mask * pred_classes)
#
#     return loc_loss, loc_ce_loss, cls_loss, loc_wt * loc_loss + cls_wt * cls_loss
#
#
# def focal_loss(gt_segmaps, pred_segmaps, gamma):
#     # gamma = torch.tensor(gamma)
#     logsoftmax = nn.LogSoftmax(dim=1)
#
#     # p = F.softmax(pred_segmaps, dim=1)
#     p = pred_segmaps
#     log_p = logsoftmax(pred_segmaps)
#
#     weight = torch.pow(torch.tensor(1.) - p, gamma)
#     focal = -weight * log_p
#
#     return torch.mean(torch.sum(gt_segmaps * focal, dim=1)) # Sum across C, mean across H, W and N
#
#
# def dice_loss(gt_segmaps, pred_segmaps):
#     # pred_probs = F.softmax(pred_segmaps, dim=1)
#     pred_probs = pred_segmaps
#
#     # compute the dice loss
#     dims = (1, 2, 3)
#     intersection = torch.sum(pred_probs * gt_segmaps, dims)
#     cardinality = torch.sum(pred_probs + gt_segmaps, dims)
#
#     dice_score = 2. * intersection / (cardinality + 1.)
#     return torch.mean(-dice_score + 1.)
#
#
#
