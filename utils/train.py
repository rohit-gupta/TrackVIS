#!/usr/bin/env python3
"""
Utility functions for use during training
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None

import pathlib
import torch

from .post_processing import reconstruct_from_tiles
from .post_processing import logits_to_probs
from .post_processing import postprocess_segmap_tensor_to_pil_img, postprocess_combined_predictions, bg_prob_levels_img, fg_cls_img
from .post_processing import input_tensor_to_pil_img, segmap_tensor_to_pil_img

def save_model(state_dict, path, epoch):
    torch.save(state_dict, path + str(epoch) + ".pth")
    

def save_val_results(save_path, epoch, loss_type, pre_pred, post_pred, tiled=True, tile_size=512):
    if tiled:
        pre_pred = reconstruct_from_tiles(pre_pred, 5, tile_size)
        post_pred = reconstruct_from_tiles(post_pred, 5, tile_size)

    if loss_type == "locaware":
        pre_prob = logits_to_probs(pre_pred)
        post_prob = logits_to_probs(post_pred)
    elif loss_type == "crossentropy":
        pre_prob = torch.nn.functional.softmax(pre_pred, dim=0)
        post_prob = torch.nn.functional.softmax(post_pred, dim=0)
    r = postprocess_segmap_tensor_to_pil_img(pre_prob, binarize=True)
    r.save(save_path + "pre_pred_epoch_" + str(epoch) + ".png")
    r = postprocess_segmap_tensor_to_pil_img(post_prob)
    r.save(save_path + "post_pred_epoch_" + str(epoch) + ".png")
    r = postprocess_combined_predictions(pre_prob, post_prob)
    r.save(save_path + "combined_pred_epoch_" + str(epoch) + ".png")
    r = bg_prob_levels_img(pre_prob)
    r.save(save_path + "pre_probs_epoch_" + str(epoch) + ".png")
    r = fg_cls_img(post_prob)
    r.save(save_path + "post_cls_epoch_" + str(epoch) + ".png")


def save_val_gt(save_path, pre_tiles, post_tiles, pre_label_tiles, post_label_tiles, tile_size):
    pre_img = reconstruct_from_tiles(pre_tiles, 3, tile_size)
    post_img = reconstruct_from_tiles(post_tiles, 3, tile_size)

    pre_gt = reconstruct_from_tiles(pre_label_tiles, 5, tile_size)
    post_gt = reconstruct_from_tiles(post_label_tiles, 5, tile_size)

    pilimg = input_tensor_to_pil_img(pre_img)
    pilimg.save(save_path + "pre_image.png")
    pilimg = input_tensor_to_pil_img(post_img)
    pilimg.save(save_path + "post_image.png")

    r = segmap_tensor_to_pil_img(pre_gt)
    r.save(save_path + "pre_gt.png")
    r = segmap_tensor_to_pil_img(post_gt)
    r.save(save_path + "post_gt.png")

def save_val_seg(save_path, pre_seg, post_seg):
    r = postprocess_segmap_tensor_to_pil_img(pre_seg, binarize=True)
    r.save(save_path + "seg_pre.png")
    r = postprocess_segmap_tensor_to_pil_img(post_seg)
    r.save(save_path + "seg_post.png")
    r = postprocess_combined_predictions(pre_seg, post_seg)
    r.save(save_path + "seg_combined.png")
