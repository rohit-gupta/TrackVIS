#!/usr/bin/env python3
"""
Utility functions for post-processing
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None

import numpy as np
import torch
import scipy.ndimage
from PIL import Image
from skimage.draw import polygon_perimeter

colors = [[  0,   0, 200], # Blue:   Background
          [  0, 200,   0], # Green:  No Damage
          [250, 125,   0], # Orange: Minor Damage
          [250,  25, 150], # Pink:   Major Damage
          [250,   0,   0]] # Red:    Destroyed
NUM_CLASSES = len(colors)
colors = np.array(colors).astype(np.uint8)
H, W = 1024, 1024

def convert_color_segmap_to_int(segmap):
    segmap_rg = segmap[:, :, 0] + segmap[:, :, 1]
    segmap_class = np.zeros((H, W))
    for i in range(NUM_CLASSES):
        segmap_class[segmap_rg == colors[i][0] + colors[i][1]] = i

    return segmap_class


def reconstruct_from_tiles(tiles, CHANNELS, TILE_SIZE, ACTUAL_SIZE = 1024):
    num_tiles = ACTUAL_SIZE // TILE_SIZE
    reconstructed = torch.zeros([CHANNELS, ACTUAL_SIZE, ACTUAL_SIZE], dtype=torch.float32)
    for x in range(num_tiles):
        for y in range(num_tiles):
            x0, x1 = x * TILE_SIZE, (x + 1) * TILE_SIZE
            y0, y1 = y * TILE_SIZE, (y + 1) * TILE_SIZE
            reconstructed[:, x0:x1, y0:y1] = tiles[x * num_tiles + y]

    return reconstructed


def input_tensor_to_pil_img(inp_tensor):
    # μ and σ for xview dataset
    MEAN = [0.309, 0.340, 0.255]
    STDDEV = [0.162, 0.144, 0.135]

    r = (255 * ((inp_tensor[0] * STDDEV[0]) + MEAN[0])).round().byte().cpu().numpy()
    g = (255 * ((inp_tensor[1] * STDDEV[1]) + MEAN[1])).round().byte().cpu().numpy()
    b = (255 * ((inp_tensor[2] * STDDEV[2]) + MEAN[2])).round().byte().cpu().numpy()

    img = np.stack([r, g, b], axis=-1)

    return Image.fromarray(img)

def _tensor_to_pil(input_tensor, apply_color=True):
    r = Image.fromarray(input_tensor)
    if apply_color:
        r.putpalette(colors)
    return r


def segmap_tensor_to_pil_img(segmap_tensor):
    return _tensor_to_pil(torch.max(segmap_tensor, 0).indices.byte().cpu().numpy())


def logits_to_probs(segmap_logits, channel_dimension=0):
    bg_logits, class_logits = torch.split(segmap_logits, [1, 4], dim=channel_dimension)
    bg_probs = torch.nn.functional.sigmoid(bg_logits)
    class_probs = torch.nn.functional.softmax(class_logits, dim=channel_dimension)
    return torch.cat([bg_probs, class_probs], dim=channel_dimension)


def postprocess_segmap_tensor_to_pil_img(segmap_tensor, apply_color=True, binarize=False, threshold=0.5):
    segmap = segmap_tensor.cpu().numpy()
    background = segmap[0, :, :] > threshold
    foreground = ~background
    damage_class = np.argmax(segmap[1:, :, :], axis=0)
    damage_class += 1
    processed_segmap = foreground * damage_class
    if binarize:
        processed_segmap[processed_segmap > 1] = 1
    return _tensor_to_pil(processed_segmap.astype(np.uint8), apply_color)


def postprocess_combined_predictions(pre_pred_tensor, post_pred_tensor, apply_color=True, threshold=0.5):
    pre_segmap = pre_pred_tensor.cpu().numpy()
    post_segmap = post_pred_tensor.cpu().numpy()
    background = pre_segmap[0, :, :] > threshold
    foreground = ~background
    damage_class = np.argmax(post_segmap[1:, :, :], axis=0)
    damage_class += 1
    processed_post_predictions = foreground * damage_class
    return _tensor_to_pil(processed_post_predictions.astype(np.uint8), apply_color)


def bg_prob_levels_img(pre_pred_tensor):
    pre_segmap = pre_pred_tensor.cpu().numpy()
    background = (255 * pre_segmap[0, :, :])

    return _tensor_to_pil(background.astype(np.uint8), apply_color=False)


def fg_cls_img(post_pred_tensor):
    post_segmap = post_pred_tensor.cpu().numpy()
    damage_class = np.argmax(post_segmap[1:, :, :], axis=0) + 1

    return _tensor_to_pil(damage_class.astype(np.uint8), apply_color=True)


def save_segmentation_map(segmap, filename):

    # get class ID from predictions array
    segmap = np.argmax(segmap, axis=0).astype("uint8")

    # plot the semantic segmentation predictions of 5 classes in each color
    # r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r = Image.fromarray(segmap)
    r.putpalette(colors)

    r.save(filename)



def bbox_to_poly(bbox):
    x0, y0, x1, y1 = bbox

    r = []
    c = []
    c.append(x0); r.append(y0)
    c.append(x1); r.append(y0)
    c.append(x1); r.append(y1)
    c.append(x0); r.append(y1)
    c.append(x0); r.append(y0)
    c = np.array(c)
    r = np.array(r)

    return r, c


def save_bboxes(image_file, bboxes, labels, filename):
    img = np.array(Image.open(image_file), dtype=np.uint8)
    # img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    for idx, bbox in enumerate(bboxes):

        label = labels[idx]
        r, c = bbox_to_poly(bbox)

        # Draw Bbox
        rr, cc = polygon_perimeter(r, c, shape=img.shape)

        img[rr, cc, :] = colors[label]

    pil_image = Image.fromarray(img)
    pil_image.save(filename)