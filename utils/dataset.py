#!/usr/bin/env python3
"""
Utility functions used in dataloaders
"""

__author__ = "Rohit Gupta"
__version__ = "0.1b"
__license__ = "GPL2"

import numpy as np
from scipy import ndimage, signal
from scipy.signal import gaussian
import json
from PIL import Image

import pycocotools.mask as mask_utils
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def detect_letterboxing(img, tol=0):
    """Detect horizontal and vertical letterboxing in images.

    Parameters
    ----------
    img : PIL Image
    Image to detect letterboxing in.

    tol: int/float
    Tolerance.

    Returns
    -------
    row_start : int
    row_end   : int
    col_start : int
    col_end   : int
    Coordinates of actual image within the letterboxed version.
    """
    img = np.array(img)
    mask = img > tol
    if img.ndim == 3:
        mask = mask.all(2)
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()  # Numpy argmax is deterministic
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()

    return row_start, row_end, col_start, col_end


def detect_letterboxing_video(images_list, mode="all"):
    """Detect horizontal and vertical letterboxing in videos.

    mode == "all" for preventing minor jitter
    mode == "first" for speed (should be fine)
    """

    if mode == "all":
        min_row_start = 0
        max_row_end = 1920
        min_col_start = 0
        max_col_end = 1920
        for image_file in images_list:
            img = Image.open(open(image_file, "rb"))
            row_start, row_end, col_start, col_end = detect_letterboxing(img)
            if row_start < min_row_start:
                min_row_start = row_start
            if col_start < min_col_start:
                min_col_start = col_start
            if row_end > max_row_end:
                max_row_end = row_end
            if col_end > max_col_end:
                max_col_end = col_end

        return min_row_start, max_row_end, min_col_start, max_col_end

    elif mode == "first":
        img = Image.open(open(images_list[0], "rb"))
        row_start, row_end, col_start, col_end = detect_letterboxing(img)

        return row_start, row_end, col_start, col_end


__YTVIS_ROOT = "/home/rohit/Downloads/YoutubeVOS/"
__MOTS_ROOT = ""
__KITTIMOTS_ROOT = ""


def get_dataset_paths(dataset, root_path=__YTVIS_ROOT):
    if root_path[-1] != "/":
        raise ValueError("Root path must have trailing slash.")
    DATASET_PATHS = dict(YTVIS=dict(ROOT=root_path,
                                    TRAIN_VIDEOS=root_path + "VIS_Videos/train_all_frames/JPEGImages/",
                                    VALID_VIDEOS=root_path + "VIS_Videos/valid_all_frames/JPEGImages/",
                                    TEST_VIDEOS=root_path + "VIS_Videos/test_all_frames/JPEGImages/",
                                    TRAIN_METADATA=root_path + "VIS_Annotations/train.json",
                                    TRAIN_SPLIT=root_path + "VIS_Annotations/train_ids.txt",
                                    TRAINVAL_SPLIT=root_path + "VIS_Annotations/trainval_ids.txt",
                                    VALID_METADATA=root_path + "VIS_Annotations/valid.json",
                                    TEST_METADATA=root_path + "VIS_Annotations/test.json"),
                         MOTS=dict(ROOT=root_path),
                         KITTIMOTS=dict(ROOT=root_path))
    return DATASET_PATHS[dataset]


def read_mask(rle_counts):
    """
    Convert RLE Counts object segmentation annotations to binary mask.
    Parameters
    ----------
    rle_counts: COCO style annotations

    Returns
    -------
    mask: binary mask
    """
    rle = mask_utils.frPyObjects(rle_counts, rle_counts.get('size')[0], rle_counts.get('size')[1])
    mask = mask_utils.decode(rle)

    return mask


def mask_centroid(mask):
    return ndimage.measurements.center_of_mass(mask)


def read_json(filepath):
    return json.load(open(filepath, "r"))


def read_txt_lines(filepath, linetype):
    return [linetype(x.strip()) for x in open(filepath, "r").readlines()]


def load_image(image_file):
    return Image.open(open(image_file, "rb")).convert("RGB")


def coord_matrices(x, y):
    x_coords = np.arange(0, x, 1)
    y_coords = np.arange(0, y, 1)
    x_coord_mat = np.tile(x_coords, [y, 1])
    y_coord_mat = np.tile(np.expand_dims(y_coords, 0).T, [1, x])

    return x_coord_mat, y_coord_mat


def gaussian_2d(kernel_size=17, std=8):
    """Returns a 2D Gaussian kernel."""
    if kernel_size % 2 == 0:
        raise ValueError("Gaussian kernel must have an odd kernel_size.")

    gaussian_1d = gaussian(kernel_size, std=std).reshape(kernel_size, 1)
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
    return gaussian_2d


def generate_object_mask_center_offsets(rle_counts, width, height, kernel_size, std):
    """

    Parameters
    ----------
    rle_counts: MS COCO style RLE Counts
    width
    height
    kernel_size: Size of kernel to represent center patch
    std: Standard Deviation of kernel to represent center patch
    """
    object_mask = read_mask(rle_counts)

    # plt.imshow(object_mask, interpolation='none')
    # plt.colorbar()
    # plt.show()
    # print(object_mask.shape)
    # print(width, height)
    cy, cx = (int(c) for c in mask_centroid(object_mask))
    # print(cx, cy)
    x, y = coord_matrices(width, height)
    # print(x.shape)
    # print(y.shape)
    # background_mask = 1 - object_mask
    x -= cx
    y -= cy
    x *= object_mask
    y *= object_mask
    # plt.imshow(x, interpolation='none')
    # plt.colorbar()
    # plt.show()
    # plt.imshow(y, interpolation='none')
    # plt.colorbar()
    # plt.show()
    center_map = np.zeros((height, width), dtype=np.float32)
    patch = gaussian_2d(kernel_size, std)
    patch_size = kernel_size // 2

    # print(cx, cy)
    # print(patch.shape)
    # print(center_map.shape)
    # print(patch.shape)
    # print(center_map.shape)
    # print(cy)
    # print(patch_size)
    x1, x2, y1, y2 = (cx - patch_size, cx + patch_size + 1, cy - patch_size, cy + patch_size + 1)
    patch_size_x1, patch_size_x2, patch_size_y1, patch_size_y2 = patch_size, patch_size, patch_size, patch_size
    # print(x1, x2, y1, y2)
    # print(patch_size_x1, patch_size_x2, patch_size_y1, patch_size_y2)
    # print(center_map.shape)
    if (x1 < 0):
        patch_size_x1 = cx
        x1 = 0
    if (y1 < 0):
        patch_size_y1 = cy
        y1 = 0
    if (x2 > width):
        patch_size_x2 = width - cx - 1
        x2 = width
    if (y2 > height):
        patch_size_y2 = height - cy - 1
        y2 = height

    # print(x1, x2, y1, y2)
    # print(patch_size_x1, patch_size_x2, patch_size_y1, patch_size_y2)
    center_map[y1:y2, x1:x2] = patch[patch_size - patch_size_y1:patch_size + patch_size_y2 + 1, patch_size - patch_size_x1:patch_size + patch_size_x2 + 1]

    return object_mask, center_map, x, y


def generate_semantic_target(masks, width, height, categories, num_categories):
    semantic_target = np.zeros((num_categories, height, width), dtype=np.int64)

    for object_num, mask in enumerate(masks):
        semantic_target[categories[object_num], :, :] += mask

    return semantic_target


def downsample_mask(mask, scale):
    # return F.interpolate(mask, size=(mask.shape[0] // scale, mask.shape[1] // scale), mode="nearest")
    # print("Mask shape in loader", mask.shape)
    return mask[range(0, mask.shape[0], scale), :][:, range(0, mask.shape[1], scale)]


def numpy_to_tensor(np_array, dtype):
    if dtype == "int":
        np_dtype = np.int64
    elif dtype == "float":
        np_dtype = np.float32

    return torch.from_numpy(np_array.astype(np_dtype))


def identify_null_masks(masks):
    null_masks = []
    for idx, mask in enumerate(masks):
        if mask.sum().item() < 1:
            null_masks.append(idx)

    return null_masks
