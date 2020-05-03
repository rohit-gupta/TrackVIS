#!/usr/bin/env python3
"""
Joint Image-Mask transforms
"""

__author__ = "Rohit Gupta"
__version__ = "0.1b"
__license__ = "GPL2"

# import numpy as np
# from PIL import Image
import random

# import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize

#
# def pad_if_smaller(img, size, fill=0):
#     min_size = min(img.size)
#     if min_size < size:
#         ow, oh = img.size
#         padh = size - oh if oh < size else 0
#         padw = size - ow if ow < size else 0
#         img = F.pad(img, (0, 0, padw, padh), fill=fill)
#     return img


class JointCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, targets):
        for t in self.transforms:
            image, targets = t(image, targets)
        return image, targets


class JointResize(object):
    def __init__(self, height, width, mode='nearest'):
        self.size = (height, width)
        self.mode = mode

    def __call__(self, image, targets):
        # Add fake batch dimension for interpolate
        # print(image.shape)
        # print(self.size)
        image = F.interpolate(image.unsqueeze(0).float(), size=self.size, mode=self.mode).squeeze(0)
        semantic_target, centers, offsets, tracking_masks, tracking_ids = targets
        semantic_target = F.interpolate(semantic_target.unsqueeze(0).float(), size=self.size, mode=self.mode).squeeze(0)
        centers = F.interpolate(centers.unsqueeze(0).float(), size=self.size, mode=self.mode).squeeze(0)
        offsets = F.interpolate(offsets.unsqueeze(0).float(), size=self.size, mode=self.mode).squeeze(0)
        tracking_masks = [F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=self.size, mode=self.mode).squeeze(0).squeeze(0) for mask in tracking_masks]
        # TODO Check JointResize Implementation
        # TODO note that mask downsampling happens after all the augmentations

        return image, (semantic_target, centers, offsets, tracking_masks, tracking_ids)


class JointRandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, targets):
        semantic_target, centers, offsets, tracking_masks, tracking_ids = targets
        if random.random() < self.flip_prob:
            image = image.flip(dims=[2])
            semantic_target = semantic_target.flip(dims=[2])
            centers = centers.flip(dims=[2])
            offsets = offsets.flip(dims=[2])
            tracking_masks = [mask.flip(dims=[1]) for mask in tracking_masks]
        return image, (semantic_target, centers, offsets, tracking_masks, tracking_ids)


class JointRandomCrop(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, targets):
        semantic_target, centers, offsets, tracking_masks, tracking_ids = targets
        _, H, W = image.shape
        Hrange, Wrange = H - self.height, W - self.width

        assert Hrange > 0 and Wrange > 0, "Inputs too small for transformation."

        h1, w1 = random.randrange(Hrange), random.randrange(Wrange)
        h2, w2 = h1 + self.height, w1 + self.width

        image = image[:, h1:h2, w1:w2]
        semantic_target = semantic_target[:, h1:h2, w1:w2]
        centers = centers[:, h1:h2, w1:w2]
        offsets = offsets[:, h1:h2, w1:w2]
        tracking_masks = [mask[h1:h2, w1:w2] for mask in tracking_masks]

        # TODO deal with completely removed objects

        return image, (semantic_target, centers, offsets, tracking_masks, tracking_ids)


class JointCenterCrop(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, targets):
        semantic_target, centers, offsets, tracking_masks, tracking_ids = targets

        _, H, W = image.shape
        HDiff, WDiff = H - self.height, W - self.width

        assert HDiff > 0 and WDiff > 0, "Inputs too small for transformation."

        h1, w1 = HDiff//2, WDiff//2
        h2, w2 = h1 + self.height, w1 + self.width

        image = image[:, h1:h2, w1:w2]
        semantic_target = semantic_target[:, h1:h2, w1:w2]
        centers = centers[:, h1:h2, w1:w2]
        offsets = offsets[:, h1:h2, w1:w2]
        tracking_masks = [mask[h1:h2, w1:w2] for mask in tracking_masks]

        # TODO deal with completely removed objects

        return image, (semantic_target, centers, offsets, tracking_masks, tracking_ids)

# class JointRandomResize(object):
#     def __init__(self, min_size, max_size=None):
#         self.min_size = min_size
#         if max_size is None:
#             max_size = min_size
#         self.max_size = max_size
#
#     def __call__(self, image, target):
#         size = random.randint(self.min_size, self.max_size)
#         image = F.resize(image, size)
#         target = F.resize(target, size, interpolation=Image.NEAREST)
#         return image, target


# class JointToTensor(object):
#     def __call__(self, image, target):
#         image = F.to_tensor(image)
#         target = torch.as_tensor(np.asarray(target), dtype=torch.int64)
#         return image, target


class JointNormalize(object):
    def __init__(self, mean, std):
        self.T = Normalize(mean, std)

    def __call__(self, image, targets):
        image = self.T(image)
        return image, targets
