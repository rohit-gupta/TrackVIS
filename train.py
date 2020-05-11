#!/usr/bin/env python3
"""
Main training script.

TODO DESCRIPTION.
"""


# TODO implement config system
# TODO implement mixed precision training
# using global variables for now



import random

import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import ToPILImage

from seg_models import TrackingHead, VideoDeepLabModel
from datasets import YTVISDataset, collate_with_labels_fn
from utils.dataset import get_dataset_paths
from transforms import JointCompose
from transforms import JointResize, JointNormalize, JointCenterCrop
from transforms import JointRandomCrop, JointRandomHorizontalFlip


aug_transforms = JointCompose([
        JointResize(height=360, width=640),
        JointRandomCrop(height=315, width=560),
        JointRandomHorizontalFlip(flip_prob=0.5),
        JointNormalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

val_transforms = JointCompose([
        JointResize(height=360, width=640),
        JointCenterCrop(height=315, width=560),
        JointNormalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

######################
# Setup data loaders #
######################

BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_FRAMES = 3
VAL_FRAMES = 16
# TODO Implement reading whole video at validation

paths = get_dataset_paths("YTVIS")
train_dataset = YTVISDataset(root=paths["ROOT"], transforms=aug_transforms, phase="train", num_frames=NUM_FRAMES)
trainval_dataset = YTVISDataset(root=paths["ROOT"], transforms=val_transforms, phase="trainval", num_frames=NUM_FRAMES)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_with_labels_fn,
                          num_workers=NUM_WORKERS, pin_memory=True)

trainval_loader = DataLoader(trainval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_with_labels_fn,
                          num_workers=NUM_WORKERS, pin_memory=True)

train_num_videos = len(train_dataset)
trainval_num_videos = len(trainval_dataset)


print(train_dataset)
print(trainval_dataset)

###############
# Setup model #
###############

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6

segmodel = VideoDeepLabModel(arch="resnet50", num_classes=41)
tracking_head = TrackingHead(256, 64, train_dataset.num_instances + train_num_videos)

if torch.cuda.is_available():
    torch.cuda.set_device("cuda:0")
else:
    torch.cuda.set_device("cpu")

# Combined optimizer needs parameters for both models
parameters = list(segmodel.parameters()) + list(tracking_head.parameters())

segmodel = segmodel.cuda()
tracking_head = tracking_head.cuda()

optimizer = optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


#################
# Training Loop #
#################


EPOCHS = 100

for epoch in range(EPOCHS):
    segmodel.train()
    tracking_head.train()
    train_pbar = tqdm.tqdm(total=len(train_loader))

    for images, (semantic_targets, centers, offsets, masks, ids) in train_loader:
        images = images.cuda()
        semantic_targets = semantic_targets.cuda()
        centers = centers.cuda()
        offsets = offsets.cuda()
        masks = [[mask.cuda() for mask in image_masks] for image_masks in masks]

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            result = segmodel(images)
            semantic_preds = result["semantic"]
            centers_pred = result["instance_center"]
            offsets_pred = result["instance_regression"]
            feature_maps = result["semantic_aspp"]
            reid_embedding, reid_classification_output, instance_ids = tracking_head(feature_maps, masks, ids)

            ##########
            # Losses #
            ##########
            # Semantic Segmentation (categorical cross-entropy)
            # Instance Center Heatmap (MSE)
            # Instance Offset Predictions (Selective L1 Loss)

            #TODO implement training losses
            #TODO implement instance masks generation from predictions



        train_pbar.update(1)

    train_pbar.close()

