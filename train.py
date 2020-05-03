


from datasets import YTVISDataset, collate_with_labels_fn
from utils.dataset import get_dataset_paths
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

import torch
from seg_models import TrackingHead, VideoDeepLabModel

from transforms import JointCompose
from transforms import JointResize, JointNormalize, JointCenterCrop
from transforms import JointRandomCrop, JointRandomHorizontalFlip


transforms = JointCompose([
        JointResize(height=360, width=640),
        JointRandomCrop(height=315, width=560),
        JointRandomHorizontalFlip(flip_prob=0.5),
        JointNormalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

paths = get_dataset_paths("YTVIS")
train_dataset = YTVISDataset(root=paths["ROOT"], transforms=transforms, phase="train", num_frames=4)
trainval_dataset = YTVISDataset(root=paths["ROOT"], transforms=transforms, phase="trainval", num_frames=4)

train_num_videos = len(train_dataset)
trainval_num_videos = len(trainval_dataset)


print(train_dataset)
print(trainval_dataset)


segmodel = VideoDeepLabModel(arch="resnet50", num_classes=41)
tracking_head = TrackingHead(256, 256, train_dataset.num_instances + train_num_videos)
