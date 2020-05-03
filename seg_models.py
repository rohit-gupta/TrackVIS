#!/usr/bin/env python3
"""
Implementation of DeepLabv3 and DeepLabv3+ using torchvision building blocks
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None

import numpy as np
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F


from torchvision.models.segmentation.deeplabv3 import ASPP, DeepLabHead as DeepLabv3Head
from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision.models.resnet import *
from torchvision.models._utils import IntermediateLayerGetter

from utils.misc import EMSG

RESNET_ARCHS = {
    'resnet18': resnet18, # Cannot be used as BasicBlock doesn't have atrous convs
    'resnet34': resnet34, # -do-
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2
}

# TODO

# RESNET_STRIDE4_FEATURE_SIZES = {
#     'resnet18': 64,
#     'resnet34': 64,
#     'resnet50': 256,
#     'resnet101': 256,
#     'resnet152': 256,
#     'wide_resnet50_2': 256,
#     'wide_resnet101_2': 256,
#     'resnext50_32x4d': 256
# }
#
# RESNET_STRIDE8_FEATURE_SIZES = {
#     'resnet18': 128,
#     'resnet34': 128,
#     'resnet50': 512,
#     'resnet101': 512,
#     'resnet152': 512,
#     'wide_resnet50_2': 512,
#     'wide_resnet101_2': 512,
#     'resnext50_32x4d': 512
# }
#
# RESNET_HIGH_FEATURE_SIZES = {
#     'resnet18': 512,
#     'resnet34': 512,
#     'resnet50': 2048,
#     'resnet101': 2048,
#     'resnet152': 2048,
#     'wide_resnet50_2': 2048,
#     'wide_resnet101_2': 2048,
#     'resnext50_32x4d': 2048
# }

# Commented out because defaul pytorch initializations seems good ? (Might test later)
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
#         if m.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)

#         m.bias.data.fill_(0.01)

def create_resnet_backbone(arch=None, block=None, layers=None,
                           groups=None, width_multiple=None,
                           replace_stride_with_dilation=None,
                           pretrained=None):
    
    if arch is None and block is None:
        raise ValueError('Specify one of ResNet name or structure.')
    if arch and block:
        raise ValueError('Specify either ResNet name or structure, not both.')

    zero_init_residual = True
    if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, True, True]
            print(EMSG("INFO"), "Using default output stride of 16")

    if arch:
        if pretrained is None:
            pretrained = True
        progress = True
        model = RESNET_ARCHS[arch](pretrained, progress,
                                   zero_init_residual=zero_init_residual,
                                   replace_stride_with_dilation=replace_stride_with_dilation)
    else:
        if layers is None:
            raise ValueError('Specify ResNet Block structure when arch is missing.')
        if groups is None:
            groups = 1
        if width_multiple is None:
            width_per_group = 64
        else:
            width_per_group = width_multiple * 64 # For WideResNets in multiples of 64
        model = ResNet(block, layers, zero_init_residual=zero_init_residual,
                       groups=groups, width_per_group=width_per_group,
                       replace_stride_with_dilation=replace_stride_with_dilation )

    return model


class DeepLabv3PlusHead(nn.Module):
    """DeepLabv3+ Head"""
    def __init__(self, low_channels, low_channels_reduced, high_channels, num_classes, atrous_rates):
        super(DeepLabv3PlusHead, self).__init__()

        aspp_channels = 256

        feature_channels = low_channels_reduced + aspp_channels # Number of ASPP features

        self.aspp_features = ASPP(high_channels, atrous_rates)
        self.low_features_reduce = nn.Sequential(nn.Conv2d(low_channels, low_channels_reduced, 1, padding=0, bias=False),
                                                 nn.BatchNorm2d(low_channels_reduced),
                                                 nn.ReLU(inplace=True))

        self.output_head = nn.Sequential(nn.Conv2d(feature_channels, aspp_channels, 3, padding=1, bias=False),
                                         nn.BatchNorm2d(aspp_channels),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.5),
                                         nn.Conv2d(aspp_channels, aspp_channels, 3, padding=1, bias=False),
                                         nn.BatchNorm2d(aspp_channels),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.1),
                                         nn.Conv2d(aspp_channels, num_classes, 1))

    def forward(self, features):

        low_features_size = features["low"].shape[-2:]
        high_features_size = features["high"].shape[-2:]

        # print(low_features_size, high_features_size)

        x = self.aspp_features(features["high"])
        x = F.interpolate(x, size=low_features_size, mode='bilinear', align_corners=False)
        
        xl = self.low_features_reduce(features["low"])

        # print(x.shape, xl.shape)

        x = torch.cat((x, xl), 1)

        x = self.output_head(x)

        return x


class DeepLabv3PlusModel(nn.Module):
    """DeepLabv3+"""
    def __init__(self, arch, pretrained_backbone=True, output_stride=16, num_classes=20):
        super(DeepLabv3PlusModel, self).__init__()

        if output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
            atrous_rates=[6, 12, 18]
        elif output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            atrous_rates=[12, 24, 36]
        else:
            raise ValueError('output_stride can be 8 or 16.')
        backbone = create_resnet_backbone(arch, pretrained=pretrained_backbone,
                                          replace_stride_with_dilation=replace_stride_with_dilation)
        return_layers = {'layer1': 'low', 'layer4': 'high'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.classifier = DeepLabv3PlusHead(RESNET_STRIDE4_FEATURE_SIZES[arch], 48, RESNET_HIGH_FEATURE_SIZES[arch], num_classes, atrous_rates)
    
    def forward(self, batch):

        result = OrderedDict()

        input_size = batch.shape[-2:]
        x = self.backbone(batch)
        x = self.classifier(x)

        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        result["out"] = x

        return result


class PanopticDeepLabDecoder(nn.Module):
    """Panoptic-DeepLab Decoder"""
    def __init__(self, low_channels=[512, 256], low_channels_reduced=[32, 16], aspp_channels=256, out_channels=256):
        super(PanopticDeepLabDecoder, self).__init__()

        #reduce stride 8 low features
        #reduce stride 16 low features
        # upsample
        # merge stride 8 low features
        # Upsample and 5x5 conv merged features
        # merge stride 4 low features
        # 5x5 conv
        self.low_features_reduce1 = nn.Sequential(nn.Conv2d(low_channels[0], low_channels_reduced[0], 1, padding=0, bias=False),
                                                 nn.BatchNorm2d(low_channels_reduced[0]),
                                                 nn.ReLU(inplace=True))
        
        self.low_features_reduce2 = nn.Sequential(nn.Conv2d(low_channels[1], low_channels_reduced[1], 1, padding=0, bias=False),
                                                 nn.BatchNorm2d(low_channels_reduced[1]),
                                                 nn.ReLU(inplace=True))

        self.upsample_refine1 = nn.Sequential(nn.Conv2d(low_channels_reduced[0] + aspp_channels, out_channels, 5, padding=2, bias=False),
                                         nn.BatchNorm2d(out_channels),
                                         nn.ReLU(inplace=True))
        self.upsample_refine2 = nn.Sequential(nn.Conv2d(low_channels_reduced[1] + out_channels, out_channels, 5, padding=2, bias=False),
                                         nn.BatchNorm2d(out_channels),
                                         nn.ReLU(inplace=True))


    def forward(self, features):

        low_features1_size = features["low1"].shape[-2:]
        low_features2_size = features["low2"].shape[-2:]
        aspp_features_size = features["aspp"].shape[-2:]
        # print(low_features1_size, low_features2_size, aspp_features_size)

        x = F.interpolate(features["aspp"], size=low_features1_size, mode='bilinear', align_corners=False)
        xl = self.low_features_reduce1(features["low1"])
        x = torch.cat((x, xl), 1)

        x = self.upsample_refine1(x)

        x = F.interpolate(x, size=low_features2_size, mode='bilinear', align_corners=False)
        xl = self.low_features_reduce2(features["low2"])
        x = torch.cat((x, xl), 1)

        x = self.upsample_refine2(x)

        return x


class PanopticDeepLabHead(nn.Sequential):
    def __init__(self, in_channels, feature_channels, num_classes):
        super(PanopticDeepLabHead, self).__init__(
            nn.Conv2d(in_channels, feature_channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, num_classes, 1)
        )


class PanopticDeepLabModel(nn.Module):
    """Panoptic Deeplab Model"""
    def __init__(self, arch, pretrained_backbone=True, num_classes=20):
        super(PanopticDeepLabModel, self).__init__()

        replace_stride_with_dilation = [False, False, True]
        atrous_rates=[6, 12, 18]
        # high_channels = RESNET_HIGH_FEATURE_SIZES[arch]
        # low1_channels = RESNET_STRIDE8_FEATURE_SIZES[arch]
        # low2_channels = RESNET_STRIDE4_FEATURE_SIZES[arch]
        high_channels = 2048
        low1_channels = 512
        low2_channels = 256
        low_channels=[low1_channels, low2_channels]
        low_channels_reduced_semantic = [64, 32]
        low_channels_reduced_instance = [32, 16]
        aspp_channels = 256
        out_channels_semantic = 256
        out_channels_instance = 256
        semantic_features = 256
        instance_features = 32

        backbone = create_resnet_backbone(arch, pretrained=pretrained_backbone,
                                          replace_stride_with_dilation=replace_stride_with_dilation)
        return_layers = {'layer1': 'low2', 'layer2': 'low1', 'layer4': 'high'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.semantic_aspp = ASPP(high_channels, atrous_rates)
        self.instance_aspp = ASPP(high_channels, atrous_rates)
        self.semantic_decoder = PanopticDeepLabDecoder(low_channels, low_channels_reduced_semantic, aspp_channels, out_channels_semantic)
        self.instance_decoder = PanopticDeepLabDecoder(low_channels, low_channels_reduced_instance, aspp_channels, out_channels_instance)
        self.semantic_head = PanopticDeepLabHead(out_channels_semantic, semantic_features, num_classes)
        self.instance_center_head = PanopticDeepLabHead(out_channels_instance, instance_features, 1)
        self.instance_regression_head = PanopticDeepLabHead(out_channels_instance, instance_features, 2)
    
    def forward(self, batch):

        result = OrderedDict()

        input_size = batch.shape[-2:]
        features = self.backbone(batch)

        # Semantic Branch
        features["aspp"] = self.semantic_aspp(features["high"])
        x = self.semantic_decoder(features)

        x = self.semantic_head(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        result["semantic"] = x

        # Instance Branch
        
        features["aspp"] = self.instance_aspp(features["high"])
        xi = self.instance_decoder(features)

        x = self.instance_center_head(xi)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        result["instance_center"] = x

        x = self.instance_regression_head(xi)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        result["instance_regression"] = x

        return result


class VideoDeepLabModel(nn.Module):
    """Panoptic Deeplab Model"""

    def __init__(self, arch, instance_aspp=False, pretrained_backbone=True, num_classes=20):
        super(VideoDeepLabModel, self).__init__()

        replace_stride_with_dilation = [False, False, True]
        atrous_rates = [6, 12, 18]
        # high_channels = RESNET_HIGH_FEATURE_SIZES[arch]
        # low1_channels = RESNET_STRIDE8_FEATURE_SIZES[arch]
        # low2_channels = RESNET_STRIDE4_FEATURE_SIZES[arch]
        high_channels = 2048
        low1_channels = 512
        low2_channels = 256
        low_channels = [low1_channels, low2_channels]
        low_channels_reduced_semantic = [64, 32]
        low_channels_reduced_instance = [32, 16]
        aspp_channels = 256
        out_channels_semantic = 256
        out_channels_instance = 256
        semantic_features = 256
        instance_features = 32

        backbone = create_resnet_backbone(arch, pretrained=pretrained_backbone,
                                          replace_stride_with_dilation=replace_stride_with_dilation)
        return_layers = {'layer1': 'low2', 'layer2': 'low1', 'layer4': 'high'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.semantic_aspp = ASPP(high_channels, atrous_rates)

        self.instance_aspp = None
        if instance_aspp:
            self.instance_aspp = ASPP(high_channels, atrous_rates)

        self.semantic_decoder = PanopticDeepLabDecoder(low_channels, low_channels_reduced_semantic, aspp_channels,
                                                       out_channels_semantic)
        self.instance_decoder = PanopticDeepLabDecoder(low_channels, low_channels_reduced_instance, aspp_channels,
                                                       out_channels_instance)
        self.semantic_head = PanopticDeepLabHead(out_channels_semantic, semantic_features, num_classes)
        self.instance_center_head = PanopticDeepLabHead(out_channels_instance, instance_features, 1)
        self.instance_regression_head = PanopticDeepLabHead(out_channels_instance, instance_features, 2)

    def forward(self, batch):
        result = OrderedDict()

        input_size = batch.shape[-2:]
        features = self.backbone(batch)

        # print(features["high"].shape)

        # Semantic Branch
        features["aspp"] = self.semantic_aspp(features["high"])
        x = self.semantic_decoder(features)

        x = self.semantic_head(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        result["semantic"] = x
        result["semantic_aspp"] = features["aspp"]

        # Instance Branch

        if self.instance_aspp:
            features["aspp"] = self.instance_aspp(features["high"])
            result["instance_aspp"] = features["aspp"]
        xi = self.instance_decoder(features)

        x = self.instance_center_head(xi)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        result["instance_center"] = x

        x = self.instance_regression_head(xi)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        result["instance_regression"] = x

        return result


class TrackingHead(nn.Module):
    """Tracking head module"""

    def __init__(self, in_channels, embedding_size, num_instances):
        super(TrackingHead, self).__init__()

        self.embedding = nn.Sequential(nn.Linear(in_channels, embedding_size),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(embedding_size, embedding_size))
        self.classifier = nn.Sequential(nn.BatchNorm1d(embedding_size),
                                        nn.Linear(embedding_size, num_instances))

    def forward(self, features, masks, ids):

        # print(features.shape)
        print(features.shape, len(masks))

        assert features.shape[0] == len(masks), "Batch size mismatch"
        for image_masks in masks:
            for mask in image_masks:
                assert features.shape[2] == mask.shape[0], "H size mismatch"
                assert features.shape[3] == mask.shape[1], "W size mismatch"
        batch_size = features.shape[0]
        feature_dim = features.shape[1]

        feature_vectors = []
        instance_ids = []

        for image_num, image_masks in enumerate(masks):
            for obj_num, mask in enumerate(image_masks):
                feature_mask = torch.stack(feature_dim * [mask])
                feature_vectors += [torch.mean(features[image_num] * feature_mask, dim=(1, 2))]
                instance_ids += [ids[image_num][obj_num]]

        feature_vectors, instance_ids = torch.stack(feature_vectors), torch.from_numpy(np.array(instance_ids))

        prebn_features = self.embedding(feature_vectors)
        classification_output = self.classifier(prebn_features)

        return prebn_features, classification_output, instance_ids



if __name__ == "__main__":
    # segmodel = DeepLabv3PlusModel(arch="resnet50", output_stride=8, pretrained_backbone=True, num_classes=5)
    segmodel = PanopticDeepLabModel(arch="resnet50", pretrained_backbone=True, num_classes=40)
    dummy_batch = torch.rand(2, 3, 1024, 1024)
    result = segmodel(dummy_batch)
    print(segmodel)
    print(result["semantic"].shape)
    print(result["instance_center"].shape)
    print(result["instance_regression"].shape)
