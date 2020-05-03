#!/usr/bin/env python3
"""
Dataloaders
"""

__author__ = "Rohit Gupta"
__version__ = "0.1b"
__license__ = "GPL2"

import random
from tqdm import tqdm
import numpy as np

import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToTensor

from utils.dataset import read_json, read_txt_lines, load_image
from utils.dataset import detect_letterboxing_video
from utils.dataset import generate_object_mask_center_offsets, generate_semantic_target, downsample_mask, identify_null_masks
from utils.dataset import get_dataset_paths
from utils.dataset import numpy_to_tensor



class YTVISDataset(VisionDataset):
    """
    Dataset Class for YTVIS
    """
    def __init__(self, root: str, transforms: object, phase: str, num_frames: int, output_stride: int = 16):
        super().__init__(root)

        phase = phase.upper()
        if phase not in ["TRAIN", "TRAINVAL", "VALID", "TEST"]:
            raise ValueError("Phase can only be train/trainval/valid/test.")
        self.phase = phase


        # if num_frames not in [1, 2, 3, 5, 7]:
        #     raise ValueError("num_frames can only be 1/2/3/5/7.")
        self.num_frames = num_frames

        paths = get_dataset_paths("YTVIS", root)

        if phase in ["TRAIN", "TRAINVAL"]:
            metadata_file = paths["TRAIN_METADATA"]
            video_dir = paths["TRAIN_VIDEOS"]
            if phase == "TRAIN":
                split_file = paths["TRAIN_SPLIT"]
            elif phase == "TRAINVAL":
                split_file = paths["TRAINVAL_SPLIT"]
            else:
                raise NotImplementedError("Dataloader only works for TRAIN & TRAINVAL for now.")
        else:
            raise NotImplementedError("Dataloader only works for TRAIN & TRAINVAL for now.")

        self.metadata, self.num_categories, self.num_instances = self.__read_ytvis_metadata(metadata_file, split_file, video_dir)
        self.output_stride = output_stride
        self.transforms = transforms
        # self.state = 0

    def __getitem__(self, item):
        vid_id = list(self.metadata.keys())[item]
        video = self.metadata[vid_id]
        vid_frames = len(video["frames"])

        # selected_frame = 14
        # selected_frame = self.state
        # self.state += 1


        selected_frames = random.sample(population=range(vid_frames), k=self.num_frames)
        print("Selected Frames", selected_frames, video["frames"][selected_frames[0]])

        # if self.num_frames == 1:
        #
        # else:
        #     raise NotImplementedError("num_frames can only be 1 for now.")

        vid_imgs = []
        vid_semantic_targets = []
        vid_centers = []
        vid_offsets = []
        vid_tracking_masks = []
        vid_tracking_ids = []

        for frame_num in selected_frames:
            img, (semantic_target, centers, offsets, tracking_masks, tracking_ids) = self.__get_frame(video, frame_num, item)
            vid_imgs += [img]
            vid_semantic_targets += [semantic_target]
            vid_centers += [centers]
            vid_offsets += [offsets]
            vid_tracking_masks += [tracking_masks]
            vid_tracking_ids += [tracking_ids]

        return vid_imgs, (vid_semantic_targets, vid_centers, vid_offsets, vid_tracking_masks, vid_tracking_ids)


    def __get_frame(self, video, selected_frame, item):

        width, height = video["width"], video["height"]
        img = load_image(video["frames"][selected_frame])

        masks = []
        categories = []
        instances = []
        centers = np.zeros((height, width), dtype=np.float32)
        x_offsets = np.zeros((height, width), dtype=np.float32)
        y_offsets = np.zeros((height, width), dtype=np.float32)

        for object_annotation in video["objects"]:
            rle_counts = object_annotation["segmentations"][selected_frame]
            if rle_counts is not None:
                mask, center, x, y = generate_object_mask_center_offsets(rle_counts, width, height, 41, 8)
                cat_id = int(object_annotation["category_id"])
                instance_id = int(object_annotation["instance_id"])
                categories += [cat_id]
                instances += [instance_id]
                masks += [mask]
                centers += center
                x_offsets += x
                y_offsets += y

        semantic_target = generate_semantic_target(masks, width, height, categories, self.num_categories)

        # Add background mask to semantic and tracking targets
        foreground_mask = semantic_target.sum(axis=0)
        background_mask = 1 - foreground_mask
        background_id = self.num_instances + item

        num_pixels = background_mask.shape[0] * background_mask.shape[1]
        num_labelled_pixels = int(np.sum(foreground_mask)) + int(np.sum(background_mask))
        if num_pixels != num_labelled_pixels:
            print(num_labelled_pixels)
            raise ValueError("All pixels must be labelled.")

        semantic_target = np.concatenate((np.expand_dims(background_mask, axis=0), semantic_target), axis=0)
        masks = [background_mask] + masks
        instances = [background_id] + instances

        # De-Letterboxing
        row_start = video["row_start"]
        row_end = video["row_end"]
        col_start = video["col_start"]
        col_end = video["col_end"]

        # print("De-Letterboxing", row_start, row_end, col_start, col_end)

        img = ToTensor()(img)
        img = img[:, row_start:row_end, col_start:col_end]

        semantic_target = semantic_target[:, row_start:row_end, col_start:col_end]
        centers = centers[row_start:row_end, col_start:col_end]
        x_offsets = x_offsets[row_start:row_end, col_start:col_end]
        y_offsets = y_offsets[row_start:row_end, col_start:col_end]
        masks = [mask[row_start:row_end, col_start:col_end] for mask in masks]

        # To Tensor

        semantic_target = numpy_to_tensor(semantic_target, "int")
        centers = numpy_to_tensor(centers, "float")
        x_offsets = numpy_to_tensor(x_offsets, "int")
        y_offsets = numpy_to_tensor(y_offsets, "int")
        tracking_masks = [numpy_to_tensor(mask, "int") for mask in masks]
        tracking_ids = instances

        # Add channel dimension to everything (except masks)
        offsets = torch.stack((x_offsets, y_offsets))
        centers = centers.unsqueeze(0)


        # print("Number of objects in frame:", len(tracking_masks))

        # Augmentations
        img, (semantic_target, centers, offsets, tracking_masks, tracking_ids) = \
            self.transforms(img, (semantic_target, centers, offsets, tracking_masks, tracking_ids))

        # Masks for tracking
        tracking_masks = [downsample_mask(mask, self.output_stride) for mask in tracking_masks]

        # Removing masks for objects cropped out
        null_masks = identify_null_masks(tracking_masks)

        for idx in null_masks:
            del tracking_masks[idx]
            del tracking_ids[idx]

        return img, (semantic_target, centers, offsets, tracking_masks, tracking_ids)

    def __len__(self):
        return len(self.metadata.keys())

    def __read_ytvis_metadata(self, metadata_file, split_file, video_dir):
        """
        Read metadata for YouTube-VIS dataset.
        """
        video_info = read_json(metadata_file)
        split_ids = read_txt_lines(split_file, int)


        # print(video_info["categories"])

        category_map = {x["id"]: x["name"] for x in video_info["categories"]}
        print(video_info["categories"])
        num_categories = len(video_info["categories"]) + 1

        metadata = {}

        pbar = tqdm(video_info["videos"])

        for video in pbar:
            vid_id = video["id"]
            if vid_id not in split_ids:
                continue
            pbar.set_description("Reading video %d" % vid_id)
            vid_hash = video["file_names"][0].split("/")[0]
            w, h = video["width"], video["height"]
            frames = [video_dir + x for x in video["file_names"]]

            row_start, row_end, col_start, col_end = detect_letterboxing_video(frames, mode="first")

            metadata[vid_id] = {"hash"     : vid_hash,
                                "width"    : w, "height": h,
                                "row_start": row_start, "row_end": row_end,
                                "col_start": col_start, "col_end": col_end,
                                "frames"   : frames,
                                "objects"  : []
                                }

        # instance_names = []
        instance_count = 0
        for object_info in video_info["annotations"]:
            vid_id = object_info["video_id"]
            if vid_id not in split_ids:
                continue
            object_category = category_map[object_info["category_id"]]
            object_info["category"] = object_category
            object_info["instance_id"] = instance_count
            metadata[vid_id]["objects"].append(object_info)
            instance_count += 1
            # instance_names += [str(vid_id) + str(object_info["id"])]
        print(str(instance_count), "objects found in", self.phase , "set.")

        return metadata, num_categories, instance_count


def collate_with_labels_fn(batch):
    images = []
    semantic_targets = []
    centers = []
    offsets = []
    masks = []
    ids = []

    for vid_imgs, (vid_semantic_targets, vid_centers, vid_offsets, vid_tracking_masks, vid_tracking_ids) in batch:
        images += vid_imgs
        semantic_targets += vid_semantic_targets
        centers += vid_centers
        offsets += vid_offsets
        masks += vid_tracking_masks
        ids += vid_tracking_ids

    images = torch.stack(images)
    semantic_targets = torch.stack(semantic_targets)
    centers = torch.stack(centers)
    offsets = torch.stack(offsets)

    return images, (semantic_targets, centers, offsets, masks, ids)
