

import torch

from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from datasets import YTVISDataset, collate_with_labels_fn
from utils.dataset import get_dataset_paths
from transforms import JointCompose
from transforms import JointResize, JointNormalize, JointCenterCrop

val_transforms = JointCompose([
        JointResize(height=360, width=640),
        JointCenterCrop(height=315, width=560),
        JointNormalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])


NUM_FRAMES = 2


paths = get_dataset_paths("YTVIS")
trainval_dataset = YTVISDataset(root=paths["ROOT"], transforms=val_transforms, phase="trainval", num_frames=NUM_FRAMES)
model = resnet50(pretrained=True, replace_stride_with_dilation=[False, False, True])
model.eval()
feature_map = IntermediateLayerGetter(model, {'layer4': 'feature_map'})
vid_imgs, (vid_semantic_targets, vid_centers, vid_offsets, vid_tracking_masks, vid_tracking_ids) = trainval_dataset[0]

print(vid_tracking_ids)

features = feature_map(vid_imgs)['feature_map']
features1 = features[0]
features2 = features[1]

feature_vectors1 = []
instance_ids1 = []
for id, mask in enumerate(vid_tracking_masks[0]):
    feature_mask = torch.stack(2048 * [mask])
    feature_vectors1 += [torch.mean(features1 * feature_mask, dim=(1, 2))]
    instance_ids1 += [[0][id]]

feature_vectors2 = []
instance_ids2 = []
for id, mask in enumerate(vid_tracking_masks[1]):
    feature_mask = torch.stack(2048 * [mask])
    feature_vectors2 += [torch.mean(features2 * feature_mask, dim=(1, 2))]
    instance_ids2 += [[1][id]]


print(instance_ids1)
print(instance_ids2)



