import cv2
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision.transforms import v2

import pandas as pd


####
def get_masks_and_boxes(mask):  #TODO: 从图像提取box和inst mask
    mask = torch.ones((1, 4))  # FloatTensor[N, 4]
    box = torch.ones((1, 512, 512))  # UInt8Tensor[N, H, W]
    return mask.type(torch.float), box.type(torch.uint8)


####
class MaskDataset(Dataset):
    def __init__(self, data_path, cases, train=True):
        self.data_path = data_path
        self.cases = cases
        self.train=True
        self.labels = pd.read_csv('%s/label.csv' % data_path)
        self.img_dir = '%s/img/' % data_path
        self.mask_dir = '%s/label/' % data_path
        self.get_masks_and_boxes = get_masks_and_boxes
        # self.shape_aug = v2.Compose([
        # ])
        # self.image_aug = v2.Compose([
        # ])

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        img_path = '%s/%s.png' % (self.img_dir, self.cases[idx])
        mask_path = '%s/%s.png' % (self.mask_dir, self.cases[idx])

        img = v2.ToDtype(torch.float32, scale=True)((read_image(img_path)))  # [3, H, W], 0-1
        mask = read_image(mask_path, ImageReadMode.GRAY)  # [H, W], 0/255
        # if self.train:
        #     img = self.shape_aug(img)
        #     img = self.image_aug(img)
        #     mask = self.shape_aug(mask)

        masks, boxes = self.get_masks_and_boxes(mask)
        label = self.labels[self.labels.iloc[:, 0] == self.cases[idx]].iloc[0, 1]
        labels = torch.tensor(label, dtype=torch.int64).repeat(masks.shape[0])  # Int64Tensor[N]

        return img, {'boxes': boxes, 'labels': labels, 'masks': masks}
