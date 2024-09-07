import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2
from pathlib import Path
import pandas as pd


####
class ClassDataset(Dataset):
    def __init__(self, data_dir, sample, img_shape, mode='train', aug=True):
        self.img_dir = '%s/img/' % data_dir
        self.img_shape = img_shape
        self.mode = mode  # mode in 'train' or 'valid'
        self.aug = aug

        labels = pd.read_csv('%s/label.csv' % data_dir)
        self.sample_labels = labels[labels.iloc[:, 0].isin(sample)]

        self.transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Grayscale(num_output_channels=3),
            v2.CenterCrop(self.img_shape),
        ])

        if aug:
            self.transforms_train = v2.Compose([
                v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # 旋转-10°~10°，x、y轴平移0.1%，缩放0.9%~1.1%，倾斜10°
                v2.RandomHorizontalFlip(p=0.5),  # 上下翻转
                v2.RandomChoice(
                    [
                        v2.GaussianBlur(kernel_size=3),  # 高斯模糊
                        v2.GaussianNoise(sigma=0.05),  # 高斯噪声
                    ]),
                v2.ColorJitter(brightness=0.1),  # 对于灰度图，只改变亮度
            ])

    def __len__(self):
        return len(self.sample_labels)

    def __getitem__(self, idx):
        img_path = Path(self.img_dir, self.sample_labels.iloc[idx, 0] + '.png')
        image = read_image(str(img_path))  # 读取3个通道
        label = self.sample_labels.iloc[idx, 1]
        image = self.transforms(image)
        if self.mode == 'train' and self.aug:
            image = self.transforms_train(image)
        return image, label


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random

    train_dir = '../../tcdata/train/'
    tp_label = pd.read_csv('%s/label.csv' % train_dir)
    basename = tp_label.iloc[:, 0].values.tolist()
    random.shuffle(basename)
    train_sample = basename[:int(len(basename) * 0.8)]
    valid_sample = basename[int(len(basename) * 0.8):]

    dataset = ClassDataset(
        train_dir, train_sample,
        img_shape=(512, 512),
        mode='train', aug=True)

    for i in range(5):
        img = dataset[i][0].numpy()
        plt.imshow(img[0, ...], cmap='gray')
        plt.show()
    print('Done!')
