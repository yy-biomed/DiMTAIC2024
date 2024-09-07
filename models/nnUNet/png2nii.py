from pathlib import Path
import cv2
import numpy as np
import SimpleITK as sitk
import pandas as pd


if __name__ == '__main__':

    ############################################################################
    dataset_name = 'thyroid'  # 数据集名称，随意命名

    imag_dir = '/tcdata/train/img/'  # 数据路径
    mask_dir = '/tcdata/train/label/'
    test_dir = '/tcdata/test/img/'
    ############################################################################

    imag_save_dir = '/app/nnUNet/data/nnUNet_raw/Dataset000_%s/imagesTr/' % dataset_name  # 保存路径
    mask_save_dir = '/app/nnUNet/data/nnUNet_raw/Dataset000_%s/labelsTr/' % dataset_name

    Path(imag_save_dir).mkdir(parents=True, exist_ok=True)
    Path(mask_save_dir).mkdir(parents=True, exist_ok=True)

    imag_list = Path(imag_dir).glob('*.png')

    n = 0
    for f in imag_list:
        base_name = f.stem
        imag = cv2.imread('%s/%s.png' % (imag_dir, base_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread('%s/%s.png' % (mask_dir, base_name), cv2.IMREAD_GRAYSCALE)
        mask = mask != 0  # 背景 0，结节 1
        mask = mask.astype(np.uint8)
        imag = sitk.GetImageFromArray(imag.reshape((1, imag.shape[0], imag.shape[1])))
        mask = sitk.GetImageFromArray(mask.reshape((1, mask.shape[0], mask.shape[1])))
        sitk.WriteImage(imag, '%s/%s_%.3d_0000.nii.gz' % (imag_save_dir, dataset_name, n))
        sitk.WriteImage(mask, '%s/%s_%.3d.nii.gz' % (mask_save_dir, dataset_name, n))
        n = n + 1

    test_save_dir = '/app/nnUNet/data/nnUNet_raw/Dataset000_%s/imagesTs/' % dataset_name  # 测试数据路径
    Path(test_save_dir).mkdir(parents=True, exist_ok=True)
    test_list = Path(test_dir).glob('*.png')
    n_t = 0
    old = []
    new = []
    for f in test_list:
        base_name=f.stem
        imag = cv2.imread('%s/%s.png' % (imag_dir, base_name), cv2.IMREAD_GRAYSCALE)
        imag = sitk.GetImageFromArray(imag.reshape((1, imag.shape[0], imag.shape[1])))
        sitk.WriteImage(imag, '%s/%s_%.3d_0000.nii.gz' % (test_save_dir, dataset_name, n_t))
        old.append(base_name)
        new.append('%s_%.3d' % (dataset_name, n_t))
        n_t = n_t + 1

    df = pd.DataFrame({'old': old, 'new': new})
    df.to_csv('/app/nnUNet/name_map.csv', index=False)

    print('Done! Total images: train %d, test %d' % (n, n_t))
