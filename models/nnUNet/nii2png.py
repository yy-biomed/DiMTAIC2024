import SimpleITK as sitk
import numpy as np
from pathlib import Path
import pandas as pd
import cv2


def nii2arr(nii_path):
    imag = sitk.ReadImage(nii_path)
    imag = sitk.GetArrayFromImage(imag)[0]
    imag = (imag > 0.5).astype(np.uint8)

    return imag


if __name__ == '__main__':

    ############################################################################
    nii_dir = '/app/nnUNet/data/post/'
    png_dir = '/app/submit/img/'
    ############################################################################

    name_map = pd.read_csv('/app/nnUNet/name_map.csv')

    Path(png_dir).mkdir(parents=True, exist_ok=True)
    for f in Path(nii_dir).glob('*.nii.gz'):
        img = nii2arr(f)
        base_name = f.stem[:-4]
        orig_name = name_map[name_map['new'] == base_name]['old'].values[0]
        cv2.imwrite('%s/%s.png' % (png_dir, orig_name), img)

    print('Done!')
