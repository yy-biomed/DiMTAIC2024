from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from pathlib import Path

if __name__ == '__main__':

    ############################################################################
    dataset_name = 'thyroid'  # 数据集名称，随意命名
    imag_dir = '/tcdata/train/img/'  # 数据路径，用于获取图片数目
    ############################################################################

    json_dir = '/app/nnUNet/data/nnUNet_raw/Dataset000_%s/' % dataset_name
    Path(json_dir).mkdir(parents=True, exist_ok=True)
    n_case = len(list(Path(imag_dir).glob('*.png')))

    generate_dataset_json(output_folder=json_dir,
                          channel_names={0: 'Ultrasound'},
                          dataset_name=dataset_name,
                          num_training_cases=n_case,
                          file_ending='.nii.gz',
                          labels={'background': 0, 'nodule': 1})

    print('Done!')
