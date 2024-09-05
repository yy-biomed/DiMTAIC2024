# DiMTAIC 2024

## model

### PretrainClassifi

ResNET50预训练分割模型








############################## OLD ##############################

## 1. 文件结构

1. data/

   储存调试用数据，包括BUSI乳腺超声图像，BUSI数据可用于模拟比赛用数据

2. models/

   拟使用模型

   nnUNet基本调试完成，拿到数据后可直接运行

3. documents/

   其他各类文档

## 2. 数据管理

为更好地对数据进行操作，在拿到数据后首先需将数据转换成我们想要的格式

1. 图像和掩码分别位于两个文件夹中
2. 对应图像和掩码的文件名相同
3. 文件名应包含足够数量的信息，例如良恶性、编号等，用`_`分割，例如malignant_000.png、malignant_001.png、benign_003.png、normal_004.png

## 3. nnUNet

### 3.1 BUSI数据

1. 首先运行png2nii.py脚本，将png图片转化为nii.gz格式
2. 其次运行dataset000.sh，nnUNet模型高度集成，无需其他参数调整

### 3.2 比赛数据

1. 由于超声影像为黑白图像，而png为三通道图像格式，在输入模型后出现通道问题，故需要先改为nii.gz格式

   通过运行png2nii.py脚本修改格式

   脚本中注意修改路径和输出文件名，输入路径imag_dir和mask_dir中的图像和掩码应当一一对应，同时文件名相同；输出图像为“`数据集名称`\_`数据编号`\_`0000`.nii.gz”（0000表示只有一个通道），输出掩码为“`数据集名称`_`数据编号`.nii.gz”

2. 输出的nii.gz文件应按照nnUNet要求格式整理，详见[Dataset conversion](documentation/dataset_format.md)，其中dataset.json文件按需改动

3. 运行dataset000.sh脚本，对数据进行预处理和训练，注意路径设置和数据集编号设置

## 4 数据增强

