# nnUNet

## 数据预处理

* 使用png作为输入出现通道错误，故转换为nii.gz格式
  
  ```shell
  python /app/nnUNet/png2nii.py
  ```

* 按照模型要求构建数据集，创建json文件

  ```shell
  python /app/nnUNet/create_json.py
  ```

* 使用nnUNetv2_plan_and_preprocess命令进行预处理

  ```shell
  nnUNetv2_plan_and_preprocess -d 0 --verify_dataset_integrity
  ```

## 训练

* 使用nnUNetv2_train命令进行训练，5折交叉验证

  修改epoch数目，1000太久：/app/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py #152 1000->200

  ```shell
  nnUNetv2_train 0 2d 0 --npz
  nnUNetv2_train 0 2d 1 --npz
  nnUNetv2_train 0 2d 2 --npz
  nnUNetv2_train 0 2d 3 --npz
  nnUNetv2_train 0 2d 4 --npz
  ```

  * 使用nnUNetv2_find_best_configuration命令获取最佳配置

  ```shell
  nnUNetv2_find_best_configuration 0 -c 2d
  ```

## 推理

使用nnUNetv2_predict命令进行推理

```shell
nnUNetv2_predict \
-d 0 \
-i /app/nnUNet/data/nnUNet_raw/Dataset000_thyroid/imagesTs \
-o /app/nnUNet/data/pred \
-f  0 1 2 3 4 \
-tr nnUNetTrainer \
-c 2d \
-p nnUNetPlans
```

## 后处理

* 按照nnUNet算法进行后处理

  ```shell
  nnUNetv2_apply_postprocessing \
  -i /app/nnUNet/data/pred \
  -o /app/nnUNet/data/post \
  -pp_pkl_file /app/nnUNet/data/nnUNet_results/Dataset000_thyroid/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl \
  -np 8 \
  -plans_json /app/nnUNet/data/nnUNet_results/Dataset000_thyroid/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json
  ```

* 将文件格式转为png

  ```shell
  python /app/nnUNet/nii2png.py
  ```

