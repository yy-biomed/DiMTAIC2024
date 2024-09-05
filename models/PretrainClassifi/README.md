# 分类预训练模型

## 模型

使用Resnet50在Imagenet上的预训练权重，替换原有全连接层

前50 epoch锁定除全连接层外全部权重，后50 epoch解除锁定，进行微调

构建DataLoader时对数据进行增强，详见loader.py

每个epoch保存权重于log文件夹，挑选前50 epoch在验证集中损失最少的权重输入后50epoch，最后保存在后50 epoch中损失最少的权重于log/best.pth

## 训练 train.py

注意修改数据路径，根据电脑情况选择合适的batch_size和n_worker

测试和正式训练时使用参数不同，详见train.py

* 检查清单：
  1. train_dir：存放训练数据的文件夹
  2. n_class：测试为3，正式为1
  3. n_epoch：测试为 (10, 10)，正式为 (50, 50)
  4. batch_size
  5. n_worker

* 开始训练：

  ```shell
  python train.py
  ```

## 推理 infer.py

注意调整参数，详见infer.py

* 检查清单：
  1. test_dir：存放测试数据的文件夹
  2. submit_dir：存放提交文件的文件夹
  3. n_class：测试为3，正式为1

* 开始推理：

  ```shell
  python infer.py
  ```

