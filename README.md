# Oxford-IIIT Pet Fine-Grained Image Classification

基于深度学习的细粒度图像分类项目，使用 Oxford-IIIT Pet 数据集和预训练 ResNet-18 完成 37 类宠物品种分类。

## 1. 项目简介

本项目是科研项目组初步培训考核项目，主要目标是完成一个可运行、可复现、结构清晰的细粒度图像分类系统。

主要内容包括：

- Oxford-IIIT Pet 数据集
- 37 个宠物品种类别
- 70% / 15% / 15% 分层划分 Train / Val / Test
- 预训练 ResNet-18
- AdamW 优化器
- CrossEntropyLoss
- Label Smoothing 消融实验
- Random Rotation 数据增强实验
- Top-1 / Top-5 / Macro-F1 评价指标
- 混淆矩阵分析
- Grad-CAM 可视化
- 模块化项目结构

## 2. 项目结构

```text
oxford_pet_classifier/
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── oxford-iiit-pet/
├── models/
│   └── model.py
├── utils/
│   ├── dataset.py
│   ├── metrics.py
│   └── gradcam.py
├── checkpoints/
│   ├── best_model.pth
│   ├── best_model_label_smoothing.pth
│   └── best_model_augmentation.pth
├── outputs/
│   ├── confusion_matrix/
│   └── gradcam/
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
3. 环境

主要依赖：

Python 3.12
PyTorch
torchvision
timm
NumPy
pandas
scikit-learn
matplotlib
TensorBoard
Pillow
PyArrow

依赖安装：

pip install -r requirements.txt
4. 数据集

使用 Oxford-IIIT Pet 数据集，共包含：

7349 张图像
37 个类别

数据采用分层随机划分：

Train：5144
Validation：1102
Test：1103

训练集使用随机裁剪和随机水平翻转等数据增强；验证集和测试集不使用随机增强。

5. 模型

基线模型采用 ImageNet 预训练 ResNet-18，并将最后的全连接层修改为 37 类输出：

ResNet-18
    ↓
Global Average Pooling
    ↓
Fully Connected
    ↓
37 classes

训练配置：

Batch Size：32
Epochs：10
Optimizer：AdamW
Learning Rate：1e-4
Loss：CrossEntropyLoss
6. 基线实验结果

基线模型最佳验证集准确率：

91.74%

测试集结果：

Metric	Result
Top-1 Accuracy	91.84%
Top-5 Accuracy	99.73%
Macro-F1	91.77%
7. 消融实验

进行了两组单变量消融实验。

7.1 Label Smoothing

在 CrossEntropyLoss 中加入：

label_smoothing=0.1

测试集结果：

Metric	Result
Top-1 Accuracy	92.66%
Top-5 Accuracy	99.27%
Macro-F1	92.65%

相比基线，Top-1 和 Macro-F1 均有所提升。

7.2 Random Rotation

在训练集加入：

RandomRotation(15)

验证集和测试集保持不变。

测试集结果：

Metric	Result
Top-1 Accuracy	92.20%
Top-5 Accuracy	99.46%
Macro-F1	92.22%

相比基线，Top-1 和 Macro-F1 均有所提升。

7.3 消融实验总结
Experiment	Val Acc	Test Top-1	Test Top-5	Macro-F1
Baseline	91.74%	91.84%	99.73%	91.77%
Label Smoothing	91.92%	92.66%	99.27%	92.65%
Random Rotation	92.11%	92.20%	99.46%	92.22%

综合测试集 Top-1 Accuracy 和 Macro-F1，Label Smoothing 实验表现最好。

8. 模型分析
Confusion Matrix

项目生成 37 类混淆矩阵，用于分析不同宠物品种之间的分类混淆情况。

生成位置：

outputs/confusion_matrix/confusion_matrix.png
Grad-CAM

使用 Grad-CAM 对模型关注区域进行可视化。

目标层：

model.layer4[-1].conv2

生成位置：

outputs/gradcam/gradcam_sample.png

Grad-CAM 结果可以用于观察模型是否主要关注宠物的头部、面部、耳朵等具有类别区分性的区域。

9. 运行方法

训练基线模型：

python train.py

测试最佳模型：

python evaluate.py

运行 Grad-CAM：

python -m utils.gradcam
10. 结果与结论

ResNet-18 基线模型在 Oxford-IIIT Pet 测试集上取得约 91.84% 的 Top-1 Accuracy 和 99.73% 的 Top-5 Accuracy，满足项目初期 80%+ 准确率目标。

两组消融实验均提升了 Top-1 Accuracy 和 Macro-F1，其中 Label Smoothing 综合表现最佳。

项目进一步加入了混淆矩阵和 Grad-CAM，可用于分析模型分类错误以及模型关注区域。

11. AI 辅助开发与问题复盘

本项目开发过程中使用 AI 辅助完成部分代码设计、模块整理和问题排查。

主要学习和修正的问题包括：

验证集和测试集不能使用随机数据增强。
使用 CrossEntropyLoss 时不需要额外添加 Softmax。
验证和测试阶段需要使用 model.eval() 和 torch.no_grad()。
PyTorch 梯度默认会累积，因此训练时需要先执行 optimizer.zero_grad()。
RTX 5070 Laptop GPU 使用 Blackwell 架构，需要安装支持 CUDA 12.8 的 PyTorch 环境。

通过上述问题排查，最终完成了可运行的训练、评估、可视化和模块化工程。