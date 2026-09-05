import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


NUM_CLASSES = 37


def create_model(num_classes=NUM_CLASSES):
    # 加载 ImageNet 预训练 ResNet-18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # 将最后的全连接层修改为 37 分类
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    print("正在创建 ResNet-18 模型...")

    model = create_model()

    print(model.fc)

    # 测试输入
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", output.shape)

    assert output.shape == (2, NUM_CLASSES)

    print("模型测试成功！")