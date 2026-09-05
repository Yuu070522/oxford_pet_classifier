import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data.dataset import create_dataloaders
from models.model import create_model


# ============================================================
# 基本配置
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "./checkpoints/best_model.pth"

OUTPUT_DIR = "./outputs/gradcam"


# ============================================================
# Grad-CAM
# ============================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # 保存前向传播的特征图
        self.forward_hook = target_layer.register_forward_hook(
            self.save_activation
        )

        # 保存反向传播的梯度
        self.backward_hook = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image, target_class):
        """
        image:
            [1, 3, 224, 224]

        target_class:
            模型预测的类别编号
        """

        self.model.zero_grad()

        # 前向传播
        output = self.model(image)

        # 取目标类别对应的分数
        score = output[:, target_class]

        # 反向传播
        score.backward()

        gradients = self.gradients
        activations = self.activations

        # ----------------------------------------------------
        # Global Average Pooling
        # ----------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # 加权求和
        # ----------------------------------------------------

        cam = (weights * activations).sum(
            dim=1,
            keepdim=True
        )

        # ReLU
        cam = F.relu(cam)

        # ----------------------------------------------------
        # Resize 到原始图片大小
        # ----------------------------------------------------

        cam = F.interpolate(
            cam,
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()

        # ----------------------------------------------------
        # 归一化到 0~1
        # ----------------------------------------------------

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam

    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


# ============================================================
# 反归一化
# ============================================================

def denormalize(image):
    """
    把 ImageNet Normalize 后的图片恢复成可以显示的 RGB 图片。
    """

    mean = np.array([
        0.485,
        0.456,
        0.406
    ])

    std = np.array([
        0.229,
        0.224,
        0.225
    ])

    image = image.cpu().numpy().transpose(
        1, 2, 0
    )

    image = image * std + mean

    image = np.clip(
        image,
        0,
        1
    )

    return image


# ============================================================
# 保存 Grad-CAM
# ============================================================

def save_gradcam(
        image_tensor,
        cam,
        true_label,
        pred_label,
        save_path
):
    """
    保存：
    左边：原始图片
    右边：Grad-CAM
    """

    image = denormalize(
        image_tensor
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    # --------------------------------------------------------
    # 左图：原始图片
    # --------------------------------------------------------

    axes[0].imshow(image)

    axes[0].set_title(
        f"Original | True: {true_label}",
        fontsize=14
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # 右图：Grad-CAM
    # --------------------------------------------------------

    axes[1].imshow(image)

    axes[1].imshow(
        cam,
        cmap="jet",
        alpha=0.5
    )

    axes[1].set_title(
        f"Grad-CAM | Pred: {pred_label}",
        fontsize=14
    )

    axes[1].axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 主程序
# ============================================================

def main():

    print("Using device:", DEVICE)

    # ========================================================
    # 1. 创建数据集
    # ========================================================

    _, _, test_loader = create_dataloaders()

    print()
    print("测试集加载完成。")
    print("开始寻找错误分类样本...")
    print()

    # ========================================================
    # 2. 创建模型
    # ========================================================

    model = create_model(
        num_classes=37
    )

    # ========================================================
    # 3. 加载训练好的最佳模型
    # ========================================================

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print("模型加载完成。")

    # ========================================================
    # 4. 设置 Grad-CAM 目标层
    # ========================================================

    target_layer = model.layer4[-1].conv2

    gradcam = GradCAM(
        model,
        target_layer
    )

    # ========================================================
    # 5. 创建输出目录
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    wrong_found = False

    # ========================================================
    # 6. 遍历测试集
    # ========================================================

    for images, labels in test_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        # ----------------------------------------------------
        # 正常预测
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                images
            )

            predictions = outputs.argmax(
                dim=1
            )

        # ----------------------------------------------------
        # 在当前 batch 中逐张寻找错误分类
        # ----------------------------------------------------

        for i in range(
                images.size(0)
        ):

            true_label = labels[i].item()

            pred_label = predictions[i].item()

            # =================================================
            # 找到错误分类
            # =================================================

            if true_label != pred_label:

                print(
                    "找到错误分类样本！"
                )

                print(
                    "True label:",
                    true_label
                )

                print(
                    "Pred label:",
                    pred_label
                )

                # ------------------------------------------------
                # 取出这一张图片
                # ------------------------------------------------

                single_image = images[
                    i:i + 1
                ]

                # ------------------------------------------------
                # 对模型“预测类别”进行 Grad-CAM
                # ------------------------------------------------

                cam = gradcam.generate(
                    single_image,
                    pred_label
                )

                # ------------------------------------------------
                # 保存
                # ------------------------------------------------

                save_path = os.path.join(
                    OUTPUT_DIR,
                    "gradcam_wrong.png"
                )

                save_gradcam(
                    single_image[0],
                    cam,
                    true_label,
                    pred_label,
                    save_path
                )

                print()
                print(
                    "Grad-CAM 已保存："
                )

                print(
                    os.path.abspath(
                        save_path
                    )
                )

                print()
                print(
                    "错误分类 Grad-CAM 生成完成！"
                )

                wrong_found = True

                break

        # =====================================================
        # 已经找到就结束整个测试集遍历
        # =====================================================

        if wrong_found:
            break

    # ========================================================
    # 关闭 Hook
    # ========================================================

    gradcam.close()

    # ========================================================
    # 最终提示
    # ========================================================

    if not wrong_found:

        print(
            "没有找到错误分类样本。"
        )


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()