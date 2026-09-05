import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from data.dataset import create_dataloaders
from models.model import create_model
from utils.metrics import top1_accuracy, top5_accuracy, macro_f1


NUM_CLASSES = 37
CHECKPOINT_PATH = "./checkpoints/best_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def evaluate(model, data_loader, criterion):
    model.eval()

    total_loss = 0.0
    total = 0

    all_labels = []
    all_predictions = []
    all_top5_predictions = []

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            total += labels.size(0)

            predictions = outputs.argmax(dim=1)

            top5_predictions = outputs.topk(
                5,
                dim=1
            ).indices

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_top5_predictions.extend(
                top5_predictions.cpu().numpy()
            )

    test_loss = total_loss / total

    top1_accuracy_value = top1_accuracy(
        all_labels,
        all_predictions
    )

    top5_accuracy_value = top5_accuracy(
        all_labels,
        all_top5_predictions
    )

    macro_f1_value = macro_f1(
        all_labels,
        all_predictions
    )

    return (
        test_loss,
        top1_accuracy_value,
        top5_accuracy_value,
        macro_f1_value,
        all_labels,
        all_predictions
    )


def save_confusion_matrix(
    all_labels,
    all_predictions
):

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES))
    )

    os.makedirs(
        "./outputs/confusion_matrix",
        exist_ok=True
    )

    plt.figure(figsize=(14, 12))

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.colorbar()

    plt.tight_layout()

    save_path = (
        "./outputs/confusion_matrix/"
        "confusion_matrix.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"混淆矩阵已保存: {save_path}"
    )


def save_results(
    test_loss,
    top1,
    top5,
    macro_f1_value
):

    os.makedirs(
        "./results",
        exist_ok=True
    )

    save_path = "./results/baseline_results.txt"

    with open(
        save_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Oxford-IIIT Pet | "
            "ResNet-18 Baseline\n"
        )

        f.write("=" * 50 + "\n")

        f.write(
            f"Test Loss: "
            f"{test_loss:.4f}\n"
        )

        f.write(
            f"Top-1 Accuracy: "
            f"{top1:.4f} "
            f"({top1 * 100:.2f}%)\n"
        )

        f.write(
            f"Top-5 Accuracy: "
            f"{top5:.4f} "
            f"({top5 * 100:.2f}%)\n"
        )

        f.write(
            f"Macro-F1: "
            f"{macro_f1_value:.4f}\n"
        )

    print(
        f"测试结果已保存: {save_path}"
    )


def main():

    print("=" * 60)

    print(
        "Oxford-IIIT Pet | "
        "ResNet-18 Test Evaluation"
    )

    print("=" * 60)

    print(
        f"使用设备: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        raise FileNotFoundError(
            f"找不到模型文件: "
            f"{CHECKPOINT_PATH}"
        )

    print(
        "\n正在创建 DataLoader..."
    )

    _, _, test_loader = (
        create_dataloaders()
    )

    print(
        "DataLoader 创建完成！"
    )

    print(
        "\n正在创建模型..."
    )

    model = create_model(
        num_classes=NUM_CLASSES
    )

    print(
        f"正在加载最佳模型: "
        f"{CHECKPOINT_PATH}"
    )

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    print(
        "\n开始测试..."
    )

    print("-" * 60)

    (
        test_loss,
        top1,
        top5,
        macro_f1_value,
        all_labels,
        all_predictions
    ) = evaluate(
        model,
        test_loader,
        criterion
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "测试集最终结果"
    )

    print(
        "=" * 60
    )

    print(
        f"Test Loss     : "
        f"{test_loss:.4f}"
    )

    print(
        f"Top-1 Accuracy: "
        f"{top1:.4f} "
        f"({top1 * 100:.2f}%)"
    )

    print(
        f"Top-5 Accuracy: "
        f"{top5:.4f} "
        f"({top5 * 100:.2f}%)"
    )

    print(
        f"Macro-F1      : "
        f"{macro_f1_value:.4f}"
    )

    save_confusion_matrix(
        all_labels,
        all_predictions
    )

    save_results(
        test_loss,
        top1,
        top5,
        macro_f1_value
    )

    print("=" * 60)

    print(
        "评估完成！"
    )


if __name__ == "__main__":
    main()