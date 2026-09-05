import os
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from utils.dataset import create_dataloaders
from models.model import create_model

NUM_CLASSES = 37
CHECKPOINT_PATH = "./checkpoints/best_model_label_smoothing.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, data_loader, criterion):
    model.eval()

    total_loss = 0.0
    total = 0
    top1_correct = 0
    top5_correct = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total += labels.size(0)

            # Top-1
            predictions = outputs.argmax(dim=1)
            top1_correct += (predictions == labels).sum().item()

            # Top-5
            top5_predictions = outputs.topk(5, dim=1).indices
            top5_correct += (
                (top5_predictions == labels.unsqueeze(1))
                .any(dim=1)
                .sum()
                .item()
            )

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    test_loss = total_loss / total
    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return test_loss, top1_accuracy, top5_accuracy, macro_f1


def main():
    print("=" * 60)
    print("Oxford-IIIT Pet | Label Smoothing Test Evaluation")
    print("=" * 60)

    print(f"使用设备: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"找不到模型文件: {CHECKPOINT_PATH}"
        )

    print("\n正在创建 DataLoader...")
    _, _, test_loader = create_dataloaders()
    print("DataLoader 创建完成！")

    print("\n正在创建模型...")
    model = create_model(num_classes=NUM_CLASSES)

    print(f"正在加载 Label Smoothing 模型:")
    print(CHECKPOINT_PATH)

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    print("\n开始测试...")
    print("-" * 60)

    test_loss, top1, top5, macro_f1 = evaluate(
        model,
        test_loader,
        criterion
    )

    print("\n" + "=" * 60)
    print("Label Smoothing 测试集最终结果")
    print("=" * 60)

    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Top-1 Accuracy: {top1:.4f} ({top1 * 100:.2f}%)")
    print(f"Top-5 Accuracy: {top5:.4f} ({top5 * 100:.2f}%)")
    print(f"Macro-F1      : {macro_f1:.4f}")

    print("=" * 60)
    print("评估完成！")


if __name__ == "__main__":
    main()