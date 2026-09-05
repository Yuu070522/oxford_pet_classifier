import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW

from utils.dataset import create_dataloaders
from models.model import create_model

NUM_CLASSES = 37
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

CHECKPOINT_DIR = "./checkpoints"
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model_label_smoothing.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def evaluate(model, data_loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def main():
    print("=" * 60)
    print("Oxford-IIIT Pet | ResNet-18 | Label Smoothing")
    print("=" * 60)

    print(f"使用设备: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("\n正在创建 DataLoader...")
    train_loader, val_loader, test_loader = create_dataloaders()
    print("DataLoader 创建完成！")

    print("\n正在创建模型...")
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    print("模型创建完成！")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_accuracy = 0.0

    print("\n开始训练...")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("Label Smoothing: 0.1")
    print("-" * 60)

    total_start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion
        )

        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_accuracy:.4f} "
            f"| Time: {epoch_time:.1f}s"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH
            )

            print(
                f"  -> 保存最佳模型 "
                f"(Val Acc: {best_val_accuracy:.4f})"
            )

    total_time = time.time() - total_start_time

    print("\n" + "=" * 60)
    print("Label Smoothing 训练完成！")
    print("=" * 60)

    print(
        f"最佳 Val Accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"最佳模型: "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"总训练时间: "
        f"{total_time / 60:.2f} 分钟"
    )


if __name__ == "__main__":
    main()