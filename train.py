import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from data.dataset import create_dataloaders
from models.model import create_model


# =========================
# 配置
# =========================
NUM_CLASSES = 37
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

CHECKPOINT_DIR = "./checkpoints"
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LOG_DIR = "./logs/baseline"
writer = SummaryWriter(LOG_DIR)


# =========================
# 验证函数
# =========================
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

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


# =========================
# 主训练流程
# =========================
def main():

    print("=" * 60)
    print("Oxford-IIIT Pet | ResNet-18 Baseline")
    print("=" * 60)

    print(f"使用设备: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 创建保存目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # -------------------------
    # 加载数据
    # -------------------------
    print("\n正在创建 DataLoader...")

    train_loader, val_loader, _ = create_dataloaders()

    print("DataLoader 创建完成！")

    # -------------------------
    # 创建模型
    # -------------------------
    print("\n正在创建模型...")

    model = create_model(
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    print("模型创建完成！")

    # -------------------------
    # Loss
    # -------------------------
    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -------------------------
    # 记录最佳验证准确率
    # -------------------------
    best_val_accuracy = 0.0

    print("\n开始训练...")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("-" * 60)

    total_start_time = time.time()

    # =========================
    # Epoch 循环
    # =========================
    for epoch in range(1, EPOCHS + 1):

        epoch_start_time = time.time()

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # -------------------------
        # Training
        # -------------------------
        for batch_idx, (images, labels) in enumerate(
            train_loader
        ):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # 清空梯度
            optimizer.zero_grad()

            # Forward
            outputs = model(images)

            # Loss
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()

            # 更新参数
            optimizer.step()

            # 统计
            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        # -------------------------
        # Validation
        # -------------------------
        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion
        )

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_accuracy, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_accuracy, epoch)

        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_accuracy:.4f} "
            f"| Time: {epoch_time:.1f}s"
        )

        # -------------------------
        # 保存最佳模型
        # -------------------------
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
    writer.close()

    print("\n" + "=" * 60)
    print("训练完成！")
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