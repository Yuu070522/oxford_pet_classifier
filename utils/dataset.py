import os
import random
import numpy as np
import pandas as pd
import torch

from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# =========================
# 基本配置
# =========================
DATA_DIR = "./data/oxford-iiit-pet/data"

IMAGE_SIZE = 224
NUM_CLASSES = 37
BATCH_SIZE = 32
SEED = 42


# =========================
# 固定随机种子
# =========================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# 读取 Parquet 数据
# =========================
def load_dataframe():
    train_path = os.path.join(
        DATA_DIR,
        "train-00000-of-00001.parquet"
    )

    test_path = os.path.join(
        DATA_DIR,
        "test-00000-of-00001.parquet"
    )

    print("正在读取数据...")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    df = pd.concat(
        [train_df, test_df],
        ignore_index=True
    )

    print(f"总图片数量: {len(df)}")
    print(f"数据列: {df.columns.tolist()}")

    return df


# =========================
# 自定义 Dataset
# =========================
class OxfordPetDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        # Hugging Face Parquet 中 image 通常是字典结构
        image_data = row["image"]

        if isinstance(image_data, dict):
            image_bytes = image_data["bytes"]
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        else:
            image = Image.open(image_data).convert("RGB")

        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# =========================
# 数据增强
# =========================
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 验证集 / 测试集
# 注意：这里绝对不能使用随机增强
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# 创建 DataLoader
# =========================
def create_dataloaders():

    set_seed(SEED)

    df = load_dataframe()

    # 使用 label 做分层抽样
    labels = df["label"]

    # 第一次：70% train，30% 临时集
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=labels
    )

    # 第二次：临时集一半 val，一半 test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"]
    )

    print("\n数据集划分:")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")

    train_dataset = OxfordPetDataset(
        train_df,
        transform=train_transform
    )

    val_dataset = OxfordPetDataset(
        val_df,
        transform=val_test_transform
    )

    test_dataset = OxfordPetDataset(
        test_df,
        transform=val_test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


# =========================
# 测试
# =========================
if __name__ == "__main__":

    print("开始测试 Oxford-IIIT Pet 数据集...")

    train_loader, val_loader, test_loader = create_dataloaders()

    images, labels = next(iter(train_loader))

    print("\nBatch 测试结果:")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Labels:", labels)

    print("\nDataset 测试成功！")