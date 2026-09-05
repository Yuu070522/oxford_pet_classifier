import numpy as np
from sklearn.metrics import f1_score


def top1_accuracy(labels, predictions):
    """
    计算 Top-1 Accuracy
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    return np.mean(labels == predictions)


def top5_accuracy(labels, top5_predictions):
    """
    计算 Top-5 Accuracy

    labels:
        真实标签，形状 [N]

    top5_predictions:
        模型 Top-5 预测结果，形状 [N, 5]
    """
    labels = np.asarray(labels)
    top5_predictions = np.asarray(top5_predictions)

    correct = np.any(
        top5_predictions == labels[:, None],
        axis=1
    )

    return np.mean(correct)


def macro_f1(labels, predictions):
    """
    计算 Macro-F1
    """
    return f1_score(
        labels,
        predictions,
        average="macro"
    )


if __name__ == "__main__":
    print("正在测试 metrics.py...")

    labels = np.array([0, 1, 2, 3, 4])

    predictions = np.array([0, 1, 2, 2, 4])

    top5_predictions = np.array([
        [0, 3, 5, 2, 1],
        [2, 1, 4, 0, 3],
        [2, 5, 1, 0, 3],
        [1, 4, 2, 0, 3],
        [4, 2, 1, 3, 0]
    ])

    print(
        f"Top-1 Accuracy: "
        f"{top1_accuracy(labels, predictions):.4f}"
    )

    print(
        f"Top-5 Accuracy: "
        f"{top5_accuracy(labels, top5_predictions):.4f}"
    )

    print(
        f"Macro-F1: "
        f"{macro_f1(labels, predictions):.4f}"
    )

    print("metrics.py 测试成功！")