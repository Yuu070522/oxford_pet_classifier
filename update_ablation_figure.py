import os
import matplotlib.pyplot as plt


# 项目中的输出位置
output_dir = "./results/figures"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "ablation_comparison.png"
)


# ============================================================
# 最终统一后的实验结果
# ============================================================

experiments = [
    "Baseline",
    "Label Smoothing",
    "Random Rotation"
]

top1 = [
    91.66,
    92.66,
    92.20
]

top5 = [
    99.55,
    99.27,
    99.46
]

macro_f1 = [
    91.63,
    92.65,
    92.22
]


# ============================================================
# 绘制对比图
# ============================================================

x = range(len(experiments))

width = 0.25

plt.figure(figsize=(10, 6))

bars1 = plt.bar(
    [i - width for i in x],
    top1,
    width=width,
    label="Top-1 Accuracy"
)

bars2 = plt.bar(
    x,
    top5,
    width=width,
    label="Top-5 Accuracy"
)

bars3 = plt.bar(
    [i + width for i in x],
    macro_f1,
    width=width,
    label="Macro-F1"
)


# ============================================================
# 图标题和坐标轴
# ============================================================

plt.title(
    "Ablation Experiment Comparison",
    fontsize=16
)

plt.ylabel(
    "Score (%)",
    fontsize=12
)

plt.xticks(
    list(x),
    experiments
)

plt.ylim(
    88,
    101
)

plt.legend()


# ============================================================
# 在柱子上显示具体数值
# ============================================================

for bars in [bars1, bars2, bars3]:

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.15,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )


plt.tight_layout()


# ============================================================
# 保存
# ============================================================

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print()
print("消融实验对比图生成完成！")
print()
print("保存位置：")
print(os.path.abspath(output_path))
print()