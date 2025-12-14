import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================
# 参数
# =====================
BATCH_SIZE = 64
NUM_CLASSES = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["Normal", "Inner Race", "Ball", "Outer Race"]

# =====================
# 数据集
# =====================
class BearingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]


# =====================
# 模型（必须和训练时完全一致）
# =====================
class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# =====================
# 主流程
# =====================
def main():
    print(f"使用设备: {DEVICE}")

    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")

    # 只为了复现“测试集”，必须和训练脚本一致
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    test_loader = DataLoader(
        BearingDataset(X_test, y_test),
        batch_size=BATCH_SIZE
    )

    model = CNN1D(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load("models/cnn1d_best.pth"))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # =====================
    # 分类报告
    # =====================
    print("\n=== Classification Report ===")
    print(classification_report(
    all_labels,
    all_preds,
    labels=[0, 1, 2, 3],
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
    ))

    # =====================
    # 混淆矩阵
    # =====================
    cm = confusion_matrix(
    all_labels,
    all_preds,
    labels=[0, 1, 2, 3]
    )

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (CNN on CWRU Dataset)")
    plt.tight_layout()

    save_path = "results/confusion_matrix.png"
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"\n混淆矩阵已保存至: {save_path}")


if __name__ == "__main__":
    main()
