# 这其实是v4版本，但是懒得重命名或者新建文件夹了（v2版本的代码应该在项目根文件里，是.txt格式的）
# v1版本就是项目里的train_cnn.py
# v2版本是每次训练都随机划分数据集
# v3版本是固定划分数据集，多次训练取平均
# v4版本是将v3版本稍作修改，实现保存结果供后续画图
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import random

# =========================
# 1. 全局参数
# =========================
NUM_RUNS = 5
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
NUM_CLASSES = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 结果保存目录（给 A-2 画图用）
RESULT_DIR = "results/cnn_multi_run"
MODEL_DIR = "models"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# 2. 固定随机种子
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# 3. 数据集定义
# =========================
class BearingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]

# =========================
# 4. CNN 模型
# =========================
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

        # ⚠️ 256 需与你的切片长度匹配
        self.classifier = nn.Sequential(
            nn.Linear(32 * 256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# =========================
# 5. 单次训练 + 测试
# =========================
def train_and_test(run_id, train_loader, val_loader, test_loader):

    print(f"\n===== Run {run_id + 1} =====")

    model = CNN1D(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_DIR, f"cnn_run{run_id + 1}_best.pth")

    # --------- 训练 ---------
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # --------- 验证 ---------
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x).argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Val Acc: {val_acc:.4f}")

    # --------- 测试 ---------
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    print(f"Test Acc (Run {run_id + 1}): {test_acc:.4f}")

    return test_acc

# =========================
# 6. 主函数
# =========================
def main():

    # --------- 加载固定数据 ---------
    X_train = np.load("data/splits/X_train.npy")
    y_train = np.load("data/splits/y_train.npy")

    X_val = np.load("data/splits/X_val.npy")
    y_val = np.load("data/splits/y_val.npy")

    X_test = np.load("data/splits/X_test.npy")
    y_test = np.load("data/splits/y_test.npy")

    train_loader = DataLoader(
        BearingDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        BearingDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        BearingDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_test_acc = []

    # --------- 多次随机训练 ---------
    for run in range(NUM_RUNS):
        set_seed(1000 + run)
        acc = train_and_test(
            run,
            train_loader,
            val_loader,
            test_loader
        )
        all_test_acc.append(acc)

    all_test_acc = np.array(all_test_acc)

    # =========================
    # ⭐ 保存结果（给 A-2 画图用）
    # =========================
    np.save(os.path.join(RESULT_DIR, "test_accs.npy"), all_test_acc)

    with open(os.path.join(RESULT_DIR, "test_accs.txt"), "w") as f:
        for i, acc in enumerate(all_test_acc):
            f.write(f"Run {i+1}: {acc:.4f}\n")
        f.write(f"\nMean: {all_test_acc.mean():.4f}\n")
        f.write(f"Std: {all_test_acc.std():.4f}\n")

    print("\n===== Final Statistics =====")
    print("Test Accuracies:", all_test_acc)
    print(f"Mean Test Acc: {all_test_acc.mean():.4f}")
    print(f"Std Test Acc: {all_test_acc.std():.4f}")

# =========================
# 7. 入口
# =========================
if __name__ == "__main__":
    main()
