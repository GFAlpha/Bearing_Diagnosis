import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import random

# =========================
# 参数
# =========================
NUM_RUNS = 5
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
NUM_CLASSES = 4

SEQ_LEN = 64      # 时间步数
FEAT_DIM = 16     # 每个时间步的特征数（64*16=1024）

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("models", exist_ok=True)

# =========================
# 固定随机种子
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# 数据集
# =========================
class BearingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].view(SEQ_LEN, FEAT_DIM)
        return x, self.y[idx]

# =========================
# LSTM 模型
# =========================
class LSTMModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=FEAT_DIM,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]      # 取最后一个时间步
        return self.fc(out)

# =========================
# 单次训练 + 测试
# =========================
def train_and_test(run_id, train_loader, val_loader, test_loader):
    print(f"\n===== Run {run_id + 1} =====")

    model = LSTMModel(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_path = f"models/lstm_run{run_id+1}_best.pth"

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

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
            torch.save(model.state_dict(), best_path)

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Val Acc: {val_acc:.4f}")

    # Test
    model.load_state_dict(torch.load(best_path))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    print(f"Test Acc (Run {run_id+1}): {test_acc:.4f}")
    return test_acc

# =========================
# 主函数
# =========================
def main():
    X_train = np.load("data/splits/X_train.npy")
    y_train = np.load("data/splits/y_train.npy")
    X_val = np.load("data/splits/X_val.npy")
    y_val = np.load("data/splits/y_val.npy")
    X_test = np.load("data/splits/X_test.npy")
    y_test = np.load("data/splits/y_test.npy")

    train_loader = DataLoader(BearingDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(BearingDataset(X_val, y_val), BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(BearingDataset(X_test, y_test), BATCH_SIZE, shuffle=False)

    accs = []
    for run in range(NUM_RUNS):
        set_seed(2000 + run)
        accs.append(train_and_test(run, train_loader, val_loader, test_loader))

    accs = np.array(accs)
    print("\n===== Final Statistics =====")
    print("Test Accuracies:", accs)
    print(f"Mean Test Acc: {accs.mean():.4f}")
    print(f"Std Test Acc: {accs.std():.4f}")

    # 保存结果
    os.makedirs("results", exist_ok=True)
    np.save("results/rnn_test_accs.npy", accs)


if __name__ == "__main__":
    main()
