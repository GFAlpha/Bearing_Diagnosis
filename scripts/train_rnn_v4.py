import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# =========================
# 参数
# =========================
NUM_RUNS = 5
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
NUM_CLASSES = 4

SEQ_LEN = 64
FEAT_DIM = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULT_DIR = "results/rnn_lstm"
MODEL_DIR = "models/rnn_lstm"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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
        out = out[:, -1, :]
        return self.fc(out)

# =========================
# 单次训练 + 测试
# =========================
def train_and_test(run_id, train_loader, val_loader, test_loader):
    print(f"\n===== Run {run_id + 1}/{NUM_RUNS} =====")

    model = LSTMModel(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_path = f"{MODEL_DIR}/best_model_run{run_id+1}.pth"

    # ---------- 训练计时 ----------
    train_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x).argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    train_time = time.time() - train_start

    # ---------- 测试 + 推理计时 ----------
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_preds, test_labels = [], []
    infer_start = time.time()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    infer_time = (time.time() - infer_start) / len(test_loader)

    test_acc = accuracy_score(test_labels, test_preds)
    print(f"Test Acc (Run {run_id+1}): {test_acc:.4f}")

    return test_acc, train_time, infer_time

# =========================
# 主函数
# =========================
def main():
    # 加载数据
    X_train = np.load("data/splits/X_train.npy")
    y_train = np.load("data/splits/y_train.npy")
    X_val = np.load("data/splits/X_val.npy")
    y_val = np.load("data/splits/y_val.npy")
    X_test = np.load("data/splits/X_test.npy")
    y_test = np.load("data/splits/y_test.npy")

    train_loader = DataLoader(BearingDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(BearingDataset(X_val, y_val), BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(BearingDataset(X_test, y_test), BATCH_SIZE, shuffle=False)

    test_accs, train_times, infer_times = [], [], []

    for run in range(NUM_RUNS):
        set_seed(3000 + run)
        acc, t_time, i_time = train_and_test(run, train_loader, val_loader, test_loader)
        test_accs.append(acc)
        train_times.append(t_time)
        infer_times.append(i_time)

    test_accs = np.array(test_accs)
    train_times = np.array(train_times)
    infer_times = np.array(infer_times)

    print("\n===== Final Statistics =====")
    print("Test Accuracies:", test_accs)
    print(f"Mean Acc: {test_accs.mean():.4f}")
    print(f"Std Acc : {test_accs.std():.4f}")
    print(f"Avg Train Time: {train_times.mean():.2f}s")
    print(f"Avg Inference Time: {infer_times.mean():.4f}s")

    # ---------- 保存 ----------
    np.save(f"{RESULT_DIR}/test_accs.npy", test_accs)
    np.save(f"{RESULT_DIR}/train_times.npy", train_times)
    np.save(f"{RESULT_DIR}/infer_times.npy", infer_times)

    with open(f"{RESULT_DIR}/summary.txt", "w", encoding="utf-8") as f:
        f.write("RNN (LSTM) Results\n")
        f.write(f"Mean Acc: {test_accs.mean():.6f}\n")
        f.write(f"Std Acc : {test_accs.std():.6f}\n")
        f.write(f"Avg Train Time: {train_times.mean():.4f}s\n")
        f.write(f"Avg Inference Time: {infer_times.mean():.6f}s\n")

if __name__ == "__main__":
    main()
