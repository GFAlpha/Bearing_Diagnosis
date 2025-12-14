import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# 1. 全局参数
# =========================
NUM_RUNS = 5
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3

DATA_DIR = "data/splits"
RESULT_DIR = "results/rnn_multi_run"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 4
INPUT_SIZE = 1      # 每个时间步 1 个特征
HIDDEN_SIZE = 64
NUM_LAYERS = 2

os.makedirs(RESULT_DIR, exist_ok=True)

# =========================
# 2. 固定随机种子工具
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# 3. Dataset
# =========================
class NpyDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)

        # CNN -> RNN 形状转换
        # (1, signal_len) -> (signal_len, 1)
        x = x.squeeze(0).unsqueeze(-1)

        return x, y

# =========================
# 4. LSTM 模型
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True
        )
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # 最后一个时间步
        out = self.fc(out)
        return out

# =========================
# 5. 加载数据
# =========================
train_ds = NpyDataset(
    f"{DATA_DIR}/X_train.npy",
    f"{DATA_DIR}/y_train.npy"
)
val_ds = NpyDataset(
    f"{DATA_DIR}/X_val.npy",
    f"{DATA_DIR}/y_val.npy"
)
test_ds = NpyDataset(
    f"{DATA_DIR}/X_test.npy",
    f"{DATA_DIR}/y_test.npy"
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# =========================
# 6. 训练 & 验证
# =========================
def train_one_run(run_id):
    print(f"\n===== Run {run_id} =====")
    set_seed(42 + run_id)

    model = LSTMClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_path = f"{RESULT_DIR}/best_model_run{run_id}.pth"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        print(f"Epoch [{epoch}/{EPOCHS}] | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    # 测试
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    test_acc = correct / total
    print(f"Test Acc (Run {run_id}): {test_acc:.4f}")
    return test_acc

# =========================
# 7. 多次运行
# =========================
all_test_accs = []

for run in range(1, NUM_RUNS + 1):
    acc = train_one_run(run)
    all_test_accs.append(acc)

all_test_accs = np.array(all_test_accs)

np.save(f"{RESULT_DIR}/test_accs.npy", all_test_accs)

with open(f"{RESULT_DIR}/test_accs.txt", "w") as f:
    for i, acc in enumerate(all_test_accs, 1):
        f.write(f"Run {i}: {acc:.4f}\n")
    f.write(f"\nMean: {all_test_accs.mean():.4f}\n")
    f.write(f"Std: {all_test_accs.std():.4f}\n")

print("\n===== Final Statistics =====")
print("Test Accuracies:", all_test_accs)
print("Mean Test Acc:", all_test_accs.mean())
print("Std Test Acc:", all_test_accs.std())
