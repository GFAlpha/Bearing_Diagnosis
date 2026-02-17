import os
import time
import random
import shutil
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

# RNN 输入：[SEQ_LEN, FEAT_DIM]，要求 SEQ_LEN * FEAT_DIM = 1024
SEQ_LEN = 32
FEAT_DIM = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULT_DIR = os.path.join("results", "rnn_lstm")
MODEL_DIR = os.path.join("models", "rnn_lstm")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# 固定随机种子
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# 数据集
# =========================
class BearingDatasetRNN(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 原始 X 是 [1024]，这里 reshape 成 [SEQ_LEN, FEAT_DIM]
        x = self.X[idx].view(SEQ_LEN, FEAT_DIM)
        return x, self.y[idx]

# =========================
# RNN(LSTM) 模型
# =========================
class RNNLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, SEQ_LEN, FEAT_DIM]
        out, _ = self.lstm(x)          # out: [B, SEQ_LEN, hidden_dim]
        out = out[:, -1, :]            # 取最后一个时间步
        out = self.fc(out)             # [B, num_classes]
        return out

# =========================
# 单次训练 + 测试
# 返回：test_acc/train_time/avg_infer_time + y_true/y_pred + best_model_path/best_val_acc
# =========================
def train_and_test(run_id, train_loader, val_loader, test_loader):
    print(f"\n===== Run {run_id + 1}/{NUM_RUNS} =====")

    model = RNNLSTM(input_dim=FEAT_DIM, hidden_dim=128, num_layers=2, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_DIR, f"best_model_run{run_id+1}.pth")

    # ---------- 训练计时 ----------
    train_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # ---------- 验证 ----------
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

        # 保存验证集最优
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    train_time = time.time() - train_start

    # ---------- 测试 + 推理计时 ----------
    # 加载该 run 的 best val 模型
    state = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    test_preds, test_labels = [], []

    # 计“每个 batch 的耗时”，最后取均值（单位：秒/step）
    infer_times = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # 推理计时（更严谨的 GPU 计时：前后都 synchronize，避免异步队列残留带来的误差）
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            preds = model(x).argmax(dim=1)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

            infer_times.append(t1 - t0)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    avg_infer_time = float(np.mean(infer_times))  # 秒/step（一个 batch）
    test_acc = accuracy_score(test_labels, test_preds)

    print(f"Test Acc (Run {run_id+1}): {test_acc:.4f}")

    return (
        test_acc,
        train_time,
        avg_infer_time,
        np.array(test_labels),
        np.array(test_preds),
        best_model_path,
        best_val_acc,
    )

# =========================
# 主函数
# =========================
def main():
    # --------------- 数据路径 ---------------
    X_train = np.load("data/splits/X_train.npy")
    y_train = np.load("data/splits/y_train.npy")
    X_val = np.load("data/splits/X_val.npy")
    y_val = np.load("data/splits/y_val.npy")
    X_test = np.load("data/splits/X_test.npy")
    y_test = np.load("data/splits/y_test.npy")
    # ------------------------------------------------------------------

    train_loader = DataLoader(BearingDatasetRNN(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(BearingDatasetRNN(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(BearingDatasetRNN(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    test_accs, train_times, infer_times = [], [], []
    seeds = []

    # 用“Test Acc 最好的一次 run”来保存 y_true/y_pred & best_overall
    best_run_idx = -1
    best_run_test_acc = -1.0
    best_run_labels = None
    best_run_preds = None
    best_run_model_path = None
    best_run_val_acc = None

    for run in range(NUM_RUNS):
        seed = 3000 + run
        seeds.append(seed)
        set_seed(seed)

        acc, t_time, i_time, labels, preds, best_model_path, best_val_acc = train_and_test(
            run, train_loader, val_loader, test_loader
        )

        test_accs.append(acc)
        train_times.append(t_time)
        infer_times.append(i_time)

        if acc > best_run_test_acc:
            best_run_test_acc = acc
            best_run_idx = run
            best_run_labels = labels
            best_run_preds = preds
            best_run_model_path = best_model_path
            best_run_val_acc = best_val_acc

    test_accs = np.array(test_accs, dtype=np.float64)
    train_times = np.array(train_times, dtype=np.float64)
    infer_times = np.array(infer_times, dtype=np.float64)

    # 额外保存一个“全局最优模型”（按 Test Acc 最好的一次 run）
    best_overall_path = os.path.join(MODEL_DIR, "best_model_overall.pth")
    if best_run_model_path is not None:
        shutil.copyfile(best_run_model_path, best_overall_path)

    print("\n===== Final Statistics =====")
    print("Test Accuracies:", test_accs)
    print(f"Mean Acc: {test_accs.mean():.4f}")
    print(f"Std Acc : {test_accs.std():.4f}")
    print(f"Avg Train Time: {train_times.mean():.2f}s")
    print(f"Avg Inference Time: {infer_times.mean():.6f}s/step (per batch)")

    # ---------- 保存（统一为 npy） ----------
    np.save(os.path.join(RESULT_DIR, "test_accs.npy"), test_accs)
    np.save(os.path.join(RESULT_DIR, "train_times.npy"), train_times)
    np.save(os.path.join(RESULT_DIR, "infer_times.npy"), infer_times)

    # y_true / y_pred 保存“最好的一次 run”
    np.save(os.path.join(RESULT_DIR, "y_true.npy"), best_run_labels)
    np.save(os.path.join(RESULT_DIR, "y_pred.npy"), best_run_preds)

    meta = {
        "model_name": "RNNLSTM",
        "num_runs": NUM_RUNS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "num_classes": NUM_CLASSES,
        "device": DEVICE,
        "seeds": seeds,
        "seq_len": SEQ_LEN,
        "feat_dim": FEAT_DIM,
        "best_run_idx_by_test_acc": int(best_run_idx + 1),  # 1-based
        "best_run_test_acc": float(best_run_test_acc),
        "best_run_val_acc": float(best_run_val_acc) if best_run_val_acc is not None else None,
        "best_model_overall_path": best_overall_path.replace("\\", "/"),
    }
    np.save(os.path.join(RESULT_DIR, "meta.npy"), meta, allow_pickle=True)

    with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("RNNLSTM Results\n")
        f.write(f"Test Accs: {test_accs.tolist()}\n")
        f.write(f"Mean Acc: {test_accs.mean():.6f}\n")
        f.write(f"Std Acc : {test_accs.std():.6f}\n")
        f.write(f"Avg Train Time (s): {train_times.mean():.6f}\n")
        f.write(f"Avg Inference Time (s/step, per batch): {infer_times.mean():.8f}\n")
        f.write(f"Best Run (by Test Acc): Run {best_run_idx+1}\n")
        f.write(f"Best Run Test Acc: {best_run_test_acc:.6f}\n")
        f.write(f"Best Model Overall: {best_overall_path}\n")

if __name__ == "__main__":
    main()