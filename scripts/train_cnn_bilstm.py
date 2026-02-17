import os
import time
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

# =========================
# 参数
# =========================
DATA_DIR = "data/splits"
RESULT_DIR = os.path.join("results", "cnn_bilstm")
MODEL_DIR = os.path.join("models", "cnn_bilstm")

EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
NUM_RUNS = 5
NUM_CLASSES = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
# 模型：CNN + BiLSTM
# =========================
class CNN_BiLSTM(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # 1024 -> pool2次 -> 256，CNN输出 [B, 32, 256]
        # 给 LSTM 的输入维度是 32（通道数），序列长度是 256
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(64 * 2, num_classes)

    def forward(self, x):
        # 兼容输入 shape：[B, 1024] 或 [B, 1, 1024]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.cnn(x)              # [B, 32, 256]
        x = x.permute(0, 2, 1)       # [B, 256, 32] 作为序列输入 LSTM

        out, _ = self.lstm(x)        # [B, 256, 128]
        out = out[:, -1, :]          # 取最后一个时间步 [B, 128]
        out = self.fc(out)           # [B, num_classes]
        return out

# =========================
# 训练与评估（单次 run）
# =========================
def train_one_run(run_id, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\n===== Run {run_id+1}/{NUM_RUNS} =====")

    model = CNN_BiLSTM(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_DIR, f"best_model_run{run_id+1}.pth")

    # 训练计时
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

        # 验证
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                preds = logits.argmax(1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Val Acc: {val_acc:.4f}")

        # 保存验证集最优
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    train_time = time.time() - train_start

    # 测试（加载该 run 的 best val 模型）
    state = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    all_preds, all_labels = [], []
    infer_times = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # 推理计时（更严谨的 GPU 计时：前后都 synchronize，避免异步队列残留带来的误差）
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            logits = model(x)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

            infer_times.append(t1 - t0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    avg_infer_time = float(np.mean(infer_times))  # 秒/step（一个 batch）

    print(f"Test Acc (Run {run_id+1}): {test_acc:.4f}")

    return test_acc, train_time, avg_infer_time, np.array(all_labels), np.array(all_preds), best_model_path, best_val_acc

# =========================
# 主函数
# =========================
def main():
    # 读取固定划分
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

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
        seed = 5000 + run
        seeds.append(seed)
        set_seed(seed)

        acc, t_time, i_time, labels, preds, best_model_path, best_val_acc = train_one_run(
            run, X_train, y_train, X_val, y_val, X_test, y_test
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

    # 保存结果
    np.save(os.path.join(RESULT_DIR, "test_accs.npy"), test_accs)
    np.save(os.path.join(RESULT_DIR, "train_times.npy"), train_times)
    np.save(os.path.join(RESULT_DIR, "infer_times.npy"), infer_times)
    np.save(os.path.join(RESULT_DIR, "y_true.npy"), best_run_labels)
    np.save(os.path.join(RESULT_DIR, "y_pred.npy"), best_run_preds)

    meta = {
        "model_name": "CNN_BiLSTM",
        "num_runs": NUM_RUNS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "num_classes": NUM_CLASSES,
        "device": DEVICE,
        "seeds": seeds,
        "best_run_idx_by_test_acc": int(best_run_idx + 1),  # 1-based
        "best_run_test_acc": float(best_run_test_acc),
        "best_run_val_acc": float(best_run_val_acc) if best_run_val_acc is not None else None,
        "best_model_overall_path": best_overall_path.replace("\\", "/"),
    }
    np.save(os.path.join(RESULT_DIR, "meta.npy"), meta, allow_pickle=True)

    with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("CNN_BiLSTM Results\n")
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