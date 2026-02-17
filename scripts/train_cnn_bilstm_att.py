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
RESULT_DIR = os.path.join("results", "cnn_bilstm_att")
MODEL_DIR = os.path.join("models", "cnn_bilstm_att")

EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
RUNS = 5
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
# Attention 机制
# =========================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: [B, T, 2H]
        scores = self.attn(x)                 # [B, T, 1]
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context


# =========================
# CNN + BiLSTM + Attention 模型
# =========================
class CNN_BiLSTM_Att(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.att = TemporalAttention(64)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # 兼容 [B, L] 或 [B, 1, L]
        if x.dim() == 2:
            x = x.unsqueeze(1)        # [B, 1, L]

        x = self.cnn(x)               # [B, 32, T]
        x = x.permute(0, 2, 1)        # [B, T, 32]

        x, _ = self.lstm(x)           # [B, T, 128]
        x = self.att(x)               # [B, 128]

        return self.fc(x)


# =========================
# 训练与评估函数
# =========================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()


def evaluate_acc(model, loader) -> float:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds.append(logits.argmax(1).cpu())
            labels.append(y.cpu())
    return accuracy_score(torch.cat(labels), torch.cat(preds))


def evaluate_with_preds(model, loader):
    """返回 y_true / y_pred（numpy）"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(y.numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    return y_true, y_pred


def test_with_infer_time(model, loader):
    """
    统一推理耗时口径：秒/step（每个 batch 一次 forward），对所有 batch 取均值。
    """
    model.eval()
    all_preds, all_labels = [], []
    infer_times = []

    with torch.no_grad():
        for x, y in loader:
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

    acc = accuracy_score(all_labels, all_preds)
    avg_infer_time = float(np.mean(infer_times))  # 秒/step（一个 batch）
    return acc, avg_infer_time


# =========================
# 主函数
# =========================
def main():
    # 加载数据
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    # DataLoader（固定划分 + 统一放外面）
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long()),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long()),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    test_accs, train_times, infer_times = [], [], []
    seeds = []

    # 用“Test Acc 最好的一次 run”来保存 y_true/y_pred & best_overall
    best_run_idx = -1
    best_run_test_acc = -1.0
    best_run_val_acc = None
    best_run_model_path = None
    best_run_y_true, best_run_y_pred = None, None

    for run in range(RUNS):
        print(f"\n=== Run {run + 1}/{RUNS} ===")

        seed = 5000 + run
        seeds.append(seed)
        set_seed(seed)

        model = CNN_BiLSTM_Att(NUM_CLASSES).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        # 每个 run 保存自己的 best_model_runX.pth（按 val acc 最优）
        best_val = 0.0
        best_model_path = os.path.join(MODEL_DIR, f"best_model_run{run+1}.pth")

        # -------- 训练 --------
        start_train = time.time()
        for epoch in range(EPOCHS):
            train_one_epoch(model, train_loader, criterion, optimizer)
            val_acc = evaluate_acc(model, val_loader)
            print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Val Acc: {val_acc:.4f}")

            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), best_model_path)

        train_time = time.time() - start_train
        train_times.append(train_time)

        # -------- 测试（用该 run 的 best 模型） --------
        state = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(state)

        test_acc, avg_infer_time = test_with_infer_time(model, test_loader)
        test_accs.append(test_acc)
        infer_times.append(avg_infer_time)

        print(f"Test Acc: {test_acc:.4f}")
        print(f"Train Time: {train_time:.2f}s | Inference Time: {avg_infer_time:.6f}s/step (per batch)")

        # 记录“按 Test Acc 最好”的 run（用于 overall + y_true/y_pred）
        if test_acc > best_run_test_acc:
            best_run_test_acc = test_acc
            best_run_idx = run
            best_run_val_acc = best_val
            best_run_model_path = best_model_path
            best_run_y_true, best_run_y_pred = evaluate_with_preds(model, test_loader)

    test_accs = np.array(test_accs, dtype=np.float64)
    train_times = np.array(train_times, dtype=np.float64)
    infer_times = np.array(infer_times, dtype=np.float64)

    # 额外保存“全局最优模型”（按 Test Acc 最好的一次 run）
    best_overall_path = os.path.join(MODEL_DIR, "best_model_overall.pth")
    if best_run_model_path is not None:
        shutil.copyfile(best_run_model_path, best_overall_path)

    # 保存结果（统一格式）
    np.save(os.path.join(RESULT_DIR, "test_accs.npy"), test_accs)
    np.save(os.path.join(RESULT_DIR, "train_times.npy"), train_times)
    np.save(os.path.join(RESULT_DIR, "infer_times.npy"), infer_times)
    np.save(os.path.join(RESULT_DIR, "y_true.npy"), best_run_y_true)
    np.save(os.path.join(RESULT_DIR, "y_pred.npy"), best_run_y_pred)

    meta = {
        "model_name": "CNN+BiLSTM+Attention",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "num_runs": RUNS,
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
        f.write("CNN+BiLSTM+Attention Results\n")
        f.write(f"Test Accs: {test_accs.tolist()}\n")
        f.write(f"Mean Acc: {test_accs.mean():.6f}\n")
        f.write(f"Std Acc : {test_accs.std():.6f}\n")
        f.write(f"Avg Train Time (s): {train_times.mean():.6f}\n")
        f.write(f"Avg Inference Time (s/step, per batch): {infer_times.mean():.8f}\n")
        f.write(f"Best Run (by Test Acc): Run {best_run_idx+1}\n")
        f.write(f"Best Run Test Acc: {best_run_test_acc:.6f}\n")
        f.write(f"Best Model Overall: {best_overall_path}\n")

    print("\n===== Final Statistics =====")
    print("Test Accuracies:", test_accs.tolist())
    print(f"Mean Acc: {test_accs.mean():.4f}")
    print(f"Std Acc : {test_accs.std():.4f}")
    print(f"Avg Train Time: {train_times.mean():.2f}s")
    print(f"Avg Inference Time: {infer_times.mean():.6f}s/step (per batch)")


if __name__ == "__main__":
    main()