import os
import time
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# =========================
# 参数
# =========================
DATA_DIR = "data/splits"
RESULT_DIR = os.path.join("results", "cnn_transformer_noiseaug")
MODEL_DIR = os.path.join("models", "cnn_transformer_noiseaug")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

NUM_RUNS = 5
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
NUM_CLASSES = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 噪声增强设置
# =========================
AUG_PROB = 0.5  # 训练 batch 加噪概率（0~1）

# 9/6/3/0 四档位噪声，噪声越来越强
AUG_SNR_CHOICES = [9, 6, 3, 0]

# 各档位采样权重（和为 1 最好，但不是硬性要求；random.choices 会按比例使用）
AUG_SNR_WEIGHTS = [0.50, 0.40, 0.09, 0.01]  

# 启用检查，防止权重填写错误
CHECK_AUG_WEIGHTS = True


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
# AWGN：对一个 batch 加噪（按每个样本单独控制 SNR）
# x: torch.Tensor, shape [B, 1, L]
# =========================
def add_awgn_batch(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    # 每个样本信号功率：mean(x^2)，形状 [B, 1, 1]
    p_signal = torch.mean(x * x, dim=-1, keepdim=True)

    # SNR(dB) -> 线性
    snr_linear = 10.0 ** (snr_db / 10.0)

    # 噪声功率与标准差
    p_noise = p_signal / snr_linear
    sigma = torch.sqrt(p_noise + 1e-12)  # 避免极小数导致数值问题

    noise = torch.randn_like(x) * sigma
    return x + noise


# =========================
# CNN + Transformer 模型
# =========================
class CNNTransformer(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [B, 1, L]
        x = self.cnn(x)           # [B, C, T]
        x = x.permute(0, 2, 1)    # [B, T, C]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# =========================
# 训练 / 验证 / 测试
# =========================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        # 训练阶段噪声增强（只作用于 train）
        if random.random() < AUG_PROB:
            snr = random.choices(AUG_SNR_CHOICES, weights=AUG_SNR_WEIGHTS, k=1)[0]
            x = add_awgn_batch(x, snr_db=snr)

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
            out = model(x).argmax(dim=1)
            preds.extend(out.cpu().numpy())
            labels.extend(y.cpu().numpy())
    return accuracy_score(labels, preds)


def test_with_infer_time(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    infer_times = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            logits = model(x)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

            infer_times.append(t1 - t0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    avg_infer_time = float(np.mean(infer_times))
    return acc, avg_infer_time, np.array(all_labels), np.array(all_preds), np.array(infer_times, dtype=np.float64)


# =========================
# 主函数
# =========================
def main():
    print("Using device:", DEVICE)

    # 基本合法性检查
    if CHECK_AUG_WEIGHTS:
        if len(AUG_SNR_CHOICES) != len(AUG_SNR_WEIGHTS):
            raise ValueError("AUG_SNR_CHOICES 和 AUG_SNR_WEIGHTS 长度必须一致。")
        if any(w < 0 for w in AUG_SNR_WEIGHTS):
            raise ValueError("AUG_SNR_WEIGHTS 不能为负数。")
        if sum(AUG_SNR_WEIGHTS) <= 0:
            raise ValueError("AUG_SNR_WEIGHTS 之和必须 > 0（否则无法采样）。")

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    def make_loader(X, y, shuffle: bool):
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [B, 1, L]
        y = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(X_train, y_train, True)
    val_loader = make_loader(X_val, y_val, False)
    test_loader = make_loader(X_test, y_test, False)

    test_accs, train_times, infer_times = [], [], []
    seeds = []

    best_run_idx = -1
    best_run_test_acc = -1.0
    best_run_val_acc = None
    best_run_model_path = None
    best_run_y_true, best_run_y_pred = None, None

    for run in range(NUM_RUNS):
        print(f"\n=== Run {run + 1}/{NUM_RUNS} ===")
        seed = 8000 + run
        seeds.append(seed)
        set_seed(seed)

        model = CNNTransformer(num_classes=NUM_CLASSES).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        best_val = 0.0
        best_model_path = os.path.join(MODEL_DIR, f"best_model_run{run+1}.pth")

        start_train = time.time()
        for epoch in range(EPOCHS):
            train_one_epoch(model, train_loader, optimizer, criterion)
            val_acc = evaluate_acc(model, val_loader)
            print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Val Acc: {val_acc:.4f}")

            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), best_model_path)

        train_time = time.time() - start_train
        train_times.append(train_time)

        # 测试：加载该 run 的 best
        state = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(state)

        test_acc, avg_infer_time, y_true, y_pred, infer_arr = test_with_infer_time(model, test_loader)
        test_accs.append(test_acc)
        infer_times.append(avg_infer_time)

        print(f"Test Acc: {test_acc:.4f}")
        print(f"Train Time: {train_time:.2f}s | Inference Time: {avg_infer_time*1000:.3f}ms/step (per batch)")

        if test_acc > best_run_test_acc:
            best_run_test_acc = test_acc
            best_run_idx = run
            best_run_val_acc = best_val
            best_run_model_path = best_model_path
            best_run_y_true, best_run_y_pred = y_true, y_pred

    test_accs = np.array(test_accs, dtype=np.float64)
    train_times = np.array(train_times, dtype=np.float64)
    infer_times = np.array(infer_times, dtype=np.float64)

    best_overall_path = os.path.join(MODEL_DIR, "best_model_overall.pth")
    if best_run_model_path is not None:
        shutil.copyfile(best_run_model_path, best_overall_path)

    np.save(os.path.join(RESULT_DIR, "test_accs.npy"), test_accs)
    np.save(os.path.join(RESULT_DIR, "train_times.npy"), train_times)
    np.save(os.path.join(RESULT_DIR, "infer_times.npy"), infer_times)
    np.save(os.path.join(RESULT_DIR, "y_true.npy"), best_run_y_true)
    np.save(os.path.join(RESULT_DIR, "y_pred.npy"), best_run_y_pred)

    meta = {
        "model_name": "CNN+Transformer (NoiseAug)",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "num_runs": NUM_RUNS,
        "num_classes": NUM_CLASSES,
        "device": DEVICE,
        "seeds": seeds,
        "aug_prob": AUG_PROB,
        "aug_snr_choices": AUG_SNR_CHOICES,
        "aug_snr_weights": AUG_SNR_WEIGHTS,
        "best_run_idx_by_test_acc": int(best_run_idx + 1),
        "best_run_test_acc": float(best_run_test_acc),
        "best_run_val_acc": float(best_run_val_acc) if best_run_val_acc is not None else None,
        "best_model_overall_path": best_overall_path.replace("\\", "/"),
    }
    np.save(os.path.join(RESULT_DIR, "meta.npy"), meta, allow_pickle=True)

    with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("CNN+Transformer (NoiseAug) Results\n")
        f.write(f"Test Accs: {test_accs.tolist()}\n")
        f.write(f"Mean Acc: {test_accs.mean():.6f}\n")
        f.write(f"Std Acc : {test_accs.std():.6f}\n")
        f.write(f"Avg Train Time (s): {train_times.mean():.6f}\n")
        f.write(f"Avg Inference Time (s/step, per batch): {infer_times.mean():.8f}\n")
        f.write(f"Best Run (by Test Acc): Run {best_run_idx+1}\n")
        f.write(f"Best Run Test Acc: {best_run_test_acc:.6f}\n")
        f.write(f"Best Model Overall: {best_overall_path}\n")
        f.write("\n[NoiseAug Settings]\n")
        f.write(f"AUG_PROB={AUG_PROB}\n")
        f.write(f"AUG_SNR_CHOICES={AUG_SNR_CHOICES}\n")
        f.write(f"AUG_SNR_WEIGHTS={AUG_SNR_WEIGHTS}\n")

    print("\n===== Final Statistics =====")
    print("Test Accuracies:", test_accs.tolist())
    print(f"Mean Acc: {test_accs.mean():.4f}")
    print(f"Std Acc : {test_accs.std():.4f}")
    print(f"Avg Train Time: {train_times.mean():.2f}s")
    print(f"Avg Inference Time: {infer_times.mean()*1000:.3f}ms/step (per batch)")


if __name__ == "__main__":
    main()