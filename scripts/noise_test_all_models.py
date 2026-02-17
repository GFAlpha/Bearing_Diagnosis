import os
import sys
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# ============================================================
# 关键修复：把 scripts 目录加入 sys.path，保证能 import 训练脚本里的模型类
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根目录
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")                         # scripts 目录
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# 现在可以直接从 scripts 目录下的 .py 文件导入（不需要 scripts. 前缀）
from train_cnn_v2 import CNN1D
from train_rnn_v4 import RNNLSTM, SEQ_LEN, FEAT_DIM
from train_cnn_bilstm import CNN_BiLSTM
from train_cnn_bilstm_att import CNN_BiLSTM_Att
from train_transformer_v2 import TransformerClassifier
from train_cnn_transformer import CNNTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NOISE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "noise_test")
OUT_ROOT = os.path.join(PROJECT_ROOT, "noise_results")

SNR_TAGS = ["clean", "snr_0", "snr_3", "snr_6", "snr_9"]

MODEL_ZOO = {
    "cnn": {
        "model_dir": os.path.join(PROJECT_ROOT, "models", "cnn"),
        "builder": lambda: CNN1D(num_classes=4),
        "input_adapter": "cnn_1d",      # [B, 1, L]
    },
    "rnn_lstm": {
        "model_dir": os.path.join(PROJECT_ROOT, "models", "rnn_lstm"),
        "builder": lambda: RNNLSTM(input_dim=FEAT_DIM, hidden_dim=128, num_layers=2, num_classes=4),
        "input_adapter": "rnn",         # [B, SEQ_LEN, FEAT_DIM]
    },
    "cnn_bilstm": {
        "model_dir": os.path.join(PROJECT_ROOT, "models", "cnn_bilstm"),
        "builder": lambda: CNN_BiLSTM(num_classes=4),
        "input_adapter": "raw_1d",      # [B, L]（模型内部也能兼容 [B,1,L]）
    },
    "cnn_bilstm_att": {
        "model_dir": os.path.join(PROJECT_ROOT, "models", "cnn_bilstm_att"),
        "builder": lambda: CNN_BiLSTM_Att(num_classes=4),
        "input_adapter": "raw_1d",      # [B, L]
    },
    "transformer": {
        "model_dir": os.path.join(PROJECT_ROOT, "models", "transformer"),
        "builder": lambda: TransformerClassifier(),
        "input_adapter": "transformer", # [B, L, 1]
    },
    "cnn_transformer": {
        "model_dir": os.path.join(PROJECT_ROOT, "models", "cnn_transformer"),
        "builder": lambda: CNNTransformer(num_classes=4),
        "input_adapter": "cnn_1d",      # [B, 1, L]
    },
}

def make_loader(X: np.ndarray, y: np.ndarray, adapter: str, batch_size: int = 64):
    """
    把 X_test 适配成不同模型需要的输入形状
    """
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    if adapter == "cnn_1d":
        # [B, 1, L]
        X_t = torch.tensor(X).unsqueeze(1)
    elif adapter == "transformer":
        # [B, L, 1]
        X_t = torch.tensor(X).unsqueeze(-1)
    elif adapter == "rnn":
        # [B, SEQ_LEN, FEAT_DIM]
        X_t = torch.tensor(X).view(-1, SEQ_LEN, FEAT_DIM)
    else:
        # raw_1d: [B, L]，让模型 forward 自己去兼容
        X_t = torch.tensor(X)

    y_t = torch.tensor(y)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

def eval_one(model, loader):
    """
    统一输出：
    - acc
    - y_true / y_pred
    - infer_times（每个 batch 一次 forward 的耗时）
    """
    model.eval()
    all_preds, all_labels = [], []
    infer_times = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            # 推理计时（更严谨：前后 synchronize）
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            logits = model(xb)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

            infer_times.append(t1 - t0)
            all_preds.extend(logits.argmax(1).detach().cpu().numpy())
            all_labels.extend(yb.detach().cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc, np.array(all_labels), np.array(all_preds), np.array(infer_times, dtype=np.float64)

def load_X_by_tag(tag: str):
    """
    从 data/noise_test 读取对应的测试集
    """
    if tag == "clean":
        return np.load(os.path.join(NOISE_DATA_DIR, "X_test_clean.npy"))
    snr = tag.split("_")[1]
    return np.load(os.path.join(NOISE_DATA_DIR, f"X_test_snr_{snr}.npy"))

def main():
    # 读取 y_test（所有 SNR 共用）
    y_test_path = os.path.join(NOISE_DATA_DIR, "y_test.npy")
    if not os.path.exists(y_test_path):
        raise FileNotFoundError(f"找不到 y_test.npy：{y_test_path}，请先运行 make_noisy_testset.py")

    y_test = np.load(y_test_path)

    for model_name, cfg in MODEL_ZOO.items():
        weight_path = os.path.join(cfg["model_dir"], "best_model_overall.pth")
        if not os.path.exists(weight_path):
            print(f"[WARN] {model_name} 缺少权重：{weight_path}，跳过。")
            continue

        model = cfg["builder"]().to(DEVICE)
        state = torch.load(weight_path, map_location=DEVICE)
        model.load_state_dict(state)

        for tag in SNR_TAGS:
            X = load_X_by_tag(tag)
            loader = make_loader(X, y_test, adapter=cfg["input_adapter"], batch_size=64)

            acc, y_true, y_pred, infer_times = eval_one(model, loader)

            out_dir = os.path.join(OUT_ROOT, model_name, tag)
            os.makedirs(out_dir, exist_ok=True)

            # 保存结果（每个模型 × 每个 SNR 都保存）
            np.save(os.path.join(out_dir, "acc.npy"), np.array([acc], dtype=np.float64))
            np.save(os.path.join(out_dir, "y_true.npy"), y_true)
            np.save(os.path.join(out_dir, "y_pred.npy"), y_pred)
            np.save(os.path.join(out_dir, "infer_times.npy"), infer_times)

            meta = {
                "model": model_name,
                "tag": tag,
                "device": DEVICE,
                "weight_path": weight_path.replace("\\", "/"),
                "acc": float(acc),
                "infer_time_mean_s_per_step": float(infer_times.mean()),
                "infer_time_std_s_per_step": float(infer_times.std()),
            }
            with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            print(f"[OK] {model_name:14s} | {tag:6s} | acc={acc:.4f} | infer={infer_times.mean()*1000:.3f}ms/step")

    print("[DONE] noise evaluation finished. 输出目录：noise_results/")

if __name__ == "__main__":
    main()