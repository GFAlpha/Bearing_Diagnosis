import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# =========================================================
# 路径配置
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = os.path.join(PROJECT_ROOT, "analysis_results", "feature_viz")
os.makedirs(OUT_DIR, exist_ok=True)

NOISE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "noise_test")


# =========================================================
# 导入模型定义（来自训练脚本）
# =========================================================
from train_cnn_transformer import CNNTransformer  # CNN+Transformer / NoiseAug 都用这个结构
from train_cnn_v2 import CNN1D
from train_rnn_v4 import RNNLSTM, SEQ_LEN, FEAT_DIM
from train_cnn_bilstm import CNN_BiLSTM
from train_cnn_bilstm_att import CNN_BiLSTM_Att
from train_transformer_v2 import TransformerClassifier


# =========================================================
# 类别名（可用于 legend）
# =========================================================
CLASS_NAMES = ["Normal", "Ball", "Inner", "Outer"]


# =========================================================
# 支持的模型池：和项目结构对齐
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ZOO = {
    "cnn": {
        "display": "CNN",
        "model_dir": os.path.join(PROJECT_ROOT, "models", "cnn"),
        "builder": lambda: CNN1D(num_classes=4),
        "input_adapter": "cnn_1d",  # [B,1,L]
    },
    "rnn_lstm": {
        "display": "LSTM",
        "model_dir": os.path.join(PROJECT_ROOT, "models", "rnn_lstm"),
        "builder": lambda: RNNLSTM(input_dim=FEAT_DIM, hidden_dim=128, num_layers=2, num_classes=4),
        "input_adapter": "rnn",     # [B,SEQ_LEN,FEAT_DIM]
    },
    "cnn_bilstm": {
        "display": "CNN+BiLSTM",
        "model_dir": os.path.join(PROJECT_ROOT, "models", "cnn_bilstm"),
        "builder": lambda: CNN_BiLSTM(num_classes=4),
        "input_adapter": "raw_1d",  # [B,L]
    },
    "cnn_bilstm_att": {
        "display": "CNN+BiLSTM+Att",
        "model_dir": os.path.join(PROJECT_ROOT, "models", "cnn_bilstm_att"),
        "builder": lambda: CNN_BiLSTM_Att(num_classes=4),
        "input_adapter": "raw_1d",
    },
    "transformer": {
        "display": "Transformer",
        "model_dir": os.path.join(PROJECT_ROOT, "models", "transformer"),
        "builder": lambda: TransformerClassifier(),
        "input_adapter": "transformer",  # [B,L,1]
    },
    "cnn_transformer": {
        "display": "CNN+Transformer",
        "model_dir": os.path.join(PROJECT_ROOT, "models", "cnn_transformer"),
        "builder": lambda: CNNTransformer(num_classes=4),
        "input_adapter": "cnn_1d",
    },
    "cnn_transformer_noiseaug": {
        "display": "CNN+Transformer(NoiseAug)",
        "model_dir": os.path.join(PROJECT_ROOT, "models", "cnn_transformer_noiseaug"),
        "builder": lambda: CNNTransformer(num_classes=4),
        "input_adapter": "cnn_1d",
    },
}


# =========================================================
# 数据读取
# =========================================================
X_FILE_MAP = {
    "clean": "X_test_clean.npy",
    "snr_0": "X_test_snr_0.npy",
    "snr_3": "X_test_snr_3.npy",
    "snr_6": "X_test_snr_6.npy",
    "snr_9": "X_test_snr_9.npy",
}
Y_FILE = "y_test.npy"


def load_xy(tag: str):
    x_path = os.path.join(NOISE_DATA_DIR, X_FILE_MAP[tag])
    y_path = os.path.join(NOISE_DATA_DIR, Y_FILE)
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"找不到 {x_path}，请先运行 make_noisy_testset.py")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"找不到 {y_path}，请确认 data/noise_test 下有 y_test.npy")

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    return X, y


def make_loader(X: np.ndarray, y: np.ndarray, adapter: str, batch_size: int = 128):
    if adapter == "cnn_1d":
        X_t = torch.tensor(X).unsqueeze(1)          # [B,1,L]
    elif adapter == "transformer":
        X_t = torch.tensor(X).unsqueeze(-1)         # [B,L,1]
    elif adapter == "rnn":
        X_t = torch.tensor(X).view(-1, SEQ_LEN, FEAT_DIM)  # [B,32,32]
    else:
        X_t = torch.tensor(X)                       # [B,L]

    y_t = torch.tensor(y)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


# =========================================================
# 关键：抽取“最后分类层之前”的特征
# 做法：找到模型中的最后一个 nn.Linear，把它的输入当作 embedding
# =========================================================
def extract_features(model: nn.Module, loader: DataLoader, max_points: int = 2000):
    model.eval()
    feats = []
    labels = []

    # 找到最后一个线性层
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linears) == 0:
        raise RuntimeError("这个模型里没找到 nn.Linear，无法用通用 hook 抽特征。")

    last_fc = linears[-1]
    cache = {"x": None}

    def _pre_hook(module, inputs):
        # inputs 是一个 tuple，inputs[0] 就是 fc 的输入特征
        cache["x"] = inputs[0].detach()

    h = last_fc.register_forward_pre_hook(_pre_hook)

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            _ = model(xb)  # forward 触发 hook

            x = cache["x"]
            if x is None:
                continue

            x = x.detach().cpu().numpy()
            y_np = yb.numpy()

            feats.append(x)
            labels.append(y_np)

            if sum(len(a) for a in labels) >= max_points:
                break

    h.remove()

    F = np.concatenate(feats, axis=0)
    Y = np.concatenate(labels, axis=0)
    # 截断到 max_points
    if len(Y) > max_points:
        F = F[:max_points]
        Y = Y[:max_points]
    return F, Y


# =========================================================
# 可视化：PCA / t-SNE
# =========================================================
def plot_2d(points_2d: np.ndarray, y: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(7, 6))

    for cls in np.unique(y):
        idx = (y == cls)
        name = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else f"class_{int(cls)}"
        plt.scatter(points_2d[idx, 0], points_2d[idx, 1], s=8, alpha=0.7, label=name)

    plt.title(title)
    plt.xlabel("Dim-1")
    plt.ylabel("Dim-2")
    plt.legend(markerscale=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_viz(F: np.ndarray, y: np.ndarray, prefix: str, seed: int = 42):
    # 1) PCA
    pca = PCA(n_components=2, random_state=seed)
    p2 = pca.fit_transform(F)
    plot_2d(p2, y, f"{prefix} | PCA(2D)", os.path.join(OUT_DIR, f"{prefix}_pca.png"))

    # 2) t-SNE（先 PCA 到 50 维再 t-SNE 更稳定）
    F50 = F
    if F.shape[1] > 50:
        pca50 = PCA(n_components=50, random_state=seed)
        F50 = pca50.fit_transform(F)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=seed
    )
    t2 = tsne.fit_transform(F50)
    plot_2d(t2, y, f"{prefix} | t-SNE(2D)", os.path.join(OUT_DIR, f"{prefix}_tsne.png"))


def main():
    parser = argparse.ArgumentParser(description="Feature visualization (PCA / t-SNE) for model embeddings.")
    parser.add_argument("--models", type=str, default="cnn_transformer,cnn_transformer_noiseaug",
                        help="要可视化的模型 key，逗号分隔。默认：cnn_transformer,cnn_transformer_noiseaug")
    parser.add_argument("--tags", type=str, default="clean,snr_0",
                        help="要可视化的数据集 tag，逗号分隔。默认：clean,snr_0（论文对比最强）")
    parser.add_argument("--max_points", type=int, default=2000,
                        help="每张图最多取多少个样本点（点太多 t-SNE 会慢）。默认 2000")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    args = parser.parse_args()

    model_keys = [s.strip() for s in args.models.split(",") if s.strip()]
    tags = [s.strip() for s in args.tags.split(",") if s.strip()]

    print("[INFO] Device:", DEVICE)
    print("[INFO] Output:", OUT_DIR.replace("\\", "/"))
    print("[INFO] Models:", model_keys)
    print("[INFO] Tags  :", tags)

    meta_all = []

    for mk in model_keys:
        if mk not in MODEL_ZOO:
            print(f"[WARN] 未知模型 key：{mk}，跳过。")
            continue

        cfg = MODEL_ZOO[mk]
        weight_path = os.path.join(cfg["model_dir"], "best_model_overall.pth")
        if not os.path.exists(weight_path):
            print(f"[WARN] 缺少权重：{weight_path}，跳过 {mk}")
            continue

        model = cfg["builder"]().to(DEVICE)
        state = torch.load(weight_path, map_location=DEVICE)
        model.load_state_dict(state)

        for tag in tags:
            X, y = load_xy(tag)
            loader = make_loader(X, y, adapter=cfg["input_adapter"], batch_size=128)

            # 抽特征
            F, Y = extract_features(model, loader, max_points=args.max_points)

            prefix = f"{mk}_{tag}"
            run_viz(F, Y, prefix=prefix, seed=args.seed)

            # 同时把特征存下来（后面你想做别的分析会很方便）
            npz_path = os.path.join(OUT_DIR, f"{prefix}_features.npz")
            np.savez_compressed(npz_path, features=F, labels=Y)

            meta = {
                "model_key": mk,
                "model_name": cfg["display"],
                "tag": tag,
                "weight_path": weight_path.replace("\\", "/"),
                "n_points": int(len(Y)),
                "feature_dim": int(F.shape[1]),
                "saved_npz": npz_path.replace("\\", "/"),
                "pca_png": os.path.join(OUT_DIR, f"{prefix}_pca.png").replace("\\", "/"),
                "tsne_png": os.path.join(OUT_DIR, f"{prefix}_tsne.png").replace("\\", "/"),
            }
            meta_all.append(meta)

            print(f"[OK] {cfg['display']:<28} | {tag:<6} | n={len(Y)} | dim={F.shape[1]}")

    # 保存一份总 meta，方便对照
    meta_path = os.path.join(OUT_DIR, "feature_viz_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_all, f, ensure_ascii=False, indent=2)

    print("\n[DONE] Feature visualization finished.")
    print(" - meta:", meta_path.replace("\\", "/"))
    print(" - figures dir:", OUT_DIR.replace("\\", "/"))
    print("\n你最建议优先看的图：")
    print("  1) cnn_transformer_clean_tsne.png  vs  cnn_transformer_noiseaug_clean_tsne.png")
    print("  2) cnn_transformer_snr_0_tsne.png  vs  cnn_transformer_noiseaug_snr_0_tsne.png")


if __name__ == "__main__":
    main()