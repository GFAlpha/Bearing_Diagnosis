import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ============================
# 配置区（后续只需要改这里）
# ============================
RESULTS_ROOT = "results"
SAVE_DIR = "analysis_results/confusion_matrices"
CLASS_NAMES = ["Normal", "Ball", "Inner", "Outer"]

MODEL_DIRS = {
    "CNN": "cnn",
    "RNN": "rnn_lstm",
    "CNN+BiLSTM": "cnn_bilstm",
    "CNN+BiLSTM+Att": "cnn_bilstm_att",
    "Transformer": "transformer",
    "CNN+Transformer": "cnn_transformer",
}

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================
# 绘图函数
# ============================

def plot_cm(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".4f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================
# 主流程
# ============================

for model_name, folder in MODEL_DIRS.items():
    model_path = os.path.join(RESULTS_ROOT, folder)
    y_true_path = os.path.join(model_path, "y_true.npy")
    y_pred_path = os.path.join(model_path, "y_pred.npy")

    if not (os.path.exists(y_true_path) and os.path.exists(y_pred_path)):
        print(f"[WARN] {model_name} 缺少 y_true / y_pred，跳过")
        continue

    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)

    save_path = os.path.join(SAVE_DIR, f"cm_{folder}.png")
    plot_cm(y_true, y_pred, f"Confusion Matrix - {model_name}", save_path)

    print(f"[OK] {model_name} 混淆矩阵已保存")

print("\n[DONE] 所有混淆矩阵生成完成")
