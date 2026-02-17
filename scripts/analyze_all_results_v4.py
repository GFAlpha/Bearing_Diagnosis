import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# ============ ① 配置区（路径 & 模型顺序） =================
# =========================================================
RESULTS_DIR = "results"
OUTPUT_DIR = "analysis_results"

MODEL_ORDER = [
    ("cnn", "CNN"),
    ("rnn_lstm", "LSTM"),
    ("cnn_bilstm", "CNN+BiLSTM"),
    ("cnn_bilstm_att", "CNN+BiLSTM+Att"),
    ("transformer", "Transformer"),
    ("cnn_transformer", "CNN+Transformer"),
    ("cnn_transformer_noiseaug", "CNN+Transformer(NoiseAug)"),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# ============ ② 工具函数：兼容推理时间文件名 =============
# =========================================================
def load_infer_times(folder_path):
    for name in ["infer_times.npy", "inference_times.npy"]:
        p = os.path.join(folder_path, name)
        if os.path.exists(p):
            return np.load(p)
    raise FileNotFoundError("Inference time file not found.")

# =========================================================
# ============ ③ 读取数据 ================================
# =========================================================
records = []

for folder, display_name in MODEL_ORDER:
    path = os.path.join(RESULTS_DIR, folder)
    if not os.path.isdir(path):
        print(f"[WARN] {folder} not found, skipped.")
        continue

    try:
        accs = np.load(os.path.join(path, "test_accs.npy"))
        train_times = np.load(os.path.join(path, "train_times.npy"))
        infer_times = load_infer_times(path)
    except FileNotFoundError:
        print(f"[WARN] Missing files in {folder}, skipped.")
        continue

    records.append({
        "Model": display_name,
        "Acc Mean": accs.mean(),
        "Acc Std": accs.std(),
        "Train Time (s)": train_times.mean(),
        "Infer Time (ms)": infer_times.mean() * 1000
    })

df = pd.DataFrame(records)

# =========================================================
# ============ ④ 保存 CSV ================================
# =========================================================
csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# =========================================================
# ============ ⑤ 保存「文本表格」（给人看的） ============
# =========================================================
txt_path = os.path.join(OUTPUT_DIR, "summary.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("Model Performance Summary\n")
    f.write("=" * 70 + "\n")
    f.write(f"{'Model':<28}{'Acc Mean':>10}{'Acc Std':>10}"
            f"{'Train(s)':>12}{'Infer(ms)':>12}\n")
    f.write("-" * 70 + "\n")

    for _, row in df.iterrows():
        f.write(f"{row['Model']:<28}"
                f"{row['Acc Mean']:>10.4f}"
                f"{row['Acc Std']:>10.4f}"
                f"{row['Train Time (s)']:>12.2f}"
                f"{row['Infer Time (ms)']:>12.3f}\n")

print(f"[INFO] Summary table saved to {txt_path}")

# =========================================================
# ============ ⑥ 图 1：准确率 ============================
# =========================================================
plt.figure(figsize=(10, 5))
plt.bar(df["Model"], df["Acc Mean"], yerr=df["Acc Std"], capsize=5)
plt.ylabel("Test Accuracy")
plt.title("Accuracy Comparison of Models")
plt.ylim(0.0, 1.01)  # ✅ 兼容更低的结果（避免以后增强实验/其他实验低于0.9时图被截断）
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_comparison.png"), dpi=300)
plt.close()

# =========================================================
# ============ 图 2：训练时间（log scale） ================
# =========================================================
plt.figure(figsize=(10, 5))
plt.bar(df["Model"], df["Train Time (s)"])
plt.yscale("log")
plt.ylabel("Training Time (s, log scale)")
plt.title("Training Time Comparison (Log Scale)")
plt.grid(axis="y", linestyle="--", alpha=0.6, which="both")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "train_time_comparison_log.png"), dpi=300)
plt.close()

# =========================================================
# ============ ⑧ 图 3：推理时间（log scale） =============
# =========================================================
plt.figure(figsize=(10, 5))
plt.bar(df["Model"], df["Infer Time (ms)"])
plt.yscale("log")
plt.ylabel("Inference Time (ms, log scale)")
plt.title("Inference Time Comparison (Log Scale)")
plt.grid(axis="y", linestyle="--", alpha=0.6, which="both")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "infer_time_comparison_log.png"), dpi=300)
plt.close()

print(f"[DONE] All analysis results saved to {OUTPUT_DIR}")