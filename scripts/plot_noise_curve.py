import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV = os.path.join(PROJECT_ROOT, "analysis_results", "noise", "noise_long_table.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "analysis_results", "noise")
os.makedirs(OUT_DIR, exist_ok=True)

# 画图时 SNR 从高到低更直观：clean, 9, 6, 3, 0
ORDER = ["clean", "snr_9", "snr_6", "snr_3", "snr_0"]
X_LABEL = ["Clean", "9", "6", "3", "0"]

def main():
    df = pd.read_csv(IN_CSV)

    # 保证顺序
    df["snr_tag"] = pd.Categorical(df["snr_tag"], categories=ORDER, ordered=True)
    df = df.sort_values(["model", "snr_tag"])

    plt.figure()
    for model_name, g in df.groupby("model"):
        y = g["acc"].tolist()
        plt.plot(X_LABEL, y, marker="o", label=model_name)

    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs SNR (AWGN on Test Set)")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    out_path = os.path.join(OUT_DIR, "acc_vs_snr.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("[OK] Saved:", out_path)

if __name__ == "__main__":
    main()