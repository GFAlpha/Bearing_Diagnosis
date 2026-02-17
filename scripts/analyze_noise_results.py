import os
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOISE_ROOT = os.path.join(PROJECT_ROOT, "noise_results")
OUT_DIR = os.path.join(PROJECT_ROOT, "analysis_results", "noise")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ORDER = [
    "cnn",
    "rnn_lstm",
    "cnn_bilstm",
    "cnn_bilstm_att",
    "transformer",
    "cnn_transformer",
    "cnn_transformer_noiseaug",
]

# ⭐ 顺序改为 9 → 6 → 3 → 0
SNR_ORDER = ["clean", "snr_9", "snr_6", "snr_3", "snr_0"]


def load_one(model_name: str, tag: str):
    d = os.path.join(NOISE_ROOT, model_name, tag)
    acc_path = os.path.join(d, "acc.npy")
    meta_path = os.path.join(d, "meta.json")
    infer_path = os.path.join(d, "infer_times.npy")

    if not os.path.exists(acc_path):
        return None

    acc = float(np.load(acc_path).reshape(-1)[0])
    infer_times = np.load(infer_path) if os.path.exists(infer_path) else None
    infer_ms = float(np.mean(infer_times) * 1000.0) if infer_times is not None else None

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return {
        "model": model_name,
        "snr_tag": tag,
        "acc": acc,
        "infer_ms_per_step": infer_ms,
        "weight_path": meta.get("weight_path", ""),
    }


def main():
    rows = []
    for m in MODEL_ORDER:
        for tag in SNR_ORDER:
            r = load_one(m, tag)
            if r is not None:
                rows.append(r)

    if len(rows) == 0:
        raise RuntimeError(f"没有读到任何 noise_results 数据，请检查路径：{NOISE_ROOT}")

    df = pd.DataFrame(rows)

    # ============================
    # 构建宽表
    # ============================
    wide = df.pivot_table(index="model", columns="snr_tag", values="acc", aggfunc="mean")

    # ⭐ 强制列顺序
    ordered_cols = ["clean", "snr_9", "snr_6", "snr_3", "snr_0"]
    wide = wide.reindex(columns=ordered_cols)

    # 计算 Drop%
    if "clean" in wide.columns:
        for tag in ["snr_9", "snr_6", "snr_3", "snr_0"]:
            if tag in wide.columns:
                wide[f"drop_{tag}"] = (wide["clean"] - wide[tag]) / wide["clean"] * 100.0

    wide = wide.reindex(MODEL_ORDER)

    # ============================
    # 保存 CSV / Excel
    # ============================
    out_csv = os.path.join(OUT_DIR, "noise_summary_table.csv")
    out_long_csv = os.path.join(OUT_DIR, "noise_long_table.csv")
    out_xlsx = os.path.join(OUT_DIR, "noise_summary_table.xlsx")

    wide.to_csv(out_csv, encoding="utf-8-sig")
    df.to_csv(out_long_csv, index=False, encoding="utf-8-sig")

    try:
        wide.to_excel(out_xlsx)
    except Exception:
        pass

    # ============================
    # 保存 TXT
    # ============================
    txt_path = os.path.join(OUT_DIR, "noise_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Noise Robustness Summary (AWGN on Test Set)\n")
        f.write("=" * 120 + "\n")

        header_cols = list(wide.columns)

        f.write(f"{'Model':<30}")
        for col in header_cols:
            f.write(f"{col:>12}")
        f.write("\n")
        f.write("-" * 120 + "\n")

        for model_name, row in wide.iterrows():
            f.write(f"{model_name:<30}")
            for col in header_cols:
                val = row[col]
                if pd.isna(val):
                    f.write(f"{'NaN':>12}")
                else:
                    f.write(f"{val:>12.4f}")
            f.write("\n")

    # ============================
    # ⭐ 精简终端输出
    # ============================
    print("[OK] Noise robustness analysis finished.")
    print("Results saved to:")
    print(" -", out_csv)
    print(" -", out_xlsx)
    print(" -", txt_path)


if __name__ == "__main__":
    main()