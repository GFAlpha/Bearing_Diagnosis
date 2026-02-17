import os
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOISE_ROOT = os.path.join(PROJECT_ROOT, "noise_results")
OUT_DIR = os.path.join(PROJECT_ROOT, "analysis_results", "noise")
os.makedirs(OUT_DIR, exist_ok=True)

# 你这次跑出来的模型目录名（以 noise_results/ 里的为准）
MODEL_ORDER = ["cnn", "rnn_lstm", "cnn_bilstm", "cnn_bilstm_att", "transformer", "cnn_transformer"]
SNR_ORDER = ["clean", "snr_9", "snr_6", "snr_3", "snr_0"]  # 论文里一般从干净到更脏：clean→9→6→3→0

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

def tag_to_snr_db(tag: str):
    if tag == "clean":
        return None
    return int(tag.split("_")[1])

def main():
    rows = []
    for m in MODEL_ORDER:
        for tag in SNR_ORDER:
            r = load_one(m, tag)
            if r is not None:
                r["snr_db"] = tag_to_snr_db(tag)
                rows.append(r)

    df = pd.DataFrame(rows)

    # 生成“宽表”：每个模型一行，每个SNR一列
    wide = df.pivot_table(index="model", columns="snr_tag", values="acc", aggfunc="mean")

    # 计算 Drop%（相对 clean）
    if "clean" in wide.columns:
        for tag in [c for c in wide.columns if c != "clean"]:
            wide[f"drop_{tag}"] = (wide["clean"] - wide[tag]) / wide["clean"] * 100.0

    # 按 MODEL_ORDER 排序
    wide = wide.reindex(MODEL_ORDER)

    # 保存
    out_csv = os.path.join(OUT_DIR, "noise_summary_table.csv")
    out_xlsx = os.path.join(OUT_DIR, "noise_summary_table.xlsx")
    wide.to_csv(out_csv, encoding="utf-8-sig")
    wide.to_excel(out_xlsx)

    # 同时保存“长表”（画图更方便）
    out_long_csv = os.path.join(OUT_DIR, "noise_long_table.csv")
    df.to_csv(out_long_csv, index=False, encoding="utf-8-sig")

    print("[OK] Saved:")
    print(" -", out_csv)
    print(" -", out_xlsx)
    print(" -", out_long_csv)
    print("\nPreview (wide table):")
    print(wide)

if __name__ == "__main__":
    main()