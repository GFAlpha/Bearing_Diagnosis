import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm

# ======================
# Config
# ======================
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

WINDOW_SIZE = 1024
STEP_SIZE = 1024

LABEL_MAP = {
    "Normal": 0,
    "Ball": 1,
    "Inner Race": 2,
    "Outer Race": 3,
}

os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# Utils
# ======================
def find_signal_from_mat(mat_dict, min_len=2048):
    """
    从 mat 文件中安全地找到一条一维振动信号
    """
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray):
            v = np.squeeze(v)
            if v.ndim == 1 and v.size >= min_len:
                return v.astype(np.float32)
    return None


def slice_signal(signal, window_size, step_size):
    segments = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        segments.append(signal[start:start + window_size])
    return segments


# ======================
# Main
# ======================
X_all, y_all = [], []

# ---------- Normal ----------
normal_dir = os.path.join(RAW_DIR, "CWRU_Normal")
print("[INFO] Loading Normal data...")

for fname in tqdm(os.listdir(normal_dir)):
    if not fname.endswith(".mat"):
        continue
    mat = sio.loadmat(os.path.join(normal_dir, fname))
    signal = find_signal_from_mat(mat)
    if signal is None:
        print(f"[WARN] No valid signal in {fname}")
        continue

    segments = slice_signal(signal, WINDOW_SIZE, STEP_SIZE)
    X_all.extend(segments)
    y_all.extend([LABEL_MAP["Normal"]] * len(segments))


# ---------- Fault ----------
fault_root = os.path.join(RAW_DIR, "CWRU_12K_DE")

for fault_type in ["Ball", "Inner Race", "Outer Race"]:
    print(f"[INFO] Loading {fault_type} data...")
    fault_label = LABEL_MAP[fault_type]
    fault_dir = os.path.join(fault_root, fault_type)

    for root, _, files in os.walk(fault_dir):
        for fname in files:
            if not fname.endswith(".mat"):
                continue
            mat = sio.loadmat(os.path.join(root, fname))
            signal = find_signal_from_mat(mat)
            if signal is None:
                continue

            segments = slice_signal(signal, WINDOW_SIZE, STEP_SIZE)
            X_all.extend(segments)
            y_all.extend([fault_label] * len(segments))


# ======================
# Save
# ======================
X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.int64)

np.save(os.path.join(OUT_DIR, "X.npy"), X_all)
np.save(os.path.join(OUT_DIR, "y.npy"), y_all)

# ======================
# Summary
# ======================
print("\n========== Dataset Summary ==========")
print(f"Total samples: {len(y_all)}")
for name, idx in LABEL_MAP.items():
    print(f"{name:<12}: {(y_all == idx).sum()}")
print("=====================================")
