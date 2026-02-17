import os
import numpy as np

DATA_SPLIT_DIR = "data/splits"
OUT_DIR = "data/noise_test"
os.makedirs(OUT_DIR, exist_ok=True)

SNR_LIST = [0, 3, 6, 9]  # dB

def add_awgn_per_sample(X: np.ndarray, snr_db: float, seed: int = 1234) -> np.ndarray:
    """
    对每个样本独立加 AWGN，使每个样本都近似满足目标 SNR（更公平、更常用）。
    X: [N, L]
    """
    rng = np.random.default_rng(seed)
    X = X.astype(np.float32, copy=False)

    # 每个样本的功率：mean(x^2)
    p_signal = np.mean(X * X, axis=1, keepdims=True)  # [N, 1]
    snr_linear = 10 ** (snr_db / 10.0)
    p_noise = p_signal / snr_linear
    sigma = np.sqrt(p_noise).astype(np.float32)       # [N, 1]

    noise = rng.normal(0.0, 1.0, size=X.shape).astype(np.float32) * sigma
    return X + noise

def main():
    X_test = np.load(os.path.join(DATA_SPLIT_DIR, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(DATA_SPLIT_DIR, "y_test.npy"))

    # clean 也存一份，后面统一读取 data/noise_test 下的文件
    np.save(os.path.join(OUT_DIR, "X_test_clean.npy"), X_test)
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)

    for snr in SNR_LIST:
        X_noisy = add_awgn_per_sample(X_test, snr_db=snr, seed=2026 + snr)
        np.save(os.path.join(OUT_DIR, f"X_test_snr_{snr}.npy"), X_noisy)
        print(f"[OK] Saved: X_test_snr_{snr}.npy")

    print("[DONE] noisy testsets prepared.")

if __name__ == "__main__":
    main()