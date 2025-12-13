import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# 读取结果
# =========================
accs = np.load("results/rnn_test_accs.npy")
runs = np.arange(1, len(accs) + 1)

mean_acc = accs.mean()
std_acc = accs.std()

os.makedirs("figures", exist_ok=True)

# =========================
# 图 1：每次训练的 Test Accuracy
# =========================
plt.figure(figsize=(8, 5))
plt.plot(runs, accs, marker='o', linewidth=2)
plt.axhline(mean_acc, linestyle='--', label=f"Mean = {mean_acc:.3f}")
plt.xlabel("Run Index")
plt.ylabel("Test Accuracy")
plt.title("RNN Test Accuracy over Multiple Runs")
plt.legend()
plt.grid(True)

plt.savefig("figures/rnn_test_accuracy_runs.png", dpi=300)
plt.show()

# =========================
# 图 2：Mean ± Std 柱状图
# =========================
plt.figure(figsize=(6, 5))
plt.bar([0], [mean_acc], yerr=[std_acc], capsize=10)
plt.xticks([0], ["LSTM"])
plt.ylabel("Test Accuracy")
plt.title("RNN Performance Stability (Mean ± Std)")
plt.grid(axis="y")

plt.savefig("figures/rnn_test_accuracy_mean_std.png", dpi=300)
plt.show()
