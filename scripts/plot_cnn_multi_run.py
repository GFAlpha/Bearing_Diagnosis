import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# 1. 路径设置
# =========================
RESULT_DIR = "results/cnn_multi_run"
ACC_PATH = os.path.join(RESULT_DIR, "test_accs.npy")

# =========================
# 2. 加载数据
# =========================
test_accs = np.load(ACC_PATH)

runs = np.arange(1, len(test_accs) + 1)
mean_acc = test_accs.mean()
std_acc = test_accs.std()

# =========================
# 3. 画图
# =========================
plt.figure(figsize=(8, 5))

# 每次训练的结果（点）
plt.scatter(runs, test_accs)
plt.plot(runs, test_accs, alpha=0.6)

# 均值 & 方差
plt.axhline(mean_acc, linestyle="--", label=f"Mean = {mean_acc:.4f}")
plt.axhline(mean_acc + std_acc, linestyle=":", label=f"+1 Std = {mean_acc + std_acc:.4f}")
plt.axhline(mean_acc - std_acc, linestyle=":", label=f"-1 Std = {mean_acc - std_acc:.4f}")

plt.xlabel("Run Index")
plt.ylabel("Test Accuracy")
plt.title("Stability of CNN over Multiple Random Runs")
plt.legend()
plt.grid(True)

# =========================
# 4. 保存 & 显示
# =========================
save_path = os.path.join(RESULT_DIR, "cnn_multi_run_stability.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Figure saved to: {save_path}")
