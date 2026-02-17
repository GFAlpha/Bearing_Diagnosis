import os
import sys
import subprocess
import argparse
from datetime import datetime


# 取 scripts 的上一级作为项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_step(py_exe: str, script_path: str, title: str):
    """运行一个脚本步骤，失败则直接退出（防止后续产物基于错误结果继续跑）"""
    print("\n" + "=" * 80)
    print(f"[STEP] {title}")
    print(f"[RUN ] {py_exe} {script_path}")
    print("=" * 80)

    start = datetime.now()
    r = subprocess.run([py_exe, script_path], cwd=PROJECT_ROOT)
    end = datetime.now()

    if r.returncode != 0:
        print(f"\n[ERROR] 步骤失败：{title}")
        print("        请查看上方报错信息（通常是缺文件/路径不对/依赖缺失）。")
        sys.exit(r.returncode)

    print(f"[OK]  步骤完成：{title}  (耗时 {(end - start)})")


def _fmt_path(p: str) -> str:
    """为了复制方便，统一输出成相对路径（如果能相对）"""
    try:
        rel = os.path.relpath(p, PROJECT_ROOT)
        return rel.replace("\\", "/")
    except Exception:
        return p.replace("\\", "/")


def _exists_mark(p: str) -> str:
    return "✅" if os.path.exists(p) else "⚠️"


def print_key_artifacts():
    """在最后输出对人最有用的关键产物清单（可直接用于写论文/查数据）"""
    analysis_dir = os.path.join(PROJECT_ROOT, "analysis_results")
    noise_dir = os.path.join(analysis_dir, "noise")
    cm_dir = os.path.join(analysis_dir, "confusion_matrices")

    # 关键文件（论文常用）
    key_files = [
        ("干净数据总表（最常引用）", os.path.join(analysis_dir, "summary.txt")),
        ("干净数据 CSV（可导入Excel）", os.path.join(analysis_dir, "summary.csv")),
        ("干净数据：准确率对比图", os.path.join(analysis_dir, "accuracy_comparison.png")),
        ("干净数据：训练时间对比图(log)", os.path.join(analysis_dir, "train_time_comparison_log.png")),
        ("干净数据：推理时间对比图(log)", os.path.join(analysis_dir, "infer_time_comparison_log.png")),

        ("噪声鲁棒性总表（最常引用）", os.path.join(noise_dir, "noise_summary.txt")),
        ("噪声鲁棒性 CSV（可导入Excel）", os.path.join(noise_dir, "noise_summary_table.csv")),
        ("噪声鲁棒性曲线图", os.path.join(noise_dir, "acc_vs_snr.png")),

        ("混淆矩阵目录（所有模型）", cm_dir),
    ]

    # 混淆矩阵常看的几张（如果你想更细，还可以继续加）
    cm_files = [
        ("混淆矩阵：CNN", os.path.join(cm_dir, "cm_cnn.png")),
        ("混淆矩阵：RNN/LSTM", os.path.join(cm_dir, "cm_rnn_lstm.png")),
        ("混淆矩阵：CNN+BiLSTM", os.path.join(cm_dir, "cm_cnn_bilstm.png")),
        ("混淆矩阵：CNN+BiLSTM+Att", os.path.join(cm_dir, "cm_cnn_bilstm_att.png")),
        ("混淆矩阵：Transformer", os.path.join(cm_dir, "cm_transformer.png")),
        ("混淆矩阵：CNN+Transformer", os.path.join(cm_dir, "cm_cnn_transformer.png")),
        ("混淆矩阵：CNN+Transformer(NoiseAug)", os.path.join(cm_dir, "cm_cnn_transformer_noiseaug.png")),
    ]

    print("\n" + "=" * 80)
    print("【关键产物清单】（优先看这些）")
    print("=" * 80)

    print("\n1) 干净数据对比（核心结论）")
    for name, p in key_files[:5]:
        print(f"  {_exists_mark(p)} {name}: {_fmt_path(p)}")

    print("\n2) 噪声鲁棒性评估（核心结论）")
    for name, p in key_files[5:8]:
        print(f"  {_exists_mark(p)} {name}: {_fmt_path(p)}")

    print("\n3) 混淆矩阵（做定性分析/举例用）")
    print(f"  {_exists_mark(cm_dir)} 混淆矩阵目录: {_fmt_path(cm_dir)}")
    for name, p in cm_files:
        print(f"  {_exists_mark(p)} {name}: {_fmt_path(p)}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="一键运行：噪声评估 + 汇总分析 + 画图 + 混淆矩阵（建议 train 完后执行）"
    )

    parser.add_argument("--skip_make_noisy", action="store_true", help="跳过生成带噪测试集 make_noisy_testset.py")
    parser.add_argument("--skip_noise_eval", action="store_true", help="跳过噪声评估 noise_test_all_models.py")
    parser.add_argument("--skip_noise_analyze", action="store_true", help="跳过噪声汇总分析 analyze_noise_results.py")
    parser.add_argument("--skip_noise_plot", action="store_true", help="跳过噪声曲线绘制 plot_noise_curve.py")
    parser.add_argument("--skip_clean_analyze", action="store_true", help="跳过干净数据汇总分析 analyze_all_results_v4.py")
    parser.add_argument("--skip_confusion", action="store_true", help="跳过混淆矩阵绘制 plot_confusion_matrices_all.py")

    args = parser.parse_args()

    py_exe = sys.executable  # 使用当前虚拟环境的 python（env）
    print("[INFO] Using python:", py_exe)
    print("[INFO] Project root:", PROJECT_ROOT)

    # 1) 生成带噪测试集
    if not args.skip_make_noisy:
        run_step(py_exe, os.path.join("scripts", "make_noisy_testset.py"),
                 "生成带噪测试集（data/noise_test/）")

    # 2) 噪声评估
    if not args.skip_noise_eval:
        run_step(py_exe, os.path.join("scripts", "noise_test_all_models.py"),
                 "噪声鲁棒性评估（输出到 noise_results/）")

    # 3) 噪声结果汇总
    if not args.skip_noise_analyze:
        run_step(py_exe, os.path.join("scripts", "analyze_noise_results.py"),
                 "噪声结果汇总（analysis_results/noise/）")

    # 4) 噪声曲线图
    if not args.skip_noise_plot:
        run_step(py_exe, os.path.join("scripts", "plot_noise_curve.py"),
                 "绘制噪声曲线图（analysis_results/noise/acc_vs_snr.png）")

    # 5) 干净数据汇总 + 对比图
    if not args.skip_clean_analyze:
        run_step(py_exe, os.path.join("scripts", "analyze_all_results_v4.py"),
                 "干净数据汇总分析（analysis_results/summary.csv + 图）")

    # 6) 混淆矩阵
    if not args.skip_confusion:
        run_step(py_exe, os.path.join("scripts", "plot_confusion_matrices_all.py"),
                 "绘制混淆矩阵（analysis_results/confusion_matrices/）")

    print("\n" + "=" * 80)
    print("[DONE] analysis_all.py 全流程执行完成 ✅")
    print("=" * 80)

    # ⭐ 新增：关键产物清单
    print_key_artifacts()


if __name__ == "__main__":
    main()