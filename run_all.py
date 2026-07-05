#!/usr/bin/env python3
"""
run_all.py
混沌教学论文 — 一键生成全部仿真图表

执行流程：
  1. 双摆仿真   (double_pendulum.py)  — 轨迹、相空间、能量、敏感性
  2. Logistic映射 (logistic_map.py)    — 分岔图、时间序列、Cobweb、返回映射
  3. Lyapunov指数 (lyapunov.py)        — 谱线、双摆λ估计、组合面板

输出目录结构:
  figures/
    dp_*.png         双摆图
    lm_*.png         Logistic映射图
    lyap_*.png       Lyapunov指数图
    chaos_overview.png  四合一摘要面板

运行时间: ~3-5 分钟 (主要在双摆长积分和Lyapunov谱计算)
"""

import os
import sys
import time

# 确保在 code/ 目录下运行
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# 将当前目录加入路径
sys.path.insert(0, SCRIPT_DIR)

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "figures")

# ============================================================
# 彩色终端输出
# ============================================================
def print_header(msg):
    print(f"\n{'=' * 65}")
    print(f"  {msg}")
    print(f"{'=' * 65}")

def print_step(msg):
    print(f"  >> {msg}")

# ============================================================
# 第一阶段：双摆仿真
# ============================================================
def run_double_pendulum():
    print_header("Phase 1: Double Pendulum Simulation")
    t0 = time.time()

    from double_pendulum import run_double_pendulum

    run_double_pendulum(OUTPUT_DIR)

    elapsed = time.time() - t0
    print_step(f"Time: {elapsed:.1f}s")
    return elapsed


# ============================================================
# 第二阶段：Logistic映射
# ============================================================
def run_logistic_map():
    print_header("Phase 2: Logistic Map Simulations")
    t0 = time.time()

    from logistic_map import run_logistic_map

    run_logistic_map(OUTPUT_DIR)

    elapsed = time.time() - t0
    print_step(f"Time: {elapsed:.1f}s")
    return elapsed


# ============================================================
# 第三阶段：Lyapunov指数
# ============================================================
def run_lyapunov():
    print_header("Phase 3: Lyapunov Exponent Calculations")
    t0 = time.time()

    from lyapunov import run_lyapunov

    lambda_dp = run_lyapunov(OUTPUT_DIR)

    elapsed = time.time() - t0
    print_step(f"Time: {elapsed:.1f}s")
    return elapsed, lambda_dp


# ============================================================
# 第四阶段：文件汇总
# ============================================================
def summarize(output_dir):
    """列出所有生成的图文件。"""
    print_header("Summary: Generated Figures")

    png_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    for i, f in enumerate(png_files, 1):
        fpath = os.path.join(output_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  [{i:2d}] {f:45s} ({size_kb:.1f} KB)")

    print(f"\n  Total: {len(png_files)} figures in {output_dir}")
    return png_files


# ============================================================
# 主入口
# ============================================================
def main():
    print("\n")
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Teaching Chaos Theory with Python                  ║")
    print("║  From the Double Pendulum to the Logistic Map       ║")
    print("║  Target: European Journal of Physics                ║")
    print("╚══════════════════════════════════════════════════════╝")

    total_t0 = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 阶段1: 双摆 ---
    t1 = run_double_pendulum()

    # --- 阶段2: Logistic映射 ---
    t2 = run_logistic_map()

    # --- 阶段3: Lyapunov指数 ---
    t3, lambda_dp = run_lyapunov()

    # --- 汇总 ---
    total_time = time.time() - total_t0
    png_files = summarize(OUTPUT_DIR)

    print_header("Execution Complete")
    print(f"  Double Pendulum:    {t1:.1f}s")
    print(f"  Logistic Map:       {t2:.1f}s")
    print(f"  Lyapunov Exponents: {t3:.1f}s")
    print(f"  ──────────────────────────")
    print(f"  Total:              {total_time:.1f}s")
    print(f"  Figures generated:  {len(png_files)}")
    print(f"  Output directory:   {OUTPUT_DIR}")
    if lambda_dp:
        print(f"  λ_max (DP):         {lambda_dp:.4f} /s")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
