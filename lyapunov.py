"""
lyapunov.py
Lyapunov指数计算 — Teaching Chaos Theory with Python

Lyapunov指数是量化混沌的数学工具，衡量相空间中相邻轨迹的
指数发散速率。λ > 0 意味着混沌。

本模块实现了两类计算：
  1. Logistic映射的Lyapunov指数谱（解析 vs 数值）
     - 用导数法: λ = lim(N→∞) (1/N) Σ ln|f'(x_n)|
  2. 双摆的最大Lyapunov指数（数值估计）
     - 用两个接近轨迹的分离速率: λ ≈ (1/t) ln(|δ(t)|/|δ(0)|)

教学目的：让学生直观理解 Lyapunov 指数作为混沌判据的含义。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

# ============================================================
# Logistic映射 Lyapunov 指数
# ============================================================

def lyapunov_logistic(r, x0=0.5, n=1000, n_transient=200):
    """
    计算Logistic映射在给定 r 下的最大Lyapunov指数。
    
    公式: λ = (1/N) Σ_{i=1}^{N} ln|r(1 - 2x_i)|
    其中 f'(x) = r(1 - 2x) 是导数。
    
    返回:
        lambda_est: Lyapunov指数估计值
    """
    x = x0
    # 舍弃暂态
    for _ in range(n_transient):
        x = logistic_map_step(r, x)

    lyap_sum = 0.0
    for _ in range(n):
        deriv = r * (1.0 - 2.0 * x)
        lyap_sum += np.log(abs(deriv) + 1e-30)  # 防零
        x = logistic_map_step(r, x)

    return lyap_sum / n


def logistic_map_step(r, x):
    """Logistic映射单步：x_{n+1} = r x (1-x)"""
    return r * x * (1.0 - x)


def lyapunov_spectrum(r_min=2.5, r_max=4.0, n_points=1000, x0=0.5,
                      n=800, n_transient=200):
    """
    计算 Lyapunov 指数谱 (λ 作为 r 的函数)。
    
    返回:
        r_vals, lyap_vals: 用于绘制谱线
    """
    r_vals = np.linspace(r_min, r_max, n_points)
    lyap_vals = np.array([lyapunov_logistic(r, x0, n, n_transient)
                          for r in r_vals])
    return r_vals, lyap_vals


# ============================================================
# 双摆最大 Lyapunov 指数（数值估计）
# ============================================================

def double_pendulum_lyapunov(theta1_0, theta2_0, dt=0.01, t_max=200,
                              delta0=1e-8, renormalize_step=50):
    """
    用轨道分离法估计双摆的最大Lyapunov指数。
    
    方法: Benettin 算法
      1. 同时积分参考轨道和扰动轨道
      2. 每隔 renormalize_step 步测量分离 d = |δ|
      3. 记录 ln(d/d₀)，然后将δ归一化到原方向
      4. λ ≈ (1/t_total) Σ ln(d_k/d₀)
    
    返回:
        t_vals, log_sep: 时间与 log₁₀(分离度)（用于绘图）
        lambda_est: 最大 Lyapunov 指数估计
    """
    from double_pendulum import simulate, derivatives, rk4_step

    n_steps = int(t_max / dt)

    # 参考轨道
    state_ref = np.array([theta1_0, 0.0, theta2_0, 0.0])

    # 扰动轨道 (只在 θ1 方向加微小扰动)
    state_delta = state_ref.copy()
    state_delta[0] += delta0

    # 归一化
    delta_vec = state_delta - state_ref
    d_norm = np.linalg.norm(delta_vec)
    if d_norm > 0:
        delta_vec /= d_norm

    log_sep_vals = []
    t_vals = []
    lyap_sum = 0.0
    n_renorm = 0

    # 暂态跳过
    transient_steps = 1000

    for i in range(n_steps):
        state_ref = rk4_step(derivatives, state_ref, dt, i * dt)
        state_delta = rk4_step(derivatives, state_delta, dt, i * dt)

        if i < transient_steps:
            continue

        # 实际分离
        delta_current = state_delta - state_ref

        # 每隔 renormalize_step 步记录和归一化
        if i % renormalize_step == 0:
            d = np.linalg.norm(delta_current)
            if d < 1e-30:
                d = 1e-30
            log_sep_vals.append(np.log10(d / delta0))
            t_vals.append(i * dt)

            lyap_sum += np.log(d / delta0)
            n_renorm += 1

            # Gram-Schmidt: 归一化扰动向量到δ₀
            if d > 0:
                delta_current = (delta_current / d) * delta0
                state_delta = state_ref + delta_current

    lambda_est = lyap_sum / (t_vals[-1] if t_vals else 1.0) if n_renorm > 0 else 0.0

    return np.array(t_vals), np.array(log_sep_vals), lambda_est


# ============================================================
# 可视化
# ============================================================

def plot_lyapunov_logistic(save_path, r_min=2.5, r_max=4.0, n_points=1000):
    """
    Logistic映射Lyapunov指数谱：λ vs r。
    与分岔图并列为论文核心图。
    """
    print("    Computing Lyapunov spectrum... (may take a moment)")
    r_vals, lyap_vals = lyapunov_spectrum(r_min, r_max, n_points)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                             gridspec_kw={'height_ratios': [1, 2.5]})

    # 上: 分岔图（缩略版）
    ax = axes[0]
    from logistic_map import bifurcation_data
    r_bif = np.linspace(r_min, r_max, 2000)
    r_data, x_data = bifurcation_data(r_bif, n_transient=300, n_sample=100)
    ax.scatter(r_data, x_data, s=0.05, c='black', alpha=0.5, rasterized=True)
    ax.set_xlim(r_min, r_max)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel('x', fontsize=11)
    ax.tick_params(labelbottom=False)
    ax.set_title('Bifurcation Diagram & Lyapunov Exponent', fontsize=15)

    # 下: Lyapunov指数谱
    ax = axes[1]
    ax.plot(r_vals, lyap_vals, 'b-', lw=0.8)
    ax.axhline(y=0, color='red', ls='-', lw=1.2, alpha=0.7)
    ax.fill_between(r_vals, lyap_vals, 0,
                    where=lyap_vals > 0, color='coral', alpha=0.3, label='Chaos (λ>0)')
    ax.fill_between(r_vals, lyap_vals, 0,
                    where=lyap_vals <= 0, color='lightblue', alpha=0.3, label='Order (λ≤0)')

    # 标注混沌阈值
    ax.axvline(x=3.5699456, color='green', ls='--', lw=1, alpha=0.6)
    ax.text(3.5699456, ax.get_ylim()[1] * 0.9, 'r_c ≈ 3.5699',
            fontsize=9, color='green', ha='left')

    ax.set_xlim(r_min, r_max)
    ax.set_ylim(min(lyap_vals) - 0.1, max(lyap_vals) + 0.1)
    ax.set_xlabel('Control parameter r', fontsize=14)
    ax.set_ylabel('Lyapunov exponent λ', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Lyapunov spectrum saved: {save_path}")


def plot_lyapunov_double_pendulum(save_path):
    """
    双摆最大Lyapunov指数估计。
    绘制 log₁₀|δ(t)| 随时间变化，及其线性拟合。
    """
    print("    Estimating double pendulum Lyapunov exponent...")
    t_vals, log_sep, lambda_est = double_pendulum_lyapunov(
        theta1_0=2.0, theta2_0=2.5,
        dt=0.01, t_max=300, delta0=1e-8)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(t_vals, log_sep, 'b-', lw=0.8, alpha=0.7, label='log₁₀|δ(t)/δ₀|')

    # 线性拟合早期部分 (前30%) — 指数发散阶段
    n_fit = max(10, int(0.3 * len(t_vals)))
    if n_fit > 2:
        coeffs = np.polyfit(t_vals[:n_fit], log_sep[:n_fit], 1)
        fit_line = np.polyval(coeffs, t_vals[:n_fit])
        ax.plot(t_vals[:n_fit], fit_line, 'r--', lw=2,
                label=f'Fit (early): λ ≈ {coeffs[0]/np.log10(np.e):.3f} /s')

    ax.axhline(y=0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Time [s]', fontsize=13)
    ax.set_ylabel('log₁₀(|δ(t)/δ₀|)', fontsize=13)
    ax.set_title(f'Double Pendulum — Maximum Lyapunov Exponent Estimate\n'
                 f'λ_max ≈ {lambda_est:.4f} /s  (θ₁₀=114.6°, θ₂₀=143.2°)',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] DP Lyapunov plot saved: {save_path}")

    return lambda_est


def plot_comparison_panel(save_path):
    """
    合并面板：Logistic映射分岔图 + Lyapunov指数谱 + 双摆敏感性的组合图。
    作为论文的"混沌摘要图"。
    """
    from logistic_map import bifurcation_data

    fig = plt.figure(figsize=(16, 12))

    # --- 面板 A: Logistic分岔图 ---
    ax1 = fig.add_axes([0.07, 0.55, 0.40, 0.40])
    r_vals = np.linspace(2.5, 4.0, 2500)
    r_data, x_data = bifurcation_data(r_vals, n_transient=400, n_sample=150)
    ax1.scatter(r_data, x_data, s=0.04, c='navy', alpha=0.5, rasterized=True)
    ax1.set_xlim(2.5, 4.0)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel('r', fontsize=12)
    ax1.set_ylabel('x', fontsize=12)
    ax1.set_title('(a) Logistic Map Bifurcation', fontsize=13)
    ax1.grid(True, alpha=0.2)

    # --- 面板 B: Lyapunov指数谱 ---
    ax2 = fig.add_axes([0.55, 0.55, 0.40, 0.40])
    r_l, l_l = lyapunov_spectrum(2.5, 4.0, n_points=800, n=600)
    ax2.plot(r_l, l_l, 'b-', lw=0.8)
    ax2.axhline(y=0, color='red', lw=1)
    ax2.fill_between(r_l, l_l, 0, where=l_l > 0, color='coral', alpha=0.3)
    ax2.set_xlim(2.5, 4.0)
    ax2.set_xlabel('r', fontsize=12)
    ax2.set_ylabel('λ', fontsize=12)
    ax2.set_title('(b) Lyapunov Exponent', fontsize=13)
    ax2.grid(True, alpha=0.2)

    # --- 面板 C: 双摆轨迹 ---
    ax3 = fig.add_axes([0.07, 0.07, 0.40, 0.40])
    from double_pendulum import simulate, to_cartesian
    t, states = simulate(2.0, 2.5, dt=0.01, t_max=100)
    _, (x2, y2) = to_cartesian(states[:, 0], states[:, 2])
    points = np.array([x2, y2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(t[0], t[-1])
    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, cmap='plasma', norm=norm, alpha=0.7, linewidth=0.6)
    lc.set_array(t)
    ax3.add_collection(lc)
    ax3.scatter(x2[0], y2[0], c='lime', s=40, zorder=5)
    ax3.set_xlim(-2.2, 2.2)
    ax3.set_ylim(-2.2, 2.2)
    ax3.set_aspect('equal')
    ax3.set_xlabel('x [m]', fontsize=12)
    ax3.set_ylabel('y [m]', fontsize=12)
    ax3.set_title('(c) Double Pendulum Trajectory', fontsize=13)
    ax3.grid(True, alpha=0.2)

    # --- 面板 D: Logistic映射时间序列（混沌）---
    ax4 = fig.add_axes([0.55, 0.07, 0.40, 0.40])
    xs = np.empty(60)
    xv = 0.4
    for i in range(60):
        xs[i] = xv
        xv = 3.9 * xv * (1 - xv)
    ax4.plot(range(60), xs, 'b-', lw=0.8)
    ax4.scatter(range(60), xs, s=12, c='darkred', zorder=3)
    ax4.set_xlabel('n', fontsize=12)
    ax4.set_ylabel('xₙ', fontsize=12)
    ax4.set_ylim(-0.05, 1.05)
    ax4.set_title('(d) Logistic Map Time Series (r=3.9)', fontsize=13)
    ax4.grid(True, alpha=0.2)

    fig.suptitle('Chaos Theory in Python: From the Double Pendulum to the Logistic Map',
                 fontsize=16, y=0.97)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Comparison panel saved: {save_path}")


# ============================================================
# 主函数
# ============================================================

def run_lyapunov(output_dir="figures"):
    """运行Lyapunov指数相关计算和绘图。"""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Lyapunov Exponent Calculations")
    print("=" * 60)

    # 1. Logistic映射 Lyapunov 谱
    print("\n--- Logistic Map Lyapunov Spectrum ---")
    plot_lyapunov_logistic(os.path.join(output_dir, "lyap_logistic.png"))

    # 2. 双摆 Lyapunov 指数
    print("\n--- Double Pendulum Lyapunov Exponent ---")
    lambda_dp = plot_lyapunov_double_pendulum(
        os.path.join(output_dir, "lyap_double_pendulum.png"))
    print(f"    Estimated λ_max = {lambda_dp:.4f} /s")

    # 3. 组合面板（论文摘要图）
    print("\n--- Comparison Panel ---")
    plot_comparison_panel(os.path.join(output_dir, "chaos_overview.png"))

    print(f"\n[Lyapunov] All figures generated. λ_max(DP) ≈ {lambda_dp:.4f}\n")
    return lambda_dp


if __name__ == "__main__":
    run_lyapunov()
