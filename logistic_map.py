"""
logistic_map.py
Logistic映射仿真 — Teaching Chaos Theory with Python

Logistic映射: x_{n+1} = r * x_n * (1 - x_n)

这是混沌理论中最经典的一维离散动力系统。虽然形式简单，
它展示了从周期倍化分岔到混沌的全部路径，是本科教学中
理解混沌的绝佳入口。

本模块实现了：
  1. 分岔图 (r从2.5到4.0，x收敛后的值)
  2. 时间序列图 (周期、倍周期、混沌三种状态)
  3. 初始条件敏感性 (两个接近的x₀，展示指数发散)
  4. Cobweb图 (蛛网图，可视化迭代过程)
  5. 返回映射 (return map, x_n vs x_{n+1})
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def logistic_map(r, x):
    """Logistic映射单步迭代: x_{n+1} = r x (1-x)"""
    return r * x * (1.0 - x)


def iterate(r, x0, n):
    """
    迭代Logistic映射 n 步。
    
    参数:
        r: 控制参数
        x0: 初始值 (0 <= x0 <= 1)
        n: 迭代步数
        
    返回:
        xs: 长度为 n 的迭代序列
    """
    xs = np.empty(n)
    xs[0] = x0
    for i in range(1, n):
        xs[i] = logistic_map(r, xs[i-1])
    return xs


def bifurcation_data(r_vals, x0=0.5, n_transient=500, n_sample=200):
    """
    批量生成分岔图数据。
    
    参数:
        r_vals: r 值数组
        x0: 初始值
        n_transient: 舍弃的暂态步数
        n_sample: 采样步数
        
    返回:
        r_list, x_list: 散点数据 (用于绘图)
    """
    r_list, x_list = [], []
    for r in r_vals:
        xs = iterate(r, x0, n_transient + n_sample)
        xs = xs[n_transient:]  # 舍弃暂态
        for x in xs:
            r_list.append(r)
            x_list.append(x)
    return np.array(r_list), np.array(x_list)


# ============================================================
# 可视化函数
# ============================================================

def plot_bifurcation(save_path, r_min=2.5, r_max=4.0, n_points=3000,
                     x0=0.5, n_transient=500, n_sample=200):
    """
    绘制Logistic映射分岔图 — 论文核心图之一。
    """
    r_vals = np.linspace(r_min, r_max, n_points)
    r_data, x_data = bifurcation_data(r_vals, x0, n_transient, n_sample)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # 散点图（小点，高密度）
    ax.scatter(r_data, x_data, s=0.08, c='navy', alpha=0.6, rasterized=True)

    # 标注周期窗口
    windows = [(3.0, 'Period-2\nbifurcation'),
               (3.449, 'Period-4'),
               (3.544, 'Period-8'),
               (3.569, 'Onset of\nchaos'),
               (3.83, 'Period-3\nwindow')]

    for r_pos, label in windows:
        ax.axvline(x=r_pos, color='red', ls='--', alpha=0.4, lw=0.8)
        ax.text(r_pos, 0.95, label, fontsize=8, ha='center', va='top',
                rotation=90, color='red', alpha=0.7)

    ax.set_xlim(r_min, r_max)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Control parameter r', fontsize=14)
    ax.set_ylabel('x (asymptotic values)', fontsize=14)
    ax.set_title('Logistic Map Bifurcation Diagram', fontsize=15)

    # 第二 x 轴 — Feigenbaum 常数标注
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Bifurcation diagram saved: {save_path}")


def plot_time_series(save_path, r_values=[2.8, 3.45, 3.9],
                     x0=0.5, n=100, labels=None):
    """
    绘制不同 r 值下的时间序列 — 周期、倍周期、混沌对比。
    """
    if labels is None:
        labels = [f'r = {r}' for r in r_values]

    fig, axes = plt.subplots(len(r_values), 1, figsize=(10, 2.5 * len(r_values)))

    for ax, r, label in zip(axes, r_values, labels):
        xs = iterate(r, x0, n)
        ax.plot(range(n), xs, 'b-', lw=1.2, alpha=0.8)
        ax.scatter(range(n), xs, s=15, c='darkred', zorder=3, alpha=0.7)
        ax.set_ylabel('xₙ', fontsize=12)
        ax.set_title(label, fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Iteration n', fontsize=13)
    fig.suptitle('Logistic Map Time Series', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Time series saved: {save_path}")


def plot_sensitivity(save_path, r=3.9, x0_1=0.4, x0_2=0.4001, n=60):
    """
    展示对初始条件的敏感性：两条 x₀ 极接近的轨迹。
    上半图：两条轨迹
    下半图：log|Δxₙ|
    """
    xs1 = iterate(r, x0_1, n)
    xs2 = iterate(r, x0_2, n)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax = axes[0]
    ax.plot(range(n), xs1, 'b-o', ms=4, lw=1, alpha=0.8, label=f'x₀ = {x0_1}')
    ax.plot(range(n), xs2, 'r-s', ms=4, lw=1, alpha=0.6, label=f'x₀ = {x0_2}')
    ax.set_ylabel('xₙ', fontsize=13)
    ax.set_title(f'Sensitivity to Initial Conditions (Logistic Map, r={r})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    diff = np.abs(xs1 - xs2)
    diff = np.where(diff < 1e-16, 1e-16, diff)
    log_diff = np.log10(diff)

    ax.plot(range(n), log_diff, 'k-', lw=1)
    ax.axhline(y=np.log10(1.0), color='gray', ls='--', alpha=0.5, label='Saturation')
    ax.set_xlabel('Iteration n', fontsize=13)
    ax.set_ylabel('log₁₀(|Δxₙ|)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Sensitivity plot saved: {save_path}")


def plot_cobweb(save_path, r=3.8, x0=0.4, n_steps=200):
    """
    Cobweb (蛛网) 图: 在 xₙ-xₙ₊₁ 平面上可视化迭代过程。
    绘制：
      - f(x) = rx(1-x) 曲线
      - y = x 对角线
      - 迭代路径 (矩形螺旋)
      - 收敛/混沌吸引子
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    # f(x) 曲线
    x_curve = np.linspace(0, 1, 500)
    y_curve = logistic_map(r, x_curve)
    ax.plot(x_curve, y_curve, 'b-', lw=2, label=f'f(x) = {r}x(1-x)')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='y = x')

    # 迭代路径
    xs = iterate(r, x0, n_steps)

    # 绘制 cobweb 路径 (最后 30 步)
    start = max(0, n_steps - 30)
    x_cw, y_cw = [], []
    for i in range(start, n_steps - 1):
        x_cw.append(xs[i])
        y_cw.append(xs[i+1])
        x_cw.append(xs[i+1])  # 水平到对角线
        y_cw.append(xs[i+1])

    ax.plot(x_cw, y_cw, 'r-', lw=1, alpha=0.7, label='Iteration path')
    ax.scatter(xs[start:-1], xs[start+1:], s=20, c='darkred', zorder=5, alpha=0.8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('xₙ', fontsize=13)
    ax.set_ylabel('xₙ₊₁', fontsize=13)
    ax.set_title(f'Cobweb Plot (r={r})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Cobweb plot saved: {save_path}")


def plot_return_map(save_path, r=3.9, x0=0.3, n=500):
    """
    返回映射 (return map): xₙ vs xₙ₊₁ 散点图。
    展示吸引子的结构。
    """
    xs = iterate(r, x0, n)
    start = 100  # 舍弃暂态

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.scatter(xs[start:-1], xs[start+1:], s=3, c='navy', alpha=0.6, rasterized=True)
    x_curve = np.linspace(0, 1, 500)
    y_curve = logistic_map(r, x_curve)
    ax.plot(x_curve, y_curve, 'r-', lw=1.5, alpha=0.5, label=f'f(x) = {r}x(1-x)')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('xₙ', fontsize=13)
    ax.set_ylabel('xₙ₊₁', fontsize=13)
    ax.set_title(f'Return Map (r={r})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Return map saved: {save_path}")


# ============================================================
# 主函数
# ============================================================

def run_logistic_map(output_dir="figures"):
    """运行完整Logistic映射仿真，生成论文所需全部图表。"""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Logistic Map Simulations")
    print("=" * 60)

    # 1. 分岔图 (核心)
    print("\n--- Bifurcation Diagram ---")
    plot_bifurcation(os.path.join(output_dir, "lm_bifurcation.png"),
                     r_min=2.5, r_max=4.0, n_points=4000)

    # 2. 时间序列对比
    print("\n--- Time Series ---")
    plot_time_series(os.path.join(output_dir, "lm_timeseries.png"),
                     r_values=[2.8, 3.45, 3.9],
                     labels=['Period-1 (r=2.8)', 'Period-2 (r=3.45)', 'Chaos (r=3.9)'])

    # 3. 初始条件敏感性
    print("\n--- Sensitivity ---")
    plot_sensitivity(os.path.join(output_dir, "lm_sensitivity.png"),
                     r=3.9, x0_1=0.4, x0_2=0.4001, n=60)

    # 4. Cobweb图 (周期 vs 混沌)
    print("\n--- Cobweb Plots ---")
    for r_val, label in [(3.2, "period2"), (3.5, "period4"), (3.9, "chaos")]:
        plot_cobweb(os.path.join(output_dir, f"lm_cobweb_{label}.png"),
                    r=r_val, x0=0.4, n_steps=100)

    # 5. 返回映射
    print("\n--- Return Maps ---")
    for r_val, label in [(3.2, "period2"), (3.83, "period3"), (3.9, "chaos")]:
        plot_return_map(os.path.join(output_dir, f"lm_return_{label}.png"),
                        r=r_val, x0=0.3, n=500)

    print("\n[Logistic Map] All figures generated.\n")


if __name__ == "__main__":
    run_logistic_map()
