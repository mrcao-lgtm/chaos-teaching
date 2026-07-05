"""
double_pendulum.py
双摆混沌动力学仿真 — Teaching Chaos Theory with Python

双摆是最简单也最直观的混沌力学系统之一。两个摆锤串联运动，
对初始条件极度敏感，是本科生物理教学中展示混沌的经典范例。

本模块实现了：
  1. 拉格朗日力学推导的四阶Runge-Kutta积分器
  2. 双摆轨迹追踪与动画生成
  3. 初始条件敏感性分析（两个微小差异的轨迹）
  4. 相空间轨迹（θ1-ω1, θ2-ω2）

物理模型:
  质量 m1 = m2 = 1.0 kg
  摆长 L1 = L2 = 1.0 m
  重力 g = 9.81 m/s²
  广义坐标: θ1 (上摆角), θ2 (下摆角)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib import colormaps
import os

# ============================================================
# 物理参数
# ============================================================
G = 9.81       # 重力加速度 m/s²
L1 = 1.0       # 上摆长度 m
L2 = 1.0       # 下摆长度 m
M1 = 1.0       # 上摆质量 kg
M2 = 1.0       # 下摆质量 kg

# ============================================================
# 运动方程 (拉格朗日力学推导)
# ============================================================
def derivatives(state, t=0):
    """
    双摆运动方程的右侧函数。
    
    state = [θ1, ω1, θ2, ω2]
    使用标准的四维一阶ODE形式。
    
    基于拉格朗日量 L = T - V 推导，
    经过符号计算得到加速度表达式。
    """
    th1, w1, th2, w2 = state
    dth = th1 - th2

    # 中间量
    den = M1 + M2 * np.sin(dth)**2
    if abs(den) < 1e-15:
        den = 1e-15  # 避免奇异

    # θ1 的角加速度
    a1 = (M2 * G * np.sin(th2) * np.cos(dth)
          - M2 * np.sin(dth) * (L1 * w1**2 * np.cos(dth) + L2 * w2**2)
          - (M1 + M2) * G * np.sin(th1))
    a1 /= L1 * den

    # θ2 的角加速度
    a2 = ((M1 + M2) * (L1 * w1**2 * np.sin(dth)
                       - G * np.sin(th2) + G * np.sin(th1) * np.cos(dth))
          + M2 * L2 * w2**2 * np.sin(dth) * np.cos(dth))
    a2 /= L2 * den

    return np.array([w1, a1, w2, a2])


def rk4_step(f, state, dt, t=0):
    """经典四阶Runge-Kutta单步积分。"""
    k1 = f(state, t)
    k2 = f(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(state + dt * k3, t + dt)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(theta1_0, theta2_0, omega1_0=0.0, omega2_0=0.0,
              dt=0.01, t_max=100.0):
    """
    双摆主积分函数。
    
    参数:
        theta1_0, theta2_0: 初始角度 (弧度)
        omega1_0, omega2_0: 初始角速度 (弧度/秒)
        dt: 时间步长
        t_max: 总仿真时间
        
    返回:
        t, states: 时间数组和状态数组 (N×4)
    """
    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)
    states = np.zeros((n_steps, 4))
    states[0] = [theta1_0, omega1_0, theta2_0, omega2_0]

    for i in range(1, n_steps):
        states[i] = rk4_step(derivatives, states[i-1], dt, t[i-1])

    return t, states


def to_cartesian(theta1, theta2):
    """
    将角度坐标转换为笛卡尔坐标 (x, y)。
    
    返回:
        (x1, y1), (x2, y2): 两个摆锤的位置
    """
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return (x1, y1), (x2, y2)


# ============================================================
# 可视化函数
# ============================================================

def plot_trajectory(t, states, save_path, title_suffix=""):
    """
    绘制双摆末端 (m2) 的轨迹。
    使用颜色渐变展示时间演进。
    """
    _, (x2, y2) = to_cartesian(states[:, 0], states[:, 2])

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # 用颜色渐变表示时间
    points = np.array([x2, y2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(t[0], t[-1])
    lc = LineCollection(segments, cmap='plasma', norm=norm, alpha=0.8, linewidth=0.8)
    lc.set_array(t)
    ax.add_collection(lc)

    # 起点和终点标记
    ax.scatter(x2[0], y2[0], c='lime', s=60, zorder=5, label='Start', edgecolors='black')
    ax.scatter(x2[-1], y2[-1], c='red', s=60, zorder=5, label='End', edgecolors='black')

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]', fontsize=13)
    ax.set_ylabel('y [m]', fontsize=13)
    ax.set_title(f'Double Pendulum Trajectory (m₂){title_suffix}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(lc, ax=ax, label='Time [s]', shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Trajectory saved: {save_path}")


def plot_snapshot(t, states, save_path, idx=-1):
    """
    绘制双摆在某一时刻的快照：摆臂 + 摆锤。
    """
    th1, _, th2, _ = states[idx]
    (x1, y1), (x2, y2) = to_cartesian(th1, th2)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # 摆臂
    ax.plot([0, x1], [0, y1], 'k-', lw=2.5, zorder=2)
    ax.plot([x1, x2], [y1, y2], 'k-', lw=2.5, zorder=2)

    # 摆锤
    ax.scatter(0, 0, c='gray', s=80, zorder=3, label='Pivot')
    ax.scatter(x1, y1, c='steelblue', s=120, zorder=3, label='m₁')
    ax.scatter(x2, y2, c='crimson', s=120, zorder=3, label='m₂')

    # 轨迹 (最近的部分)
    n_trace = min(500, len(t))
    trace_states = states[max(0, idx - n_trace):idx + 1]
    _, (tx2, ty2) = to_cartesian(trace_states[:, 0], trace_states[:, 2])
    ax.plot(tx2, ty2, 'r-', alpha=0.3, lw=0.8, label='Recent trace')

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]', fontsize=13)
    ax.set_ylabel('y [m]', fontsize=13)
    ax.set_title(f'Double Pendulum Snapshot (t={t[idx]:.1f}s)', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Snapshot saved: {save_path}")


def plot_sensitivity(t, states1, states2, save_path, delta0=1e-3):
    """
    初始条件敏感性分析。
    对比两条初始角度相差 delta0 弧度的轨迹，
    绘制 θ₁(t) 和 log|Δθ₁(t)|。
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # 上: 两条轨迹的 θ1(t)
    ax = axes[0]
    ax.plot(t, states1[:, 0], 'b-', lw=0.8, alpha=0.8, label='θ₁ (ref)')
    ax.plot(t, states2[:, 0], 'r-', lw=0.8, alpha=0.6, label=f'θ₁ (ref + {delta0:.0e} rad)')
    ax.set_ylabel('θ₁ [rad]', fontsize=13)
    ax.set_title(f'Sensitivity to Initial Conditions (Δθ₁₀ = {delta0:.0e} rad)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 下: log|Δθ1(t)|
    ax = axes[1]
    diff = np.abs(states1[:, 0] - states2[:, 0])
    diff = np.where(diff < 1e-16, 1e-16, diff)  # 防止 log(0)
    log_diff = np.log10(diff)

    ax.plot(t, log_diff, 'k-', lw=0.8)
    ax.axhline(y=np.log10(np.pi), color='gray', ls='--', alpha=0.5,
               label='Saturation (π)')
    ax.set_xlabel('Time [s]', fontsize=13)
    ax.set_ylabel('log₁₀(|Δθ₁|)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Sensitivity plot saved: {save_path}")


def plot_phase_portrait(t, states, save_path):
    """
    相空间图: (θ₁, ω₁) 和 (θ₂, ω₂)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 使用采样以降低数据密度
    step = max(1, len(t) // 3000)
    t_s = t[::step]
    th1_s = states[::step, 0]
    w1_s = states[::step, 1]
    th2_s = states[::step, 2]
    w2_s = states[::step, 3]

    for ax, th, w, label, color in zip(
        axes, [th1_s, th2_s], [w1_s, w2_s],
        ['θ₁', 'θ₂'], ['steelblue', 'crimson']):
        # 颜色渐变
        points = np.array([th, w]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(t_s[0], t_s[-1])
        lc = LineCollection(segments, cmap='viridis', norm=norm, alpha=0.7, linewidth=0.6)
        lc.set_array(t_s)
        ax.add_collection(lc)
        ax.scatter(th[0], w[0], c='lime', s=40, zorder=5, edgecolors='black')
        ax.set_xlabel(f'{label} [rad]', fontsize=13)
        ax.set_ylabel(f'd{label}/dt [rad/s]', fontsize=13)
        ax.grid(True, alpha=0.3)

    axes[0].set_title('Phase Portrait — Upper Pendulum', fontsize=13)
    axes[1].set_title('Phase Portrait — Lower Pendulum', fontsize=13)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Phase portrait saved: {save_path}")


def plot_energy(t, states, save_path):
    """
    绘制双摆的总机械能。
    混沌运动中总能量应守恒，可作为数值精度的检验。
    """
    th1, w1, th2, w2 = states.T
    (x1, y1), (x2, y2) = to_cartesian(th1, th2)

    # 动能
    v1_sq = (L1 * w1)**2
    v2_sq = (L1 * w1)**2 + (L2 * w2)**2 + 2 * L1 * L2 * w1 * w2 * np.cos(th1 - th2)
    KE = 0.5 * M1 * v1_sq + 0.5 * M2 * v2_sq

    # 势能
    PE = M1 * G * y1 + M2 * G * y2

    E_total = KE + PE
    E_rel = (E_total - E_total[0]) / abs(E_total[0]) * 100  # 百分比偏差

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax = axes[0]
    ax.plot(t, KE, 'b-', lw=0.8, alpha=0.7, label='KE')
    ax.plot(t, PE, 'r-', lw=0.8, alpha=0.7, label='PE')
    ax.plot(t, E_total, 'k-', lw=1.2, label='Total')
    ax.set_ylabel('Energy [J]', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, E_rel, 'g-', lw=0.8)
    ax.set_ylabel('ΔE/E₀ [%]', fontsize=13)
    ax.set_xlabel('Time [s]', fontsize=13)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Energy Conservation Check', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Energy plot saved: {save_path}")


# ============================================================
# 主函数 — 生成双摆全部图表
# ============================================================

def run_double_pendulum(output_dir="figures"):
    """运行完整双摆仿真，生成论文所需全部图表。"""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Double Pendulum Simulation")
    print("=" * 60)

    # --- 配置：三个关键初始条件 ---
    configs = [
        # (θ₁, θ₂, label, t_max)
        (np.pi / 2, np.pi / 2, "θ₁=90°, θ₂=90°", 60),
        (np.pi, np.pi * 0.99, "θ₁=180°, θ₂=178.2°", 80),
        (2.0, 2.5, "θ₁≈114.6°, θ₂≈143.2°", 100),
    ]

    for th1, th2, label, t_max in configs:
        print(f"\n--- Config: {label} ---")
        dt = 0.01
        t, states = simulate(th1, th2, dt=dt, t_max=t_max)

        safe_label = label.replace("°", "deg").replace("=", "_").replace("≈", "~").replace(", ", "_").replace(" ", "")
        base = f"dp_{safe_label}"

        plot_trajectory(t, states, os.path.join(output_dir, f"{base}_trajectory.png"),
                        title_suffix=f" ({label})")
        plot_snapshot(t, states, os.path.join(output_dir, f"{base}_snapshot.png"))
        plot_phase_portrait(t, states, os.path.join(output_dir, f"{base}_phase.png"))
        plot_energy(t, states, os.path.join(output_dir, f"{base}_energy.png"))

    # --- 初始化敏感性分析 ---
    print("\n--- Sensitivity Analysis ---")
    th1_ref, th2_ref = np.pi / 2, np.pi / 2
    delta = 1e-3  # 0.057°

    t, states_ref = simulate(th1_ref, th2_ref, dt=0.01, t_max=40)
    _, states_delta = simulate(th1_ref + delta, th2_ref, dt=0.01, t_max=40)

    plot_sensitivity(t, states_ref, states_delta,
                     os.path.join(output_dir, "dp_sensitivity.png"), delta0=delta)

    # --- 长时间混沌轨迹 ---
    print("\n--- Long-time Chaotic Trajectory ---")
    t_long, states_long = simulate(2.0, 2.5, dt=0.005, t_max=200)
    plot_trajectory(t_long, states_long,
                    os.path.join(output_dir, "dp_long_trajectory.png"),
                    title_suffix=" (long, chaotic)")

    print("\n[Double Pendulum] All figures generated.\n")


if __name__ == "__main__":
    run_double_pendulum()
