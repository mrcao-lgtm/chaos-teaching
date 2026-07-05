#!/usr/bin/env python3
"""Quick smoke test — verify all modules import and run a minimal example."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

print('=== Import test ===')

from double_pendulum import simulate, derivatives, to_cartesian, rk4_step
print('[OK] double_pendulum')

from logistic_map import iterate, bifurcation_data, logistic_map
print('[OK] logistic_map')

from lyapunov import lyapunov_logistic, double_pendulum_lyapunov
print('[OK] lyapunov')

# Quick functional test: Logistic iteration
xs = iterate(3.9, 0.5, 100)
assert len(xs) == 100
assert 0 <= xs[-1] <= 1
print(f'[OK] Logistic iterate: last={xs[-1]:.6f}')

# Quick functional test: Lyapunov
lyap = lyapunov_logistic(3.9, n=500, n_transient=100)
print(f'[OK] Lyapunov λ(r=3.9) = {lyap:.4f} (expect > 0 for chaos)')
assert lyap > 0, f'Chaotic regime should have positive Lyapunov exponent, got {lyap}'

# Quick DP test: short simulation
t, states = simulate(np.pi/2, np.pi/2, dt=0.01, t_max=5)
assert len(t) == 500
print(f'[OK] DP short sim: {len(t)} steps')
print()

print('All smoke tests PASSED.')
print(f'Python: {sys.version}')
