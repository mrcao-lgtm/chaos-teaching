# Chaos-Teaching

Teaching chaos theory with Python simulations: double pendulum dynamics, logistic map, and Lyapunov exponents.

Author: Xinghuan Cao, Department of Fundamental Courses, Ma'anshan University, Maanshan, China (ORCID: 0009-0008-4489-9763)

## Contents

- `double_pendulum.py` — Double pendulum chaos simulation (Lagrangian mechanics + RK4 integrator; trajectory, phase space, energy, and initial-condition sensitivity analysis)
- `logistic_map.py` — Logistic map simulation (bifurcation diagram, time series, Cobweb plot, return map)
- `lyapunov.py` — Lyapunov exponent calculation (logistic map spectrum, double pendulum λ estimation)
- `run_all.py` — Generate all simulation figures in one run
- `test_imports.py` — Import verification

## Usage

```bash
pip install numpy matplotlib scipy
python run_all.py
```

Output: `figures/` directory with all generated plots (~3–5 minutes).

## License

MIT License
