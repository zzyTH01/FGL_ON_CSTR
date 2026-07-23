#!/usr/bin/env python
"""Lorenz-63 统一入口(数据生成 + FGL 扫描)。

合并 ``lorenz/generate_lorenz.py`` 与 ``lorenz/lh_sweep.py``。

用法::

    uv run python lorenz/run.py                         # 跑所有 enabled=True
    uv run python lorenz/run.py -e generate --sweep      # 扫 ρ 生成数据
    uv run python lorenz/run.py -e lh_sweep              # ρ=60 的 L×H 扫描
    uv run python lorenz/run.py --list
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
from scipy.integrate import solve_ivp

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fgl_common import RNN, run_fgl_experiment, run_lh_sweep  # noqa: E402

_LORENZ_DIR = os.path.dirname(os.path.abspath(__file__))
SIGMA, BETA = 10.0, 8.0 / 3.0


# ==================== Lorenz ODE & data generation ====================
def _lorenz_ode(t, state, sigma, rho, beta):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def generate_lorenz(rho=28.0, sigma=SIGMA, beta=BETA, t_end=500.0, dt=0.05, seed=42):
    """Generate Lorenz-63 series. Returns (t, x, y, z). Migrated from generate_lorenz.py."""
    np.random.seed(seed)
    x0 = [np.random.randn() * 0.1 + 1.0 for _ in range(3)]
    t_eval = np.arange(0, t_end, dt)
    sol = solve_ivp(_lorenz_ode, [0, t_end], x0, args=(sigma, rho, beta),
                    t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-12)
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


def _gen_for_sweep(rho=60.0, n_points=8000):
    """Generate fixed data for the L×H sweep (initial [1,1,1]). Mirrors lh_sweep.py.
    Returns an (N, 2) tensor with both columns = x(t)."""
    def ode(t, s):
        x, y, z = s
        return [SIGMA * (y - x), x * (rho - z) - y, x * y - BETA * z]

    sol = solve_ivp(ode, [0, 500], [1.0, 1.0, 1.0], t_eval=np.arange(0, 500, 0.05),
                    method='RK45', rtol=1e-9, atol=1e-12)
    x = sol.y[0]
    start = len(x) // 5
    x = x[start:start + n_points]
    col = torch.tensor(x, dtype=torch.float64).unsqueeze(1)
    return torch.cat((col, col.clone()), dim=1)


def periodicity_score(series):
    """Autocorrelation peak (excluding lag 0)."""
    s = series - series.mean()
    ac = np.correlate(s, s, mode='full')
    ac = ac[len(ac) // 2:] / (ac[len(ac) // 2:] + 1e-10)
    return float(ac[20:min(400, len(ac))].max()) if len(ac) > 20 else 1.0


def estimate_lyap(x, dt, dim=3, delay=10):
    """Simple Lyapunov estimate via nearest-neighbor divergence."""
    x = x[len(x) // 2:]
    n = len(x) - delay * (dim - 1) - 50
    embedded = np.array([x[i:i + delay * dim:delay] for i in range(n)])
    d0_vals, d1_vals = [], []
    for i in range(0, n, 10):
        dists = np.linalg.norm(embedded - embedded[i], axis=1)
        dists[i] = 1e10
        j = np.argmin(dists)
        d0_vals.append(dists[j])
        if i + 20 < n and j + 20 < n:
            d1_vals.append(np.linalg.norm(embedded[i + 20] - embedded[j + 20]))
    if len(d1_vals) == 0 or np.mean(d0_vals) == 0:
        return 0.0
    return float(np.mean(np.log(np.array(d1_vals) / (np.array(d0_vals[:len(d1_vals)]) + 1e-10))) / (20 * dt))


def _data_dir():
    d = os.path.join(_LORENZ_DIR, "data")
    os.makedirs(d, exist_ok=True)
    return d


def _save_pkl(series, path, label=""):
    col = torch.tensor(series, dtype=torch.float64).unsqueeze(1)
    tensor = torch.cat((col, col.clone()), dim=1)
    with open(path, "wb") as f:
        pickle.dump(tensor, f)
    print(f"  Saved: {path}  shape={tensor.shape}")


# ==================== Experiments ====================
def run_generate(args):
    """Corresponds to generate_lorenz.py:generate data (single or --sweep over ρ)."""
    if args.sweep:
        rho_values = [20, 22, 24, 25, 28, 32, 40, 60, 80, 99.5, 100.0, 100.5]
        print("=" * 75 + "\n  Lorenz-63 ρ Sweep\n" + "=" * 75)
        print(f"  {'ρ':>8s}  {'Periodicity':>12s}  {'Lyap(est)':>10s}  "
              f"{'x_min':>8s}  {'x_max':>8s}  {'x_std':>8s}  {'Regime'}")
        print(f"  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*25}")
        results = []
        for rho in rho_values:
            try:
                _, x, _, _ = generate_lorenz(rho=rho, t_end=args.t_end, dt=args.dt, seed=args.seed)
                start = len(x) // 5
                x_trim = x[start:start + args.n_points]
                score = periodicity_score(x_trim)
                lyap = estimate_lyap(x_trim, args.dt)
                if score > 0.95:
                    regime = "period-1 (like CSTR)"
                elif score > 0.8:
                    regime = "periodic"
                elif score > 0.6:
                    regime = "weakly periodic"
                elif lyap > 0.01:
                    regime = "★ CHAOS (like MG τ≥17)"
                elif score < 0.6 and lyap < 0.005:
                    regime = "★ PERIOD-DOUBLING? (like MG τ=13)"
                else:
                    regime = "transitional"
                print(f"  {rho:8.1f}  {score:12.4f}  {lyap:10.6f}  "
                      f"{x_trim.min():8.2f}  {x_trim.max():8.2f}  "
                      f"{x_trim.std():8.3f}  {regime}")
                results.append({"rho": rho, "periodicity": score, "lyap": lyap, "regime": regime})
                if score < 0.85 or "PERIOD-DOUBLING" in regime:
                    _save_pkl(x_trim, os.path.join(_data_dir(), f"lorenz_rho{rho}.pkl"),
                              f"Lorenz ρ={rho}")
            except Exception as e:
                print(f"  {rho:8.1f}  {'─':>12s}  FAILED: {e}")
        print("\nCandidates for FGL testing:")
        for r in sorted(results, key=lambda r: r["periodicity"]):
            print(f"  ρ={r['rho']:6.1f}  P={r['periodicity']:.4f}  lyap≈{r['lyap']:.6f}  {r['regime']}")
    else:
        print(f"Generating Lorenz-63 with ρ={args.rho}...")
        _, x, _, _ = generate_lorenz(rho=args.rho, t_end=args.t_end, dt=args.dt, seed=args.seed)
        start = len(x) // 5
        x_trim = x[start:start + args.n_points]
        score = periodicity_score(x_trim)
        lyap = estimate_lyap(x_trim, args.dt)
        print(f"Periodicity: {score:.4f}, Lyap(est): {lyap:.6f}")
        print(f"x range: [{x_trim.min():.3f}, {x_trim.max():.3f}], std: {x_trim.std():.3f}")
        _save_pkl(x_trim, os.path.join(_data_dir(), f"lorenz_rho{args.rho}.pkl"),
                  f"Lorenz ρ={args.rho}")


def run_lh_sweep_exp(args):
    """Corresponds to lh_sweep.py:ρ=ρ L×H sweep (default ρ=60 strong chaos)."""
    data = _gen_for_sweep(rho=args.rho, n_points=args.n_points)
    print(f"Lorenz ρ={args.rho}, {args.n_points} pts")
    print(f"  Range: [{data[:, 0].min():.1f}, {data[:, 0].max():.1f}]")
    L_vals = [int(x) for x in (args.L_values or "8,15,25,40,60").split(",")]
    H_vals = [int(x) for x in (args.H_values or "5,15,30,45,60").split(",")]
    seeds = list(range(args.seeds if args.seeds else 3))

    def _run(L, H, data, seed):
        return run_fgl_experiment(
            data, lookback_window=L, forecasting_horizon=H,
            alpha=args.alpha, temperature=args.temperature, num_bins=args.bins,
            epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
            seed=seed, model_fn=RNN, verbose=False, label=f"L{L}_H{H}")

    outdir = os.path.join(_LORENZ_DIR, "results")
    run_lh_sweep(_run, data, L_vals, H_vals, seeds, outdir, name="lorenz_lh_sweep",
                 title=f"Lorenz-63 L×H Sweep (ρ={args.rho}, strong chaos)",
                 period_label=f"Lorenz ρ={args.rho} strong chaos, Lyapunov ~1.6",
                 extra_meta={"System": "Lorenz-63", "rho": args.rho,
                             "alpha": args.alpha, "T": args.temperature})


# ==================== Experiment switches ====================
EXPERIMENTS = {
    "generate": dict(fn=run_generate,    enabled=True,  note="生成 Lorenz 数据(单点或 --sweep 扫 ρ)"),
    "lh_sweep": dict(fn=run_lh_sweep_exp, enabled=True, note="L×H 扫描(默认 ρ=60 强混沌,主线)"),
}


def main():
    parser = argparse.ArgumentParser(description="Lorenz-63 统一入口(配置字典开关)")
    parser.add_argument("-e", "--experiments", type=str, default=None,
                        help="逗号分隔的实验名;不指定则跑所有 enabled=True")
    parser.add_argument("--list", action="store_true", help="列出实验及开关状态")
    # generate 参数
    parser.add_argument("--rho", type=float, default=60.0)
    parser.add_argument("--t_end", type=float, default=500.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n_points", type=int, default=8000)
    parser.add_argument("--sweep", action="store_true", help="[generate] 扫 ρ")
    # FGL 参数
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("-T", "--temperature", type=float, default=4.0, dest="temperature")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=None, help="[lh_sweep] 种子数量")
    parser.add_argument("--L_values", type=str, default=None, help="[lh_sweep] L 取值")
    parser.add_argument("--H_values", type=str, default=None, help="[lh_sweep] H 取值")
    args = parser.parse_args()

    if args.list:
        print("Lorenz experiments(开关 / 说明):")
        for name, cfg in EXPERIMENTS.items():
            flag = "✓ ON " if cfg["enabled"] else "  off"
            print(f"  {name:10s} [{flag}]  {cfg['note']}")
        return

    if args.experiments:
        names = [n.strip() for n in args.experiments.split(",")]
        for n in names:
            if n not in EXPERIMENTS:
                sys.exit(f"未知实验: {n};可用: {', '.join(EXPERIMENTS)}")
    else:
        names = [n for n, c in EXPERIMENTS.items() if c["enabled"]]

    print(f"\n将运行实验: {names}\n")
    for n in names:
        print(f"\n{'#' * 60}\n# Experiment: {n}\n{'#' * 60}")
        EXPERIMENTS[n]["fn"](args)


if __name__ == "__main__":
    main()
