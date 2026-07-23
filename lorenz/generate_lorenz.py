#!/usr/bin/env python
"""
Lorenz-63 system — data generation and FGL sweep over ρ (Rayleigh number).

The Lorenz system is a 3D ODE that exhibits a complete bifurcation sequence:
  fixed point → limit cycle → period-doubling → chaos

  ρ < 1:       stable origin
  ρ ≈ 13-24:   limit cycle (period-1)      ← CSTR equivalent
  ρ ≈ 24-28:   chaos onset                 ← MG τ≈17 equivalent
  ρ ≈ 99-101:  period-doubling windows      ← MG τ≈13 equivalent ("sweet spot")

We extract x(t) as the observed time series (single variable, like real data).

Usage:
  uv run python lorenz/generate_lorenz.py --sweep
  uv run python lorenz/generate_lorenz.py --rho 28
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
from scipy.integrate import solve_ivp

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def lorenz_ode(t, state, sigma, rho, beta):
    x, y, z = state
    return [sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z]


def generate_lorenz(rho=28.0, sigma=10.0, beta=8.0/3.0,
                    t_end=500.0, dt=0.05, seed=42):
    """
    Generate Lorenz-63 time series.

    Args:
        rho: Rayleigh number (control parameter)
        sigma: Prandtl number
        beta: geometric parameter
        t_end: total integration time
        dt: sampling interval
    Returns:
        time, x(t), y(t), z(t)
    """
    np.random.seed(seed)
    x0 = [np.random.randn() * 0.1 + 1.0 for _ in range(3)]

    t_eval = np.arange(0, t_end, dt)
    sol = solve_ivp(
        lorenz_ode, [0, t_end], x0,
        args=(sigma, rho, beta),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-9, atol=1e-12,
    )

    return sol.t, sol.y[0], sol.y[1], sol.y[2]


def periodicity_score(series):
    """Autocorrelation peak (excluding lag 0)."""
    s = series - series.mean()
    ac = np.correlate(s, s, mode='full')
    ac = ac[len(ac)//2:] / (ac[len(ac)//2] + 1e-10)
    return float(ac[20:min(400, len(ac))].max()) if len(ac) > 20 else 1.0


def estimate_lyap(x, dt, dim=3, delay=10):
    """Simple Lyapunov estimate via nearest-neighbor divergence."""
    # Skip transient, use last half
    x = x[len(x)//2:]
    n = len(x) - delay * (dim - 1) - 50

    # Reconstruct phase space via time-delay embedding
    embedded = np.array([x[i:i + delay*dim:delay] for i in range(n)])

    # Find nearest neighbors and track divergence
    d0_vals, d1_vals = [], []
    for i in range(0, n, 10):
        dists = np.linalg.norm(embedded - embedded[i], axis=1)
        dists[i] = 1e10  # exclude self
        j = np.argmin(dists)
        d0_vals.append(dists[j])
        if i + 20 < n and j + 20 < n:
            d1_vals.append(np.linalg.norm(embedded[i+20] - embedded[j+20]))

    if len(d1_vals) == 0 or np.mean(d0_vals) == 0:
        return 0.0
    return float(np.mean(np.log(np.array(d1_vals) / (np.array(d0_vals[:len(d1_vals)]) + 1e-10))) / (20 * dt))


def save_pkl(series, path, label=""):
    col = torch.tensor(series, dtype=torch.float64).unsqueeze(1)
    tensor = torch.cat((col, col.clone()), dim=1)
    with open(path, "wb") as f:
        pickle.dump(tensor, f)
    print(f"  Saved: {path}  shape={tensor.shape}")


def main():
    parser = argparse.ArgumentParser(description="Lorenz-63 data generation")
    parser.add_argument("--rho", type=float, default=28.0)
    parser.add_argument("--t_end", type=float, default=500.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n_points", type=int, default=6000,
                        help="Points to keep (after discarding transient)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep over ρ values")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.sweep:
        # ρ sweep covering all regimes
        rho_values = [20, 22, 24, 25, 28, 32, 40, 60, 80, 99.5, 100.0, 100.5]

        print("=" * 75)
        print("  Lorenz-63 ρ Sweep")
        print("=" * 75)
        print(f"  {'ρ':>8s}  {'Periodicity':>12s}  {'Lyap(est)':>10s}  "
              f"{'x_min':>8s}  {'x_max':>8s}  {'x_std':>8s}  {'Regime'}")
        print(f"  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*25}")

        results = []
        for rho in rho_values:
            try:
                t, x, y, z = generate_lorenz(rho=rho, t_end=args.t_end, dt=args.dt, seed=args.seed)
                # Discard first 20% as transient
                start = len(x) // 5
                x_trim = x[start:start + args.n_points]

                score = periodicity_score(x_trim)
                lyap = estimate_lyap(x_trim, args.dt)

                regime = ""
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

                label = f"ρ={rho}"
                if "PERIOD-DOUBLING" in regime:
                    label += " ← SWEET SPOT?"

                print(f"  {rho:8.1f}  {score:12.4f}  {lyap:10.6f}  "
                      f"{x_trim.min():8.2f}  {x_trim.max():8.2f}  "
                      f"{x_trim.std():8.3f}  {regime}")

                results.append({
                    "rho": rho, "periodicity": score, "lyap": lyap,
                    "x_min": x_trim.min(), "x_max": x_trim.max(),
                    "x_std": x_trim.std(), "regime": regime,
                })

                # Save promising candidates (periodicity < 0.85 or interesting)
                if score < 0.85 or "PERIOD-DOUBLING" in regime:
                    save_pkl(x_trim,
                             os.path.join(OUTPUT_DIR, f"lorenz_rho{rho}.pkl"),
                             f"Lorenz ρ={rho}")
            except Exception as e:
                print(f"  {rho:8.1f}  {'─':>12s}  FAILED: {e}")

        print()
        print("Candidates for FGL testing:")
        results.sort(key=lambda r: r["periodicity"])
        for r in results:
            print(f"  ρ={r['rho']:6.1f}  P={r['periodicity']:.4f}  lyap≈{r['lyap']:.6f}  {r['regime']}")

    else:
        print(f"Generating Lorenz-63 with ρ={args.rho}...")
        t, x, y, z = generate_lorenz(rho=args.rho, t_end=args.t_end, dt=args.dt, seed=args.seed)
        start = len(x) // 5
        x_trim = x[start:start + args.n_points]

        score = periodicity_score(x_trim)
        lyap = estimate_lyap(x_trim, args.dt)
        print(f"Periodicity: {score:.4f}, Lyap(est): {lyap:.6f}")
        print(f"x range: [{x_trim.min():.3f}, {x_trim.max():.3f}], std: {x_trim.std():.3f}")

        save_pkl(x_trim, os.path.join(OUTPUT_DIR, f"lorenz_rho{args.rho}.pkl"),
                 f"Lorenz ρ={args.rho}")


if __name__ == "__main__":
    main()
