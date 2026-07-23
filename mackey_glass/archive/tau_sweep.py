#!/usr/bin/env python
"""
Mackey-Glass τ-sweep: FGL effectiveness vs chaos strength.

Tests the hypothesis that FGL improvement correlates with Lyapunov exponent.
τ = 10:  limit cycle (periodic)    → prediction: FGL Δ ≈ 0%
τ = 13:  period-doubling           → prediction: FGL Δ ≈ small
τ = 17:  chaos onset (paper)       → prediction: FGL Δ ~23%
τ = 23:  moderate chaos            → prediction: FGL Δ > 23%
τ = 30:  high-dimensional chaos    → prediction: FGL Δ >> 23%

Usage:
  uv run python mackey_glass/exp/tau_sweep.py
"""

import sys
import os
import pickle
import json
import argparse
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, MG_DIR)

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.utils import MackeyGlass, RNN, create_time_series_dataset, KL

# ---- Device ----
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using {device}")


class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, current_loss, model):
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def generate_mg_data(tau, n_points=10000, dt=1.0, seed=42):
    """Generate Mackey-Glass data for a given tau. Returns (data_tensor, lyap_exp)."""
    mg = MackeyGlass(
        tau=tau,
        constant_past=0.9,
        nmg=10, beta=0.2, gamma=0.1,
        dt=dt,
        splits=(float(n_points), 0.),
        seed_id=seed,
    )
    # Build 2-column tensor (same format as data.pkl)
    vals = []
    for idx in range(len(mg)):
        _, target = mg[idx]
        vals.append(target.squeeze().item())
    col = torch.tensor(vals, dtype=torch.float64).unsqueeze(1)
    tensor = torch.cat((col, col.clone()), dim=1)
    return tensor, mg.lyap_exp


def run_fgl_on_data(data, horizon=5, alpha=0.5, num_bins=50,
                    epochs=50, temperature=4, lookback=8,
                    batch_size=128, patience=5):
    """Run one FGL experiment. Returns (teacher_mse, baseline_mse, student_mse, improvement)."""
    torch.manual_seed(42)
    hidden_size = 128
    num_layers = 2
    lr = 1e-4
    output_size = num_bins
    H = horizon
    L = lookback

    val_size, test_size = 0.2, 0.2

    # Shared bin edges (same fix as CSTR)
    x_raw = np.array([float(pt[0]) for pt in data])
    y_raw = np.array([float(pt[1]) for pt in data])
    all_y = []
    for i in range(len(x_raw) - L - 1 + 1):
        all_y.append(y_raw[i + L])
    shared_edges = np.linspace(np.array(all_y).min(), np.array(all_y).max(), num_bins - 1)

    teacher_train, teacher_val, teacher_test, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=1,
        num_bins=num_bins, val_size=val_size, test_size=test_size,
        offset=H - 1, batch_size=batch_size, bin_edges=shared_edges,
    )
    student_train, student_val, student_test, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H,
        num_bins=num_bins, val_size=val_size, test_size=test_size,
        offset=0, batch_size=batch_size, bin_edges=shared_edges,
    )

    celoss = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    # Teacher
    teacher = RNN(L, hidden_size, output_size, num_layers).to(device)
    opt_t = torch.optim.Adam(teacher.parameters(), lr=lr)
    stop_t = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        teacher.train()
        for _, x, y in teacher_train:
            x = x.float().to(device).view(-1, 1, L)
            y = y.long().to(device)
            opt_t.zero_grad()
            celoss(teacher(x), y).backward()
            opt_t.step()
        teacher.eval()
        with torch.no_grad():
            vloss = sum(celoss(teacher(x.float().to(device).view(-1, 1, L)),
                               y.long().to(device)).item()
                        for _, x, y in teacher_val) / len(teacher_val)
        if stop_t.step(vloss, teacher):
            break
    stop_t.restore(teacher)

    # Baseline
    baseline = RNN(L, hidden_size, output_size, num_layers).to(device)
    opt_b = torch.optim.Adam(baseline.parameters(), lr=lr)
    stop_b = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        baseline.train()
        for _, x, y in student_train:
            x = x.float().to(device).view(-1, 1, L)
            y = y.long().to(device)
            opt_b.zero_grad()
            celoss(baseline(x), y).backward()
            opt_b.step()
        baseline.eval()
        with torch.no_grad():
            vloss = sum(celoss(baseline(x.float().to(device).view(-1, 1, L)),
                               y.long().to(device)).item()
                        for _, x, y in student_val) / len(student_val)
        if stop_b.step(vloss, baseline):
            break
    stop_b.restore(baseline)

    # Student (FGL)
    student = RNN(L, hidden_size, output_size, num_layers).to(device)
    opt_s = torch.optim.Adam(student.parameters(), lr=lr)
    stop_s = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        student.train()
        for (_, x_s, y_s), (_, x_t, _) in zip(student_train, teacher_train):
            x_s = x_s.float().to(device).view(-1, 1, L)
            targets = y_s.long().to(device)
            outputs = student(x_s)
            x_t = x_t.float().to(device).view(-1, 1, L)
            with torch.no_grad():
                logits = teacher(x_t)
            loss = alpha * celoss(outputs, targets) + KL(outputs, logits, temperature, alpha)
            opt_s.zero_grad()
            loss.backward()
            opt_s.step()
        student.eval()
        with torch.no_grad():
            vloss = sum(celoss(student(x.float().to(device).view(-1, 1, L)),
                               y.long().to(device)).item()
                        for _, x, y in student_val) / len(student_val)
        if stop_s.step(vloss, student):
            break
    stop_s.restore(student)

    # Evaluate
    def eval_mse(model, loader):
        model.eval()
        total = 0.0
        with torch.no_grad():
            for _, x, y in loader:
                x = x.float().to(device).view(-1, 1, L)
                total += mse(model(x).argmax(dim=1).float(),
                             y.float().to(device).squeeze(-1)).item()
        return total / len(loader)

    t_mse = eval_mse(teacher, teacher_test)
    b_mse = eval_mse(baseline, student_test)
    s_mse = eval_mse(student, student_test)
    improvement = (b_mse - s_mse) / b_mse * 100 if b_mse > 0 else 0

    return {"teacher": t_mse, "baseline": b_mse, "student": s_mse, "improvement": improvement}


def main():
    parser = argparse.ArgumentParser(description="MG τ-sweep: chaos vs FGL")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--taus", type=str, default="10,13,17,23,30",
                        help="Comma-separated τ values")
    parser.add_argument("--n_points", type=int, default=10000)
    args = parser.parse_args()

    tau_values = [float(t.strip()) for t in args.taus.split(",")]

    print("=" * 70)
    print("  Mackey-Glass τ-Sweep: FGL Improvement vs Chaos Strength")
    print("=" * 70)
    print(f"  τ values: {tau_values}")
    print(f"  H={args.horizon}, α={args.alpha}, epochs={args.epochs}")
    print(f"  Prediction: τ↑ → Lyapunov↑ → FGL Δ↑")
    print("=" * 70)

    results = []
    for tau in tau_values:
        print(f"\n{'─'*50}")
        print(f"  τ = {tau}")
        print(f"{'─'*50}")

        # Generate data
        data, lyap = generate_mg_data(tau, n_points=args.n_points)
        print(f"  Lyapunov exponent: {lyap:+.6f}")

        # Compute periodicity (autocorrelation peak)
        h2o = data[:, 0].numpy()
        ac = np.correlate(h2o - h2o.mean(), h2o - h2o.mean(), mode="full")
        ac = ac[len(ac)//2:] / (ac[len(ac)//2] + 1e-10)
        periodicity = float(ac[20:200].max()) if len(ac) > 20 else 1.0
        ac_zero_lag = next((i for i in range(1, len(ac)) if ac[i] < 0), len(ac))
        print(f"  Periodicity score: {periodicity:.4f}  (AC zero at lag {ac_zero_lag})")

        # Run FGL
        r = run_fgl_on_data(
            data, horizon=args.horizon, alpha=args.alpha,
            epochs=args.epochs,
        )
        r["tau"] = tau
        r["lyap"] = lyap
        r["periodicity"] = periodicity
        r["ac_zero_lag"] = ac_zero_lag
        results.append(r)

        marker = " ★" if r["improvement"] > 0 else ""
        print(f"  Teacher={r['teacher']:.2f}  Baseline={r['baseline']:.2f}  "
              f"Student={r['student']:.2f}  Δ={r['improvement']:+.1f}%{marker}")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'τ':>6s}  {'Lyapunov':>10s}  {'Periodicity':>12s}  "
          f"{'Teacher':>9s}  {'Baseline':>9s}  {'Student':>9s}  {'FGL Δ':>8s}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*12}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}")
    for r in results:
        print(f"  {r['tau']:6.1f}  {r['lyap']:+10.6f}  {r['periodicity']:12.4f}  "
              f"{r['teacher']:9.2f}  {r['baseline']:9.2f}  {r['student']:9.2f}  "
              f"{r['improvement']:+7.1f}%")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    taus = [r["tau"] for r in results]
    lyaps = [r["lyap"] for r in results]
    improvements = [r["improvement"] for r in results]
    periodicities = [r["periodicity"] for r in results]

    # Plot 1: FGL Δ vs Lyapunov
    ax = axes[0]
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in improvements]
    ax.scatter(lyaps, improvements, c=colors, s=120, zorder=5, edgecolors="black", linewidth=0.5)
    for i, tau in enumerate(taus):
        offset = 12 if i % 2 == 0 else -18
        ax.annotate(f"τ={tau:.0f}", (lyaps[i], improvements[i]),
                    textcoords="offset points", xytext=(0, offset),
                    fontsize=10, ha="center", fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lyapunov Exponent", fontsize=12)
    ax.set_ylabel("FGL Improvement Δ (%)", fontsize=12)
    ax.set_title("FGL Effectiveness vs Chaos Strength", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Plot 2: Autocorrelation decay by τ
    ax = axes[1]
    cmap = plt.cm.viridis
    for i, r in enumerate(results):
        tau = r["tau"]
        data_t = None
        d, _ = generate_mg_data(tau, n_points=args.n_points)
        s = d[:, 0].numpy()
        ac = np.correlate(s - s.mean(), s - s.mean(), mode="full")
        ac = ac[len(ac)//2:] / (ac[len(ac)//2] + 1e-10)
        lags = np.arange(len(ac))
        color = cmap(i / (len(results) - 1))
        ax.plot(lags[:100], ac[:100], linewidth=1.2, color=color,
                label=f"τ={tau:.0f} (lyap={r['lyap']:+.4f})")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Lag (steps)", fontsize=12)
    ax.set_ylabel("Autocorrelation", fontsize=12)
    ax.set_title("Autocorrelation Decay by τ", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 100)

    plt.tight_layout()

    # Save
    output_dir = os.path.join(SCRIPT_DIR, "tau_sweep_output")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "tau_sweep.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {plot_path}")

    # Save JSON
    json_path = os.path.join(output_dir, "tau_sweep_results.json")
    # Convert numpy values
    serializable = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                sr[k] = float(v)
            elif isinstance(v, np.ndarray):
                sr[k] = v.tolist()
            else:
                sr[k] = v
        serializable.append(sr)
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
