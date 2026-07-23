#!/usr/bin/env python
"""
Geometry test: Is the MG τ=13 FGL sweet spot caused by bifurcation dynamics,
or by the geometric condition τ ≈ L+H-1?

Tests 4 (L,H) configs × ~13 τ values × 5 seeds = ~260 experiments.

Configs (L, H, L+H-1):
  config_A (original):  (8, 5, 12)
  config_B (longer L):  (12, 5, 16)
  config_C (longer H):  (8, 9, 16)
  config_D (shorter L): (5, 5, 9)

Two theories make opposite predictions:
  ① Dynamics: peak FGL Δ always at τ≈13 regardless of L,H
  ② Geometry: peak FGL Δ tracks L+H-1 (moves with config)

Usage:
  cd /tmp && source .venv/bin/activate && python mackey_glass/exp/tau_sweep_geometry.py
  python mackey_glass/exp/tau_sweep_geometry.py --quick  # fast mode: 3 seeds, 30 epochs
"""

import sys
import os
import csv
import json
import time
import argparse
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, MG_DIR)

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
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


# ==================== Configs ====================
CONFIGS = [
    {"name": "config_A", "L": 8,  "H": 5,  "label": "(8,5) L+H-1=12 [original]"},
    {"name": "config_B", "L": 12, "H": 5,  "label": "(12,5) L+H-1=16 [longer L]"},
    {"name": "config_C", "L": 8,  "H": 9,  "label": "(8,9) L+H-1=16 [longer H]"},
    {"name": "config_D", "L": 5,  "H": 5,  "label": "(5,5) L+H-1=9 [shorter L+H]"},
]

# Reference τ values from original sweep, plus geometric neighborhood
REF_TAUS = [10, 13, 17, 23, 30]
NEIGHBOR_RADIUS = 5


def build_tau_list(L, H):
    """Build τ list: geometric neighborhood of L+H-1 ± radius, plus reference τs."""
    center = L + H - 1
    geo_taus = list(range(max(5, center - NEIGHBOR_RADIUS),
                          min(35, center + NEIGHBOR_RADIUS + 1)))
    all_taus = sorted(set(REF_TAUS + geo_taus))
    return all_taus


# ==================== Early Stopper ====================
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


# ==================== MG Data Generation ====================
def generate_mg_data(tau, n_points=10000, dt=1.0, seed=42):
    """Generate MG data. Returns (tensor, lyap_exp)."""
    mg = MackeyGlass(
        tau=tau, constant_past=0.9,
        nmg=10, beta=0.2, gamma=0.1,
        dt=dt, splits=(float(n_points), 0.), seed_id=seed,
    )
    vals = []
    for idx in range(len(mg)):
        _, target = mg[idx]
        vals.append(target.squeeze().item())
    col = torch.tensor(vals, dtype=torch.float64).unsqueeze(1)
    tensor = torch.cat((col, col.clone()), dim=1)
    return tensor, mg.lyap_exp


def periodicity_score(series):
    s = series - series.mean()
    ac = np.correlate(s, s, mode="full")
    ac = ac[len(ac)//2:] / (ac[len(ac)//2] + 1e-10)
    return float(ac[20:200].max()) if len(ac) > 20 else 1.0


# ==================== Single FGL Run ====================
def run_fgl_one(data, L, H, alpha=0.5, num_bins=50, epochs=50,
                temperature=4, batch_size=128, patience=5, seed=42):
    """Run one FGL experiment. Returns dict with metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden_size = 128
    num_layers = 2
    lr = 1e-4
    output_size = num_bins
    val_size, test_size = 0.2, 0.2

    # Shared bin edges
    x_raw = np.array([float(pt[0]) for pt in data])
    y_raw = np.array([float(pt[1]) for pt in data])
    all_y = []
    for i in range(len(x_raw) - L - 1 + 1):
        all_y.append(y_raw[i + L])
    shared_edges = np.linspace(np.array(all_y).min(), np.array(all_y).max(),
                               num_bins - 1)

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

    return {"teacher_mse": t_mse, "baseline_mse": b_mse,
            "student_mse": s_mse, "fgl_delta": improvement}


# ==================== Main Sweep ====================
def main():
    parser = argparse.ArgumentParser(description="MG τ geometry test")
    parser.add_argument("--quick", action="store_true",
                        help="Fast mode: 3 seeds, 30 epochs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n_points", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    if args.quick:
        args.seeds = 3
        args.epochs = 30
        print("[QUICK MODE] 3 seeds, 30 epochs")

    SEEDS = list(range(args.seeds))
    EPOCHS = args.epochs

    # ---- Estimate total work ----
    total_runs = 0
    for cfg in CONFIGS:
        taus = build_tau_list(cfg["L"], cfg["H"])
        total_runs += len(taus) * len(SEEDS)

    # Rough timing: ~2s per FGL run on MPS
    est_seconds = total_runs * 2.5
    est_minutes = est_seconds / 60
    print(f"Total experiments: {len(CONFIGS)} configs × ~{len(build_tau_list(CONFIGS[0]['L'], CONFIGS[0]['H']))} τ × {len(SEEDS)} seeds = ~{total_runs}")
    print(f"Estimated time: ~{est_minutes:.0f} min ({est_seconds/60:.1f} min)")
    if est_minutes > 120:
        print("WARNING: >2h estimated. Consider --quick for initial screening.")
    print()

    # ---- Output setup ----
    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "geometry_test_results.csv")

    # ---- Run experiments ----
    all_rows = []
    start_time = time.time()

    for cfg in CONFIGS:
        L, H = cfg["L"], cfg["H"]
        taus = build_tau_list(L, H)
        center = L + H - 1
        print(f"\n{'='*60}")
        print(f"  {cfg['name']}: L={L}, H={H}, L+H-1={center}")
        print(f"  τ values: {taus}")
        print(f"{'='*60}")

        # Pre-generate MG data for each τ (same across seeds)
        mg_cache = {}
        for tau in tqdm(taus, desc=f"Generating MG data ({cfg['name']})"):
            data, lyap = generate_mg_data(tau, n_points=args.n_points, seed=42)
            mg_cache[tau] = (data, lyap)

        for tau in taus:
            data, lyap = mg_cache[tau]
            h2o = data[:, 0].numpy()
            periodicity = periodicity_score(h2o)

            for seed in tqdm(SEEDS, desc=f"  {cfg['name']} τ={tau}", leave=False):
                r = run_fgl_one(data, L=L, H=H, epochs=EPOCHS, seed=seed,
                                alpha=args.alpha)
                row = {
                    "config_name": cfg["name"],
                    "L": L, "H": H,
                    "L_plus_H_minus_1": center,
                    "tau": tau,
                    "seed": seed,
                    "baseline_mse": r["baseline_mse"],
                    "teacher_mse": r["teacher_mse"],
                    "student_mse": r["student_mse"],
                    "fgl_delta": r["fgl_delta"],
                    "lyapunov": lyap,
                    "periodicity": periodicity,
                }
                all_rows.append(row)

        elapsed = time.time() - start_time
        print(f"  Elapsed: {elapsed/60:.1f} min")

    # ---- Save CSV ----
    fieldnames = ["config_name", "L", "H", "L_plus_H_minus_1", "tau", "seed",
                  "baseline_mse", "teacher_mse", "student_mse", "fgl_delta",
                  "lyapunov", "periodicity"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults saved to {csv_path}")

    # ---- Aggregate: mean ± std per (config, tau) ----
    from collections import defaultdict
    agg = defaultdict(list)
    for row in all_rows:
        key = (row["config_name"], row["tau"])
        agg[key].append(row["fgl_delta"])

    agg_table = {}
    for (cfg_name, tau), deltas in agg.items():
        agg_table[(cfg_name, tau)] = {
            "mean": np.mean(deltas), "std": np.std(deltas),
            "n": len(deltas),
        }

    # ---- Find peak per config ----
    peak_info = {}
    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        center = cfg["L"] + cfg["H"] - 1
        best_tau, best_val = None, -999
        for tau in build_tau_list(cfg["L"], cfg["H"]):
            if (cfg_name, tau) in agg_table:
                m = agg_table[(cfg_name, tau)]["mean"]
                if m > best_val:
                    best_val = m
                    best_tau = tau
        if best_tau is not None:
            peak_info[cfg_name] = {
                "tau_peak": best_tau, "delta_peak": best_val,
                "L_plus_H_minus_1": center,
                "offset": best_tau - center,
            }

    # ---- Figure 1: Absolute τ ----
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"config_A": "#2ecc71", "config_B": "#3498db",
              "config_C": "#e74c3c", "config_D": "#9b59b6"}
    markers = {"config_A": "o", "config_B": "s", "config_C": "^", "config_D": "D"}

    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        center = cfg["L"] + cfg["H"] - 1
        taus = build_tau_list(cfg["L"], cfg["H"])
        means, stds, xs = [], [], []
        for tau in taus:
            if (cfg_name, tau) in agg_table:
                xs.append(tau)
                means.append(agg_table[(cfg_name, tau)]["mean"])
                stds.append(agg_table[(cfg_name, tau)]["std"])
        ax.errorbar(xs, means, yerr=stds, color=colors[cfg_name],
                    marker=markers[cfg_name], markersize=7, linewidth=1.5,
                    capsize=4, label=f"{cfg['label']}")

    # Mark τ=13 for reference
    ax.axvline(x=13, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.annotate("τ=13 (original peak)", xy=(13, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 80),
                xytext=(13.5, 75), fontsize=9, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("τ (absolute)", fontsize=12)
    ax.set_ylabel("FGL Δ (%)", fontsize=12)
    ax.set_title("Figure 1: Absolute τ — Theory ① predicts all peaks at τ≈13", fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_absolute_tau.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 2: Relative τ ----
    fig, ax = plt.subplots(figsize=(12, 6))

    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        center = cfg["L"] + cfg["H"] - 1
        taus = build_tau_list(cfg["L"], cfg["H"])
        means, stds, xs_rel = [], [], []
        for tau in taus:
            if (cfg_name, tau) in agg_table:
                xs_rel.append(tau - center)
                means.append(agg_table[(cfg_name, tau)]["mean"])
                stds.append(agg_table[(cfg_name, tau)]["std"])
        ax.errorbar(xs_rel, means, yerr=stds, color=colors[cfg_name],
                    marker=markers[cfg_name], markersize=7, linewidth=1.5,
                    capsize=4, label=f"{cfg['label']}")

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.annotate("τ = L+H-1", xy=(0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 80),
                xytext=(0.5, 75), fontsize=9, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("τ − (L+H−1)", fontsize=12)
    ax.set_ylabel("FGL Δ (%)", fontsize=12)
    ax.set_title("Figure 2: Relative offset — Theory ② predicts all peaks align at same offset",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_relative_offset.png"), dpi=150)
    plt.close(fig)

    # ---- Summary markdown ----
    md_path = os.path.join(output_dir, "geometry_test_summary.md")
    with open(md_path, "w") as f:
        f.write("# MG τ Geometry Test — Summary\n\n")
        f.write(f"**Run date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Seeds:** {SEEDS}\n")
        f.write(f"**Epochs per run:** {EPOCHS}\n")
        f.write(f"**Total experiments:** {len(all_rows)}\n\n")

        f.write("## Peak Analysis per Config\n\n")
        f.write("| Config | L | H | L+H−1 | τ_peak | Δ_peak | offset = τ_peak − (L+H−1) |\n")
        f.write("|--------|---|---|:---:|:---:|:---:|:---:|\n")
        offsets = []
        for cfg in CONFIGS:
            cfg_name = cfg["name"]
            if cfg_name in peak_info:
                pi = peak_info[cfg_name]
                f.write(f"| {cfg_name} | {cfg['L']} | {cfg['H']} | {pi['L_plus_H_minus_1']} | "
                        f"{pi['tau_peak']} | {pi['delta_peak']:+.1f}% | {pi['offset']:+d} |\n")
                offsets.append(pi["offset"])

        f.write("\n")

        # Judgment
        if len(offsets) >= 3:
            offset_std = np.std(offsets)
            offset_range = max(offsets) - min(offsets)
            tau_peaks = [peak_info[c["name"]]["tau_peak"] for c in CONFIGS if c["name"] in peak_info]
            tau_range = max(tau_peaks) - min(tau_peaks)

            f.write("## Judgment\n\n")

            if offset_std < 2 and tau_range > 3:
                f.write("**Result: SUPPORTS Theory ② (Geometry Resonance).**\n\n")
                f.write(f"- Offset std = {offset_std:.2f} (< 2): peak positions align in relative coordinates\n")
                f.write(f"- τ_peak range = {tau_range}: peaks moved with L+H-1 across configs\n")
                f.write("- The original τ=13 sweet spot is likely a hyperparameter geometric artifact.\n")
                f.write("- Recommendation: redesign experiments to decouple τ from (L,H).\n")
            elif tau_range <= 2:
                f.write("**Result: SUPPORTS Theory ① (Dynamical Bifurcation).**\n\n")
                f.write(f"- τ_peak range = {tau_range} (≤ 2): all peaks near τ≈13 regardless of L,H config\n")
                f.write("- The sweet spot is robust against hyperparameter changes.\n")
                f.write("- The 'feedback-loop complexity' theory stands.\n")
            else:
                f.write("**Result: AMBIGUOUS — neither theory fully explains the data.**\n\n")
                f.write(f"- τ_peak range = {tau_range} (peaks partly moved)\n")
                f.write(f"- Offset std = {offset_std:.2f} (peaks partly aligned in relative coords)\n")
                f.write("- Both geometric and dynamical factors may contribute.\n")
                f.write("- Recommend: further experiments to disentangle, e.g. fixing τ=13 and varying L+H-1 across a wider range.\n")

        f.write(f"\n## Full data: `geometry_test_results.csv` ({len(all_rows)} rows)\n")

    print(f"Summary saved to {md_path}")

    # ---- Print summary to console ----
    print("\n" + "=" * 60)
    print("  PEAK ANALYSIS")
    print("=" * 60)
    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        if cfg_name in peak_info:
            pi = peak_info[cfg_name]
            print(f"  {cfg_name}: τ_peak={pi['tau_peak']}, Δ={pi['delta_peak']:+.1f}%, "
                  f"offset={pi['offset']:+d}")
    print(f"\n  All outputs in: {output_dir}/")


if __name__ == "__main__":
    main()
