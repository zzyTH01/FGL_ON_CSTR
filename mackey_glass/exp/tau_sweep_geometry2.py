#!/usr/bin/env python
"""
Geometry Test 2: Fix τ=13, scan L+H-1 to test if geometric alignment independently
affects FGL Δ magnitude.

If geometric alignment matters: FGL Δ should peak when L+H-1 ≈ τ (=13).
If dynamics dominate: FGL Δ should be consistently high regardless of L+H-1.

7 configs × 5 seeds × 50 epochs = 35 full FGL trainings.

Usage:
  cd /tmp && source .venv/bin/activate && python mackey_glass/exp/tau_sweep_geometry2.py
"""

import sys
import os
import csv
import time
import argparse
from datetime import datetime
from collections import defaultdict

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
print(f"Using {device}")

TAU = 13  # Fixed τ at the bifurcation sweet spot

# ---- (L,H) configs covering L+H-1 from 5 to 30 ----
CONFIGS = [
    {"L": 3,  "H": 3},   # L+H-1 = 5,  distance = -8
    {"L": 5,  "H": 5},   # L+H-1 = 9,  distance = -4
    {"L": 7,  "H": 7},   # L+H-1 = 13, distance = 0  (aligned)
    {"L": 9,  "H": 9},   # L+H-1 = 17, distance = +4
    {"L": 11, "H": 11},  # L+H-1 = 21, distance = +8
    {"L": 13, "H": 13},  # L+H-1 = 25, distance = +12
    {"L": 15, "H": 16},  # L+H-1 = 30, distance = +17
]


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


def generate_mg_data(n_points=10000, seed=42):
    """Generate MG data at τ=TAU. Returns (tensor, lyap_exp)."""
    mg = MackeyGlass(
        tau=TAU, constant_past=0.9,
        nmg=10, beta=0.2, gamma=0.1,
        dt=1.0, splits=(float(n_points), 0.), seed_id=seed,
    )
    vals = []
    for idx in range(len(mg)):
        _, target = mg[idx]
        vals.append(target.squeeze().item())
    col = torch.tensor(vals, dtype=torch.float64).unsqueeze(1)
    tensor = torch.cat((col, col.clone()), dim=1)
    return tensor, mg.lyap_exp


def run_fgl_one(data, L, H, alpha=0.5, num_bins=50, epochs=50,
                temperature=4, batch_size=128, patience=5, seed=42):
    """Run one FGL experiment at τ=13. Returns dict with metrics."""
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


def main():
    parser = argparse.ArgumentParser(description="MG τ=13 geometry test 2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n_points", type=int, default=10000)
    args = parser.parse_args()

    SEEDS = list(range(args.seeds))
    EPOCHS = args.epochs

    total_runs = len(CONFIGS) * len(SEEDS)
    est_minutes = total_runs * 1.0  # ~1 min per full run (50 epochs)
    print(f"τ = {TAU} (fixed)")
    print(f"Configs: {len(CONFIGS)}, Seeds: {len(SEEDS)}")
    print(f"Total runs: {total_runs}, Estimated: ~{est_minutes:.0f} min")
    print()

    # ---- Pre-generate MG data once (same across configs) ----
    print("Generating MG τ=13 data (shared across all configs)...")
    data, lyap = generate_mg_data(n_points=args.n_points, seed=42)
    h2o = data[:, 0].numpy()
    ac = np.correlate(h2o - h2o.mean(), h2o - h2o.mean(), mode="full")
    ac = ac[len(ac)//2:] / (ac[len(ac)//2] + 1e-10)
    periodicity = float(ac[20:200].max()) if len(ac) > 200 else 1.0
    print(f"  Lyapunov={lyap:+.6f}, Periodicity={periodicity:.4f}")
    print()

    # ---- Output ----
    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "geometry_test2_results.csv")

    # ---- Run experiments ----
    all_rows = []
    start_time = time.time()

    for cfg in CONFIGS:
        L, H = cfg["L"], cfg["H"]
        center = L + H - 1
        distance = center - TAU
        label = f"L={L},H={H} (L+H-1={center}, dist={distance:+d})"
        print(f"\n{'='*55}")
        print(f"  {label}")
        print(f"{'='*55}")

        for seed in tqdm(SEEDS, desc=f"  Seeds"):
            r = run_fgl_one(data, L=L, H=H, epochs=EPOCHS, seed=seed)
            row = {
                "L": L, "H": H,
                "L_plus_H_minus_1": center,
                "distance_from_13": distance,
                "seed": seed,
                "baseline_mse": r["baseline_mse"],
                "teacher_mse": r["teacher_mse"],
                "student_mse": r["student_mse"],
                "fgl_delta": r["fgl_delta"],
            }
            all_rows.append(row)

        elapsed = time.time() - start_time
        print(f"  Elapsed: {elapsed/60:.1f} min")

    # ---- Save CSV ----
    fieldnames = ["L", "H", "L_plus_H_minus_1", "distance_from_13", "seed",
                  "baseline_mse", "teacher_mse", "student_mse", "fgl_delta"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults saved to {csv_path}")

    # ---- Aggregate ----
    agg = defaultdict(list)
    for row in all_rows:
        key = row["L_plus_H_minus_1"]
        agg[key].append(row["fgl_delta"])

    agg_table = {}
    for center, deltas in agg.items():
        agg_table[center] = {
            "mean": np.mean(deltas), "std": np.std(deltas),
            "distance": center - TAU,
        }

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(10, 6))
    centers = sorted(agg_table.keys())
    distances = [agg_table[c]["distance"] for c in centers]
    means = [agg_table[c]["mean"] for c in centers]
    stds = [agg_table[c]["std"] for c in centers]

    ax.errorbar(distances, means, yerr=stds, color="#2ecc71", marker="o",
                markersize=10, linewidth=2, capsize=5, capthick=1.5,
                label=f"τ={TAU} (fixed)")

    # Annotate each point with (L,H)
    for cfg, center, m in zip(CONFIGS, distances, means):
        ax.annotate(f"({cfg['L']},{cfg['H']})", (center, m),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=9, ha="center")

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.annotate("Geometric alignment\n(L+H-1 = τ = 13)", xy=(0, ax.get_ylim()[1]),
                xytext=(2, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 60),
                fontsize=9, color="gray")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("L+H−1 − 13 (distance from geometric alignment)", fontsize=12)
    ax.set_ylabel("FGL Δ (%)", fontsize=12)
    ax.set_title(f"Geometry Test 2: τ={TAU} fixed, varying L+H−1\n"
                 "Flat curve = dynamics dominate  |  Peak at 0 = geometry matters",
                 fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "geometry_test2_figure.png"), dpi=150)
    plt.close(fig)
    print(f"Figure saved to {output_dir}/geometry_test2_figure.png")

    # ---- Summary Markdown ----
    md_path = os.path.join(output_dir, "geometry_test2_summary.md")

    # Key comparison: distance=0 vs distance=±8
    d0_center = 13
    d_neg8_center = 5   # distance=-8
    d_pos8_center = 21  # distance=+8

    d0_mean = agg_table.get(d0_center, {}).get("mean", None)
    d0_std = agg_table.get(d0_center, {}).get("std", None)
    d_neg8_mean = agg_table.get(d_neg8_center, {}).get("mean", None)
    d_neg8_std = agg_table.get(d_neg8_center, {}).get("std", None)
    d_pos8_mean = agg_table.get(d_pos8_center, {}).get("mean", None)
    d_pos8_std = agg_table.get(d_pos8_center, {}).get("std", None)

    # Determine which scenario
    judgment = ""
    if d0_mean is not None and d_neg8_mean is not None and d_pos8_mean is not None:
        d0_lower = d0_mean - d0_std
        d0_upper = d0_mean + d0_std
        d8_avg = (d_neg8_mean + d_pos8_mean) / 2

        # Check "flat" scenario: error bars overlap
        flat_neg = abs(d0_mean - d_neg8_mean) < max(d0_std, d_neg8_std) * 1.5
        flat_pos = abs(d0_mean - d_pos8_mean) < max(d0_std, d_pos8_std) * 1.5

        # Check "peak at 0" scenario: d0 significantly higher
        peak_neg = d0_mean > d_neg8_mean + d_neg8_std + d0_std  # error bars don't overlap
        peak_pos = d0_mean > d_pos8_mean + d_pos8_std + d0_std

        ratio_neg = d0_mean / d_neg8_mean if d_neg8_mean > 0 else float('inf')
        ratio_pos = d0_mean / d_pos8_mean if d_pos8_mean > 0 else float('inf')

        if flat_neg and flat_pos:
            judgment = (
                "**Result: FLAT CURVE — dynamics dominate.**\n\n"
                "FGL Δ at distance=0 is NOT significantly higher than at distance=±8. "
                "The error bars overlap substantially. "
                "Once τ is in the dynamical sweet spot (τ=13), the magnitude of FGL benefit "
                "is robust and does NOT depend on geometric alignment of L+H-1 with τ.\n\n"
                "This is a **stronger** confirmation of the dynamical bifurcation theory "
                "than Test 1 — it directly shows that geometric alignment has no independent "
                "effect on FGL Δ magnitude, even when τ itself is at the optimal value."
            )
        elif peak_neg or peak_pos:
            judgment = (
                "**Result: PEAK AT DISTANCE=0 — geometric alignment matters independently.**\n\n"
                "FGL Δ at distance=0 is significantly higher than at distance=±8 "
                "(error bars do not overlap). This means geometric alignment of L+H-1 with τ "
                "contributes to FGL Δ magnitude independently of τ's dynamical state.\n\n"
                "The ~1/3 geometry + 2/3 dynamics estimate from Test 1 is confirmed "
                "and can be quantified more precisely in the magnitude dimension."
            )
        else:
            judgment = (
                "**Result: NEITHER FLAT NOR SIMPLE PEAK — additional variables at play.**\n\n"
                "The curve does not cleanly fit either the 'flat' or 'peak at 0' pattern. "
                "There may be other variables (e.g., L and H individually, not just their sum) "
                "that affect FGL Δ at fixed τ. This warrants further investigation."
            )

    with open(md_path, "w") as f:
        f.write("# MG Geometry Test 2 — τ=13 fixed, L+H−1 scan\n\n")
        f.write(f"**Run date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**τ:** {TAU} (fixed)\n")
        f.write(f"**Seeds:** {SEEDS}\n")
        f.write(f"**Epochs per run:** {EPOCHS}\n")
        f.write(f"**Total runs:** {total_runs}\n\n")

        f.write("## Results\n\n")
        f.write("| L | H | L+H−1 | distance from 13 | FGL Δ (mean ± std) |\n")
        f.write("|---|---|:---:|:---:|:---:|\n")
        for c in sorted(agg_table.keys()):
            a = agg_table[c]
            f.write(f"| {a.get('L', '?')} | {a.get('H', '?')} | {c} | {c-13:+d} | "
                    f"{a['mean']:+.1f}% ± {a['std']:.1f}% |\n")

        f.write("\n## Key Comparison\n\n")
        if d0_mean is not None:
            f.write(f"- distance=0 (L+H-1=13): Δ = {d0_mean:+.1f}% ± {d0_std:.1f}%\n")
        if d_neg8_mean is not None:
            f.write(f"- distance=−8 (L+H-1=5): Δ = {d_neg8_mean:+.1f}% ± {d_neg8_std:.1f}%\n")
        if d_pos8_mean is not None:
            f.write(f"- distance=+8 (L+H-1=21): Δ = {d_pos8_mean:+.1f}% ± {d_pos8_std:.1f}%\n")
        f.write("\n")

        if judgment:
            f.write("## Judgment\n\n")
            f.write(judgment)
            f.write("\n")
            if d0_mean and d_neg8_mean and d_pos8_mean:
                f.write(f"\n**Numerical basis:**\n")
                f.write(f"- distance=0: {d0_mean:+.1f}% ± {d0_std:.1f}%\n")
                f.write(f"- distance=−8: {d_neg8_mean:+.1f}% ± {d_neg8_std:.1f}%\n")
                f.write(f"- distance=+8: {d_pos8_mean:+.1f}% ± {d_pos8_std:.1f}%\n")
                f.write(f"- Ratio (d0 / avg(d±8)): {d0_mean / max(d8_avg, 0.01):.2f}×\n")

        f.write(f"\n## Full data: `geometry_test2_results.csv` ({total_runs} rows)\n")

    print(f"Summary saved to {md_path}")

    # ---- Console summary ----
    print("\n" + "=" * 55)
    print("  RESULTS: τ=13 fixed, L+H-1 scan")
    print("=" * 55)
    for c in sorted(agg_table.keys()):
        a = agg_table[c]
        marker = " ← ALIGNED" if c == TAU else ""
        print(f"  L+H-1={c:2d} (dist={c-13:+d}): Δ = {a['mean']:+.1f}% ± {a['std']:.1f}%{marker}")
    print()
    if judgment:
        first_line = judgment.split("\n")[0]
        print(f"  {first_line}")
    print(f"\n  All outputs in: {output_dir}/")


if __name__ == "__main__":
    main()
