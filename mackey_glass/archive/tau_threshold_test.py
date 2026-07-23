#!/usr/bin/env python
"""
Threshold Test: Fix τ=13, H=5. Scan L across τ threshold.
Tests whether FGL benefit vanishes when L ≥ τ (Student can see the delayed term).

Prediction:
  1. L < 13: FGL Δ significantly positive and stable
  2. L ≥ 13: FGL Δ drops to ≈0 or negative (step-like transition at L≈13)
  3. If transition point deviates from L=13, threshold model is imprecise

Usage:
  cd /tmp && source .venv/bin/activate && python mackey_glass/exp/tau_threshold_test.py
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
from scipy import stats
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

TAU = 13      # MG equation delay parameter (fixed)
H = 5         # Student horizon (fixed)
L_VALUES = [5, 7, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25]

# Long sequence: max_L * 1000 ≈ 25000 points to ensure ample data
N_POINTS = 30000


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


def generate_mg_data(n_points=N_POINTS, seed=42):
    """Generate long MG sequence at τ=TAU."""
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


def run_fgl_one(data, L, alpha=0.5, num_bins=50, epochs=50,
                temperature=4, batch_size=128, patience=5, seed=42):
    """Run one FGL experiment. Returns dict with metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden_size = 128
    num_layers = 2
    lr = 1e-4
    output_size = num_bins
    val_size, test_size = 0.2, 0.2

    x_raw = np.array([float(pt[0]) for pt in data])
    y_raw = np.array([float(pt[1]) for pt in data])

    # Check data sufficiency
    n_windows_student = len(x_raw) - L - H + 1
    if n_windows_student < 500:
        print(f"  WARNING: Only {n_windows_student} student windows for L={L}")

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
    abs_imp = b_mse - s_mse
    improvement = abs_imp / b_mse * 100 if b_mse > 0 else 0

    return {"teacher_mse": t_mse, "baseline_mse": b_mse,
            "student_mse": s_mse, "abs_improvement": abs_imp,
            "fgl_delta": improvement}


# ---- Simple changepoint detection (no external lib needed) ----
def find_changepoint(L_sorted, values):
    """Find L value that minimizes total piecewise SSE with a single break."""
    best_L, best_sse = None, float('inf')
    total_sse_single = np.sum((values - values.mean())**2)  # no-break baseline
    for i in range(2, len(L_sorted) - 2):
        left = values[:i]
        right = values[i:]
        sse = np.sum((left - left.mean())**2) + np.sum((right - right.mean())**2)
        if sse < best_sse:
            best_sse = sse
            best_L = (L_sorted[i-1] + L_sorted[i]) / 2
    return best_L, best_sse, total_sse_single


def piecewise_fit(L_arr, y_arr, breakpoint=13):
    """Fit piecewise linear: y = a1 + b1*L (L<bp) + a2 + b2*L (L>=bp)."""
    below = L_arr < breakpoint
    above = L_arr >= breakpoint
    X_below = np.column_stack([np.ones(below.sum()), L_arr[below]])
    X_above = np.column_stack([np.ones(above.sum()), L_arr[above]])
    y_below = y_arr[below]
    y_above = y_arr[above]

    # Fit separately
    coef_below = np.linalg.lstsq(X_below, y_below, rcond=None)[0]
    coef_above = np.linalg.lstsq(X_above, y_above, rcond=None)[0]
    pred = np.concatenate([
        X_below @ coef_below,
        X_above @ coef_above,
    ])
    sse = np.sum((y_arr - pred)**2)
    n_params = 4
    aic = len(y_arr) * np.log(sse / len(y_arr)) + 2 * n_params
    return sse, aic


def main():
    parser = argparse.ArgumentParser(description="MG τ=13 L-threshold test")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    SEEDS = list(range(args.seeds))
    EPOCHS = args.epochs
    total_runs = len(L_VALUES) * len(SEEDS)
    est_minutes = total_runs * 0.6
    print(f"τ = {TAU}, H = {H}")
    print(f"L values: {L_VALUES}")
    print(f"Total runs: {total_runs}, Estimated: ~{est_minutes:.0f} min")
    print()

    # ---- Generate long MG sequence ----
    print(f"Generating MG τ={TAU} sequence ({N_POINTS} points)...")
    data, lyap = generate_mg_data(n_points=N_POINTS, seed=42)
    h2o = data[:, 0].numpy()
    ac = np.correlate(h2o - h2o.mean(), h2o - h2o.mean(), mode="full")
    ac = ac[len(ac)//2:] / (ac[len(ac)//2] + 1e-10)
    periodicity = float(ac[20:200].max()) if len(ac) > 200 else 1.0
    print(f"  Lyapunov={lyap:+.6f}, Periodicity={periodicity:.4f}")
    # Data sufficiency check
    for L in L_VALUES:
        n_win = N_POINTS - L - H + 1
        n_train = int(n_win * 0.6)
        print(f"  L={L:2d}: {n_win} windows, ~{n_train} train samples")
    print()

    # ---- Output ----
    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "threshold_test_results.csv")

    # ---- Run experiments ----
    all_rows = []
    start_time = time.time()

    for L in L_VALUES:
        label = f"L={L}"
        if L < TAU:
            label += " (below τ)"
        elif L == TAU:
            label += " (AT τ)"
        else:
            label += " (above τ)"
        print(f"\n{'='*55}")
        print(f"  {label}")
        print(f"{'='*55}")

        for seed in tqdm(SEEDS, desc=f"  Seeds"):
            r = run_fgl_one(data, L=L, epochs=EPOCHS, seed=seed)
            row = {
                "L": L, "tau": TAU, "H": H, "seed": seed,
                "baseline_mse": r["baseline_mse"],
                "teacher_mse": r["teacher_mse"],
                "student_mse": r["student_mse"],
                "abs_improvement": r["abs_improvement"],
                "fgl_delta": r["fgl_delta"],
            }
            all_rows.append(row)

        elapsed = time.time() - start_time
        print(f"  Elapsed: {elapsed/60:.1f} min")

    # ---- Save CSV ----
    fieldnames = ["L", "tau", "H", "seed", "baseline_mse", "teacher_mse",
                  "student_mse", "abs_improvement", "fgl_delta"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults saved to {csv_path}")

    # ---- Aggregate ----
    agg = defaultdict(list)
    for row in all_rows:
        agg[row["L"]].append(row)

    L_sorted = sorted(agg.keys())
    agg_means = {}
    for L in L_sorted:
        rows_L = agg[L]
        b_mses = [r["baseline_mse"] for r in rows_L]
        abs_imps = [r["abs_improvement"] for r in rows_L]
        deltas = [r["fgl_delta"] for r in rows_L]
        agg_means[L] = {
            "baseline_mean": np.mean(b_mses), "baseline_std": np.std(b_mses),
            "abs_mean": np.mean(abs_imps), "abs_std": np.std(abs_imps),
            "delta_mean": np.mean(deltas), "delta_std": np.std(deltas),
        }

    # ---- Baseline MSE diagnostic ----
    print("\nBaseline MSE check:")
    prev_b = None
    baseline_ok = True
    for L in L_sorted:
        b = agg_means[L]["baseline_mean"]
        flag = ""
        if prev_b is not None:
            ratio = max(b, prev_b) / min(b, prev_b)
            if ratio > 2.0:
                flag = f" ← WARNING: {ratio:.1f}× change from L={prev_L}"
                baseline_ok = False
        print(f"  L={L:2d}: Baseline MSE = {b:.2f} ± {agg_means[L]['baseline_std']:.2f}{flag}")
        prev_b = b
        prev_L = L

    # ---- Statistical tests ----
    below_13 = []
    above_13 = []
    for row in all_rows:
        if row["L"] < TAU:
            below_13.append(row["abs_improvement"])
        else:
            above_13.append(row["abs_improvement"])

    t_stat, p_val = stats.ttest_ind(below_13, above_13, equal_var=False)
    print(f"\nWelch t-test (L<13 vs L≥13 on abs_improvement):")
    print(f"  L<13: mean={np.mean(below_13):.3f}, std={np.std(below_13):.3f}, n={len(below_13)}")
    print(f"  L≥13: mean={np.mean(above_13):.3f}, std={np.std(above_13):.3f}, n={len(above_13)}")
    print(f"  t = {t_stat:.4f}, p = {p_val:.6f}")

    # ---- Changepoint detection ----
    L_arr = np.array(L_sorted)
    abs_means_arr = np.array([agg_means[L]["abs_mean"] for L in L_sorted])
    cp_L, cp_sse, sse_single = find_changepoint(L_arr, abs_means_arr)
    r2_reduction = (sse_single - cp_sse) / sse_single * 100 if sse_single > 0 else 0
    print(f"\nChangepoint detection (abs_improvement vs L):")
    print(f"  Best breakpoint: L ≈ {cp_L:.1f}")
    print(f"  SSE reduction: {r2_reduction:.1f}%")
    print(f"  Deviation from τ=13: {cp_L - TAU:+.1f}")

    # ---- Regression comparison ----
    # Linear: abs_imp = a + b*L
    X_lin = np.column_stack([np.ones(len(L_arr)), L_arr])
    coef_lin = np.linalg.lstsq(X_lin, abs_means_arr, rcond=None)[0]
    pred_lin = X_lin @ coef_lin
    sse_lin = np.sum((abs_means_arr - pred_lin)**2)
    aic_lin = len(L_arr) * np.log(sse_lin / len(L_arr)) + 2 * 2

    # Piecewise at L=13
    bp = TAU
    sse_pw, aic_pw = piecewise_fit(L_arr, abs_means_arr, breakpoint=bp)
    delta_aic = aic_lin - aic_pw

    print(f"\nRegression comparison (abs_improvement vs L):")
    print(f"  Linear model:      SSE={sse_lin:.4f}, AIC={aic_lin:.2f}")
    print(f"  Piecewise (L=13):  SSE={sse_pw:.4f}, AIC={aic_pw:.2f}")
    print(f"  ΔAIC (linear − piecewise): {delta_aic:+.2f}")
    if delta_aic > 10:
        print(f"  → Piecewise model strongly preferred (supports threshold)")
    elif delta_aic > 2:
        print(f"  → Piecewise model weakly preferred")
    else:
        print(f"  → Linear model adequate (does not support sharp threshold)")

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Figure 1: Absolute improvement vs L (primary)
    ax = axes[0]
    ax.errorbar(L_arr, abs_means_arr,
                yerr=[agg_means[L]["abs_std"] for L in L_sorted],
                color="#2ecc71", marker="o", markersize=9, linewidth=2,
                capsize=5, capthick=1.5)
    ax.axvline(x=TAU, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.annotate(f"L=τ={TAU}", xy=(TAU, ax.get_ylim()[1]),
                xytext=(TAU + 0.5, ax.get_ylim()[1] * 0.95),
                fontsize=10, color="red", fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.5)
    # Linear fit
    L_fine = np.linspace(L_arr.min(), L_arr.max(), 100)
    ax.plot(L_fine, coef_lin[0] + coef_lin[1] * L_fine, 'k--', linewidth=0.8, alpha=0.5, label="Linear fit")
    ax.set_xlabel("L (lookback window)", fontsize=12)
    ax.set_ylabel("Absolute improvement (baseline − student)", fontsize=11)
    ax.set_title(f"Fig 1: Abs Improvement vs L (τ={TAU}, H={H})\ndashed=linear fit, red=L=τ threshold",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Figure 2: Baseline MSE vs L (diagnostic)
    ax = axes[1]
    b_means = [agg_means[L]["baseline_mean"] for L in L_sorted]
    b_stds = [agg_means[L]["baseline_std"] for L in L_sorted]
    ax.errorbar(L_arr, b_means, yerr=b_stds,
                color="#3498db", marker="s", markersize=8, linewidth=2,
                capsize=5, capthick=1.5)
    ax.set_xlabel("L (lookback window)", fontsize=12)
    ax.set_ylabel("Baseline MSE", fontsize=12)
    ax.set_title("Fig 2: Baseline MSE vs L (task difficulty check)",
                 fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.2)
    # Check for >2× fluctuations
    b_arr = np.array(b_means)
    for i in range(1, len(b_arr)):
        ratio = max(b_arr[i], b_arr[i-1]) / min(b_arr[i], b_arr[i-1])
        if ratio > 2:
            ax.annotate(f"{ratio:.1f}×", (L_arr[i], b_arr[i]),
                        color="red", fontsize=8, fontweight="bold")

    # Figure 3: Δ% vs L (reference)
    ax = axes[2]
    d_means = [agg_means[L]["delta_mean"] for L in L_sorted]
    d_stds = [agg_means[L]["delta_std"] for L in L_sorted]
    colors = ["#2ecc71" if m > 0 else "#e74c3c" for m in d_means]
    ax.bar(L_arr, d_means, yerr=d_stds, color=colors, edgecolor="white",
           linewidth=0.5, capsize=4, width=1.2)
    ax.axvline(x=TAU, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("L (lookback window)", fontsize=12)
    ax.set_ylabel("FGL Δ (%)", fontsize=12)
    ax.set_title(f"Fig 3: FGL Δ% vs L (τ={TAU}, H={H})\nred=L=τ threshold",
                 fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "threshold_test_figures.png"), dpi=150)
    plt.close(fig)
    print(f"\nFigures saved to {output_dir}/threshold_test_figures.png")

    # ---- Summary Markdown ----
    md_path = os.path.join(output_dir, "threshold_test_summary.md")

    # Judgment
    cp_in_range = abs(cp_L - TAU) <= 1.5 if cp_L is not None else False
    pw_better = delta_aic > 10
    p_sig = p_val < 0.05

    if cp_in_range and pw_better:
        verdict = (
            f"**SUPPORTS threshold hypothesis.**\n\n"
            f"- Changepoint at L≈{cp_L:.1f} (within L=12~14 interval, deviation={cp_L-TAU:+.1f})\n"
            f"- Segmented regression strongly preferred (ΔAIC={delta_aic:+.1f})\n"
            f"- Welch t-test: {'significant' if p_sig else 'not significant'} (p={p_val:.6f})\n\n"
            f"The data supports a step-like transition at L≈τ where FGL benefit vanishes."
        )
    elif cp_in_range and not pw_better:
        verdict = (
            f"**PARTIALLY supports threshold — changepoint aligned but regression ambiguous.**\n\n"
            f"- Changepoint at L≈{cp_L:.1f} (within interval, deviation={cp_L-TAU:+.1f})\n"
            f"- But segmented regression NOT clearly preferred (ΔAIC={delta_aic:+.1f})\n"
            f"- Welch t-test: {'significant' if p_sig else 'not significant'} (p={p_val:.06f})\n\n"
            f"The location matches prediction but the shape of the transition may be more gradual than a sharp step."
        )
    elif not cp_in_range:
        verdict = (
            f"**DOES NOT support simple threshold hypothesis.**\n\n"
            f"- Changepoint at L≈{cp_L:.1f} (deviation from τ={TAU}: {cp_L-TAU:+.1f})\n"
            f"- The transition does NOT occur at L≈τ as predicted\n"
            f"- Effect appears more continuous than step-like\n\n"
            f"The threshold model at L=τ is imprecise; a continuous decay model (effect ∝ τ−L) may be more appropriate."
        )
    else:
        verdict = (
            f"**AMBIGUOUS — mixed evidence.**\n\n"
            f"- Changepoint at L≈{cp_L:.1f} (deviation={cp_L-TAU:+.1f})\n"
            f"- ΔAIC = {delta_aic:+.1f}\n"
            f"- p = {p_val:.6f}\n"
        )

    if not baseline_ok:
        verdict += (
            f"\n\n**CAVEAT: Task difficulty was NOT adequately controlled.** "
            f"Baseline MSE fluctuated >2× between adjacent L values. "
            f"Results should be interpreted with caution. "
            f"Future experiments need longer sequences or different data generation."
        )

    with open(md_path, "w") as f:
        f.write("# MG L-Threshold Test — τ=13 fixed, H=5, L scan\n\n")
        f.write(f"**Run date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**τ:** {TAU} (MG equation delay, fixed)\n")
        f.write(f"**H:** {H} (student horizon, fixed)\n")
        f.write(f"**L values:** {L_VALUES}\n")
        f.write(f"**Seeds:** {SEEDS}\n")
        f.write(f"**Epochs:** {EPOCHS}\n")
        f.write(f"**MG sequence length:** {N_POINTS}\n")
        f.write(f"**Total runs:** {total_runs}\n\n")

        f.write("## Predictions (pre-registered)\n\n")
        f.write("1. L < τ: Δ significantly positive and stable\n")
        f.write("2. L ≥ τ: Δ drops to ≈0 or negative (step at L≈τ)\n")
        f.write("3. Deviation from L=τ → threshold model imprecise\n\n")

        f.write("## Results\n\n")
        f.write("| L | Baseline MSE | Abs Improvement | FGL Δ% |\n")
        f.write("|:---:|:---:|:---:|:---:|\n")
        for L in L_sorted:
            a = agg_means[L]
            marker = " ← τ" if L == TAU else (" < τ" if L < TAU else " > τ")
            f.write(f"| **{L}**{marker} | {a['baseline_mean']:.3f} ± {a['baseline_std']:.3f} | "
                    f"{a['abs_mean']:+.3f} ± {a['abs_std']:.3f} | "
                    f"{a['delta_mean']:+.1f}% ± {a['delta_std']:.1f}% |\n")

        f.write("\n## Statistical Tests\n\n")
        f.write(f"**Welch t-test** (L<{TAU} vs L≥{TAU} on abs_improvement):\n")
        f.write(f"- L<{TAU}: mean={np.mean(below_13):.4f}, std={np.std(below_13):.4f}, n={len(below_13)}\n")
        f.write(f"- L≥{TAU}: mean={np.mean(above_13):.4f}, std={np.std(above_13):.4f}, n={len(above_13)}\n")
        f.write(f"- t = {t_stat:.4f}, p = {p_val:.6f}\n\n")

        f.write(f"**Changepoint detection:**\n")
        f.write(f"- Best breakpoint: L ≈ {cp_L:.1f}\n")
        f.write(f"- Deviation from τ={TAU}: {cp_L - TAU:+.1f}\n")
        f.write(f"- SSE reduction vs single-line: {r2_reduction:.1f}%\n\n")

        f.write(f"**Regression comparison:**\n")
        f.write(f"- Linear model AIC = {aic_lin:.2f}\n")
        f.write(f"- Piecewise model AIC = {aic_pw:.2f}\n")
        f.write(f"- ΔAIC = {delta_aic:+.2f}\n\n")

        f.write(f"**Baseline MSE control:** {'PASSED' if baseline_ok else 'FAILED — >2× fluctuations detected'}\n\n")

        f.write("## Judgment\n\n")
        f.write(verdict)
        f.write(f"\n\n## Full data: `threshold_test_results.csv` ({total_runs} rows)\n")

    print(f"Summary saved to {md_path}")

    # ---- Console summary ----
    print("\n" + "=" * 55)
    print("  RESULTS: L threshold test (τ=13, H=5)")
    print("=" * 55)
    for L in L_sorted:
        a = agg_means[L]
        marker = " ← τ" if L == TAU else ""
        print(f"  L={L:2d}{marker}: Abs={a['abs_mean']:+.3f}±{a['abs_std']:.3f}  "
              f"Δ={a['delta_mean']:+.1f}%±{a['delta_std']:.1f}%  "
              f"Base={a['baseline_mean']:.3f}")
    print(f"\n  t-test: t={t_stat:.2f}, p={p_val:.6f}")
    print(f"  Changepoint: L≈{cp_L:.1f}  |  ΔAIC={delta_aic:+.1f}")
    print(f"  Baseline control: {'OK' if baseline_ok else 'FAILED'}")
    print(f"\n  Verdict: {verdict.split(chr(10))[0]}")
    print(f"\n  All outputs in: {output_dir}/")


if __name__ == "__main__":
    main()
