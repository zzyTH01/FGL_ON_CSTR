#!/usr/bin/env python
"""
H-Threshold Test: Fix τ=13, L=5. Scan H across the critical point L+H-1=τ → H=9.

Symmetric counterpart to threshold_test (fixed H=5, scanned L).
Together they test the formula: FGL benefit vanishes when L+H-1 ≥ τ.

Prediction: Step-like drop at H=9 (where L+H-1 = 5+9-1 = 13 = τ).

Usage:
  cd /tmp && source .venv/bin/activate && python mackey_glass/exp/h_threshold_test.py
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

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using {device}")

TAU = 13
L = 5  # Fixed lookback
H_VALUES = [2, 4, 6, 7, 8, 9, 10, 11, 12, 14, 16]
N_POINTS = 10000


class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience; self.min_delta = min_delta
        self.best_loss = float("inf"); self.counter = 0; self.best_state = None
    def step(self, current_loss, model):
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss = current_loss; self.counter = 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return False
        self.counter += 1; return self.counter >= self.patience
    def restore(self, model):
        if self.best_state: model.load_state_dict(self.best_state)


def run_fgl_one(data, H, alpha=0.5, num_bins=50, epochs=50,
                temperature=4, batch_size=128, patience=5, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    hidden_size, num_layers, lr = 128, 2, 1e-4
    output_size, val_size, test_size = num_bins, 0.2, 0.2

    x_raw = np.array([float(pt[0]) for pt in data])
    y_raw = np.array([float(pt[1]) for pt in data])
    all_y = [y_raw[i + L] for i in range(len(x_raw) - L - 1 + 1)]
    shared_edges = np.linspace(np.array(all_y).min(), np.array(all_y).max(), num_bins - 1)

    teacher_train, teacher_val, teacher_test, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=1,
        num_bins=num_bins, val_size=val_size, test_size=test_size,
        offset=H - 1, batch_size=batch_size, bin_edges=shared_edges)
    student_train, student_val, student_test, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H,
        num_bins=num_bins, val_size=val_size, test_size=test_size,
        offset=0, batch_size=batch_size, bin_edges=shared_edges)

    celoss = nn.CrossEntropyLoss(); mse = nn.MSELoss()

    def train_model(model, loader, val_loader, opt):
        es = EarlyStopper(patience=patience)
        for _ in range(epochs):
            model.train()
            for _, x, y in loader:
                x = x.float().to(device).view(-1, 1, L); y = y.long().to(device)
                opt.zero_grad(); celoss(model(x), y).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vl = sum(celoss(model(x.float().to(device).view(-1, 1, L)),
                                y.long().to(device)).item() for _, x, y in val_loader) / len(val_loader)
            if es.step(vl, model): break
        es.restore(model)

    teacher = RNN(L, hidden_size, output_size, num_layers).to(device)
    train_model(teacher, teacher_train, teacher_val, torch.optim.Adam(teacher.parameters(), lr=lr))

    baseline = RNN(L, hidden_size, output_size, num_layers).to(device)
    train_model(baseline, student_train, student_val, torch.optim.Adam(baseline.parameters(), lr=lr))

    student = RNN(L, hidden_size, output_size, num_layers).to(device)
    opt_s = torch.optim.Adam(student.parameters(), lr=lr)
    es_s = EarlyStopper(patience=patience)
    for _ in range(epochs):
        student.train()
        for (_, xs, ys), (_, xt, _) in zip(student_train, teacher_train):
            xs = xs.float().to(device).view(-1, 1, L); ys = ys.long().to(device)
            out = student(xs)
            xt = xt.float().to(device).view(-1, 1, L)
            with torch.no_grad(): lt = teacher(xt)
            loss = alpha * celoss(out, ys) + KL(out, lt, temperature, alpha)
            opt_s.zero_grad(); loss.backward(); opt_s.step()
        student.eval()
        with torch.no_grad():
            vl = sum(celoss(student(x.float().to(device).view(-1, 1, L)),
                            y.long().to(device)).item() for _, x, y in student_val) / len(student_val)
        if es_s.step(vl, student): break
    es_s.restore(student)

    def ev(model, loader):
        model.eval(); t = 0.0
        with torch.no_grad():
            for _, x, y in loader:
                x = x.float().to(device).view(-1, 1, L)
                t += mse(model(x).argmax(1).float(), y.float().to(device).squeeze(-1)).item()
        return t / len(loader)

    bm = ev(baseline, student_test); sm = ev(student, student_test); tm = ev(teacher, teacher_test)
    return {"teacher_mse": tm, "baseline_mse": bm, "student_mse": sm,
            "abs_improvement": bm - sm, "fgl_delta": (bm - sm) / bm * 100 if bm > 0 else 0}


def find_changepoint(x_sorted, values):
    best_x, best_sse = None, float('inf')
    sse_single = np.sum((values - values.mean())**2)
    for i in range(2, len(x_sorted) - 2):
        left, right = values[:i], values[i:]
        sse = np.sum((left - left.mean())**2) + np.sum((right - right.mean())**2)
        if sse < best_sse: best_sse = sse; best_x = (x_sorted[i-1] + x_sorted[i]) / 2
    return best_x, best_sse, sse_single


def piecewise_fit(x_arr, y_arr, bp):
    below, above = x_arr < bp, x_arr >= bp
    Xb = np.column_stack([np.ones(below.sum()), x_arr[below]])
    Xa = np.column_stack([np.ones(above.sum()), x_arr[above]])
    cb = np.linalg.lstsq(Xb, y_arr[below], rcond=None)[0]
    ca = np.linalg.lstsq(Xa, y_arr[above], rcond=None)[0]
    pred = np.concatenate([Xb @ cb, Xa @ ca])
    sse = np.sum((y_arr - pred)**2)
    aic = len(y_arr) * np.log(sse / len(y_arr)) + 2 * 4
    return sse, aic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    SEEDS = list(range(args.seeds)); EPOCHS = args.epochs

    total_runs = len(H_VALUES) * len(SEEDS)
    print(f"τ = {TAU}, L = {L} (fixed)")
    print(f"H values: {H_VALUES}")
    print(f"Critical H: L+H-1=τ → H={TAU - L + 1}")
    print(f"Total runs: {total_runs}, Estimated: ~{total_runs * 0.8:.0f} min\n")

    print(f"Generating MG τ={TAU} sequence ({N_POINTS} points)...")
    mg = MackeyGlass(tau=TAU, constant_past=0.9, nmg=10, beta=0.2, gamma=0.1,
                     dt=1.0, splits=(float(N_POINTS), 0.), seed_id=42)
    vals = [mg[idx][1].squeeze().item() for idx in range(len(mg))]
    col = torch.tensor(vals, dtype=torch.float64).unsqueeze(1)
    data = torch.cat((col, col.clone()), dim=1)
    print(f"  Lyapunov={mg.lyap_exp:+.6f}\n")

    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "h_threshold_results.csv")

    all_rows = []; start_time = time.time()
    for H in H_VALUES:
        crit = L + H - 1
        label = f"H={H} (L+H-1={crit})"
        if crit < TAU: label += " < τ"
        elif crit == TAU: label += " = τ ← CRITICAL"
        else: label += " > τ"
        print(f"\n{'='*55}\n  {label}\n{'='*55}")

        for seed in tqdm(SEEDS, desc="  Seeds"):
            r = run_fgl_one(data, H=H, epochs=EPOCHS, seed=seed)
            all_rows.append({"L": L, "H": H, "tau": TAU, "L_plus_H_minus_1": crit,
                             "seed": seed, **r})
        print(f"  Elapsed: {(time.time()-start_time)/60:.1f} min")

    # Save CSV
    fns = ["L", "H", "tau", "L_plus_H_minus_1", "seed",
           "baseline_mse", "teacher_mse", "student_mse", "abs_improvement", "fgl_delta"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fns); w.writeheader(); w.writerows(all_rows)
    print(f"\nSaved to {csv_path}")

    # Aggregate
    agg = defaultdict(list)
    for row in all_rows: agg[row["H"]].append(row)
    H_sorted = sorted(agg.keys())
    am = {}
    for H in H_sorted:
        rs = agg[H]; bm = [r["baseline_mse"] for r in rs]
        ai = [r["abs_improvement"] for r in rs]; dd = [r["fgl_delta"] for r in rs]
        am[H] = {"bm": np.mean(bm), "bs": np.std(bm),
                 "ai": np.mean(ai), "ais": np.std(ai),
                 "dm": np.mean(dd), "ds": np.std(dd)}

    # Baseline diagnostic
    print("\nBaseline MSE check:")
    b_arr = np.array([am[H]["bm"] for H in H_sorted])
    baseline_ok = True
    for i in range(1, len(b_arr)):
        ratio = max(b_arr[i], b_arr[i-1]) / min(b_arr[i], b_arr[i-1])
        if ratio > 2:
            print(f"  WARNING: {ratio:.1f}× jump between H={H_sorted[i-1]} and H={H_sorted[i]}")
            baseline_ok = False
    if baseline_ok: print("  OK — no >2× jumps between adjacent H values")

    # Stats
    crit_H = TAU - L + 1  # = 9
    below = [r["abs_improvement"] for r in all_rows if r["H"] < crit_H]
    above = [r["abs_improvement"] for r in all_rows if r["H"] >= crit_H]
    t_stat, p_val = stats.ttest_ind(below, above, equal_var=False)
    print(f"\nWelch t-test (H<{crit_H} vs H≥{crit_H}):")
    print(f"  H<{crit_H}: mean={np.mean(below):.4f}, n={len(below)}")
    print(f"  H≥{crit_H}: mean={np.mean(above):.4f}, n={len(above)}")
    print(f"  t={t_stat:.4f}, p={p_val:.6f}")

    # Changepoint
    H_arr = np.array(H_sorted); ai_arr = np.array([am[H]["ai"] for H in H_sorted])
    cp_H, cp_sse, sse1 = find_changepoint(H_arr, ai_arr)
    print(f"\nChangepoint: H≈{cp_H:.1f} (predicted={crit_H}, deviation={cp_H-crit_H:+.1f})")

    # Regression
    Xl = np.column_stack([np.ones(len(H_arr)), H_arr])
    cl = np.linalg.lstsq(Xl, ai_arr, rcond=None)[0]
    sse_l = np.sum((ai_arr - Xl @ cl)**2)
    aic_l = len(H_arr) * np.log(sse_l / len(H_arr)) + 2 * 2
    sse_p, aic_p = piecewise_fit(H_arr, ai_arr, bp=crit_H)
    daic = aic_l - aic_p
    print(f"Linear AIC={aic_l:.1f}, Piecewise AIC={aic_p:.1f}, ΔAIC={daic:+.1f}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.errorbar(H_arr, ai_arr, yerr=[am[H]["ais"] for H in H_sorted],
                color="#2ecc71", marker="o", markersize=9, linewidth=2, capsize=5)
    ax.axvline(x=crit_H, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.annotate(f"H={crit_H}\n(L+H-1=τ)", xy=(crit_H, ax.get_ylim()[1]),
                xytext=(crit_H+0.3, ax.get_ylim()[1]*0.9), fontsize=9, color="red", fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.5)
    Hf = np.linspace(H_arr.min(), H_arr.max(), 100)
    ax.plot(Hf, cl[0] + cl[1]*Hf, 'k--', linewidth=0.8, alpha=0.5, label="Linear")
    ax.set_xlabel("H (forecast horizon)"); ax.set_ylabel("Absolute improvement")
    ax.set_title(f"Fig 1: Abs Improvement vs H (τ={TAU}, L={L})\nL+H-1≥τ at H={crit_H}", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.errorbar(H_arr, b_arr, yerr=[am[H]["bs"] for H in H_sorted],
                color="#3498db", marker="s", markersize=8, linewidth=2, capsize=5)
    ax.axvline(x=crit_H, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("H (forecast horizon)"); ax.set_ylabel("Baseline MSE")
    ax.set_title("Fig 2: Baseline MSE vs H (task difficulty check)", fontweight="bold")
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    dm_arr = [am[H]["dm"] for H in H_sorted]; ds_arr = [am[H]["ds"] for H in H_sorted]
    colors = ["#2ecc71" if m > 0 else "#e74c3c" for m in dm_arr]
    ax.bar(H_arr, dm_arr, yerr=ds_arr, color=colors, edgecolor="white",
           linewidth=0.5, capsize=4, width=0.6)
    ax.axvline(x=crit_H, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("H (forecast horizon)"); ax.set_ylabel("FGL Δ (%)")
    ax.set_title(f"Fig 3: FGL Δ% vs H (τ={TAU}, L={L})", fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout(); fig.savefig(os.path.join(output_dir, "h_threshold_figures.png"), dpi=150)
    plt.close(fig)
    print(f"Figures saved to {output_dir}/h_threshold_figures.png")

    # Summary MD
    md_path = os.path.join(output_dir, "h_threshold_summary.md")
    cp_ok = abs(cp_H - crit_H) <= 1.5
    pw_ok = daic > 10

    if cp_ok and pw_ok and baseline_ok:
        verdict = (f"**SUPPORTS L+H-1≥τ formula — bidirectional verification.**\n\n"
                   f"- Changepoint at H≈{cp_H:.1f} (predicted H={crit_H}, deviation={cp_H-crit_H:+.1f})\n"
                   f"- Segmented regression strongly preferred (ΔAIC={daic:+.1f})\n"
                   f"- Baseline MSE stable across H → task difficulty controlled\n"
                   f"- Combined with threshold_test (fixed H=5, scanned L): the formula L+H-1≥τ "
                   f"predicts the FGL benefit threshold from both directions.\n\n"
                   f"This is the strongest evidence in the series: two independent experiments "
                   f"converge on the same threshold formula.")
    elif cp_ok and pw_ok:
        verdict = (f"**SUPPORTS formula — but baseline MSE unstable.**\n\n"
                   f"- Changepoint at H≈{cp_H:.1f} (deviation={cp_H-crit_H:+.1f})\n"
                   f"- BUT baseline MSE showed unexpected fluctuations → task difficulty not fully controlled.")
    elif not baseline_ok:
        verdict = (f"**Cannot interpret — baseline MSE instability.**\n\n"
                   f"First fix the coupling between baseline data and H before interpreting abs_improvement.")
    else:
        verdict = (f"**MIXED — changepoint deviation={cp_H-crit_H:+.1f}, ΔAIC={daic:+.1f}.**\n\n"
                   f"Does not cleanly support or refute the formula.")

    with open(md_path, "w") as f:
        f.write("# MG H-Threshold Test — τ=13, L=5 fixed, scan H\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**τ:** {TAU} | **L:** {L} | **Seeds:** {SEEDS} | **Epochs:** {EPOCHS}\n")
        f.write(f"**Critical H:** L+H-1=τ → H={crit_H}\n\n")

        f.write("## Results\n\n| H | L+H−1 | Baseline MSE | Abs Improvement | FGL Δ% |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|\n")
        for H in H_sorted:
            a = am[H]; crit = L + H - 1
            m = " ← CRITICAL" if crit == TAU else ""
            f.write(f"| **{H}**{m} | {crit} | {a['bm']:.3f} ± {a['bs']:.3f} | "
                    f"{a['ai']:+.3f} ± {a['ais']:.3f} | {a['dm']:+.1f}% ± {a['ds']:.1f}% |\n")

        f.write(f"\n## Statistics\n\n")
        f.write(f"- Welch t-test: t={t_stat:.4f}, p={p_val:.6f}\n")
        f.write(f"- Changepoint: H≈{cp_H:.1f} (predicted {crit_H}, deviation {cp_H-crit_H:+.1f})\n")
        f.write(f"- ΔAIC (linear − piecewise): {daic:+.1f}\n")
        f.write(f"- Baseline MSE stability: {'OK' if baseline_ok else 'FAILED'}\n\n")

        f.write("## Judgment\n\n" + verdict + f"\n\n## Data: `h_threshold_results.csv` ({total_runs} rows)\n")

    print(f"Summary saved to {md_path}")

    # Console
    print("\n" + "=" * 55)
    print(f"  RESULTS: H-threshold (τ={TAU}, L={L})")
    print("=" * 55)
    for H in H_sorted:
        a = am[H]; crit = L + H - 1
        m = " ← CRITICAL" if crit == TAU else ""
        print(f"  H={H:2d} (L+H-1={crit:2d}){m}: Abs={a['ai']:+.3f}±{a['ais']:.3f}  "
              f"Δ={a['dm']:+.1f}%±{a['ds']:.1f}%  Base={a['bm']:.3f}")
    print(f"\n  t-test: p={p_val:.6f}  |  CP@H≈{cp_H:.1f}  |  ΔAIC={daic:+.1f}")
    print(f"  {verdict.split(chr(10))[0]}")


if __name__ == "__main__":
    main()
