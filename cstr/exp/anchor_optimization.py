#!/usr/bin/env python
"""
Batch orchestrator for CSTR anchor-point optimization experiments.

Tasks:
  1. Fine-grid L×H sweep around L=20,H=15 (25 configs × 5 seeds = 125 runs)
  2. LSTM architecture test at L=20,H=15 (5 seeds)
  3. Regression mode test at L=20,H=15 with alpha sweep (5 alphas × 5 seeds = 25 runs)
  4. (Optional) alpha×T grid search at L=20,H=15 (36 combos × 3 seeds = 108 runs)

Usage:
  uv run python cstr/exp/anchor_optimization.py --task 1          # Task 1 only
  uv run python cstr/exp/anchor_optimization.py --task 2          # Task 2 only
  uv run python cstr/exp/anchor_optimization.py --task 3          # Task 3 only
  uv run python cstr/exp/anchor_optimization.py --task 4          # Task 4 only
  uv run python cstr/exp/anchor_optimization.py --task all        # All tasks
  uv run python cstr/exp/anchor_optimization.py --task all --max-workers 4  # Limit parallelism
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Use venv Python directly (subprocess doesn't inherit 'uv run' PATH)
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python3")

FGL_CSTR_PY = os.path.join(SCRIPT_DIR, "fgl_cstr.py")
FGL_LSTM_PY = os.path.join(SCRIPT_DIR, "fgl_cstr_lstm.py")
FGL_REG_PY = os.path.join(SCRIPT_DIR, "fgl_cstr_regression.py")

# --- Shared config ---
SHARED_ARGS_CLS = ["--epochs", "30", "--num_bins", "50", "--temperature", "4",
                   "--batch_size", "64", "--patience", "5", "--dataset", "h2o"]
SHARED_ARGS_REG = ["--epochs", "30", "--batch_size", "64", "--patience", "5", "--dataset", "h2o"]

# ========================================================================
# Task definitions
# ========================================================================

def task1_configs():
    """Generate (L, H, seed) tuples for fine-grid search."""
    L_vals = [15, 18, 20, 22, 25]
    H_vals = [10, 12, 15, 18, 20]
    seeds = list(range(5))  # 0,1,2,3,4
    configs = []
    for L in L_vals:
        for H in H_vals:
            for s in seeds:
                configs.append((L, H, s))
    return configs


def task2_configs():
    """Generate (seed,) tuples for LSTM test at L=20,H=15."""
    return [(s,) for s in range(5)]


def task3_configs():
    """Generate (alpha, seed) tuples for regression test at L=20,H=15."""
    alphas = [0.0, 0.3, 0.5, 0.7, 0.9]
    seeds = list(range(5))
    return [(a, s) for a in alphas for s in seeds]


def task4_configs():
    """Generate (alpha, T, seed) tuples for grid search at L=20,H=15."""
    alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    temps = [2, 4, 6, 8, 10, 12]
    seeds = [0, 1, 2]
    return [(a, T, s) for a in alphas for T in temps for s in seeds]


# ========================================================================
# Runner helpers
# ========================================================================

def parse_output(stdout):
    """Parse the final evaluation line from fgl_cstr.py output.
    Format: '  Teacher:  <val>  Baseline: <val>  Student: <val>  (Δ=+XX.X%)'
    """
    pattern = r"Teacher:\s+([\d.]+)\s+Baseline:\s+([\d.]+)\s+Student:\s+([\d.]+)\s+.*?Δ=([+-]?[\d.]+)%"
    m = re.search(pattern, stdout)
    if m:
        return {
            "teacher_mse": float(m.group(1)),
            "baseline_mse": float(m.group(2)),
            "student_mse": float(m.group(3)),
            "fgl_delta": float(m.group(4)),
        }
    # Fallback: try to find any line with the pattern
    for line in stdout.splitlines():
        m = re.search(pattern, line)
        if m:
            return {
                "teacher_mse": float(m.group(1)),
                "baseline_mse": float(m.group(2)),
                "student_mse": float(m.group(3)),
                "fgl_delta": float(m.group(4)),
            }
    return None


def run_single(script_path, extra_args, label="", shared_args=None):
    """Run a single experiment via subprocess. Returns parsed result dict or None."""
    if shared_args is None:
        shared_args = SHARED_ARGS_CLS
    cmd_parts = [VENV_PYTHON, "-u", script_path] + shared_args + extra_args
    try:
        env = os.environ.copy()
        env["VIRTUAL_ENV"] = os.path.join(PROJECT_ROOT, ".venv")
        env["PATH"] = os.path.join(PROJECT_ROOT, ".venv", "bin") + ":" + env.get("PATH", "")
        result = subprocess.run(
            cmd_parts, capture_output=True, text=True, timeout=600,
            cwd=PROJECT_ROOT, env=env
        )
        stdout = result.stdout + result.stderr
        parsed = parse_output(stdout)
        if parsed is None:
            # If regex failed, print output for debugging
            print(f"  [WARN] Parse failure for {label}: cmd={' '.join(cmd_parts)}")
            print(f"  stdout tail: {stdout[-300:]}")
        return parsed
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {label}")
        return None
    except Exception as e:
        print(f"  [ERROR] {label}: {e}")
        return None


# ========================================================================
# CSV helpers
# ========================================================================

def write_csv_rows(filepath, fieldnames, rows, append=True):
    """Write rows to CSV. Creates file with header if not appending or if new."""
    mode = "a" if (append and os.path.exists(filepath)) else "w"
    with open(filepath, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_existing_keys(filepath, key_cols):
    """Load set of already-completed experiment keys from CSV to skip re-runs."""
    if not os.path.exists(filepath):
        return set()
    keys = set()
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals = tuple(row[c] for c in key_cols)
            if any(v == "NA" for v in vals):
                continue  # skip failed runs so they get retried
            keys.add(vals)
    return keys


# ========================================================================
# Task execution
# ========================================================================

def run_task1(max_workers=4):
    """Fine-grid L×H sweep around L=20,H=15."""
    print("\n" + "=" * 70)
    print("TASK 1: Fine-grid L×H sweep around L=20,H=15")
    print("=" * 70)

    csv_path = os.path.join(RESULTS_DIR, "anchor_refine_results.csv")
    fieldnames = ["L", "H", "seed", "baseline_mse", "teacher_mse",
                  "student_mse", "abs_improvement", "fgl_delta"]

    configs = task1_configs()
    existing = load_existing_keys(csv_path, ["L", "H", "seed"])
    pending = [(L, H, s) for (L, H, s) in configs
               if (str(L), str(H), str(s)) not in existing]

    print(f"  Total configs: {len(configs)} | Already done: {len(configs)-len(pending)} | Pending: {len(pending)}")

    if not pending:
        print("  All done. Skipping.")
        return

    results_lock = []  # accumulate for batch CSV write
    completed = 0

    def run_one(L, H, seed):
        extra = ["--horizon", str(H), "--lookback_window", str(L),
                 "--alpha", "0.5", "--seed", str(seed)]
        label = f"L={L},H={H},seed={seed}"
        parsed = run_single(FGL_CSTR_PY, extra, label)
        if parsed:
            return {
                "L": L, "H": H, "seed": seed,
                "baseline_mse": parsed["baseline_mse"],
                "teacher_mse": parsed["teacher_mse"],
                "student_mse": parsed["student_mse"],
                "abs_improvement": parsed["baseline_mse"] - parsed["student_mse"],
                "fgl_delta": parsed["fgl_delta"],
            }
        else:
            return {
                "L": L, "H": H, "seed": seed,
                "baseline_mse": "NA", "teacher_mse": "NA",
                "student_mse": "NA", "abs_improvement": "NA", "fgl_delta": "NA",
            }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, L, H, s): (L, H, s) for (L, H, s) in pending}
        for fut in as_completed(futures):
            row = fut.result()
            results_lock.append(row)
            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                write_csv_rows(csv_path, fieldnames, results_lock)
                results_lock.clear()
                print(f"  Progress: {completed}/{len(pending)}")

    # Flush remaining
    if results_lock:
        write_csv_rows(csv_path, fieldnames, results_lock)

    print(f"  Task 1 complete. Results → {csv_path}")
    _print_task1_heatmaps(csv_path)


def _print_task1_heatmaps(csv_path):
    """Print heatmaps from Task 1 results."""
    import numpy as np
    rows = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            if row["fgl_delta"] == "NA":
                continue
            rows.append(row)

    if not rows:
        print("  No valid data for heatmaps.")
        return

    # Aggregate
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        key = (int(r["L"]), int(r["H"]))
        agg[key].append(float(r["fgl_delta"]))

    L_vals = sorted(set(k[0] for k in agg))
    H_vals = sorted(set(k[1] for k in agg))

    print("\n  FGL Δ% Heatmap (mean ± std):")
    header = "L\\H  " + "  ".join(f"{h:>6}" for h in H_vals)
    print(f"  {header}")
    for L in L_vals:
        line = f"  {L:>3}  "
        for H in H_vals:
            vals = agg.get((L, H), [])
            if vals:
                line += f" {np.mean(vals):+.1f}±{np.std(vals):.0f} "
            else:
                line += "   NA   "
        print(line)

    # Baseline MSE heatmap
    agg_b = defaultdict(list)
    for r in rows:
        key = (int(r["L"]), int(r["H"]))
        agg_b[key].append(float(r["baseline_mse"]))

    print("\n  Baseline MSE Heatmap (mean):")
    print(f"  {header}")
    for L in L_vals:
        line = f"  {L:>3}  "
        for H in H_vals:
            vals = agg_b.get((L, H), [])
            if vals:
                line += f" {np.mean(vals):6.1f} "
            else:
                line += "   NA   "
        print(line)

    # Find best
    best_key = max(agg, key=lambda k: np.mean(agg[k]))
    best_mean = np.mean(agg[best_key])
    best_std = np.std(agg[best_key])
    print(f"\n  Best point: L={best_key[0]}, H={best_key[1]} → Δ={best_mean:+.1f}% ± {best_std:.1f}%")


def run_task2(max_workers=4):
    """LSTM architecture test at L=20,H=15."""
    print("\n" + "=" * 70)
    print("TASK 2: LSTM architecture test at L=20,H=15")
    print("=" * 70)

    csv_path = os.path.join(RESULTS_DIR, "anchor_lstm_results.csv")
    fieldnames = ["architecture", "L", "H", "seed", "baseline_mse",
                  "teacher_mse", "student_mse", "abs_improvement", "fgl_delta"]

    configs = task2_configs()
    existing = load_existing_keys(csv_path, ["seed"])
    pending = [(s,) for (s,) in configs if (str(s),) not in existing]

    print(f"  Total: {len(configs)} | Done: {len(configs)-len(pending)} | Pending: {len(pending)}")

    if not pending:
        print("  All done. Skipping.")
        return

    results = []
    completed = 0

    def run_one(seed):
        extra = ["--horizon", "15", "--lookback_window", "20",
                 "--alpha", "0.5", "--seed", str(seed)]
        label = f"LSTM L=20,H=15,seed={seed}"
        parsed = run_single(FGL_LSTM_PY, extra, label)
        if parsed:
            return {
                "architecture": "LSTM", "L": 20, "H": 15, "seed": seed,
                "baseline_mse": parsed["baseline_mse"],
                "teacher_mse": parsed["teacher_mse"],
                "student_mse": parsed["student_mse"],
                "abs_improvement": parsed["baseline_mse"] - parsed["student_mse"],
                "fgl_delta": parsed["fgl_delta"],
            }
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, s): s for (s,) in pending}
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                results.append(row)
            completed += 1
            print(f"  Progress: {completed}/{len(pending)}")

    write_csv_rows(csv_path, fieldnames, results, append=False)
    _print_task2_summary(csv_path)


def _print_task2_summary(csv_path):
    """Print LSTM vs RNN comparison."""
    import numpy as np
    rows = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        print("  No data for comparison.")
        return

    deltas = [float(r["fgl_delta"]) for r in rows]
    abs_imps = [float(r["abs_improvement"]) for r in rows]
    b_mse = [float(r["baseline_mse"]) for r in rows]
    t_mse = [float(r["teacher_mse"]) for r in rows]
    s_mse = [float(r["student_mse"]) for r in rows]

    print(f"\n  LSTM Summary (n={len(rows)} seeds):")
    print(f"    Baseline MSE:  {np.mean(b_mse):.1f} ± {np.std(b_mse):.1f}")
    print(f"    Teacher MSE:   {np.mean(t_mse):.1f} ± {np.std(t_mse):.1f}")
    print(f"    Student MSE:   {np.mean(s_mse):.1f} ± {np.std(s_mse):.1f}")
    print(f"    Abs Improvement: {np.mean(abs_imps):.1f} ± {np.std(abs_imps):.1f}")
    print(f"    FGL Δ%:        {np.mean(deltas):+.1f}% ± {np.std(deltas):.1f}%")


def run_task3(max_workers=4):
    """Regression mode at L=20,H=15 with alpha sweep."""
    print("\n" + "=" * 70)
    print("TASK 3: Regression mode at L=20,H=15 with alpha sweep")
    print("=" * 70)

    csv_path = os.path.join(RESULTS_DIR, "anchor_regression_results.csv")
    fieldnames = ["mode", "L", "H", "alpha", "seed", "baseline_mse",
                  "teacher_mse", "student_mse", "abs_improvement", "fgl_delta"]

    configs = task3_configs()
    existing = load_existing_keys(csv_path, ["alpha", "seed"])
    pending = [(a, s) for (a, s) in configs
               if (str(a), str(s)) not in existing]

    print(f"  Total: {len(configs)} | Done: {len(configs)-len(pending)} | Pending: {len(pending)}")

    if not pending:
        print("  All done. Skipping.")
        return

    results = []
    completed = 0

    def run_one(alpha, seed):
        extra = ["--horizon", "15", "--lookback_window", "20",
                 "--alpha", str(alpha), "--seed", str(seed)]
        label = f"Reg L=20,H=15,α={alpha},seed={seed}"
        parsed = run_single(FGL_REG_PY, extra, label, shared_args=SHARED_ARGS_REG)
        if parsed:
            return {
                "mode": "regression", "L": 20, "H": 15,
                "alpha": alpha, "seed": seed,
                "baseline_mse": parsed["baseline_mse"],
                "teacher_mse": parsed["teacher_mse"],
                "student_mse": parsed["student_mse"],
                "abs_improvement": parsed["baseline_mse"] - parsed["student_mse"],
                "fgl_delta": parsed["fgl_delta"],
            }
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, a, s): (a, s) for (a, s) in pending}
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                results.append(row)
            completed += 1
            print(f"  Progress: {completed}/{len(pending)}")

    write_csv_rows(csv_path, fieldnames, results, append=False)
    _print_task3_summary(csv_path)


def _print_task3_summary(csv_path):
    """Print regression alpha sweep summary."""
    import numpy as np
    from collections import defaultdict
    rows = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        print("  No data.")
        return

    agg = defaultdict(list)
    for r in rows:
        agg[float(r["alpha"])].append(float(r["fgl_delta"]))

    print(f"\n  Regression α Sweep Summary:")
    print(f"  {'α':>5}  {'Mean Δ%':>10}  {'Std':>8}  {'Abs Imp Mean':>14}")
    for a in sorted(agg):
        vals = agg[a]
        abs_vals = [float(r2["abs_improvement"]) for r2 in rows if float(r2["alpha"]) == a]
        print(f"  {a:5.1f}  {np.mean(vals):+10.1f}%  {np.std(vals):8.1f}  {np.mean(abs_vals):14.4f}")


def run_task4(max_workers=4):
    """α×T grid search at L=20,H=15."""
    print("\n" + "=" * 70)
    print("TASK 4: α×T grid search at L=20,H=15")
    print("=" * 70)

    csv_path = os.path.join(RESULTS_DIR, "anchor_alpha_T_results.csv")
    fieldnames = ["L", "H", "alpha", "temperature", "seed", "baseline_mse",
                  "teacher_mse", "student_mse", "abs_improvement", "fgl_delta"]

    configs = task4_configs()
    existing = load_existing_keys(csv_path, ["alpha", "temperature", "seed"])
    pending = [(a, T, s) for (a, T, s) in configs
               if (str(a), str(T), str(s)) not in existing]

    print(f"  Total: {len(configs)} | Done: {len(configs)-len(pending)} | Pending: {len(pending)}")

    if not pending:
        print("  All done. Skipping.")
        return

    results_lock = []
    completed = 0

    def run_one(alpha, T, seed):
        extra = ["--horizon", "15", "--lookback_window", "20",
                 "--alpha", str(alpha), "--temperature", str(T),
                 "--seed", str(seed)]
        label = f"α={alpha},T={T},seed={seed}"
        parsed = run_single(FGL_CSTR_PY, extra, label)
        if parsed:
            return {
                "L": 20, "H": 15, "alpha": alpha, "temperature": T, "seed": seed,
                "baseline_mse": parsed["baseline_mse"],
                "teacher_mse": parsed["teacher_mse"],
                "student_mse": parsed["student_mse"],
                "abs_improvement": parsed["baseline_mse"] - parsed["student_mse"],
                "fgl_delta": parsed["fgl_delta"],
            }
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, a, T, s): (a, T, s) for (a, T, s) in pending}
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                results_lock.append(row)
            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                write_csv_rows(csv_path, fieldnames, results_lock)
                results_lock.clear()
                print(f"  Progress: {completed}/{len(pending)}")

    if results_lock:
        write_csv_rows(csv_path, fieldnames, results_lock)

    _print_task4_heatmap(csv_path)


def _print_task4_heatmap(csv_path):
    """Print α×T heatmap."""
    import numpy as np
    from collections import defaultdict
    rows = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        print("  No data.")
        return

    agg = defaultdict(list)
    for r in rows:
        key = (float(r["alpha"]), float(r["temperature"]))
        agg[key].append(float(r["fgl_delta"]))

    alphas = sorted(set(k[0] for k in agg))
    temps = sorted(set(k[1] for k in agg))

    print("\n  FGL Δ% α×T Heatmap:")
    header = "α\\T  " + "  ".join(f"{t:>6}" for t in temps)
    print(f"  {header}")
    for a in alphas:
        line = f"  {a:.1f}  "
        for T in temps:
            vals = agg.get((a, T), [])
            if vals:
                line += f" {np.mean(vals):+.1f} "
            else:
                line += "   NA   "
        print(line)

    best_key = max(agg, key=lambda k: np.mean(agg[k]))
    print(f"\n  Best (α,T): α={best_key[0]}, T={best_key[1]} → Δ={np.mean(agg[best_key]):+.1f}%")


# ========================================================================
# Main
# ========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSTR Anchor Optimization Batch Runner")
    parser.add_argument("--task", type=str, required=True,
                        choices=["1", "2", "3", "4", "all"],
                        help="Which task to run")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Max parallel workers (default: 4)")
    args = parser.parse_args()

    print(f"Anchor Optimization Batch Runner")
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Max workers: {args.max_workers}")
    print(f"  Project root: {PROJECT_ROOT}")

    task = args.task
    mw = args.max_workers

    if task in ("1", "all"):
        run_task1(max_workers=mw)

    if task in ("2", "all"):
        run_task2(max_workers=mw)

    if task in ("3", "all"):
        run_task3(max_workers=mw)

    if task in ("4", "all"):
        run_task4(max_workers=mw)

    print(f"\n  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  All tasks complete.")
