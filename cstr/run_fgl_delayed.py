#!/usr/bin/env python
"""
Run FGL (3-stage teacher -> baseline -> student) on the delayed-feedback CSTR
datasets, to test whether FGL gain recovers as the data goes from periodic
(base period-1 CSTR) to aperiodic (delayed-feedback τ sweep).

Same FGL configuration as the mainline CSTR baseline (classification/bins,
L=20, H=15), so the comparison to the known period-1 "floor" is apples-to-apples.

Usage:
  uv run python cstr/run_fgl_delayed.py
  uv run python cstr/run_fgl_delayed.py --seeds 5
"""
import os
import sys
import pickle
import csv

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for generate_delayed_stable

from fgl_common import RNN, run_fgl_experiment
from generate_delayed_stable import periodicity_score

CSTR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(CSTR, "data")
RESULTS = os.path.join(CSTR, "results")
TAG = "s1_A0.9_b0.03"

# (label, filename): base period-1 control + representative τ across the transition
DATASETS = [
    ("base_h2o(period-1)", "data_h2o.pkl"),
    ("tau30",  f"data_delayed_stable_h2o_tau30_{TAG}.pkl"),   # per ~0.79 (mild)
    ("tau50",  f"data_delayed_stable_h2o_tau50_{TAG}.pkl"),   # per ~0.56 (aperiodic)
    ("tau80",  f"data_delayed_stable_h2o_tau80_{TAG}.pkl"),   # per ~0.79 (quasi-periodic window)
    ("tau100", f"data_delayed_stable_h2o_tau100_{TAG}.pkl"),  # per ~0.49 (aperiodic)
    ("tau150", f"data_delayed_stable_h2o_tau150_{TAG}.pkl"),  # per ~0.47 (strongly aperiodic)
]


def load_series(fn):
    with open(os.path.join(DATA, fn), "rb") as f:
        t = pickle.load(f)
    return t, np.asarray(t[:, 0], dtype=float)


def run_all(args):
    """核心逻辑:6 个延迟数据集 × seeds 跑 FGL,写 fgl_delayed_summary.csv。

    接受 argparse Namespace(与 cstr/run.py 的字段兼容),供 run.py 统一入口调用。
    """
    rows = []
    for label, fn in DATASETS:
        path = os.path.join(DATA, fn)
        if not os.path.exists(path):
            print(f"  skip {label}: {fn} not found", flush=True)
            continue
        data, series = load_series(fn)
        burn = min(1000, len(series) // 5)
        per, _ = periodicity_score(series[burn:])
        bases, stus, imps = [], [], []
        print(f"\n=== {label}  (periodicity={per:.3f}, N={len(series)}) ===", flush=True)
        for s in range(args.seeds):
            r = run_fgl_experiment(
                data, lookback_window=args.L, forecasting_horizon=args.H,
                alpha=args.alpha, temperature=args.temperature, num_bins=args.bins,
                epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                seed=s, model_fn=RNN, verbose=False, label=f"{label}/s{s}")
            bases.append(r["baseline"]); stus.append(r["student"]); imps.append(r["improvement"])
            print(f"  seed{s}: base={r['baseline']:.4f} stu={r['student']:.4f} "
                  f"Δ={r['improvement']:+.1f}%", flush=True)

        def _ms(a):
            a = np.array(a)
            return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)
        bm, bsd = _ms(bases); sm, ssd = _ms(stus); im, isd = _ms(imps)
        print(f"  -> base={bm:.4f}±{bsd:.4f} stu={sm:.4f}±{ssd:.4f} "
              f"Δ={im:+.1f}%±{isd:.1f}%", flush=True)
        rows.append({"dataset": label, "periodicity": per, "N": len(series),
                     "baseline_mse": bm, "baseline_sd": bsd,
                     "student_mse": sm, "student_sd": ssd,
                     "improvement": im, "improvement_sd": isd})

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "fgl_delayed_summary.csv")
    with open(out, "w", newline="") as f:
        cols = ["dataset", "periodicity", "N", "baseline_mse", "baseline_sd",
                "student_mse", "student_sd", "improvement", "improvement_sd"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n{'=' * 68}\nFGL on delayed-feedback CSTR "
          f"(L={args.L} H={args.H}, {args.seeds} seeds)\n{'=' * 68}")
    print(f"{'dataset':22s} {'period':>6s} {'baseline':>10s} {'student':>10s} {'Δ%':>12s}")
    for r in rows:
        print(f"{r['dataset']:22s} {r['periodicity']:6.3f} {r['baseline_mse']:10.4f} "
              f"{r['student_mse']:10.4f} {r['improvement']:+7.1f}%±{r['improvement_sd']:.1f}")
    print(f"\nwrote {out}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=20)
    p.add_argument("--H", type=int, default=15)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("-T", "--temperature", type=float, default=4.0, dest="temperature")
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
