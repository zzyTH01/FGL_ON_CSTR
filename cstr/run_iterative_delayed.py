#!/usr/bin/env python
"""
Adaptive continuous distillation (run_iterative_distillation, E variant) on the
delayed-feedback CSTR datasets — does iterating lower student MSE below the
standard one-pass FGL student?

4 arms (shared round-0 student = standard FGL):
  A_single : round-0 student (standard FGL, no iteration)   <-- reference
  E_single : 1 round, adaptive-E weights
  A_iter   : <=K rounds, uniform weights (attribution control)
  E_iter   : <=K rounds, re-estimated adaptive-E weights   <-- 自适应连续蒸馏

init_delta = (round0_mse - arm_mse)/round0_mse   (>0 => arm BEAT standard FGL)

Usage:
  uv run python cstr/run_iterative_delayed.py
  uv run python cstr/run_iterative_delayed.py --seeds 5 --K 5
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

from fgl_common import run_iterative_distillation
from generate_delayed_stable import periodicity_score

CSTR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(CSTR, "data")
RESULTS = os.path.join(CSTR, "results")
TAG = "s1_A0.9_b0.03"

DATASETS = [
    ("base_h2o(period-1)", "data_h2o.pkl"),
    ("tau50",  f"data_delayed_stable_h2o_tau50_{TAG}.pkl"),
    ("tau100", f"data_delayed_stable_h2o_tau100_{TAG}.pkl"),
    ("tau150", f"data_delayed_stable_h2o_tau150_{TAG}.pkl"),
]
ARMS = ("A_single", "E_single", "A_iter", "E_iter")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=20)
    p.add_argument("--H", type=int, default=15)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("-T", "--temperature", type=float, default=4.0, dest="temperature")
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--round_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    for label, fn in DATASETS:
        path = os.path.join(DATA, fn)
        if not os.path.exists(path):
            print(f"  skip {label}: {fn} not found", flush=True)
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        series = np.asarray(data[:, 0], dtype=float)
        burn = min(1000, len(series) // 5)
        per, _ = periodicity_score(series[burn:])

        arm_mse = {a: [] for a in ARMS}
        arm_init = {a: [] for a in ARMS}
        arm_rounds = {a: [] for a in ARMS}
        base_mse = []
        print(f"\n=== {label}  (periodicity={per:.3f}) ===", flush=True)
        for s in range(args.seeds):
            res = run_iterative_distillation(
                data, L=args.L, H=args.H, alpha=args.alpha, temperature=args.temperature,
                num_bins=args.bins, epochs=args.epochs, round_epochs=args.round_epochs,
                batch_size=args.batch_size, patience=args.patience, K=args.K,
                seed=s, variant="E", verbose=False)
            base_mse.append(res["A_single"]["baseline_mse"])
            for a in ARMS:
                arm_mse[a].append(res[a]["student_mse"])
                arm_init[a].append(res[a]["init_delta"])
                arm_rounds[a].append(res[a]["rounds_used"])
            print(f"  seed{s}: A_single={res['A_single']['student_mse']:.2f} "
                  f"E_iter={res['E_iter']['student_mse']:.2f} "
                  f"(E_iter Δinit={res['E_iter']['init_delta']:+.1f}%, "
                  f"rounds={res['E_iter']['rounds_used']})", flush=True)

        def _ms(a):
            a = np.array(a)
            return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)
        bm, _ = _ms(base_mse)
        line = f"  -> baseline={bm:.2f}  "
        for a in ARMS:
            m, _ = _ms(arm_mse[a]); mi, _ = _ms(arm_init[a])
            line += f"{a}={m:.2f}(Δinit{mi:+.1f}%) "
        print(line, flush=True)
        for a in ARMS:
            m, sd = _ms(arm_mse[a]); mi, sid = _ms(arm_init[a]); mr, _ = _ms(arm_rounds[a])
            rows.append({"dataset": label, "periodicity": per, "arm": a,
                         "student_mse": m, "student_sd": sd,
                         "init_delta": mi, "init_delta_sd": sid,
                         "rounds_used": mr, "baseline_mse": bm})

    out = os.path.join(RESULTS, "iterative_delayed_summary.csv")
    cols = ["dataset", "periodicity", "arm", "student_mse", "student_sd",
            "init_delta", "init_delta_sd", "rounds_used", "baseline_mse"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n{'=' * 74}\nAdaptive continuous distillation (E variant) on delayed CSTR "
          f"(L={args.L} H={args.H}, K<={args.K}, {args.seeds} seeds)\n{'=' * 74}")
    print(f"{'dataset':20s} {'arm':9s} {'stu_mse':>10s} {'Δinit%':>10s} {'rounds':>6s}")
    for r in rows:
        print(f"{r['dataset']:20s} {r['arm']:9s} {r['student_mse']:10.2f} "
              f"{r['init_delta']:+9.1f}% {r['rounds_used']:6.1f}")
    print(f"\nwrote {out}")
    print("\nKey: E_iter Δinit% > 0  =>  continuous adaptive distillation lowered")
    print("student MSE below the standard one-pass FGL student (A_single).")


if __name__ == "__main__":
    main()
