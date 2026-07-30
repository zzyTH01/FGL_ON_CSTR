#!/usr/bin/env python
"""地板成因战役主驱动:在延迟 CSTR 上对每个 (dataset, L, H, seed) 同时记录
{baseline, baseline_converged, teacher, fgl_student, A_iter, E_iter} MSE,
写入一张 floor_sweep.csv,供 analyze_floor.py 检验 H1-H4。

用法::

    uv run python cstr/run_floor_sweep.py --datasets tau100 --seeds 3 --K 5
    uv run python cstr/run_floor_sweep.py --datasets tau100,tau50,tau150 --seeds 3 --K 5
    uv run python cstr/run_floor_sweep.py --anchors
"""
import argparse
import csv
import os
import pickle
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for generate_delayed_stable

from fgl_common import (  # noqa: E402
    run_fgl_experiment, run_iterative_distillation, run_baseline_converged,
)
from generate_delayed_stable import periodicity_score  # noqa: E402

CSTR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(CSTR, "data")
TAG = "s1_A0.9_b0.03"

COLUMNS = ["dataset", "tau", "periodicity", "L", "H", "LplusH_minus_1", "seed",
           "baseline_mse", "baseline_converged_mse", "teacher_mse",
           "fgl_student_mse", "A_iter_mse", "E_iter_mse"]

DATASET_REGISTRY = {
    "base":   ("data_h2o.pkl", None),
    "tau50":  (f"data_delayed_stable_h2o_tau50_{TAG}.pkl", 50),
    "tau100": (f"data_delayed_stable_h2o_tau100_{TAG}.pkl", 100),
    "tau150": (f"data_delayed_stable_h2o_tau150_{TAG}.pkl", 150),
}


def default_tau100_grid():
    """9 surface cells (L×H) + 4 extra H=15 cells = 13 unique cells straddling τ=100."""
    surface = [(L, H) for L in (20, 50, 100) for H in (5, 15, 30)]
    extra_h15 = [(L, 15) for L in (40, 70, 85, 120)]
    seen, cells = set(), []
    for c in surface + extra_h15:
        if c not in seen:
            seen.add(c)
            cells.append(c)
    return cells


def default_anchor_cells():
    return [(20, 15)]


def load_entries(names):
    """names: list of registry keys (e.g. ['tau100']). Returns list of entry dicts."""
    entries = []
    for name in names:
        if name not in DATASET_REGISTRY:
            print(f"  skip unknown dataset key: {name}", flush=True)
            continue
        fn, tau = DATASET_REGISTRY[name]
        path = os.path.join(DATA, fn)
        if not os.path.exists(path):
            print(f"  skip {name}: {fn} not found", flush=True)
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        series = np.asarray(data[:, 0], dtype=float)
        burn = min(1000, len(series) // 5)
        per, _ = periodicity_score(series[burn:])
        entries.append({"label": name, "data": data, "tau": tau if tau is not None else "",
                        "periodicity": per})
    return entries


def run(entries, cells_by_dataset, seeds, *, alpha=0.5, temperature=4.0, bins=50,
        epochs=30, round_epochs=15, batch_size=64, patience=5,
        conv_epochs=100, conv_patience=10, K=5, outdir="cstr/results", verbose=True):
    """Run the floor sweep. entries: list of {label,data,tau,periodicity}.
    cells_by_dataset: {label: [(L,H), ...]}. Returns list of row dicts.
    """
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for entry in entries:
        label = entry["label"]
        data = entry["data"]
        tau = entry["tau"]
        per = entry["periodicity"]
        cells = cells_by_dataset.get(label, [])
        if verbose:
            print(f"\n=== {label}  (tau={tau}, per={per:.3f}, {len(cells)} cells) ===",
                  flush=True)
        for (L, H) in cells:
            for s in seeds:
                r_fgl = run_fgl_experiment(
                    data, lookback_window=L, forecasting_horizon=H,
                    alpha=alpha, temperature=temperature, num_bins=bins, epochs=epochs,
                    batch_size=batch_size, patience=patience, seed=s, verbose=False,
                    label=f"{label}/L{L}_H{H}/s{s}")
                r_conv = run_baseline_converged(
                    data, lookback_window=L, forecasting_horizon=H, num_bins=bins,
                    epochs=conv_epochs, patience=conv_patience, batch_size=batch_size,
                    seed=s, verbose=False, label=f"{label}/L{L}_H{H}/s{s}")
                r_it = run_iterative_distillation(
                    data, L=L, H=H, alpha=alpha, temperature=temperature, num_bins=bins,
                    epochs=epochs, round_epochs=round_epochs, batch_size=batch_size,
                    patience=patience, K=K, seed=s, variant="E", verbose=False)
                row = {"dataset": label, "tau": tau, "periodicity": per,
                       "L": L, "H": H, "LplusH_minus_1": L + H - 1, "seed": s,
                       "baseline_mse": r_fgl["baseline"],
                       "baseline_converged_mse": r_conv["baseline_mse"],
                       "teacher_mse": r_fgl["teacher"],
                       "fgl_student_mse": r_fgl["student"],
                       "A_iter_mse": r_it["A_iter"]["student_mse"],
                       "E_iter_mse": r_it["E_iter"]["student_mse"]}
                rows.append(row)
                if verbose:
                    print(f"  L={L:3d} H={H:2d} s{s}: base={row['baseline_mse']:.1f} "
                          f"baseC={row['baseline_converged_mse']:.1f} "
                          f"tch={row['teacher_mse']:.1f} "
                          f"E_iter={row['E_iter_mse']:.1f}", flush=True)

    out = os.path.join(outdir, "floor_sweep.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if verbose:
        print(f"\nwrote {out}  ({len(rows)} rows)")
    return rows


def main():
    p = argparse.ArgumentParser(description="Floor-determinants sweep on delayed CSTR")
    p.add_argument("--datasets", type=str, default="tau100",
                   help="逗号分隔的 registry 键(默认 tau100)")
    p.add_argument("--anchors", action="store_true",
                   help="跑横向锚点 tau50/tau150 @ L20H15(忽略 --datasets)")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("-T", "--temperature", type=float, default=4.0, dest="temperature")
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--round_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--conv_epochs", type=int, default=100)
    p.add_argument("--conv_patience", type=int, default=10)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--outdir", type=str, default=os.path.join(CSTR, "results"))
    args = p.parse_args()

    seeds = list(range(args.seeds))
    if args.anchors:
        entries = load_entries(["tau50", "tau150"])
        cells_by_dataset = {e["label"]: default_anchor_cells() for e in entries}
    else:
        names = [n.strip() for n in args.datasets.split(",")]
        entries = load_entries(names)
        # tau100 用深挖网格;其余数据集默认只跑 L20H15 锚
        cells_by_dataset = {}
        for e in entries:
            cells_by_dataset[e["label"]] = (default_tau100_grid()
                                            if e["label"] == "tau100"
                                            else default_anchor_cells())

    run(entries, cells_by_dataset, seeds,
        alpha=args.alpha, temperature=args.temperature, bins=args.bins,
        epochs=args.epochs, round_epochs=args.round_epochs,
        batch_size=args.batch_size, patience=args.patience,
        conv_epochs=args.conv_epochs, conv_patience=args.conv_patience,
        K=args.K, outdir=args.outdir, verbose=True)


if __name__ == "__main__":
    main()
