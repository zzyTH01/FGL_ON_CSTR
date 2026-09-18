"""Run iterative distillation and report bin-index-to-physical remapped MSE."""
from __future__ import annotations

import csv
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

from fgl_common.training import run_iterative_distillation

_CSTR_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CSTR_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_FIELDS = [
    "dataset", "L", "H", "seed", "arm",
    "student_mse_bin_index", "baseline_mse_bin_index",
    "student_mse_centroid", "student_mse_bin_center",
    "baseline_mse_centroid", "baseline_mse_bin_center",
    "continuous_delta", "rounds_used", "total_epochs",
]
_DEFAULT_OUT = os.path.join(_CSTR_DIR, "results", "cstr_iterative_continuous_remap.csv")
_DATASETS = {
    "temperature": os.path.join(_CSTR_DIR, "data", "data.pkl"),
    "h2o": os.path.join(_CSTR_DIR, "data", "data_h2o.pkl"),
}


def _load_data(dataset: str):
    if dataset not in _DATASETS:
        raise ValueError(f"unknown CSTR dataset {dataset!r}")
    with open(_DATASETS[dataset], "rb") as f:
        return pickle.load(f)


def _write(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = os.fspath(path) + ".tmp"
    with open(temporary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_all(args, out_path: str = _DEFAULT_OUT):
    """Run all requested distillation arms and retain both MSE representations."""
    variants = tuple(item.strip() for item in args.distill_variants.split(",") if item.strip())
    old_rows = []
    if os.path.exists(out_path):
        with open(out_path, newline="") as f:
            old_rows = list(csv.DictReader(f))
    rows = [row for row in old_rows
            if not (row["dataset"] == args.dataset and int(row["L"]) == args.L and
                    int(row["H"]) == args.H)]

    data = _load_data(args.dataset)
    for seed in range(args.seeds):
        arms = run_iterative_distillation(
            data, L=args.L, H=args.H, alpha=args.alpha, temperature=args.temperature,
            num_bins=args.bins, epochs=args.epochs, round_epochs=args.round_epochs,
            batch_size=args.batch_size, K=args.K, patience=args.patience, seed=seed,
            weight_distributions=variants, w_floor=args.w_floor, verbose=False,
        )
        for arm, result in arms.items():
            row = {
                "dataset": args.dataset, "L": args.L, "H": args.H, "seed": seed,
                "arm": arm,
                "student_mse_bin_index": result["student_mse"],
                "baseline_mse_bin_index": result["baseline_mse"],
                "student_mse_centroid": result["student_continuous_mse"],
                "student_mse_bin_center": result["student_bin_center_mse"],
                "baseline_mse_centroid": result["baseline_continuous_mse"],
                "baseline_mse_bin_center": result["baseline_bin_center_mse"],
                "continuous_delta": result["continuous_delta"],
                "rounds_used": result["rounds_used"],
                "total_epochs": result["total_epochs"],
            }
            rows.append(row)
            print(f"  {args.dataset} L{args.L} H{args.H} {arm:14s} seed={seed}: "
                  f"centroid={row['student_mse_centroid']:.8g} "
                  f"bin-center={row['student_mse_bin_center']:.8g}")

    _write(out_path, rows)
    print(f"  → {out_path} ({len(rows)} total rows)")
    summary = defaultdict(list)
    for row in rows:
        if (row["dataset"] == args.dataset and int(row["L"]) == args.L and
                int(row["H"]) == args.H):
            summary[row["arm"]].append(float(row["student_mse_centroid"]))
    print("\nSUMMARY remapped iterative distillation (physical centroid MSE):")
    for arm in sorted(summary):
        values = np.asarray(summary[arm])
        sd = values.std(ddof=1) if len(values) > 1 else 0.0
        print(f"  {arm:14s}: {values.mean():.8g}±{sd:.8g} (n={len(values)})")
    return rows
