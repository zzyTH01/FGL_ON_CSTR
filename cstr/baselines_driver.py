"""Batch runner for CSTR external continuous-MSE forecasting baselines."""
from __future__ import annotations

import csv
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

from fgl_common.baselines import run_forecasting_baseline

_CSTR_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CSTR_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_FIELDS = ["dataset", "L", "H", "method", "seed", "val_mse", "test_mse",
           "n_test", "epochs"]
_DEFAULT_OUT = os.path.join(_CSTR_DIR, "results", "cstr_external_baselines.csv")
_DATASETS = {
    "temperature": os.path.join(_CSTR_DIR, "data", "data.pkl"),
    "h2o": os.path.join(_CSTR_DIR, "data", "data_h2o.pkl"),
}


def _resolve_data(name: str) -> str:
    if name in _DATASETS:
        return _DATASETS[name]
    raise ValueError(f"unknown CSTR dataset {name!r}; expected {sorted(_DATASETS)}")


def _load_data(dataset: str):
    with open(_resolve_data(dataset), "rb") as f:
        return pickle.load(f)


def _read_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = os.fspath(path) + ".tmp"
    with open(temporary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_all(args, out_path: str = _DEFAULT_OUT):
    """Run methods × seeds for one CSTR L/H grid point.

    Existing rows for the same ``(dataset, L, H)`` are replaced so repeated
    pilot runs are reproducible, while rows at other L/H points are retained.
    """
    methods = tuple(
        item.strip().lower() for item in args.baseline_methods.split(",")
        if item.strip()
    )
    if not methods:
        raise ValueError("baseline_methods cannot be empty")

    rows = [row for row in _read_rows(out_path)
            if not (row["dataset"] == args.dataset and
                    int(row["L"]) == args.L and int(row["H"]) == args.H)]
    data = _load_data(args.dataset)
    for seed in range(args.seeds):
        for method in methods:
            result = run_forecasting_baseline(
                data,
                method=method,
                lookback_window=args.L,
                forecasting_horizon=args.H,
                seed=seed,
                epochs=args.baseline_epochs,
                batch_size=args.batch_size,
                patience=args.patience,
            )
            row = {
                "dataset": args.dataset,
                "L": args.L,
                "H": args.H,
                "method": method,
                "seed": seed,
                "val_mse": result["val_mse"],
                "test_mse": result["test_mse"],
                "n_test": result["n_test"],
                "epochs": result["epochs"],
            }
            rows.append(row)
            print(f"  {args.dataset} L{args.L} H{args.H} {method:8s} seed={seed}: "
                  f"val={result['val_mse']:.6g} test={result['test_mse']:.6g}")
    _write_rows(out_path, rows)
    print(f"  → {out_path} ({len(rows)} total rows)")

    summary = defaultdict(list)
    for row in rows:
        if (row["dataset"] == args.dataset and int(row["L"]) == args.L and
                int(row["H"]) == args.H):
            summary[row["method"]].append(float(row["test_mse"]))
    print(f"\nSUMMARY external baselines ({args.dataset}, L={args.L}, H={args.H}):")
    for method in methods:
        values = np.asarray(summary[method])
        sd = values.std(ddof=1) if len(values) > 1 else 0.0
        print(f"  {method:8s}: test_mse={values.mean():.6g}±{sd:.6g} (n={len(values)})")
    return rows
