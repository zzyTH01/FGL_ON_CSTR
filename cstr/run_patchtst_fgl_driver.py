#!/usr/bin/env python
"""Run continuous PatchTST teacher / baseline / FGL student on CSTR.

This is a non-mainline experiment and is exposed through
``python cstr/run.py -e patchtst_fgl``.
"""
from __future__ import annotations

import csv
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fgl_common.patchtst_fgl import run_patchtst_fgl  # noqa: E402

_CSTR_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_CSTR_DIR, "results")
_DEFAULT_OUT = os.path.join(_RESULTS_DIR, "cstr_patchtst_fgl.csv")
_DEFAULT_SUMMARY = os.path.join(_RESULTS_DIR, "cstr_patchtst_fgl_summary.csv")
_DEFAULT_PLOT = os.path.join(_RESULTS_DIR, "plots", "cstr_patchtst_fgl_summary.png")
_FIELDS = [
    "dataset", "L", "H", "seed", "alpha", "teacher_mse", "baseline_mse",
    "student_mse", "improvement", "teacher_epochs", "baseline_epochs",
    "student_epochs", "teacher_val_mse_normalized", "baseline_val_mse_normalized",
    "student_val_mse_normalized", "n_test", "d_model", "nhead",
    "dim_feedforward", "dropout", "lr", "epochs", "batch_size", "patience",
]


def _resolve_data(name: str) -> str:
    for directory in ("data", "."):
        path = os.path.join(_CSTR_DIR, directory, name)
        if os.path.exists(path):
            return path
    return os.path.join(_CSTR_DIR, "data", name)


def _load_data(dataset: str = "h2o"):
    filename = "data.pkl" if dataset == "temperature" else "data_h2o.pkl"
    with open(_resolve_data(filename), "rb") as f:
        return pickle.load(f)


def _read_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _parse_int_list(value: str | None, default: str) -> list[int]:
    raw = value if value is not None and value.strip() else default
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("integer list cannot be empty")
    return values


def _summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = (int(row["L"]), int(row["H"]))
        for arm in ("teacher_mse", "baseline_mse", "student_mse"):
            grouped[key][arm].append(float(row[arm]))
    summary = []
    for (L, H), values in sorted(grouped.items()):
        teacher = np.asarray(values["teacher_mse"])
        baseline = np.asarray(values["baseline_mse"])
        student = np.asarray(values["student_mse"])
        ratio = float(np.mean(student) / np.mean(baseline)) if np.mean(baseline) else np.nan
        summary.append({
            "L": L,
            "H": H,
            "n_seeds": len(values["baseline_mse"]),
            "teacher_mse_mean": float(teacher.mean()),
            "teacher_mse_sd": float(teacher.std(ddof=1)) if len(teacher) > 1 else 0.0,
            "baseline_mse_mean": float(baseline.mean()),
            "baseline_mse_sd": float(baseline.std(ddof=1)) if len(baseline) > 1 else 0.0,
            "student_mse_mean": float(student.mean()),
            "student_mse_sd": float(student.std(ddof=1)) if len(student) > 1 else 0.0,
            "student_over_baseline": ratio,
            "student_improvement_percent": (1.0 - ratio) * 100.0,
        })
    return summary


def _write_summary(summary: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(summary[0].keys()) if summary else ["L", "H"]
    temporary = f"{path}.tmp"
    with open(temporary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    os.replace(temporary, path)


def _plot_summary(summary: list[dict], path: str) -> None:
    if not summary:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    labels = [f"L{item['L']}/H{item['H']}" for item in summary]
    arms = ("teacher_mse_mean", "baseline_mse_mean", "student_mse_mean")
    sd_arms = ("teacher_mse_sd", "baseline_mse_sd", "student_mse_sd")
    arm_labels = ("teacher", "PatchTST baseline", "PatchTST FGL")
    x = np.arange(len(summary))
    width = 0.26

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for offset, (arm, sd_arm, label) in enumerate(zip(arms, sd_arms, arm_labels)):
        means = np.asarray([item[arm] for item in summary], dtype=float)
        sds = np.asarray([item[sd_arm] for item in summary], dtype=float)
        ax.bar(x + (offset - 1) * width, means, width, yerr=sds, capsize=3, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("test physical MSE")
    ax.set_title("PatchTST continuous FGL on CSTR")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_all(args, out_path: str | os.PathLike[str] = _DEFAULT_OUT,
            summary_path: str | os.PathLike[str] | None = None,
            plot_path: str | os.PathLike[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Run the requested grid; replace rows for the selected (dataset,L,H)."""
    os.makedirs(os.path.join(_RESULTS_DIR, "logs"), exist_ok=True)
    out_path = os.fspath(out_path)
    summary_path = os.fspath(summary_path or _DEFAULT_SUMMARY)
    plot_path = os.fspath(plot_path or _DEFAULT_PLOT)
    L_values = _parse_int_list(getattr(args, "L_values", None), "20")
    H_values = _parse_int_list(getattr(args, "H_values", None), "12,15")
    seeds = int(args.seeds) if getattr(args, "seeds", None) is not None else 5
    data = _load_data(args.dataset)
    existing = _read_rows(out_path)

    new_rows: list[dict] = []
    for L in L_values:
        for H in H_values:
            for seed in range(seeds):
                result = run_patchtst_fgl(
                    data,
                    lookback_window=L,
                    forecasting_horizon=H,
                    alpha=float(args.alpha),
                    epochs=int(args.baseline_epochs),
                    batch_size=int(args.batch_size),
                    patience=int(args.patience),
                    lr=float(args.lr),
                    seed=seed,
                    d_model=int(args.d_model),
                    nhead=int(args.nhead),
                    dim_feedforward=int(args.dim_feedforward),
                    dropout=float(args.dropout),
                    verbose=True,
                    label=f"{args.dataset}/L{L}H{H}/s{seed}",
                )
                row = {
                    "dataset": args.dataset, "L": L, "H": H, "seed": seed,
                    **result,
                    "d_model": int(args.d_model), "nhead": int(args.nhead),
                    "dim_feedforward": int(args.dim_feedforward),
                    "dropout": float(args.dropout), "lr": float(args.lr),
                    "epochs": int(args.baseline_epochs),
                    "batch_size": int(args.batch_size),
                    "patience": int(args.patience),
                }
                print("  row:", {key: row[key] for key in
                                  ("dataset", "L", "H", "seed", "baseline_mse",
                                   "student_mse", "improvement")}, flush=True)
                new_rows.append(row)

    selected = {(args.dataset, L, H) for L in L_values for H in H_values}
    rows = [row for row in existing
            if (row["dataset"], int(row["L"]), int(row["H"])) not in selected]
    rows.extend(new_rows)
    rows.sort(key=lambda row: (row["dataset"], int(row["L"]), int(row["H"]), int(row["seed"])))
    _write_rows(out_path, rows)

    requested = {(args.dataset, L, H) for L in L_values for H in H_values}
    summary = _summarize([row for row in rows
                          if (row["dataset"], int(row["L"]), int(row["H"])) in requested])
    _write_summary(summary, summary_path)
    _plot_summary(summary, plot_path)

    print("\nSUMMARY PatchTST-FGL (physical test MSE)")
    for item in summary:
        print(f"  L{item['L']}/H{item['H']}: teacher={item['teacher_mse_mean']:.6g} "
              f"baseline={item['baseline_mse_mean']:.6g} student={item['student_mse_mean']:.6g} "
              f"student/baseline={item['student_over_baseline']:.3f} "
              f"(delta={item['student_improvement_percent']:+.2f}%)")
    print(f"  -> {out_path}")
    print(f"  -> {summary_path}")
    print(f"  -> {plot_path}")
    return rows, summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L_values", default="20")
    parser.add_argument("--H_values", default="12,15")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--baseline_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dataset", choices=("h2o", "temperature"), default="h2o")
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
