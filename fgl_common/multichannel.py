"""Run existing FGL independently on exported complex CSTR channels."""

import csv
import os
import pickle
import re
from pathlib import Path

from .training import run_fgl_experiment


_CHANNEL_RE = re.compile(r"data_complex_cstr_(?P<channel>.+)_tau(?P<tau>\d+)\.pkl$")


def list_complex_channels(data_dir, tau=None):
    root = Path(data_dir)
    rows = []
    for path in sorted(root.glob("data_complex_cstr_*_tau*.pkl")):
        match = _CHANNEL_RE.search(path.name)
        if not match:
            continue
        row_tau = int(match.group("tau"))
        if tau is not None and row_tau != tau:
            continue
        rows.append({"path": str(path), "channel": match.group("channel"), "tau": row_tau})
    return rows


def run_channel_grid(data_dir, cells, seeds, out_csv, quick=False, epochs=30,
                     batch_size=64, patience=5, alpha=0.5, temperature=4.0,
                     num_bins=50):
    channels = list_complex_channels(data_dir)
    rows = []
    for item in channels:
        with open(item["path"], "rb") as f:
            data = pickle.load(f)
        for L, H in cells:
            for seed in seeds:
                result = run_fgl_experiment(
                    data,
                    lookback_window=L,
                    forecasting_horizon=H,
                    alpha=alpha,
                    temperature=temperature,
                    num_bins=num_bins,
                    epochs=1 if quick else epochs,
                    batch_size=batch_size,
                    patience=patience,
                    seed=seed,
                    verbose=False,
                    label=f"complex-{item['channel']}-tau{item['tau']}",
                )
                rows.append({
                    **item,
                    "L": L,
                    "H": H,
                    "seed": seed,
                    "baseline_mse": result["baseline"],
                    "teacher_mse": result["teacher"],
                    "student_mse": result["student"],
                    "fgl_delta": result["improvement"],
                })
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fields = list(rows[0].keys()) if rows else [
        "path", "channel", "tau", "L", "H", "seed", "baseline_mse",
        "teacher_mse", "student_mse", "fgl_delta",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows
