#!/usr/bin/env python
"""
Plot the delayed-feedback CSTR H2O datasets (the aperiodic τ-sweep outputs).

Small multiples of the H2O mass-fraction time series for each saved τ,
ordered by τ and labelled with the periodicity score, to visualize the
periodic -> aperiodic transition across the τ sweep.

Usage:
  uv run python cstr/plot_delayed_stable.py
"""
import os
import re
import csv
import glob
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
TAG = "s1_A0.9_b0.03"
DT = 0.1


def load_periodicity():
    """τ -> periodicity from the sweep CSV (matching TAG)."""
    csv_path = os.path.join(RESULTS, f"delayed_tau_sweep_{TAG}.csv")
    per = {}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row["status"] == "ok":
                    per[int(row["tau"])] = float(row["periodicity"])
    return per


def main():
    import torch  # only needed to unpickle the tensors
    per = load_periodicity()
    pattern = os.path.join(DATA, f"data_delayed_stable_h2o_tau*_{TAG}.pkl")
    files = glob.glob(pattern)
    if not files:
        raise SystemExit(f"no datasets matching {pattern}; run the sweep first")

    items = []
    for fp in files:
        tau = int(re.search(r"tau(\d+)", os.path.basename(fp)).group(1))
        with open(fp, "rb") as f:
            t = pickle.load(f)
        items.append((tau, np.asarray(t[:, 0], dtype=float)))
    items.sort(key=lambda x: x[0])

    n = len(items)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 1.9),
                             sharex=True)
    # settled segment: past burn-in (100s) + onset (50s), ~25 natural cycles
    lo, hi = int(100 / DT), int(300 / DT)
    tt = np.arange(lo, hi) * DT
    for ax, (tau, s) in zip(axes.flat, items):
        ax.plot(tt, s[lo:hi], lw=0.6, color="C0")
        score = per.get(tau, float("nan"))
        ax.set_title(f"τ={tau}  (per={score:.2f})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("H₂O mass frac", fontsize=8)
    fig.suptitle("Delayed-feedback CSTR — H₂O across the τ sweep "
                 "(sign=+1, A=0.9, β=0.03; window 100–300 s)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(RESULTS, "delayed_stable_h2o_panels.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}  ({n} panels, τ = {[t for t,_ in items]})")
    plt.close(fig)


if __name__ == "__main__":
    main()
