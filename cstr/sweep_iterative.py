#!/usr/bin/env python
"""L×H / 显式 cells:迭代蒸馏 4 臂对比 + 逐轮 MSE 曲线。

用法::

    # Phase 0 试点:3 典型点
    uv run python cstr/sweep_iterative.py --cells "20,15;8,30;72,15" --seeds 3 --epochs 20 --round_epochs 10 --K 3
    # Phase 1 全网格
    uv run python cstr/sweep_iterative.py --grid --seeds 2 --epochs 30 --round_epochs 15 --K 5

输出 cstr/results/iterative_sweep.csv + 逐轮曲线 png。
"""
import argparse
import csv
import os
import pickle
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from fgl_common import run_iterative_distillation

_CSTR_DIR = os.path.dirname(os.path.abspath(__file__))
ARMS = ("A_single", "E_single", "A_iter", "E_iter")


def _load(name="data_h2o.pkl"):
    for d in ("data", "."):
        p = os.path.join(_CSTR_DIR, d, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(name)


def _parse_cells(args):
    if args.grid:
        Ls = [8, 20, 35, 50, 72]
        Hs = [5, 15, 30, 45, 60]
        return [(L, H) for L in Ls for H in Hs]
    pairs = []
    for tok in args.cells.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        L, H = tok.split(",")
        pairs.append((int(L), int(H)))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="iterative distillation 4-arm sweep")
    ap.add_argument("--cells", default="20,15;8,30;72,15", help="semicolon-separated L,H pairs")
    ap.add_argument("--grid", action="store_true", help="5x5 full grid (overrides --cells)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--round_epochs", type=int, default=10)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("-T", "--temperature", type=float, default=4.0, dest="temperature")
    ap.add_argument("--bins", type=int, default=50)
    args = ap.parse_args()

    cells = _parse_cells(args)
    seeds = list(range(args.seeds))
    data = _load()
    outdir = os.path.join(_CSTR_DIR, "results")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "iterative_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["L", "H", "seed", "arm", "baseline_mse", "student_mse", "fgl_delta", "init_delta",
             "rounds_used", "mse_curve_val", "mse_curve_test"])

    cell_results = {}
    total = len(cells) * len(seeds)
    done = 0
    for (L, H) in cells:
        per_arm = {a: [] for a in ARMS}
        curves_val = {a: [] for a in ARMS}
        for s in seeds:
            res = run_iterative_distillation(
                data, L=L, H=H, alpha=args.alpha, temperature=args.temperature,
                num_bins=args.bins, epochs=args.epochs, round_epochs=args.round_epochs,
                K=args.K, seed=s, verbose=False)
            for arm, r in res.items():
                per_arm[arm].append(r["student_mse"])
                curves_val[arm].append(r["mse_curve_val"])
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [L, H, s, arm, r["baseline_mse"], r["student_mse"], r["fgl_delta"],
                         r["init_delta"], r["rounds_used"],
                         ";".join(f"{v:.3f}" for v in r["mse_curve_val"]),
                         ";".join(f"{v:.3f}" for v in r["mse_curve_test"])])
            done += 1
        cell_results[(L, H)] = (per_arm, curves_val)
        e_iter = np.mean(per_arm["E_iter"]); e_single = np.mean(per_arm["E_single"])
        a_iter = np.mean(per_arm["A_iter"])
        rel = (a_iter - e_iter) / a_iter * 100 if a_iter > 0 else float("nan")
        print(f"[{done}/{total}] L={L:<3} H={H:<3}: E_iter={e_iter:6.1f}  "
              f"E_single={e_single:6.1f}  A_iter={a_iter:6.1f}  "
              f"(E_iter vs A_iter {rel:+.1f}%)", flush=True)

    _report(cell_results, outdir)


def _report(cell_results, outdir):
    print(f"\n{'=' * 70}\nE-iter 相对对照的 student MSE 下降 (%)  [+ = E-iter 更好]\n{'=' * 70}")
    print(f"{'L,H':>10} | {'vs A-single':>12} | {'vs E-single':>12} | {'vs A-iter':>10}")
    print("-" * 60)
    for (L, H), (per_arm, _) in cell_results.items():
        a_s = np.mean(per_arm["A_single"]); e_s = np.mean(per_arm["E_single"])
        a_i = np.mean(per_arm["A_iter"]); e_i = np.mean(per_arm["E_iter"])

        def rel(base):
            return (base - e_i) / base * 100 if base > 0 else float("nan")
        print(f"({L:>2},{H:<3})    | {rel(a_s):>+11.1f}% | {rel(e_s):>+11.1f}% | {rel(a_i):>+9.1f}%")
    _plot_curves(cell_results, outdir)
    print(f"\nCSV: {os.path.join(outdir, 'iterative_sweep.csv')}")


def _plot_curves(cell_results, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_cells = len(cell_results)
    fig, axes = plt.subplots(1, n_cells, figsize=(5 * n_cells, 4), squeeze=False)
    for ax, ((L, H), (_, curves_val)) in zip(axes[0], cell_results.items()):
        for arm in ARMS:
            curves = curves_val[arm]
            if not curves:
                continue
            max_len = max(len(c) for c in curves)
            arr = np.full((len(curves), max_len), np.nan)
            for i, c in enumerate(curves):
                arr[i, :len(c)] = c
            mean = np.nanmean(arr, axis=0)
            ax.plot(range(max_len), mean, marker="o", label=arm)
        ax.set_title(f"L={L}, H={H}")
        ax.set_xlabel("round (0 = round-0 student)")
        ax.set_ylabel("val MSE")
        ax.legend(fontsize=8)
    fig.suptitle("Per-round val MSE by arm (lower = better; E-iter shape answers H1)")
    fig.tight_layout()
    png = os.path.join(outdir, "iterative_curves.png")
    fig.savefig(png, dpi=120)
    print(f"逐轮曲线: {png}")


if __name__ == "__main__":
    main()
