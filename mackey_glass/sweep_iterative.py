#!/usr/bin/env python
"""MG τ=13:迭代蒸馏 4 臂对比 + 逐轮 MSE 曲线(镜像 cstr/sweep_iterative.py,换 MG 数据源)。

用法::
    # Phase 0 试点:3 跨阈值点
    FGL_DEVICE=cpu uv run python mackey_glass/sweep_iterative.py --cells "4,7;4,10;13,7" --seeds 3 --epochs 50 --round_epochs 20 --K 5
输出 mackey_glass/results/iterative_mg_sweep.csv + iterative_mg_curves.png。
"""
import argparse
import csv
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_MG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _MG_DIR)  # utils.utils 在此

import numpy as np
import torch
from fgl_common import run_iterative_distillation
from utils.utils import MackeyGlass

_RESULTS_DIR = os.path.join(_MG_DIR, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)
ARMS = ("A_single", "E_single", "A_iter", "E_iter")


def generate_mg_data(tau=13.0, n_points=10000, seed=42):
    """生成 MG 序列,返回 (N,2) 张量(两列均为序列值,自回归)。与 run.py 同口径。"""
    mg = MackeyGlass(tau=tau, constant_past=0.9, nmg=10, beta=0.2, gamma=0.1,
                     dt=1.0, splits=(float(n_points), 0.0), seed_id=seed)
    vals = [mg[idx][1].squeeze().item() for idx in range(len(mg))]
    col = torch.tensor(vals, dtype=torch.float64).unsqueeze(1)
    return torch.cat((col, col.clone()), dim=1), mg.lyap_exp


def _parse_cells(args):
    if args.grid:
        Ls = [4, 7, 10, 13, 16]
        Hs = [4, 7, 10, 13, 16]
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
    ap = argparse.ArgumentParser(description="MG iterative distillation 4-arm sweep")
    ap.add_argument("--cells", default="4,7;4,10;13,7", help="semicolon-separated L,H pairs")
    ap.add_argument("--grid", action="store_true", help="5x5 grid (overrides --cells)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--round_epochs", type=int, default=20)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("-T", "--temperature", type=float, default=4.0, dest="temperature")
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--tau", type=float, default=13.0)
    ap.add_argument("--n_points", type=int, default=10000)
    ap.add_argument("--variant", default="E", help="weighting variant: E / E-soft / C / D")
    ap.add_argument("--w_floor", type=float, default=None, help="[E-soft] 软地板(默认 0.2)")
    args = ap.parse_args()

    _parts = []
    if args.variant != "E":
        _parts.append(args.variant)
    if args.w_floor is not None:
        _parts.append(f"wf{args.w_floor}")
    tag = ("_" + "_".join(_parts)) if _parts else ""
    cells = _parse_cells(args)
    seeds = list(range(args.seeds))
    data, lyap = generate_mg_data(tau=args.tau, n_points=args.n_points)
    print(f"MG τ={args.tau}, Lyapunov={lyap:+.6f}, {args.n_points} pts, variant={args.variant}", flush=True)
    csv_path = os.path.join(_RESULTS_DIR, f"iterative_mg_sweep{tag}.csv")
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
                batch_size=args.batch_size, K=args.K, seed=s, variant=args.variant,
                w_floor=args.w_floor, verbose=False)
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
        print(f"[{done}/{total}] L={L:<3} H={H:<3}: E_iter={e_iter:6.3f}  "
              f"E_single={e_single:6.3f}  A_iter={a_iter:6.3f}  "
              f"(E_iter vs A_iter {rel:+.1f}%)", flush=True)

    _report(cell_results, tag)


def _report(cell_results, tag=""):
    print(f"\n{'=' * 70}\nE-iter 相对对照的 student MSE 下降 (%)  [+ = E-iter 更好]\n{'=' * 70}")
    print(f"{'L,H':>10} | {'vs A-single':>12} | {'vs E-single':>12} | {'vs A-iter':>10}")
    print("-" * 60)
    for (L, H), (per_arm, _) in cell_results.items():
        a_s = np.mean(per_arm["A_single"]); e_s = np.mean(per_arm["E_single"])
        a_i = np.mean(per_arm["A_iter"]); e_i = np.mean(per_arm["E_iter"])

        def rel(base):
            return (base - e_i) / base * 100 if base > 0 else float("nan")
        print(f"({L:>2},{H:<3})    | {rel(a_s):>+11.1f}% | {rel(e_s):>+11.1f}% | {rel(a_i):>+9.1f}%")
    _plot_curves(cell_results, tag)
    print(f"\nCSV: {os.path.join(_RESULTS_DIR, f'iterative_mg_sweep{tag}.csv')}")


def _plot_curves(cell_results, tag=""):
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
            ax.plot(range(max_len), np.nanmean(arr, axis=0), marker="o", label=arm)
        ax.set_title(f"L={L}, H={H}")
        ax.set_xlabel("round (0 = round-0 student)")
        ax.set_ylabel("val MSE")
        ax.legend(fontsize=8)
    fig.suptitle("MG τ=13: Per-round val MSE by arm (E-iter shape answers H1)")
    fig.tight_layout()
    png = os.path.join(_RESULTS_DIR, f"iterative_mg_curves{tag}.png")
    fig.savefig(png, dpi=120)
    print(f"逐轮曲线: {png}")


if __name__ == "__main__":
    main()
