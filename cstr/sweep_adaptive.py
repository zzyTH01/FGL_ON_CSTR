#!/usr/bin/env python
"""L×H 网格:自适应蒸馏变体 E vs A(标准 FGL student)的 student MSE 对比。

每个 (L,H,seed) 配对跑 A 与 E。同 seed 下,``run_adaptive_weight`` 内部
``torch.manual_seed(seed)`` 使 teacher/baseline/prelim 完全一致,两者只差最终 student
的 KL 加权 → E−A 是低方差的**配对比较**,少量 seed 即可看出网格趋势。

用法::

    uv run python cstr/sweep_adaptive.py                                  # 默认 5×5, A/E, 2 seeds
    uv run python cstr/sweep_adaptive.py --seeds 3 --epochs 30
    uv run python cstr/sweep_adaptive.py --L_values 8,20 --H_values 5,15 --seeds 1 --epochs 3   # 冒烟

输出 ``cstr/results/adaptive_lh_sweep.csv`` + 热力图 ``adaptive_lh_E_vs_A.png``。
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
from fgl_common import run_adaptive_weight

_CSTR_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name="data_h2o.pkl"):
    for d in ("data", "."):
        p = os.path.join(_CSTR_DIR, d, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(name)


def run_all(args):
    """核心逻辑:L×H 网格 × variants × seeds 跑自适应蒸馏,写 adaptive_lh_sweep.csv + 热力图。

    接受 argparse Namespace;与 cstr/run.py 的字段共用部分直接读取,专有字段
    (L_values/H_values/variants)用 getattr 缺省,供 run.py 统一入口调用。
    """
    Ls = [int(x) for x in (getattr(args, "L_values", None) or "8,20,35,50,72").split(",")]
    Hs = [int(x) for x in (getattr(args, "H_values", None) or "5,15,30,45,60").split(",")]
    variants = [v.strip() for v in (getattr(args, "variants", None) or "A,E").split(",")]
    seeds = list(range(getattr(args, "seeds", 2)))
    data = _load()

    outdir = os.path.join(_CSTR_DIR, "results")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "adaptive_lh_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["L", "H", "seed", "variant", "baseline_mse",
                                "student_mse_init", "student_mse", "fgl_delta", "init_delta"])

    results = {}  # (L,H) -> {variant -> [student_mse per seed]}
    total = len(Ls) * len(Hs) * len(variants) * len(seeds)
    done = 0
    for L in Ls:
        for H in Hs:
            cell = {}
            for v in variants:
                mses = []
                for s in seeds:
                    r = run_adaptive_weight(data, L=L, H=H, alpha=args.alpha,
                                            temperature=args.temperature, num_bins=args.bins,
                                            epochs=args.epochs, seed=s, variant=v, verbose=False)
                    with open(csv_path, "a", newline="") as f:
                        csv.writer(f).writerow([L, H, s, v, r["baseline_mse"],
                                                r["student_mse_init"], r["student_mse"],
                                                r["fgl_delta"], r["init_delta"]])
                    mses.append(r["student_mse"])
                    done += 1
                cell[v] = np.array(mses)
            results[(L, H)] = cell
            if "A" in cell and "E" in cell:
                A = cell["A"].mean(); E = cell["E"].mean()
                rel = (A - E) / A * 100 if A > 0 else float("nan")
                wins = int(np.sum(cell["E"] < cell["A"]))
                print(f"[{done}/{total}] L={L:<3} H={H:<3}: A={A:6.1f}  E={E:6.1f}  "
                      f"E低 {rel:+5.1f}%  (E<A 在 {wins}/{len(seeds)} seed)", flush=True)
            else:
                print(f"[{done}/{total}] L={L:<3} H={H:<3}: done", flush=True)

    _report(results, Ls, Hs, seeds, outdir)


def main():
    ap = argparse.ArgumentParser(description="L×H sweep: adaptive variant E vs A")
    ap.add_argument("--L_values", default="8,20,35,50,72")
    ap.add_argument("--H_values", default="5,15,30,45,60")
    ap.add_argument("--variants", default="A,E")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("-T", "--temperature", type=float, default=4.0, dest="temperature")
    ap.add_argument("--bins", type=int, default=50)
    args = ap.parse_args()
    run_all(args)


def _report(results, Ls, Hs, seeds, outdir):
    # 聚合表
    print(f"\n{'='*70}\nL×H 网格:E 相对 A 的 student MSE 下降 (%)  [+ = E 更好]\n{'='*70}")
    header = "L\\H | " + " | ".join(f"H={h:<3}" for h in Hs)
    print(header)
    print("-" * len(header))
    M = np.full((len(Ls), len(Hs)), np.nan)
    W = np.full((len(Ls), len(Hs)), 0)     # E<A 的 seed 总数
    Sn = np.full((len(Ls), len(Hs)), 0)    # seed 总数
    for i, L in enumerate(Ls):
        cells = [results.get((L, H)) for H in Hs]
        row = []
        for j, H in enumerate(Hs):
            cell = cells[j]
            if cell and "A" in cell and "E" in cell:
                A = cell["A"].mean(); E = cell["E"].mean()
                rel = (A - E) / A * 100 if A > 0 else float("nan")
                M[i, j] = rel
                W[i, j] = int(np.sum(cell["E"] < cell["A"]))
                Sn[i, j] = len(cell["E"])
                row.append(f"{rel:+5.1f}" if not np.isnan(rel) else "  n/a")
            else:
                row.append("  n/a")
        print(f"L={L:<3}| " + " | ".join(row))

    if np.isfinite(M).any():
        finite = M[np.isfinite(M)]
        wins_total = int(W.sum()); seeds_total = int(Sn.sum())
        print(f"\n网格汇总:{len(finite)} 个 cell,E 平均下降 {np.nanmean(M):+.1f}%,"
              f"中位数 {np.nanmedian(M):+.1f}%。E<A 在 {wins_total}/{seeds_total} 个 (cell×seed)。")
        _heatmap(M, W, Sn, Ls, Hs, outdir)
    print(f"\nCSV: {os.path.join(outdir, 'adaptive_lh_sweep.csv')}")


def _heatmap(M, W, Sn, Ls, Hs, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    vmax = max(20, np.nanmax(np.abs(M)))
    im = ax.imshow(M, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(Hs))); ax.set_xticklabels([f"H={h}" for h in Hs])
    ax.set_yticks(range(len(Ls))); ax.set_yticklabels([f"L={L}" for L in Ls])
    ax.set_xlabel("horizon"); ax.set_ylabel("lookback")
    ax.set_title("E vs A: student MSE relative reduction (%)  [+ = E better]\n"
                 "(cell = mean over seeds; annotation = E<A seed count)")
    for i in range(len(Ls)):
        for j in range(len(Hs)):
            val = M[i, j]
            txt = f"{val:+.0f}%\n{int(W[i,j])}/{int(Sn[i,j])}" if np.isfinite(val) else "n/a"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="E vs A student MSE reduction (%) [+ = E better]")
    fig.tight_layout()
    png = os.path.join(outdir, "adaptive_lh_E_vs_A.png")
    fig.savefig(png, dpi=120)
    print(f"热力图: {png}")


if __name__ == "__main__":
    main()
