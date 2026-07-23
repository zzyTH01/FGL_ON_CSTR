"""通用 L×H 扫描框架。

把原本在 ``cstr/exp/fgl_cstr_lh_sweep.py``、``mackey_glass/exp/mg_lh_sweep.py``、
``lorenz/lh_sweep.py`` 三处近乎相同的扫描逻辑收敛为 ``run_lh_sweep``。

输出与原脚本一致:``{name}.csv`` + ``{name}.png``(三联热力图) + ``{name}_report.md``。
"""
import csv
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _get(d, *keys, default=None):
    """Fetch the first present key from a dict (tolerates baseline / baseline_mse)."""
    for k in keys:
        if k in d:
            return d[k]
    return default


def run_lh_sweep(run_fn, data, L_values, H_values, seeds,
                 outdir, name="lh_sweep",
                 title=None, period_label=None,
                 extra_meta=None, verbose=True):
    """Generic L×H sweep.

    Args:
        run_fn: callable ``(L, H, data, seed) -> dict`` returning at least
            baseline / student / teacher MSE (keys ``baseline[_mse]`` etc.).
        data: raw dataset passed through to ``run_fn``.
        L_values / H_values / seeds: sweep grid.
        outdir: output directory for CSV / PNG / MD.
        name: filename stem (``{name}.csv|.png|_report.md``).
    Returns:
        list of result rows (dicts).
    """
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"{name}.csv")

    if verbose:
        total = len(L_values) * len(H_values) * len(seeds)
        print(f"\nL×H Sweep: L={list(L_values)}  H={list(H_values)}  seeds={list(seeds)}")
        print(f"Configs: {len(L_values) * len(H_values)}  Total runs: {total}")
        if period_label:
            print(f"({period_label})")

    rows = []
    for L in L_values:
        for H in H_values:
            if verbose:
                print(f"\n{'=' * 50}\n  L={L:2d} H={H:2d} (L+H-1={L + H - 1:3d})\n{'=' * 50}")
            for s in seeds:
                r = run_fn(L, H, data, s)
                bm = _get(r, 'baseline_mse', 'baseline')
                sm = _get(r, 'student_mse', 'student')
                tm = _get(r, 'teacher_mse', 'teacher')
                d = (bm - sm) / bm * 100 if bm > 0 else 0
                rows.append({"L": L, "H": H, "seed": s,
                             "baseline_mse": bm, "teacher_mse": tm, "student_mse": sm,
                             "abs_improvement": bm - sm, "fgl_delta": d})
                if verbose:
                    print(f"  seed={s}: Base={bm:.2f} Stu={sm:.2f} Δ={d:+.1f}%")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["L", "H", "seed", "baseline_mse", "teacher_mse",
                                          "student_mse", "abs_improvement", "fgl_delta"])
        w.writeheader()
        w.writerows(rows)
    if verbose:
        print(f"\nSaved: {csv_path}")

    # aggregate
    agg = defaultdict(list)
    for r in rows:
        agg[(r["L"], r["H"])].append(r)
    L_s = sorted(set(r["L"] for r in rows))
    H_s = sorted(set(r["H"] for r in rows))

    grid_d = np.zeros((len(L_s), len(H_s)))
    grid_b = np.zeros((len(L_s), len(H_s)))
    grid_a = np.zeros((len(L_s), len(H_s)))
    for i, L in enumerate(L_s):
        for j, H in enumerate(H_s):
            rs = agg[(L, H)]
            grid_d[i, j] = np.mean([r["fgl_delta"] for r in rs])
            grid_b[i, j] = np.mean([r["baseline_mse"] for r in rs])
            grid_a[i, j] = np.mean([r["abs_improvement"] for r in rs])

    _plot_heatmaps(grid_d, grid_a, grid_b, L_s, H_s,
                   os.path.join(outdir, f"{name}.png"), title or name)
    _write_report(os.path.join(outdir, f"{name}_report.md"), agg, L_s, H_s,
                  name, extra_meta, len(rows))
    if verbose:
        print(f"Figure + report saved to: {outdir}")
    return rows


def _plot_heatmaps(grid_d, grid_a, grid_b, L_s, H_s, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for ax, grid, t, cmap, label in [
        (axes[0], grid_d, "FGL Δ%", "RdYlGn", "Δ%"),
        (axes[1], grid_a, "Abs Improvement (Base−Stu)", "RdYlGn", "Abs Imp"),
        (axes[2], grid_b, "Baseline MSE (task difficulty)", "YlOrRd", "Base MSE"),
    ]:
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap,
                       extent=[H_s[0] - 0.5, H_s[-1] + 0.5, L_s[0] - 0.5, L_s[-1] + 0.5])
        ax.set_xticks(H_s)
        ax.set_yticks(L_s)
        ax.set_xlabel("H (forecast horizon)")
        ax.set_ylabel("L (lookback)")
        ax.set_title(t, fontweight="bold")
        for i in range(len(L_s)):
            for j in range(len(H_s)):
                v = grid[i, j]
                ax.text(H_s[j], L_s[i], f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > (grid.max() + grid.min()) / 3 else "black")
        plt.colorbar(im, ax=ax, label=label)
    plt.suptitle(title, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_report(path, agg, L_s, H_s, name, extra_meta, n_rows):
    with open(path, "w") as f:
        f.write(f"# {name}\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        if extra_meta:
            for k, v in extra_meta.items():
                f.write(f"**{k}:** {v}\n")
            f.write("\n")
        f.write("## FGL Δ% Heatmap\n\n")
        f.write("| L\\H | " + " | ".join(f"{h}" for h in H_s) + " |\n")
        f.write("|" + "---|" * (len(H_s) + 1) + "\n")
        for L in L_s:
            f.write(f"| {L} | " + " | ".join(
                f"{np.mean([r['fgl_delta'] for r in agg[(L, H)]]):+.1f}%" for H in H_s) + " |\n")
        f.write(f"\n## Data: `{name}.csv` ({n_rows} rows)\n")
