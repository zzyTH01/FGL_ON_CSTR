#!/usr/bin/env python
"""分析迭代蒸馏 sweep CSV:per-seed 最小 test MSE + 配对统计(E_iter vs A_iter 等)。

用法::
    uv run python cstr/analyze_iterative.py cstr/results/iterative_phase1_anchor.csv
    uv run python cstr/analyze_iterative.py cstr/results/iterative_phase1_grid.csv
"""
import sys
import csv
from collections import defaultdict
from math import sqrt

import numpy as np
from scipy import stats

ARMS = ("A_single", "E_single", "A_iter", "E_iter")


def _per_seed_min_test(rows):
    """(L,H) -> seed -> {arm: min test MSE over rounds}。"""
    cells = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        c = [float(x) for x in r["mse_curve_test"].split(";")]
        cells[(r["L"], r["H"])][r["seed"]][r["arm"]] = min(c)
    return cells


def _paired(a, b):
    """配对统计:返回 (mean_diff, t, p_t, p_wilcoxon, n)。a-b 正 = a 更大。"""
    a = np.array(a, float); b = np.array(b, float)
    diff = a - b
    n = len(diff)
    if n < 2 or diff.std(ddof=1) == 0:
        return diff.mean(), float("nan"), float("nan"), float("nan"), n
    t = diff.mean() / (diff.std(ddof=1) / sqrt(n))
    p_t = stats.ttest_rel(a, b).pvalue
    try:
        p_w = stats.wilcoxon(a, b, zero_method="wilcox").pvalue
    except ValueError:
        p_w = float("nan")
    return diff.mean(), t, p_t, p_w, n


def analyze(path):
    rows = list(csv.DictReader(open(path)))
    cells = _per_seed_min_test(rows)
    print(f"\n{'=' * 78}\n分析: {path}\n{'=' * 78}")

    all_e, all_a, all_es, all_as = [], [], [], []
    for (L, H), seeds in sorted(cells.items(), key=lambda x: (int(x[0][0]), int(x[0][1]))):
        e = [seeds[s]["E_iter"] for s in seeds]
        a = [seeds[s]["A_iter"] for s in seeds]
        es = [seeds[s]["E_single"] for s in seeds]
        asg = [seeds[s]["A_single"] for s in seeds]
        n = len(e)
        e_win = int(np.sum(np.array(e) < np.array(a)))
        # 配对:E_iter vs A_iter(正差异 = A 更大 = E 更好)
        md, t, p_t, p_w, _ = _paired(a, e)
        rel = (np.array(a) - np.array(e)) / np.array(a) * 100
        print(f"\n  L={L} H={H} (n={n} seeds, per-seed-min test MSE):")
        print(f"    E_iter  per-seed: {np.array(e).round(2).tolist()}")
        print(f"    A_iter  per-seed: {np.array(a).round(2).tolist()}")
        print(f"    E_iter median={np.median(e):.2f}  A_iter median={np.median(a):.2f}  "
              f"| E<A in {e_win}/{n} seeds")
        print(f"    E_iter vs A_iter: mean rel reduction {rel.mean():+.1f}% | "
              f"paired t={t:+.2f} p={p_t:.4g} | Wilcoxon p={p_w:.4g}")
        # 次判据
        _, _, p_es, _, _ = _paired(es, e)
        _, _, p_as, _, _ = _paired(asg, a)
        print(f"    E_iter vs E_single: p_t={p_es:.4g}  |  A_iter vs A_single: p_t={p_as:.4g}")
        all_e += e; all_a += a; all_es += es; all_as += asg

    if len(cells) > 1:
        md, t, p_t, p_w, n = _paired(all_a, all_e)
        wins = int(np.sum(np.array(all_e) < np.array(all_a)))
        rel = (np.array(all_a) - np.array(all_e)) / np.array(all_a) * 100
        print(f"\n  {'─' * 60}")
        print(f"  GRID-WIDE E_iter vs A_iter (n={n} cell×seed):")
        print(f"    E<A in {wins}/{n} ({wins/n*100:.0f}%) | mean rel reduction {rel.mean():+.1f}% "
              f"(median {np.median(rel):+.1f}%) | paired t={t:+.2f} p={p_t:.4g}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: analyze_iterative.py <csv> [<csv> ...]")
    for p in sys.argv[1:]:
        analyze(p)
