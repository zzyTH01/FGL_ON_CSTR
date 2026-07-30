#!/usr/bin/env python
"""Rosenstein 最大 Lyapunov 指数估计(纯 numpy/scipy,无 nolds 依赖)。

对延迟反馈 CSTR 数据集批量估计 λ_max,补"低周期性≠严格混沌,未算 Lyapunov"缺口,
并为 floor_determinants 战役 H3(floor ~ teacher_mse * exp(λ·H))提供 λ。

用法::

    uv run python cstr/lyapunov_delayed.py
    uv run python cstr/lyapunov_delayed.py --glob 'cstr/data/data_delayed_stable_h2o_tau*_s1_A0.9_b0.03.pkl'
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
from scipy.signal import correlate


def _autocorr_zero_lag(series, max_lag=None):
    """首个自相关过零的 lag(嵌入延迟估计)。"""
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 2, 300)
    full = correlate(x, x, mode="full")[n - 1:]  # autocorr for lag>=0
    denom = full[0]
    if abs(denom) < 1e-12:
        return 1
    full = full / denom
    lag = 1
    while lag < max_lag and full[lag] > 0:
        lag += 1
    return max(lag, 1)


def largest_lyapunov_rosenstein(series, emb_dim=5, emb_lag=None,
                                min_tsep=1, max_k=None, fit_range=None):
    """Rosenstein 最大 Lyapunov 指数(每样本)。

    Args:
        series: 1D array.
        emb_dim: 嵌入维数。
        emb_lag: 嵌入延迟;None 则由自相关过零估计。
        min_tsep: 最近邻搜索时排除的时间邻近点数(避免时间相关)。
        max_k: 跟踪发散的最大步数;None 则 min(M//2, 100)。
        fit_range: (a, b) 线性拟合区(1-indexed 步数);None 则 (1, max_k//2)。
    Returns:
        ``(lyap, ks, S)``。lyap = <log d(k)> 线性区斜率;正=>混沌发散。
        退化输入(常数/太短)返回 ``(0.0, ks, S)``。
    """
    s = np.asarray(series, dtype=float)
    s = s - s.mean()
    n = len(s)
    if emb_lag is None:
        emb_lag = _autocorr_zero_lag(s)
    M = n - (emb_dim - 1) * emb_lag          # 嵌入向量数
    if M <= emb_dim + 2:
        return 0.0, np.array([]), np.array([])
    Y = np.empty((M, emb_dim))
    for j in range(emb_dim):
        Y[:, j] = s[j * emb_lag: j * emb_lag + M]
    if max_k is None:
        max_k = min(M // 2, 100)
    if max_k < 2:
        return 0.0, np.array([]), np.array([])

    # 每个点 i 找最近邻 j(排除时间邻近)
    upper = M - max_k
    pairs = []
    for i in range(upper):
        diff = Y[:upper] - Y[i]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        dist2[i] = np.inf
        lo = max(0, i - min_tsep); hi = min(upper, i + min_tsep + 1)
        dist2[lo:hi] = np.inf
        j = int(np.argmin(dist2))
        d0 = dist2[j]
        if np.isfinite(d0) and d0 > 0:
            pairs.append((i, j))

    if not pairs:
        return 0.0, np.arange(1, max_k + 1), np.full(max_k, np.nan)

    S = np.zeros(max_k)
    counts = np.zeros(max_k)
    for (i, j) in pairs:
        for k in range(1, max_k + 1):
            if i + k < M and j + k < M:
                dk = np.linalg.norm(Y[i + k] - Y[j + k])
                if dk > 0:
                    S[k - 1] += np.log(dk)
                    counts[k - 1] += 1
    valid = counts > 0
    S[valid] /= counts[valid]
    S[~valid] = np.nan
    ks = np.arange(1, max_k + 1)

    if fit_range is None:
        fit_range = (1, max(2, max_k // 2))
    a, b = fit_range
    seg_k = ks[a - 1:b]
    seg_S = S[a - 1:b]
    mask = np.isfinite(seg_S)
    if mask.sum() < 2:
        return 0.0, ks, S
    lyap = float(np.polyfit(seg_k[mask], seg_S[mask], 1)[0])
    return lyap, ks, S


def _tau_from_name(fn):
    m = re.search(r"tau(\d+)_", os.path.basename(fn))
    return int(m.group(1)) if m else ""


def estimate_all(glob_pattern, out_csv, burn_frac=0.2):
    """对 glob 匹配的所有延迟数据集估计 λ,写 CSV。

    Returns 列表 of ``{"file", "tau", "lyap", "N"}``。
    """
    import pickle
    rows = []
    for fn in sorted(glob.glob(glob_pattern)):
        with open(fn, "rb") as f:
            d = pickle.load(f)
        series = np.asarray(d[:, 0], dtype=float)
        burn = min(1000, max(50, int(len(series) * burn_frac)))
        lyap, _, _ = largest_lyapunov_rosenstein(series[burn:])
        rows.append({"file": os.path.basename(fn), "tau": _tau_from_name(fn),
                     "lyap": lyap, "N": len(series)})
        print(f"  {os.path.basename(fn)}  tau={_tau_from_name(fn)}  "
              f"λ={lyap:+.4f}  N={len(series)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "tau", "lyap", "N"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}  ({len(rows)} rows)")
    return rows


def main():
    p = argparse.ArgumentParser(description="Rosenstein λ for delayed-feedback CSTR datasets")
    p.add_argument("--glob", default="cstr/data/data_delayed_stable_h2o_tau*_s1_A0.9_b0.03.pkl")
    p.add_argument("--out", default="cstr/results/lyapunov_tau.csv")
    args = p.parse_args()
    estimate_all(args.glob, args.out)


if __name__ == "__main__":
    main()
