#!/usr/bin/env python
"""Mackey-Glass 统一实验入口。

通过 ``EXPERIMENTS`` 配置字典的 ``enabled`` 字段控制实验是否运行(开关),
命令行 ``-e`` 可覆盖。训练与分析逻辑在 ``fgl_common`` 包与 ``utils/utils.py``
(MackeyGlass 数据集类)中;旧单用途脚本在 ``archive/``(仅溯源)。

用法::

    uv run python mackey_glass/run.py                  # 跑所有 enabled=True
    uv run python mackey_glass/run.py -e base,lh_sweep
    uv run python mackey_glass/run.py --list
    uv run python mackey_glass/run.py -e h_threshold --seeds 3

研究结论(见 ``conclusion/final_conclusions.md``):MG τ=13 是 FGL 最佳验证集
(倍周期分岔,2 个反馈环)。几何条件 L+H-1≥τ 在固定 H 扫 L 时严格成立
(CSTR 上该巧合被证伪,见 ``conclusion/floor_study_final_report.md``)。
"""
import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from scipy import stats

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_MG_DIR = os.path.dirname(os.path.abspath(__file__))
# MackeyGlass 类在 mackey_glass/utils/utils.py
sys.path.insert(0, _MG_DIR)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fgl_common import RNN, run_fgl_experiment, run_lh_sweep, run_iterative_distillation  # noqa: E402
from utils.utils import MackeyGlass  # noqa: E402

_RESULTS_DIR = os.path.join(_MG_DIR, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

# Mackey-Glass 默认参数
N_POINTS = 10000
DEFAULT_TAU = 13.0


# ==================== MG data generation ====================
def generate_mg_data(tau=DEFAULT_TAU, n_points=N_POINTS, seed=42):
    """Generate MG series at given τ. Returns (data_tensor (N,2), lyap_exp)."""
    mg = MackeyGlass(tau=tau, constant_past=0.9, nmg=10, beta=0.2, gamma=0.1,
                     dt=1.0, splits=(float(n_points), 0.), seed_id=seed)
    vals = [mg[idx][1].squeeze().item() for idx in range(len(mg))]
    col = torch.tensor(vals, dtype=torch.float64).unsqueeze(1)
    return torch.cat((col, col.clone()), dim=1), mg.lyap_exp


def _load_pkl_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _mg_periodicity(data):
    h2o = data[:, 0].numpy()
    ac = np.correlate(h2o - h2o.mean(), h2o - h2o.mean(), mode="full")
    ac = ac[len(ac) // 2:] / (ac[len(ac) // 2:] + 1e-10)
    score = float(ac[20:200].max()) if len(ac) > 200 else 1.0
    ac_zero = next((i for i in range(1, len(ac)) if ac[i] < 0), len(ac))
    return score, ac_zero


# ==================== Threshold analysis helpers ====================
def find_changepoint(x_sorted, values):
    """Find x that minimizes piecewise SSE with a single break."""
    best_x, best_sse = None, float('inf')
    sse_single = np.sum((values - values.mean()) ** 2)
    for i in range(2, len(x_sorted) - 2):
        left, right = values[:i], values[i:]
        sse = np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2)
        if sse < best_sse:
            best_sse = sse
            best_x = (x_sorted[i - 1] + x_sorted[i]) / 2
    return best_x, best_sse, sse_single


def piecewise_fit(x_arr, y_arr, bp):
    """Piecewise linear fit at breakpoint bp; returns (sse, aic)."""
    below, above = x_arr < bp, x_arr >= bp
    Xb = np.column_stack([np.ones(below.sum()), x_arr[below]])
    Xa = np.column_stack([np.ones(above.sum()), x_arr[above]])
    cb = np.linalg.lstsq(Xb, y_arr[below], rcond=None)[0]
    ca = np.linalg.lstsq(Xa, y_arr[above], rcond=None)[0]
    pred = np.concatenate([Xb @ cb, Xa @ ca])
    sse = np.sum((y_arr - pred) ** 2)
    aic = len(y_arr) * np.log(sse / len(y_arr)) + 2 * 4
    return sse, aic


# ==================== Experiments ====================
def run_base(args):
    """对应 base_exp.py:标准 FGL(用预生成 data.pkl)。"""
    data = _load_pkl_data(os.path.join(_MG_DIR, "data.pkl"))
    run_fgl_experiment(data, lookback_window=args.L, forecasting_horizon=args.H,
                       alpha=args.alpha, temperature=args.temperature, num_bins=args.bins,
                       epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                       seed=args.seed, model_fn=RNN, label="base")


def run_drift(args):
    """对应 drift_exp.py:FGL + Page-Hinkley 漂移评估。"""
    data = _load_pkl_data(os.path.join(_MG_DIR, "data.pkl"))
    run_fgl_experiment(data, lookback_window=args.L, forecasting_horizon=args.H,
                       alpha=args.alpha, temperature=args.temperature, num_bins=args.bins,
                       epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                       seed=args.seed, model_fn=RNN, use_ph=True, label="drift")


def run_lh_sweep_exp(args):
    """对应 mg_lh_sweep.py:τ=13 的 L×H 扫描。"""
    data, lyap = generate_mg_data(tau=args.tau, n_points=args.n_points)
    print(f"MG τ={args.tau}, Lyapunov={lyap:+.6f}, {args.n_points} pts")
    L_vals = [int(x) for x in (args.L_values or "4,7,10,13,16").split(",")]
    H_vals = [int(x) for x in (args.H_values or "4,7,10,13,16").split(",")]
    seeds = list(range(args.seeds if args.seeds else 3))

    def _run(L, H, data, seed):
        return run_fgl_experiment(
            data, lookback_window=L, forecasting_horizon=H,
            alpha=args.alpha, temperature=args.temperature, num_bins=args.bins,
            epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
            seed=seed, model_fn=RNN, verbose=False, label=f"L{L}_H{H}")

    run_lh_sweep(_run, data, L_vals, H_vals, seeds, _RESULTS_DIR, name="mg_lh_sweep",
                 title=f"Mackey-Glass L×H Sweep (τ={args.tau})",
                 period_label=f"MG τ={args.tau} (period-doubling)",
                 extra_meta={"System": "Mackey-Glass", "tau": args.tau, "lyap": lyap,
                             "alpha": args.alpha, "T": args.temperature})


def run_tau_sweep(args):
    """对应 tau_sweep.py:扫 τ,FGL vs Lyapunov/周期性。"""
    taus = [float(t.strip()) for t in (args.taus or "10,13,17,23,30").split(",")]
    print("=" * 70 + f"\n  MG τ-Sweep: FGL vs Chaos ({taus})\n" + "=" * 70)
    results = []
    for tau in taus:
        print(f"\n{'─'*50}\n  τ = {tau}\n{'─'*50}")
        data, lyap = generate_mg_data(tau=tau, n_points=args.n_points)
        periodicity, ac_zero = _mg_periodicity(data)
        print(f"  Lyapunov={lyap:+.6f}  Periodicity={periodicity:.4f} (AC zero @ lag {ac_zero})")
        r = run_fgl_experiment(data, lookback_window=8, forecasting_horizon=args.H,
                               alpha=args.alpha, temperature=args.temperature, num_bins=args.bins,
                               epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                               seed=42, model_fn=RNN, verbose=False, label=f"tau{tau}")
        r.update(tau=tau, lyap=lyap, periodicity=periodicity, ac_zero_lag=ac_zero)
        results.append(r)
        print(f"  Base={r['baseline']:.2f} Stu={r['student']:.2f} Δ={r['improvement']:+.1f}%")

    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    print(f"  {'τ':>6} {'Lyap':>10} {'Period':>8} {'Base':>9} {'Stu':>9} {'Δ':>8}")
    for r in results:
        print(f"  {r['tau']:6.1f} {r['lyap']:+10.6f} {r['periodicity']:8.4f} "
              f"{r['baseline']:9.2f} {r['student']:9.2f} {r['improvement']:+7.1f}%")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lyaps = [r["lyap"] for r in results]
    imps = [r["improvement"] for r in results]
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in imps]
    axes[0].scatter(lyaps, imps, c=colors, s=120, zorder=5, edgecolors="black", linewidth=0.5)
    for i, r in enumerate(results):
        axes[0].annotate(f"τ={r['tau']:.0f}", (lyaps[i], imps[i]),
                         textcoords="offset points", xytext=(0, 12 if i % 2 == 0 else -18),
                         fontsize=10, ha="center", fontweight="bold")
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].set_xlabel("Lyapunov Exponent"); axes[0].set_ylabel("FGL Δ (%)")
    axes[0].set_title("FGL vs Chaos Strength", fontweight="bold"); axes[0].grid(alpha=0.2)
    cmap = plt.cm.viridis
    for i, r in enumerate(results):
        d, _ = generate_mg_data(tau=r["tau"], n_points=args.n_points)
        s = d[:, 0].numpy()
        ac = np.correlate(s - s.mean(), s - s.mean(), mode="full")
        ac = ac[len(ac) // 2:] / (ac[len(ac) // 2:] + 1e-10)
        axes[1].plot(np.arange(len(ac))[:100], ac[:100], lw=1.2,
                     color=cmap(i / max(1, len(results) - 1)),
                     label=f"τ={r['tau']:.0f} (lyap={r['lyap']:+.4f})")
    axes[1].axhline(0, color="black", lw=0.5, ls="--")
    axes[1].set_xlabel("Lag"); axes[1].set_ylabel("Autocorrelation")
    axes[1].set_title("AC Decay by τ", fontweight="bold"); axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.2); axes[1].set_xlim(0, 100)
    plt.tight_layout()
    out = os.path.join(_RESULTS_DIR, "tau_sweep.png")
    plt.savefig(out, dpi=150); plt.close(fig)
    print(f"\nPlot saved: {out}")
    with open(os.path.join(_RESULTS_DIR, "tau_sweep_results.json"), "w") as f:
        json.dump([{k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in r.items()} for r in results], f, indent=2)


def _run_threshold_scan(args, fixed_name, fixed_val, scan_values, tau):
    """通用阈值扫描:固定一个轴(fixed_name='L' or 'H')扫另一个。
    返回 (all_rows, agg, scan_sorted, am_dict)。对应 tau_threshold_test / h_threshold_test。"""
    data, lyap = generate_mg_data(tau=tau, n_points=args.n_points)
    print(f"MG τ={tau}, Lyapunov={lyap:+.6f}")
    seeds = list(range(args.seeds if args.seeds else 5))
    all_rows = []
    for v in scan_values:
        L, H = (fixed_val, v) if fixed_name == "L" else (v, fixed_val)
        crit = L + H - 1
        tag = "<τ" if crit < tau else ("=τ ←CRIT" if crit == tau else ">τ")
        print(f"\n{'='*50}\n  {fixed_name}={fixed_val} scan={fixed_name=='H' and L or H} "
              f"(L={L} H={H} L+H-1={crit} {tag})\n{'='*50}")
        for s in seeds:
            r = run_fgl_experiment(data, lookback_window=L, forecasting_horizon=H,
                                   alpha=args.alpha, temperature=args.temperature, num_bins=args.bins,
                                   epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                                   seed=s, model_fn=RNN, verbose=False)
            all_rows.append({"L": L, "H": H, "tau": tau, "L_plus_H_minus_1": crit,
                             "seed": s, "baseline_mse": r["baseline"], "teacher_mse": r["teacher"],
                             "student_mse": r["student"], "abs_improvement": r["baseline"] - r["student"],
                             "fgl_delta": r["improvement"]})
    return all_rows


def _threshold_report(all_rows, scan_key, fixed_name, fixed_val, tau, out_prefix):
    """写 CSV + 三联图 + 判定报告(对应 h_threshold/tau_threshold 的输出)。"""
    scan_sorted = sorted({r[scan_key] for r in all_rows})
    agg = defaultdict(list)
    for r in all_rows:
        agg[r[scan_key]].append(r)
    am = {v: {"bm": np.mean([r["baseline_mse"] for r in agg[v]]),
              "bs": np.std([r["baseline_mse"] for r in agg[v]]),
              "ai": np.mean([r["abs_improvement"] for r in agg[v]]),
              "ais": np.std([r["abs_improvement"] for r in agg[v]]),
              "dm": np.mean([r["fgl_delta"] for r in agg[v]])} for v in scan_sorted}

    csv_path = os.path.join(_RESULTS_DIR, f"{out_prefix}_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["L", "H", "tau", "L_plus_H_minus_1", "seed",
                                          "baseline_mse", "teacher_mse", "student_mse",
                                          "abs_improvement", "fgl_delta"])
        w.writeheader(); w.writerows(all_rows)
    print(f"Saved: {csv_path}")

    x_arr = np.array(scan_sorted)
    ai_arr = np.array([am[v]["ai"] for v in scan_sorted])
    crit = int(tau - fixed_val + 1) if fixed_name == "L" else int(tau - fixed_val + 1)
    # critical scan value where L+H-1 = tau, given the fixed axis = fixed_val
    crit_scan = tau - fixed_val + 1
    cp_x, _, _ = find_changepoint(x_arr, ai_arr)
    Xl = np.column_stack([np.ones(len(x_arr)), x_arr])
    cl = np.linalg.lstsq(Xl, ai_arr, rcond=None)[0]
    aic_l = len(x_arr) * np.log(np.sum((ai_arr - Xl @ cl) ** 2) / len(x_arr)) + 2 * 2
    _, aic_p = piecewise_fit(x_arr, ai_arr, bp=crit_scan)
    print(f"Changepoint {scan_key}≈{cp_x:.1f} (predicted {crit_scan})  "
          f"ΔAIC(linear−piecewise)={aic_l - aic_p:+.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].errorbar(x_arr, ai_arr, yerr=[am[v]["ais"] for v in scan_sorted],
                     color="#2ecc71", marker="o", ms=9, lw=2, capsize=5)
    axes[0].axvline(crit_scan, color="red", ls="--", lw=1.5, alpha=0.7)
    axes[0].axhline(0, color="black", lw=0.5)
    xf = np.linspace(x_arr.min(), x_arr.max(), 100)
    axes[0].plot(xf, cl[0] + cl[1] * xf, 'k--', lw=0.8, alpha=0.5, label="Linear")
    axes[0].set_xlabel(scan_key); axes[0].set_ylabel("Abs improvement")
    axes[0].set_title(f"Abs Improvement vs {scan_key}", fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.2)
    axes[1].errorbar(x_arr, [am[v]["bm"] for v in scan_sorted],
                     yerr=[am[v]["bs"] for v in scan_sorted], color="#3498db", marker="s", ms=8, capsize=5)
    axes[1].axvline(crit_scan, color="red", ls="--", lw=1.5, alpha=0.7)
    axes[1].set_xlabel(scan_key); axes[1].set_ylabel("Baseline MSE")
    axes[1].set_title("Baseline MSE (difficulty check)", fontweight="bold"); axes[1].grid(alpha=0.2)
    dm = [am[v]["dm"] for v in scan_sorted]
    axes[2].bar(x_arr, dm, color=["#2ecc71" if m > 0 else "#e74c3c" for m in dm], edgecolor="white")
    axes[2].axvline(crit_scan, color="red", ls="--", lw=1.5, alpha=0.7)
    axes[2].axhline(0, color="black", lw=0.5)
    axes[2].set_xlabel(scan_key); axes[2].set_ylabel("FGL Δ (%)")
    axes[2].set_title(f"FGL Δ% vs {scan_key}", fontweight="bold"); axes[2].grid(alpha=0.2, axis="y")
    plt.tight_layout()
    png_path = os.path.join(_RESULTS_DIR, f"{out_prefix}_figures.png")
    plt.savefig(png_path, dpi=150); plt.close(fig)
    print(f"Figures saved: {png_path}")


def run_l_threshold(args):
    """对应 tau_threshold_test.py:固定 H 扫 L(干净因果检验)。"""
    H_fixed = args.H
    L_values = [int(x) for x in (args.L_values or "4,6,8,9,10,11,12,13,14,16").split(",")]
    rows = _run_threshold_scan(args, fixed_name="H", fixed_val=H_fixed,
                               scan_values=L_values, tau=args.tau)
    _threshold_report(rows, scan_key="L", fixed_name="H", fixed_val=H_fixed,
                      tau=args.tau, out_prefix="l_threshold")


def run_h_threshold(args):
    """对应 h_threshold_test.py:固定 L 扫 H。"""
    L_fixed = args.L
    H_values = [int(x) for x in (args.H_values or "2,4,6,7,8,9,10,11,12,14,16").split(",")]
    rows = _run_threshold_scan(args, fixed_name="L", fixed_val=L_fixed,
                               scan_values=H_values, tau=args.tau)
    _threshold_report(rows, scan_key="H", fixed_name="L", fixed_val=L_fixed,
                      tau=args.tau, out_prefix="h_threshold")


def run_geometry(args):
    """对应 tau_sweep_geometry2.py:固定 τ,扫 (L,H) 配置检验 L+H-1 与 τ 距离假说。"""
    # configs: (L,H) 覆盖 L+H-1 从远小于 τ 到远大于 τ
    configs = [(4, 4), (6, 6), (8, 8), (10, 10), (13, 13), (15, 16), (4, 16), (16, 4)]
    data, lyap = generate_mg_data(tau=args.tau, n_points=args.n_points)
    print(f"MG τ={args.tau}, Lyapunov={lyap:+.6f}")
    seeds = list(range(args.seeds if args.seeds else 5))
    rows = []
    for L, H in configs:
        center = L + H - 1
        dist = center - args.tau
        print(f"\n  L={L} H={H} (L+H-1={center}, dist from τ={dist:+.0f})")
        for s in seeds:
            r = run_fgl_experiment(data, lookback_window=L, forecasting_horizon=H,
                                   alpha=args.alpha, temperature=args.temperature, num_bins=args.bins,
                                   epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                                   seed=s, model_fn=RNN, verbose=False)
            rows.append({"L": L, "H": H, "center": center, "distance": dist, "seed": s,
                         "baseline_mse": r["baseline"], "student_mse": r["student"],
                         "fgl_delta": r["improvement"]})
    csv_path = os.path.join(_RESULTS_DIR, "geometry_test2_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["L", "H", "center", "distance", "seed",
                                          "baseline_mse", "student_mse", "fgl_delta"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {csv_path}")
    # 按 distance 聚合
    agg = defaultdict(list)
    for r in rows:
        agg[r["distance"]].append(r["fgl_delta"])
    print("\nFGL Δ vs (L+H-1 − τ) distance:")
    for dist in sorted(agg):
        arr = np.array(agg[dist])
        print(f"  dist={dist:+3.0f}: Δ={arr.mean():+7.1f}% ± {arr.std():.1f} (n={len(arr)})")


def run_iterative_distill_exp(args):
    """迭代自适应蒸馏:变体列表 × 单/迭代臂,双权重分布(E 硬 / E-soft 稍软化)。"""
    import csv
    data, lyap = generate_mg_data(tau=args.tau, n_points=args.n_points)
    print(f"MG τ={args.tau}, Lyapunov={lyap:+.6f}")
    n_seeds = args.seeds if args.seeds else 3
    variants = tuple(v.strip() for v in args.distill_variants.split(","))
    rows = []
    for s in range(n_seeds):
        arms = run_iterative_distillation(
            data, L=args.L, H=args.H, alpha=args.alpha, temperature=args.temperature,
            num_bins=args.bins, epochs=args.epochs, round_epochs=args.round_epochs,
            batch_size=args.batch_size, K=args.K, patience=args.patience, seed=s,
            weight_distributions=variants, w_floor=args.w_floor, verbose=False)
        for arm, r in arms.items():
            rows.append({"seed": s, "arm": arm, "student_mse": r["student_mse"],
                         "baseline_mse": r["baseline_mse"], "teacher_mse": r["teacher_mse"],
                         "fgl_delta": r["fgl_delta"], "init_delta": r["init_delta"],
                         "rounds_used": r["rounds_used"], "total_epochs": r["total_epochs"]})

    out = os.path.join(_RESULTS_DIR, "iterative_distill.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  → {out} ({len(rows)} rows)")

    print(f"\n{'=' * 60}\nSUMMARY: iterative_distill MG (L={args.L} H={args.H} τ={args.tau})\n{'=' * 60}")
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("student_mse", "fgl_delta", "init_delta"):
            agg[r["arm"]][k].append(r[k])

    def _sd(a):
        a = np.array(a); return a.std(ddof=1) if len(a) > 1 else 0.0

    for arm in sorted(agg):
        sm = np.array(agg[arm]["student_mse"])
        fd = np.array(agg[arm]["fgl_delta"])
        idt = np.array(agg[arm]["init_delta"])
        print(f"  {arm:14s}: student_mse={sm.mean():.3f}±{_sd(sm):.3f}  "
              f"Δbase={fd.mean():+.1f}%±{_sd(fd):.1f}  "
              f"Δinit={idt.mean():+.1f}%±{_sd(idt):.1f}  (n={len(sm)})")


# ==================== Experiment switches ====================
EXPERIMENTS = {
    "base":        dict(fn=run_base,         enabled=True,  note="标准 FGL(data.pkl)"),
    "drift":       dict(fn=run_drift,        enabled=False, note="FGL + Page-Hinkley 漂移评估"),
    "lh_sweep":    dict(fn=run_lh_sweep_exp, enabled=True,  note="L×H 扫描(τ=13 主线)"),
    "tau_sweep":   dict(fn=run_tau_sweep,    enabled=False, note="τ 扫描:FGL vs Lyapunov/周期性"),
    "l_threshold": dict(fn=run_l_threshold,  enabled=False, note="固定 H 扫 L —— 干净因果检验"),
    "h_threshold": dict(fn=run_h_threshold,  enabled=False, note="固定 L 扫 H —— 阈值对称检验"),
    "geometry":    dict(fn=run_geometry,     enabled=False, note="(L,H) 配置几何证伪 L+H-1≥τ"),
    "iterative_distill": dict(fn=run_iterative_distill_exp, enabled=False,
                              note="迭代自适应蒸馏(E/E-soft 双权重分布);MG 负结果,默认关"),
}


def main():
    parser = argparse.ArgumentParser(description="Mackey-Glass 统一实验入口(配置字典开关)")
    parser.add_argument("-e", "--experiments", type=str, default=None,
                        help="逗号分隔的实验名;不指定则跑所有 enabled=True")
    parser.add_argument("--list", action="store_true", help="列出实验及开关状态")
    # FGL 参数
    parser.add_argument("--L", type=int, default=5, help="历史窗口(亦作 h_threshold 固定 L)")
    parser.add_argument("--H", type=int, default=5, help="预测步长(亦作 l_threshold 固定 H)")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("-T", "--temperature", type=float, default=4.0, dest="temperature")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=None, help="多种子实验的种子数")
    parser.add_argument("--seed_single", type=int, default=42)
    # MG / 扫描参数
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--n_points", type=int, default=N_POINTS)
    parser.add_argument("--taus", type=str, default=None, help="[tau_sweep] τ 列表")
    parser.add_argument("--L_values", type=str, default=None, help="[lh_sweep/l_threshold] L 列表")
    parser.add_argument("--H_values", type=str, default=None, help="[lh_sweep/h_threshold] H 列表")
    parser.add_argument("--round_epochs", type=int, default=20, help="[iterative_distill] 每轮 epoch")
    parser.add_argument("--K", type=int, default=5, help="[iterative_distill] 最大迭代轮数")
    parser.add_argument("--distill_variants", type=str, default="E,E-soft",
                        help="[iterative_distill] 权重分布变体,逗号分隔(如 E,E-soft)")
    parser.add_argument("--w_floor", type=float, default=0.2,
                        help="[iterative_distill] E-soft 软地板(默认 0.2)")
    args = parser.parse_args()

    if args.list:
        print("Mackey-Glass experiments(开关 / 说明):")
        for name, cfg in EXPERIMENTS.items():
            flag = "✓ ON " if cfg["enabled"] else "  off"
            print(f"  {name:12s} [{flag}]  {cfg['note']}")
        return

    if args.experiments:
        names = [n.strip() for n in args.experiments.split(",")]
        for n in names:
            if n not in EXPERIMENTS:
                sys.exit(f"未知实验: {n};可用: {', '.join(EXPERIMENTS)}")
    else:
        names = [n for n, c in EXPERIMENTS.items() if c["enabled"]]

    print(f"\n将运行实验: {names}\n")
    for n in names:
        print(f"\n{'#' * 60}\n# Experiment: {n}\n{'#' * 60}")
        EXPERIMENTS[n]["fn"](args)


if __name__ == "__main__":
    main()
