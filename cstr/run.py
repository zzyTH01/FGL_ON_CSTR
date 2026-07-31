#!/usr/bin/env python
"""CSTR 统一实验入口。

把原本散落在 ``cstr/exp/`` 下的 8 个实验脚本合并为单文件,通过 ``EXPERIMENTS``
配置字典的 ``enabled`` 字段控制是否运行(内置开关),命令行 ``-e`` 可覆盖。

用法::

    uv run python cstr/run.py                      # 跑所有 enabled=True 的实验
    uv run python cstr/run.py -e baseline,lh_sweep  # 只跑指定实验
    uv run python cstr/run.py --list                # 列出所有实验 + 开关 + 说明
    uv run python cstr/run.py -e baseline -H 5 --alpha 0.5   # 带参数

研究结论(见 ``conclusion/final_conclusions.md``):CSTR 单反馈环不适合 FGL,
多数优化实验为"失败探索",故默认 ``enabled=False``,仅保留 baseline 与 lh_sweep 主线。
所有训练逻辑实现在 ``fgl_common`` 包中,本文件仅做薄包装与开关调度。
"""
import argparse
import os
import pickle
import sys

import numpy as np

# 确保项目根在 sys.path,以便 import fgl_common
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fgl_common import (  # noqa: E402
    RNN, LSTMModel,
    run_fgl_experiment, run_adaptive_weight, run_adaptive_inference, run_seq2seq,
    run_iterative_distillation,
    run_lh_sweep,
)

# -------------------- 数据路径 --------------------
_CSTR_DIR = os.path.dirname(os.path.abspath(__file__))
def _resolve_data(name):
    """数据路径:data/ 优先(归并后),其次根目录(向后兼容)。"""
    for d in ("data", "."):
        p = os.path.join(_CSTR_DIR, d, name)
        if os.path.exists(p):
            return p
    return os.path.join(_CSTR_DIR, "data", name)


_AVAILABLE_DATASETS = {
    "temperature": _resolve_data("data.pkl"),
    "h2o":         _resolve_data("data_h2o.pkl"),
}
DEFAULT_DATASET = "h2o"


def _load_data(dataset=DEFAULT_DATASET):
    with open(_AVAILABLE_DATASETS[dataset], "rb") as f:
        return pickle.load(f)


# ================================================================
#  实验薄包装 —— 每个对应原 cstr/exp/ 下的一个脚本
# ================================================================
def run_baseline(args):
    """对应 cstr/exp/fgl_cstr.py:标准 RNN 分类蒸馏。"""
    data = _load_data(args.dataset)
    run_fgl_experiment(data, lookback_window=args.L, forecasting_horizon=args.H,
                       alpha=args.alpha, temperature=args.temperature, num_bins=args.bins,
                       epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                       seed=args.seed, model_fn=RNN, label="baseline")


def run_lstm(args):
    """对应 cstr/exp/fgl_cstr_lstm.py:LSTM 模型对照。"""
    data = _load_data(args.dataset)
    run_fgl_experiment(data, lookback_window=args.L, forecasting_horizon=args.H,
                       alpha=args.alpha, temperature=args.temperature, num_bins=args.bins,
                       epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                       seed=args.seed, model_fn=LSTMModel, label="lstm")


def run_regression(args):
    """对应 cstr/exp/fgl_cstr_regression.py:连续值回归(跳过离散化)。"""
    data = _load_data(args.dataset)
    run_fgl_experiment(data, lookback_window=args.L, forecasting_horizon=args.H,
                       alpha=args.alpha, epochs=args.epochs, batch_size=args.batch_size,
                       patience=args.patience, seed=args.seed, regression=True, label="regression")


def run_seq2seq_exp(args):
    """对应 cstr/exp/fgl_cstr_seq2seq.py:多步序列预测。"""
    data = _load_data(args.dataset)
    run_seq2seq(data, student_horizon=args.H, teacher_steps=args.teacher_steps,
                alpha=args.alpha, temperature=args.temperature, num_bins=args.bins,
                epochs=args.epochs, lookback_window=args.L, batch_size=args.batch_size,
                patience=args.patience)


def run_adaptive(args):
    """对应 cstr/exp/fgl_cstr_adaptive.py:推理时 teacher-student 融合。"""
    data = _load_data(args.dataset)
    run_adaptive_inference(data, student_horizon=args.H, base_alpha=args.alpha,
                           num_bins=args.bins, epochs=args.epochs, temperature=args.temperature,
                           lookback_window=args.L, batch_size=args.batch_size,
                           patience=args.patience, seed=args.seed)


def run_adaptive_weight_exp(args):
    """对应 cstr/exp/adaptive_weight_exp.py:自适应蒸馏权重 A/B/C/D(teacher−student MSE 差距)。"""
    data = _load_data(args.dataset)
    variants = (args.variants or "A,B,C").split(",")
    n_seeds = args.seeds if args.seeds else 5
    rows = []
    for variant in variants:
        variant = variant.strip()
        for s in range(n_seeds):
            r = run_adaptive_weight(data, L=args.L, H=args.H, alpha=args.alpha,
                                    temperature=args.temperature, num_bins=args.bins,
                                    epochs=args.epochs, batch_size=args.batch_size,
                                    patience=args.patience, seed=s, variant=variant)
            rows.append(r)
    # 汇总:vs baseline 以及 vs 初始 student(自适应是否真下降)
    print(f"\n{'='*60}\nSUMMARY: adaptive_weight (L={args.L} H={args.H})\n{'='*60}")
    from collections import defaultdict
    agg_fgl = defaultdict(list)
    agg_init = defaultdict(list)
    for r in rows:
        agg_fgl[r["variant"]].append(r["fgl_delta"])
        agg_init[r["variant"]].append(r["init_delta"])

    def _sd(a):
        a = np.array(a)
        return a.std(ddof=1) if len(a) > 1 else 0.0

    for v in sorted(agg_fgl.keys()):
        fgl = np.array(agg_fgl[v]); ini = np.array(agg_init[v])
        print(f"  {v}: vs baseline Δ={fgl.mean():+.1f}%±{_sd(fgl):.1f}  "
              f"vs initStudent Δ={ini.mean():+.1f}%±{_sd(ini):.1f}  (n={len(fgl)})")


def run_iterative_distill_exp(args):
    """迭代自适应蒸馏:变体列表 × 单/迭代臂,双权重分布(E 硬 / E-soft 稍软化)。"""
    import csv
    data = _load_data(args.dataset)
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

    os.makedirs(os.path.join(_CSTR_DIR, "results"), exist_ok=True)
    out = os.path.join(_CSTR_DIR, "results", "iterative_distill.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  → {out} ({len(rows)} rows)")

    print(f"\n{'=' * 60}\nSUMMARY: iterative_distill (L={args.L} H={args.H})\n{'=' * 60}")
    from collections import defaultdict
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
        print(f"  {arm:14s}: student_mse={sm.mean():.1f}±{_sd(sm):.1f}  "
              f"Δbase={fd.mean():+.1f}%±{_sd(fd):.1f}  "
              f"Δinit={idt.mean():+.1f}%±{_sd(idt):.1f}  (n={len(sm)})")


def run_lh_sweep_exp(args):
    """对应 cstr/exp/fgl_cstr_lh_sweep.py:L×H 网格扫描(主线)。"""
    data = _load_data(args.dataset)
    L_vals = [int(x) for x in (args.L_values or "8,20,35,50,72").split(",")]
    H_vals = [int(x) for x in (args.H_values or "5,15,30,45,60").split(",")]
    n_seeds = args.seeds if args.seeds else 3
    seeds = list(range(n_seeds))

    def _run(L, H, data, seed):
        return run_fgl_experiment(
            data, lookback_window=L, forecasting_horizon=H,
            alpha=args.alpha, temperature=args.temperature, num_bins=args.bins,
            epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
            seed=seed, model_fn=RNN, verbose=False, label=f"L{L}_H{H}")

    outdir = os.path.join(_CSTR_DIR, "results")
    run_lh_sweep(_run, data, L_vals, H_vals, seeds, outdir, name="cstr_lh_sweep",
                 title="CSTR L×H Sweep (sub-cycle ≈ 72 steps)",
                 period_label="CSTR H2O sub-cycle ≈ 72 steps",
                 extra_meta={"Dataset": f"{args.dataset}.pkl", "alpha": args.alpha,
                             "T": args.temperature, "epochs": args.epochs})


def run_floor_sweep_exp(args):
    """对应 cstr/run_floor_sweep.py:地板成因战役(τ=100 深挖 L×H 网格)。

    记录 {baseline, baseline_converged, teacher, fgl_student, A_iter, E_iter} 每
    (dataset, L, H, seed),写 cstr/results/floor_sweep.csv,供 H1-H4 检验。
    """
    sys.path.insert(0, _CSTR_DIR)
    import run_floor_sweep
    entries = run_floor_sweep.load_entries(["tau100"])
    cells = {"tau100": run_floor_sweep.default_tau100_grid()}
    seeds = list(range(args.seeds if args.seeds else 3))
    run_floor_sweep.run(
        entries, cells, seeds,
        alpha=args.alpha, temperature=args.temperature, bins=args.bins,
        epochs=args.epochs, round_epochs=args.round_epochs,
        batch_size=args.batch_size, patience=args.patience, K=args.K,
        conv_epochs=100, conv_patience=10,
        outdir=os.path.join(_CSTR_DIR, "results"), verbose=True)


# ================================================================
#  实验开关配置  —— enabled=True 的实验在 `python cstr/run.py` 时运行
# ================================================================
EXPERIMENTS = {
    "baseline":        dict(fn=run_baseline,        enabled=True,  note="标准 RNN 分类蒸馏(基准)"),
    "lstm":            dict(fn=run_lstm,            enabled=False, note="LSTM 对照 —— 失败:模型容量非瓶颈"),
    "regression":      dict(fn=run_regression,      enabled=False, note="连续值回归 —— 失败"),
    "seq2seq":         dict(fn=run_seq2seq_exp,     enabled=False, note="多步序列预测 —— 失败"),
    "adaptive":        dict(fn=run_adaptive,        enabled=False, note="推理时 teacher-student 融合 —— 失败"),
    "adaptive_weight": dict(fn=run_adaptive_weight_exp, enabled=False, note="自适应蒸馏权重(teacher−student MSE 差距)A~E;E=零地板放大版,实测有效(L20H15)"),
    "iterative_distill": dict(fn=run_iterative_distill_exp, enabled=True,
                              note="迭代自适应蒸馏(E 硬 / E-soft 稍软化,双权重分布);CSTR 已验证有效"),
    "lh_sweep":        dict(fn=run_lh_sweep_exp,    enabled=True,  note="L×H 网格扫描(主线)"),
    "floor_sweep":      dict(fn=run_floor_sweep_exp, enabled=False, note="地板成因战役:τ=100 深挖 L×H,记录 baseline/teacher/E_iter 等地板量(H1-H4)"),
}


# ================================================================
#  CLI
# ================================================================
def _add_common_args(p):
    p.add_argument("--L", type=int, default=20, help="历史窗口 lookback")
    p.add_argument("--H", type=int, default=15, help="预测步长 horizon")
    p.add_argument("--alpha", type=float, default=0.5, help="CE 权重 α(0=纯蒸馏)")
    p.add_argument("-T", "--temperature", type=float, default=4.0, dest="temperature")
    p.add_argument("--bins", type=int, default=50, help="离散化 bin 数")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                   choices=list(_AVAILABLE_DATASETS))
    # 特定实验的参数
    p.add_argument("--teacher_steps", type=int, default=10, help="[seq2seq] 教师步数 K")
    p.add_argument("--variants", type=str, default=None, help="[adaptive_weight] 变体,如 A,B,C")
    p.add_argument("--seeds", type=int, default=None, help="[adaptive_weight/lh_sweep] 种子数量")
    p.add_argument("--round_epochs", type=int, default=15, help="[iterative_distill] 每轮 epoch")
    p.add_argument("--K", type=int, default=5, help="[iterative_distill] 最大迭代轮数")
    p.add_argument("--distill_variants", type=str, default="E,E-soft",
                   help="[iterative_distill] 权重分布变体,逗号分隔(如 E,E-soft)")
    p.add_argument("--w_floor", type=float, default=0.2,
                   help="[iterative_distill] E-soft 软地板(默认 0.2)")
    p.add_argument("--L_values", type=str, default=None, help="[lh_sweep] L 取值,逗号分隔")
    p.add_argument("--H_values", type=str, default=None, help="[lh_sweep] H 取值,逗号分隔")


def main():
    parser = argparse.ArgumentParser(description="CSTR 统一实验入口(配置字典开关)")
    _add_common_args(parser)
    parser.add_argument("-e", "--experiments", type=str, default=None,
                        help="逗号分隔的实验名;不指定则跑所有 enabled=True")
    parser.add_argument("--list", action="store_true", help="列出所有实验及开关状态")
    args = parser.parse_args()

    if args.list:
        print("CSTR experiments(开关 / 说明):")
        for name, cfg in EXPERIMENTS.items():
            flag = "✓ ON " if cfg["enabled"] else "  off"
            print(f"  {name:16s} [{flag}]  {cfg['note']}")
        print("\n用法: python cstr/run.py [-e name1,name2] [--list]")
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
