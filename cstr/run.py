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

# 确保项目根在 sys.path,以便 import fgl_common
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fgl_common import (  # noqa: E402
    RNN, LSTMModel,
    run_fgl_experiment, run_adaptive_weight, run_adaptive_inference, run_seq2seq,
    run_lh_sweep,
)

# -------------------- 数据路径 --------------------
_CSTR_DIR = os.path.dirname(os.path.abspath(__file__))
_AVAILABLE_DATASETS = {
    "temperature": os.path.join(_CSTR_DIR, "data.pkl"),
    "h2o":         os.path.join(_CSTR_DIR, "data_h2o.pkl"),
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


def run_seq2seq(args):
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


def run_adaptive_weight(args):
    """对应 cstr/exp/adaptive_weight_exp.py:自适应蒸馏权重 A/B/C/D。"""
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
    # 汇总
    print(f"\n{'='*60}\nSUMMARY: adaptive_weight (L={args.L} H={args.H})\n{'='*60}")
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[r["variant"]].append(r["fgl_delta"])
    for v, deltas in sorted(agg.items()):
        arr = np.array(deltas)
        print(f"  {v}: Δ={arr.mean():+.1f}% ± {arr.std(ddof=1) if len(arr) > 1 else 0:.1f}% (n={len(arr)})")


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


# ================================================================
#  实验开关配置  —— enabled=True 的实验在 `python cstr/run.py` 时运行
# ================================================================
EXPERIMENTS = {
    "baseline":        dict(fn=run_baseline,        enabled=True,  note="标准 RNN 分类蒸馏(基准)"),
    "lstm":            dict(fn=run_lstm,            enabled=False, note="LSTM 对照 —— 失败:模型容量非瓶颈"),
    "regression":      dict(fn=run_regression,      enabled=False, note="连续值回归 —— 失败"),
    "seq2seq":         dict(fn=run_seq2seq,         enabled=False, note="多步序列预测 —— 失败"),
    "adaptive":        dict(fn=run_adaptive,        enabled=False, note="推理时 teacher-student 融合 —— 失败"),
    "adaptive_weight": dict(fn=run_adaptive_weight, enabled=False, note="自适应蒸馏权重 A/B/C/D —— 失败"),
    "lh_sweep":        dict(fn=run_lh_sweep_exp,    enabled=True,  note="L×H 网格扫描(主线)"),
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
    p.add_argument("--L_values", type=str, default=None, help="[lh_sweep] L 取值,逗号分隔")
    p.add_argument("--H_values", type=str, default=None, help="[lh_sweep] H 取值,逗号分隔")


def main():
    import numpy as np  # noqa: F401  (used by run_adaptive_weight summary)
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
