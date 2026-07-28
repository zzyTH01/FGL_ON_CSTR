"""FGL 共享底座 (Future-Guided Learning common library)。

收敛了三个回归域 (cstr / mackey_glass / lorenz) 共用的模型、数据、蒸馏、训练、扫描组件。

典型用法::

    from fgl_common import RNN, KL, create_time_series_dataset, run_fgl_experiment

    result = run_fgl_experiment(data, lookback_window=8, forecasting_horizon=5,
                                alpha=0.5, temperature=4)

模块结构:
    models.py         —— RNN / LSTMModel / RNNRegression / SeqRNN
    data.py           —— create_time_series_dataset / create_seq_dataset
    distillation.py   —— KL / KL_weighted / seq_KL / compute_weights
    training.py       —— device / EarlyStopper / evaluate* / run_fgl_experiment /
                         run_adaptive_weight / run_adaptive_inference / run_seq2seq
    sweep.py          —— run_lh_sweep (通用 L×H 扫描 + 热力图 + 报告)
"""
from .models import RNN, LSTMModel, RNNRegression, SeqRNN
from .data import create_time_series_dataset, create_seq_dataset
from .distillation import KL, KL_weighted, seq_KL, compute_weights
from .training import (
    device, EarlyStopper,
    evaluate, evaluate_with_ph, evaluate_regression, evaluate_seq,
    page_hinkley_update, compute_shared_bin_edges, compute_per_sample_errors,
    compute_per_sample_mse,
    run_fgl_experiment, run_iterative_distillation, run_adaptive_weight, run_adaptive_inference, run_seq2seq,
)
from .sweep import run_lh_sweep

__all__ = [
    # models
    "RNN", "LSTMModel", "RNNRegression", "SeqRNN",
    # data
    "create_time_series_dataset", "create_seq_dataset",
    # distillation
    "KL", "KL_weighted", "seq_KL", "compute_weights",
    # training
    "device", "EarlyStopper",
    "evaluate", "evaluate_with_ph", "evaluate_regression", "evaluate_seq",
    "page_hinkley_update", "compute_shared_bin_edges", "compute_per_sample_errors",
    "compute_per_sample_mse",
    "run_fgl_experiment", "run_iterative_distillation", "run_adaptive_weight", "run_adaptive_inference", "run_seq2seq",
    # sweep
    "run_lh_sweep",
]
