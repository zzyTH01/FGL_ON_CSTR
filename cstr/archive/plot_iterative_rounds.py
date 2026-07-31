#!/usr/bin/env python
"""可视化:E_iter 在 L=20,H=15 上逐轮的 student 预测 vs 真实值。

每轮(含 round-0)在 test 集上画 真实值(灰线) vs 预测值(红点),
展示迭代蒸馏如何让预测逐轮贴合真实曲线。值空间映射:预测 bin → bin 中心。

用法::
    uv run python cstr/plot_iterative_rounds.py
输出 cstr/results/iterative_round_predictions.png
"""
import os
import sys
import pickle

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from fgl_common import compute_shared_bin_edges, create_time_series_dataset, run_iterative_distillation
from fgl_common.training import device

_CSTR_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name="data_h2o.pkl"):
    for d in ("data", "."):
        p = os.path.join(_CSTR_DIR, d, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(name)


def _bin_centers(edges, num_bins):
    """digitize 的 bin(0..num_bins-1)→ 代表性值(相邻边中点,端点取边本身)。"""
    c = np.zeros(num_bins)
    c[0] = edges[0]
    for b in range(1, num_bins - 1):
        c[b] = (edges[b - 1] + edges[b]) / 2
    c[num_bins - 1] = edges[-1]
    return c


def main():
    L, H, num_bins = 20, 15, 50
    data = _load()
    bin_edges, _, _ = compute_shared_bin_edges(data, L, num_bins)
    centers = _bin_centers(bin_edges, num_bins)

    # 外部 test loader 必须与 run_iterative_distillation 内部 student_test 同参数
    _, _, test_loader, _, orig_test = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H, num_bins=num_bins,
        val_size=0.2, test_size=0.2, offset=0, batch_size=64, bin_edges=bin_edges)
    n_test = len(orig_test)

    snapshots = []  # (round, preds_value[np], mse_test)

    def snap(student, round_idx, mse_val, mse_test):
        student.eval()
        preds = np.full(n_test, np.nan)
        with torch.no_grad():
            for idx, x, y in test_loader:
                x = x.float().to(device).view(-1, 1, L)
                p = student(x).argmax(dim=1).cpu().numpy()
                for j, ii in enumerate(idx.numpy()):
                    preds[ii] = centers[p[j]]
        snapshots.append((round_idx, preds.copy(), float(mse_test)))
        print(f"  snapshot round {round_idx}: test MSE={mse_test:.1f}", flush=True)

    run_iterative_distillation(data, L=L, H=H, num_bins=num_bins, epochs=20,
                               round_epochs=10, K=6, seed=0, verbose=False,
                               e_iter_snapshot_fn=snap)

    _plot(orig_test, snapshots)


def _plot(actual, snapshots, window=180):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    actual = np.asarray(actual, dtype=float)
    idx = np.arange(min(window, len(actual)))
    act = actual[idx]
    n = len(snapshots)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for k, (r, preds, mse) in enumerate(snapshots):
        ax = axes[k // cols][k % cols]
        ax.plot(idx, act, color="dimgray", lw=1.6, label="真实值 actual")
        ax.scatter(idx, preds[idx], s=7, color="crimson", alpha=0.75, label="预测 predicted")
        ax.set_title(f"round {r}  (test MSE = {mse:.1f})", fontsize=10)
        ax.set_xlabel("test 样本序号", fontsize=8)
        ax.set_ylabel("H₂O 质量分数(值空间)", fontsize=8)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=8, loc="best")
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("E_iter 逐轮 student 预测 vs 真实值  (L=20, H=15, seed=0; 红点逐轮贴向灰线)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(_CSTR_DIR, "results", "iterative_round_predictions.png")
    fig.savefig(out, dpi=130)
    print(f"\n图已保存: {out}")


if __name__ == "__main__":
    main()
