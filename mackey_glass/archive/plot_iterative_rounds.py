#!/usr/bin/env python
"""可视化:E_iter 在 MG τ=13 锚点 L=4,H=10 上逐轮的 student 预测 vs 真实值。

镜像 cstr/plot_iterative_rounds.py。每轮(含 round-0)在 test 集上画
真实值(灰线) vs 预测值(红点),展示迭代蒸馏如何让预测逐轮贴合真实曲线。
值空间映射:预测 bin → bin 中心。

用法::
    FGL_DEVICE=cpu uv run python mackey_glass/plot_iterative_rounds.py
输出 mackey_glass/results/iterative_mg_round_predictions.png
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_MG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _MG_DIR)

import numpy as np
import torch
from fgl_common import compute_shared_bin_edges, create_time_series_dataset, run_iterative_distillation
from fgl_common.training import device
from sweep_iterative import generate_mg_data


def _bin_centers(edges, num_bins):
    c = np.zeros(num_bins)
    c[0] = edges[0]
    for b in range(1, num_bins - 1):
        c[b] = (edges[b - 1] + edges[b]) / 2
    c[num_bins - 1] = edges[-1]
    return c


def main():
    L, H, num_bins = 4, 10, 50
    data, lyap = generate_mg_data(tau=13.0, n_points=10000)
    print(f"MG τ=13, Lyapunov={lyap:+.6f}", flush=True)
    bin_edges, _, _ = compute_shared_bin_edges(data, L, num_bins)
    centers = _bin_centers(bin_edges, num_bins)

    # 外部 test loader 必须与 run_iterative_distillation 内部 student_test 同参数
    _, _, test_loader, _, orig_test = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H, num_bins=num_bins,
        val_size=0.2, test_size=0.2, offset=0, batch_size=128, bin_edges=bin_edges)
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
        print(f"  snapshot round {round_idx}: test MSE={mse_test:.3f}", flush=True)

    run_iterative_distillation(data, L=L, H=H, num_bins=num_bins, epochs=50,
                               round_epochs=20, K=5, batch_size=128, seed=0,
                               verbose=False, e_iter_snapshot_fn=snap)

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
        ax.set_title(f"round {r}  (test MSE = {mse:.3f})", fontsize=10)
        ax.set_xlabel("test 样本序号", fontsize=8)
        ax.set_ylabel("MG τ=13 序列值(值空间)", fontsize=8)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=8, loc="best")
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("MG τ=13  E_iter 逐轮 student 预测 vs 真实值  (L=4, H=10, seed=0)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(_MG_DIR, "results", "iterative_mg_round_predictions.png")
    fig.savefig(out, dpi=130)
    print(f"\n图已保存: {out}")


if __name__ == "__main__":
    main()
