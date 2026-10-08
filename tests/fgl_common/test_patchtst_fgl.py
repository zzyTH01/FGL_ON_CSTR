import numpy as np
import pytest
import torch

from fgl_common.baselines import PatchTST
from fgl_common.patchtst_fgl import run_patchtst_fgl


def _series(n=60):
    x = np.linspace(0.0, 1.0, n)
    y = np.sin(6.0 * np.pi * x)
    return list(zip(x, y))


def test_patchtst_fgl_returns_three_continuous_arms():
    result = run_patchtst_fgl(
        _series(48),
        lookback_window=6,
        forecasting_horizon=2,
        epochs=1,
        batch_size=8,
        patience=1,
        seed=0,
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        val_size=0.2,
        test_size=0.2,
        verbose=False,
    )
    assert set(result) >= {"teacher_mse", "baseline_mse", "student_mse", "improvement"}
    for key in ("teacher_mse", "baseline_mse", "student_mse"):
        assert np.isfinite(result[key])
        assert result[key] >= 0.0
    assert np.isfinite(result["improvement"])
