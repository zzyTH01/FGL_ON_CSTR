import numpy as np
import pytest


def test_baseline_contract_reports_mean_squared_error_not_sqrt_mse():
    from fgl_common.baselines import _physical_mse

    # _physical_mse receives scalar summaries of normalized MSE, so the known
    # 2x physical error must be reported as 4x normalized squared error.
    assert _physical_mse(np.array([2.0]), np.array([0.0]), y_std=1.0) == pytest.approx(4.0)
    assert _physical_mse(np.array([2.0]), np.array([0.0]), y_std=3.0) == pytest.approx(36.0)


def test_deep_forecasting_baseline_does_not_square_normalized_mse_twice(monkeypatch):
    import torch
    from fgl_common import baselines
    from fgl_common.baselines import _evaluate, _arrays_to_loader, build_continuous_windows, run_forecasting_baseline

    class ZeroModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Adam needs a parameter, but predictions must remain zero.
            self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def forward(self, x):
            return torch.zeros(len(x), 1, requires_grad=True)

    monkeypatch.setitem(baselines.MODEL_BUILDERS, "zero", lambda _L, output_size: ZeroModel())
    monkeypatch.setitem(baselines._LEARNING_RATES, "zero", 0.0)
    data = []
    for i in range(80):
        x = i / 79.0
        data.append((x, np.sin(6 * np.pi * x)))
    result = run_forecasting_baseline(
        data, method="zero", lookback_window=4, forecasting_horizon=2,
        seed=0, epochs=1, batch_size=8, patience=1, lr=0.0,
    )
    windows = build_continuous_windows(data, 4, 2)
    # Reproduce the exact mini-batch mean-of-means used during validation.
    val_loader = _arrays_to_loader(windows.x_val, windows.y_val, batch_size=8,
                                   shuffle=False, seed=0)
    expected = _evaluate(ZeroModel(), val_loader) * windows.y_std ** 2
    assert result["val_mse"] == pytest.approx(expected)
