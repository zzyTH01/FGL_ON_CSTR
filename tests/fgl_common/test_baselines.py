import numpy as np
import torch
import pytest

from fgl_common.baselines import (
    MODEL_BUILDERS,
    build_continuous_windows,
    run_forecasting_baseline,
    run_ridge_baseline,
)


def _series(n=80):
    x = np.linspace(0.0, 1.0, n)
    y = np.sin(6.0 * np.pi * x)
    return list(zip(x, y))


def test_continuous_windows_use_chronological_split_and_train_only_scaling():
    windows = build_continuous_windows(_series(80), lookback_window=4, forecasting_horizon=2,
                                        val_size=0.2, test_size=0.2)
    n_windows = 80 - 4 - 2 + 1
    n_train = 45
    n_val = 15
    n_test = 15
    assert windows.x_train.shape == (n_train, 1, 4)
    assert windows.x_val.shape == (n_val, 1, 4)
    assert windows.x_test.shape == (n_test, 1, 4)
    assert windows.y_test.shape == (n_test,)
    # Target is the value H-1 positions after the final input point.
    raw_x, raw_y = zip(*_series(80))
    expected_first_target = raw_y[4 + 2 - 1]
    np.testing.assert_allclose(windows.y_train[0] * windows.y_std + windows.y_mean,
                               expected_first_target)
    # Standardization statistics must not be computed from val/test data.
    train_x = np.stack([np.asarray(raw_x[i:i + 4]) for i in range(n_train)])
    np.testing.assert_allclose(windows.x_mean, train_x.mean(axis=0))
    np.testing.assert_allclose(windows.x_std, train_x.std(axis=0))
    expected_test_x = np.stack([np.asarray(raw_x[i:i + 4]) for i in range(60, 75)])
    np.testing.assert_allclose(
        windows.x_test,
        ((expected_test_x - windows.x_mean) / windows.x_std)[:, None, :],
    )


def test_ridge_baseline_returns_continuous_physical_mse():
    result = run_ridge_baseline(_series(80), lookback_window=4, forecasting_horizon=2,
                                val_size=0.2, test_size=0.2)
    assert set(result) >= {"method", "val_mse", "test_mse", "n_test"}
    assert result["method"] == "ridge"
    assert result["n_test"] == 15
    assert np.isfinite(result["test_mse"])
    assert result["test_mse"] >= 0.0


@pytest.mark.parametrize("method", ["dlinear", "patchtst", "gru", "tcn"])
def test_deep_model_builders_have_common_contract(method):
    model = MODEL_BUILDERS[method](4, output_size=1)
    out = model(torch.zeros(3, 1, 4))
    assert out.shape == (3, 1)


@pytest.mark.parametrize("method", ["dlinear", "patchtst", "gru", "tcn"])
def test_deep_baselines_smoke_train_and_evaluate(method):
    result = run_forecasting_baseline(
        _series(60), method=method, lookback_window=4, forecasting_horizon=2,
        seed=0, epochs=1, batch_size=8, patience=1, val_size=0.2, test_size=0.2,
    )
    assert result["method"] == method
    assert result["n_test"] == 11
    assert result["epochs"] == 1
    assert np.isfinite(result["val_mse"])
    assert np.isfinite(result["test_mse"])
    assert result["test_mse"] >= 0.0

