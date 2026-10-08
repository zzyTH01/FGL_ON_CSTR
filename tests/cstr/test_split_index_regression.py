"""Regression test: verify the test split index offset matches create_time_series_dataset.

Bug context (2026-10-08): plot_method_comparison.py computed ``actual`` using
``n_test`` as the window offset instead of ``n_train + n_val``, causing the
ground truth to come from the middle of the dataset rather than the test tail.
"""

import numpy as np
import pytest

from fgl_common.data import create_time_series_dataset


def _make_synthetic(n=300, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.randn(n).astype(np.float64)
    y = np.cumsum(rng.randn(n) * 0.01).astype(np.float64)
    return list(zip(x, y))


@pytest.mark.parametrize("L,H,val_size,test_size", [
    (20, 15, 0.2, 0.2),
    (20, 12, 0.2, 0.2),
    (10, 5, 0.15, 0.25),
])
def test_actual_index_matches_dataloader(L, H, val_size, test_size):
    data = _make_synthetic()
    x_raw = np.array([pt[0] for pt in data])
    y_raw = np.array([pt[1] for pt in data])

    n_windows = len(x_raw) - L - H + 1
    n_test = int(n_windows * test_size)
    n_val = int(n_windows * val_size)
    n_train = n_windows - n_val - n_test

    _, _, test_loader, _, orig_test = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H, num_bins=50,
        val_size=val_size, test_size=test_size, offset=0, batch_size=64)

    # Correct index: test window i → global window (n_train + n_val + i)
    # target time step = (n_train + n_val + i) + L + H - 1
    correct_actual = np.array([
        y_raw[(n_train + n_val) + i + L + H - 1] for i in range(n_test)
    ])

    np.testing.assert_allclose(orig_test, correct_actual, rtol=1e-12)

    # Verify the WRONG index (bug) produces different values
    wrong_actual = np.array([
        y_raw[n_test + i + L + H - 1] for i in range(n_test)
    ])
    assert not np.allclose(orig_test, wrong_actual), \
        "Wrong index should NOT match; test offset formula has changed"
