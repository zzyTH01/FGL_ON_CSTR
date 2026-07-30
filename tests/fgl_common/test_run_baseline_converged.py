import math
from fgl_common import run_baseline_converged


def _series(n=600, seed=0):
    import numpy as np
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    s = np.sin(t * 0.2) * 50.0 + 100.0 + 1.0 * rng.randn(n)
    return [(float(s[i]), float(s[i])) for i in range(n)]


def test_returns_finite_baseline_mse():
    r = run_baseline_converged(_series(), lookback_window=20, forecasting_horizon=15,
                               num_bins=20, epochs=5, patience=3, seed=0, verbose=False)
    assert "baseline_mse" in r and "epochs_run" in r
    assert math.isfinite(r["baseline_mse"]) and r["baseline_mse"] >= 0
    assert r["epochs_run"] >= 1


def test_more_epochs_no_worse_than_one():
    # same seed => same init; 20 epochs should not be worse than 1 epoch
    data = _series()
    r_long = run_baseline_converged(data, 20, 15, num_bins=20, epochs=20,
                                    patience=5, seed=0, verbose=False)
    r_short = run_baseline_converged(data, 20, 15, num_bins=20, epochs=1,
                                     patience=5, seed=0, verbose=False)
    assert r_long["baseline_mse"] <= r_short["baseline_mse"] + 1e-6
