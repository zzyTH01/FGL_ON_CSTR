import csv
from types import SimpleNamespace

import numpy as np
import pytest


def _series():
    x = np.linspace(0.0, 1.0, 60)
    return list(zip(x, np.sin(6 * np.pi * x)))


def _args():
    return SimpleNamespace(
        dataset="h2o", L=4, H=2, seeds=2, baseline_methods="ridge,dlinear",
        baseline_epochs=1, batch_size=8, patience=1,
    )


def test_driver_writes_one_row_per_method_and_seed(tmp_path, monkeypatch):
    from cstr import baselines_driver

    monkeypatch.setattr(baselines_driver, "_load_data", lambda _name: _series())
    calls = []

    def fake_baseline(data, method, lookback_window, forecasting_horizon, seed, epochs, **kwargs):
        calls.append((method, lookback_window, forecasting_horizon, seed, epochs))
        return {"method": method, "val_mse": 0.2, "test_mse": 0.1,
                "n_test": 11, "epochs": 1}

    monkeypatch.setattr(baselines_driver, "run_forecasting_baseline", fake_baseline)
    out = tmp_path / "external.csv"
    baselines_driver.run_all(_args(), out_path=out)

    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert [(int(r["seed"]), r["method"]) for r in rows] == [
        (0, "ridge"), (0, "dlinear"), (1, "ridge"), (1, "dlinear")
    ]
    assert calls == [
        ("ridge", 4, 2, 0, 1), ("dlinear", 4, 2, 0, 1),
        ("ridge", 4, 2, 1, 1), ("dlinear", 4, 2, 1, 1),
    ]
    assert rows[0]["dataset"] == "h2o"
    assert float(rows[0]["test_mse"]) == 0.1


def test_driver_replaces_selected_grid_but_preserves_other_rows(tmp_path, monkeypatch):
    from cstr import baselines_driver

    monkeypatch.setattr(baselines_driver, "_load_data", lambda _name: _series())
    monkeypatch.setattr(
        baselines_driver, "run_forecasting_baseline",
        lambda _data, method, **kwargs: {"method": method, "val_mse": 1.0,
                                         "test_mse": 0.5, "n_test": 11, "epochs": 1},
    )
    out = tmp_path / "external.csv"
    baselines_driver.run_all(_args(), out_path=out)
    changed = _args()
    changed.H = 3
    baselines_driver.run_all(changed, out_path=out)

    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    assert sum(r["H"] == "2" for r in rows) == 4
    assert sum(r["H"] == "3" for r in rows) == 4


def test_external_baselines_experiment_is_off_by_default():
    from cstr.run import EXPERIMENTS

    assert "external_baselines" in EXPERIMENTS
    assert EXPERIMENTS["external_baselines"]["enabled"] is False
