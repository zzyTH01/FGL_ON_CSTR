import csv
from types import SimpleNamespace

import numpy as np


def _series():
    x = np.linspace(0.0, 1.0, 48)
    return list(zip(x, np.sin(6 * np.pi * x)))


def _args():
    return SimpleNamespace(
        dataset="h2o", L_values="4", H_values="2", seeds=2,
        baseline_epochs=1, batch_size=8, patience=1, alpha=0.5,
        d_model=8, nhead=2, dim_feedforward=16, dropout=0.0, lr=5e-4,
    )


def test_patchtst_driver_writes_one_row_per_grid_and_seed(tmp_path, monkeypatch):
    from cstr import run_patchtst_fgl_driver

    monkeypatch.setattr(run_patchtst_fgl_driver, "_load_data", lambda _name: _series())
    calls = []

    def fake_run(data, *, lookback_window, forecasting_horizon, seed, **kwargs):
        calls.append((lookback_window, forecasting_horizon, seed, kwargs["epochs"]))
        return {
            "lookback": lookback_window, "horizon": forecasting_horizon,
            "alpha": kwargs["alpha"], "teacher_mse": 0.3, "baseline_mse": 0.2,
            "student_mse": 0.1, "improvement": 50.0,
            "teacher_epochs": 1, "baseline_epochs": 1, "student_epochs": 1,
            "teacher_val_mse_normalized": 0.3, "baseline_val_mse_normalized": 0.2,
            "student_val_mse_normalized": 0.1, "n_test": 8,
        }

    monkeypatch.setattr(run_patchtst_fgl_driver, "run_patchtst_fgl", fake_run)
    out = tmp_path / "patchtst.csv"
    run_patchtst_fgl_driver.run_all(
        _args(), out_path=out,
        summary_path=tmp_path / "summary.csv", plot_path=tmp_path / "summary.png",
    )

    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert [(int(r["H"]), int(r["seed"]), int(r["epochs"])) for r in rows] == [
        (2, 0, 1), (2, 1, 1)
    ]
    assert calls == [(4, 2, 0, 1), (4, 2, 1, 1)]


def test_patchtst_fgl_experiment_is_off_by_default():
    from cstr.run import EXPERIMENTS

    assert EXPERIMENTS["patchtst_fgl"]["enabled"] is False
