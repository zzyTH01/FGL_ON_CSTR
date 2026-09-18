import csv
from types import SimpleNamespace

import numpy as np


def _args():
    return SimpleNamespace(
        dataset="h2o", L=20, H=15, seeds=2, alpha=0.5, temperature=4.0,
        bins=50, epochs=1, batch_size=8, patience=1, round_epochs=1, K=1,
        distill_variants="E", w_floor=0.2,
    )


def _arms(seed, **kwargs):
    return {
        "E_iter": {
            "student_mse": 100.0 + seed,
            "baseline_mse": 120.0,
            "student_continuous_mse": 0.02 + 0.001 * seed,
            "student_bin_center_mse": 0.03 + 0.001 * seed,
            "baseline_continuous_mse": 0.04,
            "baseline_bin_center_mse": 0.05,
            "continuous_delta": 50.0,
            "rounds_used": 2,
            "total_epochs": 45,
        }
    }


def test_remap_driver_writes_continuous_metrics(tmp_path, monkeypatch):
    from cstr import remap_compare

    monkeypatch.setattr(remap_compare, "_load_data", lambda _name: [(0.0, 0.0)] * 50)
    monkeypatch.setattr(remap_compare, "run_iterative_distillation", lambda data, seed, **kwargs: _arms(seed))
    out = tmp_path / "remap.csv"
    remap_compare.run_all(_args(), out_path=out)
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["arm"] == "E_iter"
    assert float(rows[1]["student_mse_centroid"]) == 0.021
    assert float(rows[0]["baseline_mse_centroid"]) == 0.04


def test_iterative_remap_experiment_is_off_by_default():
    from cstr.run import EXPERIMENTS

    assert EXPERIMENTS["iterative_remap"]["enabled"] is False
