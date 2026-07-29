import sys, pathlib
_CSTR = pathlib.Path(__file__).resolve().parents[2] / "cstr"
sys.path.insert(0, str(_CSTR))

import numpy as np
from generate_delayed_stable import control_law, onset_factor


def test_control_law_zero_at_center():
    # delayed_h2o == center  =>  feedback term is 0  =>  mdot == mdot_base
    assert control_law(0.48, mdot_base=2.0) == 2.0


def test_control_law_bounded_for_extreme_inputs():
    # tanh saturates: |mdot/mdot_base - 1| <= amplitude for ANY delayed_h2o
    mdot_base = 3.0
    A = 0.3
    for x in [-1e6, -10.0, 0.0, 0.48, 1.0, 10.0, 1e6]:
        for s in (-1, 1):
            mdot = control_law(x, mdot_base, sign=s, amplitude=A)
            assert abs(mdot / mdot_base - 1.0) <= A + 1e-9, (x, s, mdot)


def test_control_law_sign_monotonicity():
    # at delayed_h2o above center, sign=+1 increases mdot, sign=-1 decreases it
    hi = control_law(0.9, 1.0, sign=+1)
    lo = control_law(0.9, 1.0, sign=-1)
    assert hi > 1.0 > lo


def test_onset_factor_ramp():
    assert onset_factor(0.0, 50.0) == 0.0
    assert onset_factor(25.0, 50.0) == 0.5
    assert onset_factor(50.0, 50.0) == 1.0
    assert onset_factor(999.0, 50.0) == 1.0  # capped


def test_onset_factor_zero_onset_is_full():
    assert onset_factor(0.001, 0.0) == 1.0


# ==================== Task 2: periodicity_score ====================
from generate_delayed_stable import periodicity_score


def test_periodicity_pure_sine_is_high_with_correct_period():
    # sine with period 50 samples -> score ~1, dom_lag == 50
    t = np.arange(4000)
    sig = np.sin(2 * np.pi * t / 50.0)
    score, dom = periodicity_score(sig)
    # finite-window autocorrelation tapers as (N-k)/N, so a clean 4000-sample
    # sine scores ~0.9875 at lag 50 — still clearly "periodic" vs noise (<0.3)
    assert score > 0.98
    assert dom == 50


def test_periodicity_white_noise_is_low():
    rng = np.random.RandomState(0)
    sig = rng.randn(4000)
    score, _ = periodicity_score(sig)
    assert score < 0.3


def test_periodicity_detects_period_doubling():
    # period-50 + half-frequency (period-100) component -> dominant period 100
    t = np.arange(4000)
    sig = np.sin(2 * np.pi * t / 50.0) + 0.5 * np.sin(2 * np.pi * t / 100.0)
    _, dom = periodicity_score(sig)
    assert dom == 100


# ==================== Task 3: simulate (smoke) ====================
import pytest
from generate_delayed_stable import simulate_delayed_feedback_stable


@pytest.mark.slow
def test_simulate_short_run_is_stable_and_sane():
    # short run: must not crash, must stay bounded
    res = simulate_delayed_feedback_stable(
        t_end=30.0, dt=0.1, tau_delay=20,
        sign=-1, amplitude=0.3, t_onset=10.0,
    )
    assert res["h2o"].shape[0] > 0
    assert np.all(np.isfinite(res["T"]))
    assert np.all(np.isfinite(res["h2o"]))
    assert res["h2o"].min() >= 0.0 and res["h2o"].max() <= 1.0      # mass fraction
    assert res["T"].max() < 5000.0                                   # no thermal runaway
    # bounded mdot by construction: |mdot/mdot_base - 1| <= amplitude (=0.3 here)
    base = res["mdot_base"]
    assert res["mdot_max"] <= base * (1 + 0.3) + 1e-12
    assert res["mdot_min"] >= base * (1 - 0.3) - 1e-12


# ==================== Task 4: sweep orchestration ====================
import csv
from generate_delayed_stable import build_tau_grid, run_sweep, write_sweep_csv


def test_build_tau_grid_default():
    assert build_tau_grid() == [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150]


def test_build_tau_grid_fine_around_merges():
    grid = build_tau_grid(fine_around=50)
    assert 35 in grid and 45 in grid and 55 in grid and 65 in grid
    assert grid == sorted(set(grid))


def test_run_sweep_with_fake_simulate():
    def fake(tau):
        t = np.arange(3000)
        return {"h2o": np.sin(2 * np.pi * t / 50.0), "T": np.full(3000, 800.0)}
    rows = run_sweep(fake, grid=[5, 50], burn_steps=100, dt=0.1)
    assert len(rows) == 2
    assert all(r["status"] == "ok" for r in rows)
    # finite-window taper (~0.983 at lag 50 for 2900 samples) — see Task 2
    assert all(r["periodicity"] > 0.98 for r in rows)
    assert all(r["dom_period"] == 50 for r in rows)
    assert all(r["dom_period_s"] == 5.0 for r in rows)


def test_run_sweep_records_failure():
    def fake(tau):
        if tau == 50:
            raise RuntimeError("boom")
        t = np.arange(3000)
        return {"h2o": np.sin(2 * np.pi * t / 50.0), "T": np.full(3000, 800.0)}
    rows = run_sweep(fake, grid=[5, 50], burn_steps=100, dt=0.1)
    assert rows[0]["status"] == "ok"
    assert rows[1]["status"].startswith("fail")


def test_write_sweep_csv_roundtrip(tmp_path):
    rows = [{"tau": 5, "periodicity": 0.99, "dom_period": 50, "dom_period_s": 5.0,
             "T_min": 770.0, "T_max": 2900.0, "status": "ok"}]
    p = tmp_path / "out.csv"
    write_sweep_csv(rows, str(p))
    with open(p) as f:
        data = list(csv.DictReader(f))
    assert len(data) == 1
    assert float(data[0]["periodicity"]) == 0.99
    assert data[0]["status"] == "ok"
