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
