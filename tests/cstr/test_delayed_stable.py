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
