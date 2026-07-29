#!/usr/bin/env python
"""
Stable delayed-feedback CSTR + tau-sweep.

Same physical topology as generate_delayed_feedback.py (single CSTR, H2O sensor,
delay line, inlet-mdot actuator) but with a BOUNDED, SMOOTH-ONSET control law so
the Cantera integrator no longer diverges at the ignition spike. See
docs/superpowers/specs/2026-07-29-cstr-delayed-feedback-stable-design.md.
"""
import os
import numpy as np

try:
    import cantera as ct
except ImportError:
    ct = None

_H2O2_PATH = (
    os.path.join(os.path.dirname(ct.__file__), "data", "h2o2.yaml")
    if ct is not None else ""
)


def control_law(delayed_h2o, mdot_base, sign=-1, amplitude=0.3, center=0.48, width=0.1):
    """Bounded feedback: mdot = mdot_base * (1 + sign*A*tanh((h2o-center)/width)).

    tanh keeps the feedback term in [-A, +A], so mdot is guaranteed in
    [mdot_base*(1-A), mdot_base*(1+A)] for any input.
    """
    return mdot_base * (1.0 + sign * amplitude * np.tanh((delayed_h2o - center) / width))


def onset_factor(t, t_onset):
    """Linear ramp of feedback strength from 0 to 1 over t_onset seconds."""
    if t_onset <= 0.0:
        return 1.0
    return float(min(1.0, t / t_onset))


def periodicity_score(series, lag_min=20, lag_max=400):
    """Normalized autocorrelation peak over lag [lag_min, lag_max).

    Matches the metric used by the other cstr/generate*.py scripts: mean-remove,
    full autocorrelation, right half, normalize by the zero-lag value, then take
    the max over the lag window. Returns (score, dom_lag); dom_lag*dt is the
    dominant period in seconds.
    """
    s = np.asarray(series, dtype=float)
    s = s - s.mean()
    ac = np.correlate(s, s, mode="full")
    ac = ac[len(ac) // 2:]            # non-negative lags
    ac = ac / (ac[0] + 1e-10)         # normalize by zero-lag
    hi = min(lag_max, len(ac))
    if hi <= lag_min:
        return 1.0, lag_min
    window = ac[lag_min:hi]
    idx = int(np.argmax(window))
    return float(window[idx]), lag_min + idx
