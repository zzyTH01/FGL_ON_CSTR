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
