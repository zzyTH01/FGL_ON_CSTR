#!/usr/bin/env python
"""
Stable delayed-feedback CSTR + tau-sweep.

Same physical topology as generate_delayed_feedback.py (single CSTR, H2O sensor,
delay line, inlet-mdot actuator) but with a BOUNDED, SMOOTH-ONSET control law so
the Cantera integrator no longer diverges at the ignition spike. See
docs/superpowers/specs/2026-07-29-cstr-delayed-feedback-stable-design.md.
"""
import os
import pickle
from collections import deque

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


def simulate_delayed_feedback_stable(
    temperature=770.0,
    pressure=60.0 * 133.3,
    composition="H2:2, O2:1",
    reactor_volume=10.0e-6,
    mass_flow_sccm=12.0,
    wall_area=1.0,
    heat_transfer_coeff=0.02,
    valve_coeff=1.0e-9,
    t_end=600.0,
    dt=0.1,
    tau_delay=50,
    sign=-1,
    amplitude=0.3,
    center=0.48,
    width=0.1,
    t_onset=50.0,
    filter_beta=0.1,
    random_seed=42,
):
    """Single CSTR with bounded delayed feedback on inlet mdot.

    Same topology as generate_delayed_feedback.simulate_delayed_feedback, but the
    feedback is bounded (tanh) and ramps in smoothly (onset_factor), so the
    integrator no longer diverges when the delay line first engages at the
    ignition spike.
    """
    if ct is None:
        raise RuntimeError("cantera is required")
    if not os.path.exists(_H2O2_PATH):
        raise FileNotFoundError(f"h2o2.yaml not found at {_H2O2_PATH}")

    gas = ct.Solution(_H2O2_PATH)
    gas.TPX = temperature, pressure, composition

    upstream = ct.Reservoir(gas, clone=True)
    cstr = ct.IdealGasReactor(gas, clone=True)
    cstr.volume = reactor_volume
    env = ct.Reservoir(gas, clone=True)
    ct.Wall(cstr, env, A=wall_area, U=heat_transfer_coeff)

    sccm_base = mass_flow_sccm
    vdot_base = sccm_base * 1.0e-6 / 60.0 * ((ct.one_atm / gas.P) * (gas.T / 273.15))
    mdot_base = gas.density * vdot_base

    downstream = ct.Reservoir(gas, clone=True)
    ct.Valve(cstr, downstream, K=valve_coeff)

    network = ct.ReactorNet([cstr])
    mfc = ct.MassFlowController(upstream, cstr, mdot=mdot_base)

    if random_seed is not None:
        np.random.seed(random_seed)

    species_names = gas.species_names
    try:
        h2o_idx = species_names.index("H2O")
    except ValueError:
        h2o_idx = species_names.index("H2O(L)") if "H2O(L)" in species_names else 0

    states = ct.SolutionArray(gas, extra=["t"])
    t = 0.0
    h2o_buffer = deque(maxlen=tau_delay) if tau_delay > 0 else None
    mdot_min, mdot_max = mdot_base, mdot_base
    dh_smooth = center  # EMA state: low-pass the sensed signal (actuator bandwidth)

    while t < t_end:
        t += dt
        current_h2o = cstr.phase.Y[h2o_idx]
        if h2o_buffer is not None:
            h2o_buffer.append(current_h2o)

        if h2o_buffer is not None and len(h2o_buffer) == tau_delay:
            delayed_h2o = h2o_buffer[0]
            # H2O spikes sharply at ignition; feeding it raw makes mdot swing in
            # lockstep with the spike and sharpen the ignition past CVODES tolerance.
            # An EMA smooths the sensed signal so mdot moves gradually (filter_beta=1
            # disables filtering). Filter is part of the controller, not the plant.
            dh_smooth = filter_beta * delayed_h2o + (1 - filter_beta) * dh_smooth
            a_eff = onset_factor(t, t_onset)
            mdot_full = control_law(dh_smooth, mdot_base, sign, amplitude, center, width)
            mdot = mdot_base + a_eff * (mdot_full - mdot_base)   # onset interpolates
        else:
            mdot = mdot_base

        mdot_min = min(mdot_min, mdot)
        mdot_max = max(mdot_max, mdot)
        mfc.mass_flow_rate = mdot
        network.advance(t)
        states.append(cstr.phase.state, t=t)

    return {
        "t": states.t,
        "T": states.T,
        "h2o": states.Y[:, h2o_idx],
        "mdot_base": float(mdot_base),
        "mdot_min": float(mdot_min),
        "mdot_max": float(mdot_max),
    }


def save_pkl(series, output_path, label=""):
    """Save a 1-D series as a 2-column float64 tensor (same format as other generators)."""
    import torch
    col = torch.tensor(np.asarray(series), dtype=torch.float64).unsqueeze(1)
    tensor = torch.cat((col, col.clone()), dim=1)
    with open(output_path, "wb") as f:
        pickle.dump(tensor, f)
    print(f"  {label}: {output_path}  shape={tuple(tensor.shape)}")
