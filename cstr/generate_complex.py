#!/usr/bin/env python
"""Generate complex CSTR datasets: multi-delay, forcing+delay, parameter modulation.

New complexity dimensions beyond existing generate_forced/dual/delayed_stable:
  1. multi_delay   — two or three simultaneous delayed feedback loops
  2. forced_delay  — external sinusoidal forcing + delayed feedback combined
  3. param_mod     — slowly varying reactor volume (parameter drift)
  4. noisy_control — stochastic perturbation on the control input
  5. multichannel  — (H2O, temperature) two-column output instead of duplicate

Usage:
  uv run python cstr/generate_complex.py --sweep
  uv run python cstr/generate_complex.py --mode multi_delay --tau1 30 --tau2 70
"""
from __future__ import annotations

import argparse
import os
import pickle
import torch
import sys
import time

import numpy as np

try:
    import cantera as ct
except ImportError:
    print("Error: cantera is required")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
_H2O2_PATH = os.path.join(os.path.dirname(ct.__file__), "data", "h2o2.yaml")
if not os.path.exists(_H2O2_PATH):
    print(f"Error: h2o2.yaml not found at {_H2O2_PATH}")
    sys.exit(1)


def periodicity_score(series, lag_min=20, lag_max=400):
    """Same metric as generate_delayed_stable.py."""
    s = np.asarray(series, dtype=float)
    s = s - s.mean()
    ac = np.correlate(s, s, mode="full")
    ac = ac[len(ac) // 2:]
    ac = ac / (ac[0] + 1e-10)
    hi = min(lag_max, len(ac))
    if hi <= lag_min:
        return 1.0, lag_min
    window = ac[lag_min:hi]
    idx = int(np.argmax(window))
    return float(window[idx]), lag_min + idx


def _make_gas(temperature=770.0, pressure_pa=60.0 * 133.3, composition="H2:2, O2:1"):
    gas = ct.Solution(_H2O2_PATH)
    gas.TPX = temperature, pressure_pa, composition
    return gas


def _base_mdot(gas, sccm=12.0):
    vdot = sccm * 1.0e-6 / 60.0 * ((ct.one_atm / gas.P) * (gas.T / 273.15))
    return gas.density * vdot


# ==============================================================================
# Mode 1: multi-delay feedback (two or three τ loops simultaneously)
# ==============================================================================

def simulate_multi_delay(
    taus=(30, 70), amplitudes=(0.15, 0.15), signs=(-1, -1),
    centers=(0.48, 0.48), widths=(0.1, 0.1),
    temperature=770.0, pressure_pa=60.0 * 133.3, composition="H2:2, O2:1",
    reactor_volume=10.0e-6, mass_flow_sccm=12.0,
    wall_area=1.0, heat_transfer_coeff=0.02, valve_coeff=1.0e-9,
    t_end=600.0, dt=0.1, t_onset=50.0, random_seed=42,
    multichannel=False,
):
    """CSTR with N simultaneous delayed feedback loops on inlet mdot.

    Each loop i uses its own delay tau_i, amplitude A_i, sign, center, width.
    Total feedback: mdot = mdot_base * (1 + sum_i(sign_i * A_i * tanh(...)))
    with a global bound to keep |total| < 0.9.
    """
    gas = _make_gas(temperature, pressure_pa, composition)
    mdot_base = _base_mdot(gas, mass_flow_sccm)

    upstream = ct.Reservoir(gas, clone=True)
    cstr = ct.IdealGasReactor(gas, clone=True)
    mfc = ct.MassFlowController(upstream, cstr, mdot=mdot_base)
    cstr.volume = reactor_volume
    env = ct.Reservoir(gas, clone=True)
    ct.Wall(cstr, env, A=wall_area, U=heat_transfer_coeff)
    downstream = ct.Reservoir(gas, clone=True)
    ct.Valve(cstr, downstream, K=valve_coeff)
    net = ct.ReactorNet([cstr])

    n_steps = int(t_end / dt)
    delay_bufs = [np.zeros(max(1, int(tau / dt))) for tau in taus]
    h2o_series = np.zeros(n_steps)
    temp_series = np.zeros(n_steps)

    dh_smooth = centers[0] if centers else 0.48
    filter_beta = 0.1
    for step in range(n_steps):
        t = step * dt
        h2o = cstr.thermo["H2O"].Y[0]
        temp = cstr.T

        # Push current h2o into each delay buffer
        for buf in delay_bufs:
            buf[:-1] = buf[1:]
            buf[-1] = h2o

        # EMA smooth (same as delayed_stable)
        dh_smooth = filter_beta * h2o + (1 - filter_beta) * dh_smooth

        # Onset ramp
        onset = min(1.0, t / t_onset) if t_onset > 0 else 1.0

        # Multi-loop feedback (EMA-smoothed)
        total_feedback = 0.0
        for i, buf in enumerate(delay_bufs):
            delayed_val = buf[0]
            if delayed_val != 0:  # buffer has data
                fb = signs[i] * amplitudes[i] * np.tanh((dh_smooth - centers[i]) / widths[i])
                total_feedback += fb
        total_feedback = np.clip(total_feedback, -0.9, 0.9)

        # Onset interpolation from mdot_base
        mdot_target = mdot_base * (1.0 + total_feedback)
        mdot = mdot_base + onset * (mdot_target - mdot_base)

        mfc.mass_flow_rate = mdot
        net.advance(t + dt)

        h2o_series[step] = cstr.thermo["H2O"].Y[0]
        temp_series[step] = cstr.T

    if multichannel:
        return np.column_stack([h2o_series, temp_series])
    return np.column_stack([h2o_series, h2o_series])


# ==============================================================================
# Mode 2: external forcing + delayed feedback combined
# ==============================================================================

def simulate_forced_delay(
    tau_delay=50, sign=-1, amplitude=0.2, center=0.48, width=0.1,
    drive_amplitude=0.15, drive_freq=0.03,
    temperature=770.0, pressure_pa=60.0 * 133.3, composition="H2:2, O2:1",
    reactor_volume=10.0e-6, mass_flow_sccm=12.0,
    wall_area=1.0, heat_transfer_coeff=0.02, valve_coeff=1.0e-9,
    t_end=600.0, dt=0.1, t_onset=50.0, random_seed=42,
    multichannel=False,
):
    """CSTR with both sinusoidal forcing AND delayed feedback on inlet mdot."""
    gas = _make_gas(temperature, pressure_pa, composition)
    mdot_base = _base_mdot(gas, mass_flow_sccm)

    upstream = ct.Reservoir(gas, clone=True)
    cstr = ct.IdealGasReactor(gas, clone=True)
    mfc = ct.MassFlowController(upstream, cstr, mdot=mdot_base)
    cstr.volume = reactor_volume
    env = ct.Reservoir(gas, clone=True)
    ct.Wall(cstr, env, A=wall_area, U=heat_transfer_coeff)
    downstream = ct.Reservoir(gas, clone=True)
    ct.Valve(cstr, downstream, K=valve_coeff)
    net = ct.ReactorNet([cstr])

    n_steps = int(t_end / dt)
    delay_buf = np.zeros(max(1, int(tau_delay / dt)))
    h2o_series = np.zeros(n_steps)
    temp_series = np.zeros(n_steps)

    dh_smooth = center  # EMA state (same as delayed_stable)
    filter_beta = 0.1
    for step in range(n_steps):
        t = step * dt
        h2o = cstr.thermo["H2O"].Y[0]
        temp = cstr.T

        delay_buf[:-1] = delay_buf[1:]
        delay_buf[-1] = h2o

        if len(delay_buf) > 0 and delay_buf[-1] != 0:
            delayed_val = delay_buf[0]
            dh_smooth = filter_beta * delayed_val + (1 - filter_beta) * dh_smooth
            a_eff = min(1.0, t / t_onset) if t_onset > 0 else 1.0
            # Delayed feedback (bounded tanh, same as delayed_stable)
            fb = sign * amplitude * np.tanh((dh_smooth - center) / width)
            # Sinusoidal forcing
            forcing = drive_amplitude * np.sin(2 * np.pi * drive_freq * t)
            # Onset interpolation from mdot_base
            mdot_target = mdot_base * (1.0 + fb + forcing)
            mdot = mdot_base + a_eff * (mdot_target - mdot_base)
        else:
            mdot = mdot_base

        mfc.mass_flow_rate = mdot
        net.advance(t + dt)

        h2o_series[step] = cstr.thermo["H2O"].Y[0]
        temp_series[step] = cstr.T

    if multichannel:
        return np.column_stack([h2o_series, temp_series])
    return np.column_stack([h2o_series, h2o_series])


# ==============================================================================
# Mode 3: slowly varying reactor volume (parameter drift / nonstationarity)
# ==============================================================================

def simulate_param_mod(
    tau_delay=50, sign=-1, amplitude=0.2, center=0.48, width=0.1,
    volume_start=8.0e-6, volume_end=14.0e-6, mod_period=200.0,
    temperature=770.0, pressure_pa=60.0 * 133.3, composition="H2:2, O2:1",
    mass_flow_sccm=12.0,
    wall_area=1.0, heat_transfer_coeff=0.02, valve_coeff=1.0e-9,
    t_end=600.0, dt=0.1, t_onset=50.0, random_seed=42,
    multichannel=False,
):
    """CSTR with delayed feedback AND slowly oscillating reactor volume."""
    gas = _make_gas(temperature, pressure_pa, composition)
    mdot_base = _base_mdot(gas, mass_flow_sccm)

    upstream = ct.Reservoir(gas, clone=True)
    cstr = ct.IdealGasReactor(gas, clone=True)
    mfc = ct.MassFlowController(upstream, cstr, mdot=mdot_base)
    cstr.volume = volume_start
    env = ct.Reservoir(gas, clone=True)
    ct.Wall(cstr, env, A=wall_area, U=heat_transfer_coeff)
    downstream = ct.Reservoir(gas, clone=True)
    ct.Valve(cstr, downstream, K=valve_coeff)
    net = ct.ReactorNet([cstr])

    n_steps = int(t_end / dt)
    delay_buf = np.zeros(max(1, int(tau_delay / dt)))
    h2o_series = np.zeros(n_steps)
    temp_series = np.zeros(n_steps)
    dh_smooth = center
    filter_beta = 0.1

    for step in range(n_steps):
        t = step * dt
        h2o = cstr.thermo["H2O"].Y[0]
        temp = cstr.T

        # Slowly modulate reactor volume (sinusoidal drift)
        vol_mid = (volume_start + volume_end) / 2.0
        vol_amp = (volume_end - volume_start) / 2.0
        cstr.volume = vol_mid + vol_amp * np.sin(2 * np.pi * t / mod_period)

        delay_buf[:-1] = delay_buf[1:]
        delay_buf[-1] = h2o

        onset = min(1.0, t / t_onset) if t_onset > 0 else 1.0
        delayed_val = delay_buf[0]
        dh_smooth = filter_beta * delayed_val + (1 - filter_beta) * dh_smooth
        fb = sign * amplitude * np.tanh((dh_smooth - center) / width)
        mdot_target = mdot_base * (1.0 + fb)
        mdot = mdot_base + onset * (mdot_target - mdot_base)
        mfc.mass_flow_rate = mdot
        net.advance(t + dt)

        h2o_series[step] = cstr.thermo["H2O"].Y[0]
        temp_series[step] = cstr.T

    if multichannel:
        return np.column_stack([h2o_series, temp_series])
    return np.column_stack([h2o_series, h2o_series])


# ==============================================================================
# Mode 4: stochastic control perturbation (noisy feedback)
# ==============================================================================

def simulate_noisy_control(
    tau_delay=50, sign=-1, amplitude=0.2, center=0.48, width=0.1,
    noise_std=0.02,
    temperature=770.0, pressure_pa=60.0 * 133.3, composition="H2:2, O2:1",
    reactor_volume=10.0e-6, mass_flow_sccm=12.0,
    wall_area=1.0, heat_transfer_coeff=0.02, valve_coeff=1.0e-9,
    t_end=600.0, dt=0.1, t_onset=50.0, random_seed=42,
    multichannel=False,
):
    """CSTR with delayed feedback + Gaussian noise on control signal."""
    gas = _make_gas(temperature, pressure_pa, composition)
    mdot_base = _base_mdot(gas, mass_flow_sccm)

    upstream = ct.Reservoir(gas, clone=True)
    cstr = ct.IdealGasReactor(gas, clone=True)
    mfc = ct.MassFlowController(upstream, cstr, mdot=mdot_base)
    cstr.volume = reactor_volume
    env = ct.Reservoir(gas, clone=True)
    ct.Wall(cstr, env, A=wall_area, U=heat_transfer_coeff)
    downstream = ct.Reservoir(gas, clone=True)
    ct.Valve(cstr, downstream, K=valve_coeff)
    net = ct.ReactorNet([cstr])

    n_steps = int(t_end / dt)
    delay_buf = np.zeros(max(1, int(tau_delay / dt)))
    h2o_series = np.zeros(n_steps)
    temp_series = np.zeros(n_steps)
    dh_smooth = center
    filter_beta = 0.1
    rng = np.random.RandomState(random_seed)

    for step in range(n_steps):
        t = step * dt
        h2o = cstr.thermo["H2O"].Y[0]
        temp = cstr.T

        delay_buf[:-1] = delay_buf[1:]
        delay_buf[-1] = h2o

        onset = min(1.0, t / t_onset) if t_onset > 0 else 1.0
        delayed_val = delay_buf[0]
        dh_smooth = filter_beta * delayed_val + (1 - filter_beta) * dh_smooth
        fb = sign * amplitude * np.tanh((dh_smooth - center) / width)
        noise = rng.normal(0, noise_std)
        mdot_target = mdot_base * (1.0 + fb + noise)
        mdot = mdot_base + onset * (mdot_target - mdot_base)
        mfc.mass_flow_rate = mdot
        net.advance(t + dt)

        h2o_series[step] = cstr.thermo["H2O"].Y[0]
        temp_series[step] = cstr.T

    if multichannel:
        return np.column_stack([h2o_series, temp_series])
    return np.column_stack([h2o_series, h2o_series])


# ==============================================================================

def _save(arr, name):
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(torch.tensor(arr, dtype=torch.float64), f)
    score, lag = periodicity_score(arr[:, 0])
    print(f"  -> {name}  (N={len(arr)}, periodicity={score:.4f}, dom_lag={lag})")
    return path


def _sweep():
    """Generate a batch of complex datasets across all modes."""
    import torch
    print("=== CSTR complex dataset sweep ===\n")
    t0 = time.time()

    # --- Multi-delay: different tau combinations ---
    print("--- Mode 1: multi-delay feedback ---")
    for taus, amps in [
        ((30, 70), (0.1, 0.1)),
        ((50, 120), (0.1, 0.1)),
        ((30, 70, 120), (0.05, 0.05, 0.05)),
    ]:
        signs = tuple(-1 for _ in taus)
        centers = tuple(0.48 for _ in taus)
        widths = tuple(0.1 for _ in taus)
        tag = "_".join(str(t) for t in taus)
        try:
            arr = simulate_multi_delay(taus=taus, amplitudes=amps, signs=signs,
                                       centers=centers, widths=widths)
            _save(arr, f"data_complex_multidelay_tau{tag}.pkl")
        except Exception as e:
            print(f"  SKIPPED tau{tag}: {type(e).__name__}")

    # --- Forced + delay ---
    print("\n--- Mode 2: forcing + delayed feedback ---")
    for tau, da, df in [
        (50, 0.05, 0.03),
        (50, 0.1, 0.05),
        (80, 0.08, 0.02),
        (100, 0.05, 0.07),
    ]:
        try:
            arr = simulate_forced_delay(tau_delay=tau, drive_amplitude=da, drive_freq=df)
            _save(arr, f"data_complex_forceddelay_tau{tau}_A{da}_f{df}.pkl")
        except Exception as e:
            print(f"  SKIPPED tau{tau}_A{da}_f{df}: {e}")

    # --- Parameter modulation ---
    print("\n--- Mode 3: parameter modulation (volume drift) ---")
    for tau, vs, ve, mp in [
        (50, 8e-6, 14e-6, 200.0),
        (80, 8e-6, 14e-6, 150.0),
        (50, 6e-6, 16e-6, 300.0),
    ]:
        try:
            arr = simulate_param_mod(tau_delay=tau, volume_start=vs, volume_end=ve, mod_period=mp)
            _save(arr, f"data_complex_parammod_tau{tau}_V{vs:.0e}-{ve:.0e}_T{mp:.0f}.pkl")
        except Exception as e:
            print(f"  SKIPPED tau{tau}: {e}")

    # --- Noisy control ---
    print("\n--- Mode 4: noisy control ---")
    for tau, ns in [
        (50, 0.01),
        (50, 0.05),
        (100, 0.02),
        (100, 0.08),
    ]:
        try:
            arr = simulate_noisy_control(tau_delay=tau, noise_std=ns)
            _save(arr, f"data_complex_noisy_tau{tau}_s{ns}.pkl")
        except Exception as e:
            print(f"  SKIPPED tau{tau}_s{ns}: {e}")

    print(f"\nTotal time: {time.time() - t0:.0f}s")


def _single(args):
    import torch
    print(f"=== Single complex dataset: mode={args.mode} ===")
    if args.mode == "multi_delay":
        taus = tuple(int(x) for x in args.taus.split(","))
        amps = tuple(float(x) for x in args.amplitudes.split(","))
        signs = tuple(-1 for _ in taus)
        arr = simulate_multi_delay(taus=taus, amplitudes=amps, signs=signs)
        tag = "_".join(str(t) for t in taus)
        _save(arr, f"data_complex_multidelay_tau{tag}.pkl")
    elif args.mode == "forced_delay":
        arr = simulate_forced_delay(tau_delay=args.tau, drive_amplitude=args.drive_amp,
                                    drive_freq=args.drive_freq)
        _save(arr, f"data_complex_forceddelay_tau{args.tau}_A{args.drive_amp}_f{args.drive_freq}.pkl")
    elif args.mode == "param_mod":
        arr = simulate_param_mod(tau_delay=args.tau, volume_start=args.vol_start,
                                 volume_end=args.vol_end, mod_period=args.mod_period)
        _save(arr, f"data_complex_parammod_tau{args.tau}.pkl")
    elif args.mode == "noisy":
        arr = simulate_noisy_control(tau_delay=args.tau, noise_std=args.noise_std)
        _save(arr, f"data_complex_noisy_tau{args.tau}_s{args.noise_std}.pkl")
    else:
        raise ValueError(f"unknown mode {args.mode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true", help="generate all complex variants")
    parser.add_argument("--mode", choices=("multi_delay", "forced_delay", "param_mod", "noisy"))
    parser.add_argument("--taus", default="30,70", help="comma-separated delays for multi_delay")
    parser.add_argument("--amplitudes", default="0.15,0.15", help="comma-separated amplitudes")
    parser.add_argument("--tau", type=int, default=50, help="delay for single-delay modes")
    parser.add_argument("--drive_amp", type=float, default=0.15)
    parser.add_argument("--drive_freq", type=float, default=0.03)
    parser.add_argument("--vol_start", type=float, default=8e-6)
    parser.add_argument("--vol_end", type=float, default=14e-6)
    parser.add_argument("--mod_period", type=float, default=200.0)
    parser.add_argument("--noise_std", type=float, default=0.02)
    args = parser.parse_args()

    if args.sweep:
        _sweep()
    elif args.mode:
        _single(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
