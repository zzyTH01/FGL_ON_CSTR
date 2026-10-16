#!/usr/bin/env python
"""Generate complex CSTR datasets: multi-delay, forcing+delay, parameter modulation.

Two generation pipelines in one file:

**Mode generator (writes to ``cstr/data/``):**
  1. multi_delay   — two or three simultaneous delayed feedback loops
  2. forced_delay  — external sinusoidal forcing + delayed feedback combined
  3. param_mod     — slowly varying reactor volume (parameter drift)
  4. noisy_control — stochastic perturbation on the control input

**Multichannel generator (writes to ``cstr/data/complex/``):**
  Selects top-variance species channels and exports FGL-compatible PKLs
  plus a combined NPZ.

Usage:
  # Mode generator
  uv run python cstr/generate_complex.py --sweep
  uv run python cstr/generate_complex.py --mode multi_delay --tau1 30 --tau2 70

  # Multichannel generator
  uv run python cstr/generate_complex.py --multichannel --tau 50
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    import cantera as ct
except ImportError:
    ct = None

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _require_cantera():
    if ct is None or not hasattr(ct, "__version__"):
        raise RuntimeError("Cantera is required (>= 3.2.0)")
    version = tuple(int(x) for x in ct.__version__.split(".")[:2])
    if version < (3, 2):
        raise RuntimeError(f"Cantera >= 3.2.0 required, found {ct.__version__}")


def _h2o2_path():
    _require_cantera()
    path = os.path.join(os.path.dirname(ct.__file__), "data", "h2o2.yaml")
    if not os.path.exists(path):
        raise RuntimeError(f"h2o2.yaml not found at {path}")
    return path


# ==============================================================================
# Periodicity score
# ==============================================================================

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
    gas = ct.Solution(_h2o2_path())
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
    """CSTR with N simultaneous delayed feedback loops on inlet mdot."""
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

        for buf in delay_bufs:
            buf[:-1] = buf[1:]
            buf[-1] = h2o

        dh_smooth = filter_beta * h2o + (1 - filter_beta) * dh_smooth

        onset = min(1.0, t / t_onset) if t_onset > 0 else 1.0

        total_feedback = 0.0
        for i, buf in enumerate(delay_bufs):
            delayed_val = buf[0]
            if delayed_val != 0:
                fb = signs[i] * amplitudes[i] * np.tanh((dh_smooth - centers[i]) / widths[i])
                total_feedback += fb
        total_feedback = np.clip(total_feedback, -0.9, 0.9)

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
# Mode 2: forced + delayed feedback
# ==============================================================================

def simulate_forced_delay(
    tau_delay=50, drive_amplitude=0.1, drive_freq=0.03,
    feedback_amplitude=0.15, center=0.48, width=0.1,
    temperature=770.0, pressure_pa=60.0 * 133.3,
    reactor_volume=10.0e-6, mass_flow_sccm=12.0,
    wall_area=1.0, heat_transfer_coeff=0.02, valve_coeff=1.0e-9,
    t_end=600.0, dt=0.1, t_onset=50.0,
):
    """CSTR with sinusoidal forcing + delayed feedback on inlet mdot."""
    gas = _make_gas(temperature, pressure_pa, "H2:2, O2:1")
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
    for step in range(n_steps):
        t = step * dt
        h2o = cstr.thermo["H2O"].Y[0]

        delay_buf[:-1] = delay_buf[1:]
        delay_buf[-1] = h2o
        dh_smooth = filter_beta * h2o + (1 - filter_beta) * dh_smooth

        onset = min(1.0, t / t_onset) if t_onset > 0 else 1.0
        drive = drive_amplitude * np.sin(2 * np.pi * drive_freq * t)
        delayed_val = delay_buf[0]
        fb = 0.0
        if delayed_val != 0:
            fb = -feedback_amplitude * np.tanh((dh_smooth - center) / width)
        total = np.clip(drive + fb, -0.9, 0.9)

        mdot_target = mdot_base * (1.0 + total)
        mdot = mdot_base + onset * (mdot_target - mdot_base)

        mfc.mass_flow_rate = mdot
        net.advance(t + dt)

        h2o_series[step] = cstr.thermo["H2O"].Y[0]
        temp_series[step] = cstr.T

    return np.column_stack([h2o_series, h2o_series])


# ==============================================================================
# Mode 3: parameter modulation (slowly varying reactor volume)
# ==============================================================================

def simulate_param_mod(
    tau_delay=50, volume_start=8e-6, volume_end=14e-6, mod_period=200.0,
    feedback_amplitude=0.15, center=0.48, width=0.1,
    temperature=770.0, pressure_pa=60.0 * 133.3,
    mass_flow_sccm=12.0,
    wall_area=1.0, heat_transfer_coeff=0.02, valve_coeff=1.0e-9,
    t_end=600.0, dt=0.1, t_onset=50.0,
):
    """CSTR with delayed feedback + sinusoidal reactor-volume drift."""
    gas = _make_gas(temperature, pressure_pa, "H2:2, O2:1")
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
    vol_mid = (volume_start + volume_end) / 2
    vol_amp = (volume_end - volume_start) / 2
    for step in range(n_steps):
        t = step * dt
        h2o = cstr.thermo["H2O"].Y[0]

        delay_buf[:-1] = delay_buf[1:]
        delay_buf[-1] = h2o
        dh_smooth = filter_beta * h2o + (1 - filter_beta) * dh_smooth

        onset = min(1.0, t / t_onset) if t_onset > 0 else 1.0
        cstr.volume = vol_mid + vol_amp * np.sin(2 * np.pi * t / mod_period)

        delayed_val = delay_buf[0]
        fb = 0.0
        if delayed_val != 0:
            fb = -feedback_amplitude * np.tanh((dh_smooth - center) / width)
        total = np.clip(fb, -0.9, 0.9)

        mdot_target = mdot_base * (1.0 + total)
        mdot = mdot_base + onset * (mdot_target - mdot_base)

        mfc.mass_flow_rate = mdot
        net.advance(t + dt)

        h2o_series[step] = cstr.thermo["H2O"].Y[0]
        temp_series[step] = cstr.T

    return np.column_stack([h2o_series, h2o_series])


# ==============================================================================
# Mode 4: noisy control
# ==============================================================================

def simulate_noisy_control(
    tau_delay=50, noise_std=0.02,
    feedback_amplitude=0.15, center=0.48, width=0.1,
    temperature=770.0, pressure_pa=60.0 * 133.3,
    reactor_volume=10.0e-6, mass_flow_sccm=12.0,
    wall_area=1.0, heat_transfer_coeff=0.02, valve_coeff=1.0e-9,
    t_end=600.0, dt=0.1, t_onset=50.0, random_seed=42,
):
    """CSTR with delayed feedback + additive Gaussian noise on control input."""
    rng = np.random.RandomState(random_seed)
    gas = _make_gas(temperature, pressure_pa, "H2:2, O2:1")
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
    for step in range(n_steps):
        t = step * dt
        h2o = cstr.thermo["H2O"].Y[0]

        delay_buf[:-1] = delay_buf[1:]
        delay_buf[-1] = h2o
        dh_smooth = filter_beta * h2o + (1 - filter_beta) * dh_smooth

        onset = min(1.0, t / t_onset) if t_onset > 0 else 1.0
        noise = rng.normal(0, noise_std)
        delayed_val = delay_buf[0]
        fb = 0.0
        if delayed_val != 0:
            fb = -feedback_amplitude * np.tanh((dh_smooth - center) / width)
        total = np.clip(fb + noise, -0.9, 0.9)

        mdot_target = mdot_base * (1.0 + total)
        mdot = mdot_base + onset * (mdot_target - mdot_base)

        mfc.mass_flow_rate = mdot
        net.advance(t + dt)

        h2o_series[step] = cstr.thermo["H2O"].Y[0]
        temp_series[step] = cstr.T

    return np.column_stack([h2o_series, h2o_series])


# ==============================================================================
# Multichannel generator (species selection + FGL-compatible export)
# ==============================================================================

from cstr.generate_delayed_stable import simulate_delayed_feedback_stable

_DEFAULT_OUTDIR = Path(__file__).resolve().parent / "data" / "complex"


def select_channels(species_names, species_arrays, temperature, k=6):
    """Select top-variance species, always including H2O and temperature."""
    names = list(species_names)
    arrays = [np.asarray(a, dtype=float) for a in species_arrays]
    scored = sorted(
        (
            (float(np.ptp(a) + np.var(a)), i)
            for i, a in enumerate(arrays)
            if names[i] != "H2O"
        ),
        reverse=True,
    )
    selected_indices = [i for _, i in scored[: max(0, k - 2)]]
    h2o_index = names.index("H2O") if "H2O" in names else None
    if h2o_index is not None:
        selected_indices.append(h2o_index)
    selected_names = ["temperature"] + [names[i] for i in dict.fromkeys(selected_indices)]
    return selected_names[: max(2, k)]


def save_multichannel(path, time, temperature, species_names, species_arrays, metadata):
    """Save time, temperature, selected species arrays and metadata to NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"time": np.asarray(time), "temperature": np.asarray(temperature)}
    for name, array in zip(species_names, species_arrays):
        arrays[name] = np.asarray(array)
    arrays["species_names"] = np.array(list(species_names))
    arrays["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(path, **arrays)


def export_single_channel_pkls(npz_path, outdir):
    """Export temperature and each species as existing FGL-compatible PKLs."""
    npz_path = Path(npz_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path, allow_pickle=False) as payload:
        tau = int(json.loads(str(payload["metadata_json"]))["tau"])
        channels = ["temperature"] + [str(x) for x in payload["species_names"]]
        created = []
        for channel in channels:
            series = np.asarray(payload[channel])
            column = torch.tensor(series, dtype=torch.float64).unsqueeze(1)
            tensor = torch.cat((column, column.clone()), dim=1)
            filename = f"data_complex_cstr_{channel}_tau{tau}.pkl"
            output = outdir / filename
            with open(output, "wb") as f:
                pickle.dump(tensor, f)
            created.append(str(output))
    return created


def simulate_complex_cstr(tau=50, t_end=300.0, dt=0.1, **kwargs):
    """Simulate the stable delayed CSTR and return all species channels."""
    _require_cantera()
    if tau <= 0:
        raise ValueError("tau must be positive")
    return simulate_delayed_feedback_stable(
        t_end=t_end,
        dt=dt,
        tau_delay=tau,
        **kwargs,
    )


def generate(tau, outdir=_DEFAULT_OUTDIR, t_end=300.0, dt=0.1, k=6, overwrite=False):
    """Generate one multichannel NPZ and export compatible channel PKLs."""
    _require_cantera()
    outdir = Path(outdir)
    npz_path = outdir / f"data_complex_cstr_multichannel_tau{tau}.npz"
    if npz_path.exists() and not overwrite:
        return npz_path, export_single_channel_pkls(npz_path, outdir)
    result = simulate_complex_cstr(tau=tau, t_end=t_end, dt=dt)
    species_names = list(result["species_names"])
    mass_fractions = result["species_matrix"]
    if mass_fractions is None or mass_fractions.size == 0:
        raise RuntimeError("delayed CSTR simulator did not return species matrix")
    selected = select_channels(species_names, mass_fractions.T, result["T"], k=k)
    species_arrays = [mass_fractions[:, species_names.index(name)] for name in selected[1:]]
    metadata = {
        "mechanism": "h2o2.yaml",
        "cantera_version": ct.__version__,
        "tau": tau,
        "t_end": t_end,
        "dt": dt,
        "composition": "H2:2, O2:1",
        "units": {"time": "s", "temperature": "K", "species": "mass fraction"},
    }
    save_multichannel(
        npz_path,
        result["t"],
        result["T"],
        selected[1:],
        species_arrays,
        metadata,
    )
    return npz_path, export_single_channel_pkls(npz_path, outdir)


# ==============================================================================
# CLI
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


def _multichannel_single(args):
    print(f"=== Multichannel generation: tau={args.mc_tau} ===")
    npz_path, pkls = generate(
        args.mc_tau,
        outdir=Path(args.mc_outdir) if args.mc_outdir else _DEFAULT_OUTDIR,
        t_end=args.mc_t_end,
        dt=args.mc_dt,
        k=args.mc_k,
        overwrite=args.mc_overwrite,
    )
    print(f"NPZ: {npz_path}")
    for path in pkls:
        print(f"PKL: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Mode generator options
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
    # Multichannel generator options
    parser.add_argument("--multichannel", action="store_true", help="generate multichannel NPZ + PKLs")
    parser.add_argument("--mc_tau", type=int, default=50, help="delay for multichannel mode")
    parser.add_argument("--mc_t_end", type=float, default=300.0)
    parser.add_argument("--mc_dt", type=float, default=0.1)
    parser.add_argument("--mc_k", type=int, default=6)
    parser.add_argument("--mc_outdir", type=str, default=None)
    parser.add_argument("--mc_overwrite", action="store_true")
    args = parser.parse_args()

    if args.sweep:
        _sweep()
    elif args.mode:
        _single(args)
    elif args.multichannel:
        _multichannel_single(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
