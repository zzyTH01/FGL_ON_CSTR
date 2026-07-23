#!/usr/bin/env python
"""
Generate CSTR dataset with EXTERNAL PERIODIC FORCING.

Adds sinusoidal perturbation to the inlet mass flow rate, producing
quasi-periodic dynamics when the driving frequency is incommensurate
with the natural oscillation frequency (~0.14 Hz, period ~7.15s).

Default: f_drive = 0.05 Hz (period 20s), amplitude A = 0.3 (±30% flow)

Usage:
  uv run python cstr/generate_forced.py
  uv run python cstr/generate_forced.py --amplitude 0.5 --freq 0.05 --t_end 600
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch

try:
    import cantera as ct
except ImportError:
    print("Error: cantera is required. Install with: uv add cantera")
    sys.exit(1)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
_H2O2_PATH = os.path.join(os.path.dirname(ct.__file__), "data", "h2o2.yaml")
if not os.path.exists(_H2O2_PATH):
    print(f"Error: h2o2.yaml not found at {_H2O2_PATH}")
    sys.exit(1)


def simulate_forced_cstr(
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
    drive_amplitude=0.3,
    drive_freq=0.05,
    random_seed=42,
):
    """
    Simulate CSTR with sinusoidal inlet flow perturbation.

    mdot(t) = mdot_0 * (1 + A * sin(2 * pi * f * t))

    Args:
        drive_amplitude (float): A — relative flow amplitude (0 = no forcing)
        drive_freq (float): f — driving frequency in Hz
    """
    gas = ct.Solution(_H2O2_PATH)
    gas.TPX = temperature, pressure, composition

    upstream = ct.Reservoir(gas, clone=True)
    cstr = ct.IdealGasReactor(gas, clone=True)
    cstr.volume = reactor_volume

    env = ct.Reservoir(gas, clone=True)
    ct.Wall(cstr, env, A=wall_area, U=heat_transfer_coeff)

    # Base mass flow rate (constant part)
    sccm_base = mass_flow_sccm
    vdot_base = sccm_base * 1.0e-6 / 60.0 * ((ct.one_atm / gas.P) * (gas.T / 273.15))
    mdot_base = gas.density * vdot_base

    downstream = ct.Reservoir(gas, clone=True)
    ct.Valve(cstr, downstream, K=valve_coeff)

    network = ct.ReactorNet([cstr])

    if random_seed is not None:
        np.random.seed(random_seed)

    states = ct.SolutionArray(gas, extra=["t"])
    t = 0.0

    # Create a MassFlowController that we'll update at each step
    mfc = ct.MassFlowController(upstream, cstr, mdot=mdot_base)

    print(f"Simulating FORCED CSTR for {t_end}s with dt={dt}s ...")
    print(f"  Drive amplitude A = {drive_amplitude}")
    print(f"  Drive frequency  f = {drive_freq} Hz (period = {1.0/drive_freq:.1f}s)")
    print(f"  Natural freq     ≈ 0.14 Hz (period ≈ 7.15s)")
    print(f"  Frequency ratio  ≈ {drive_freq/0.14:.3f}")

    while t < t_end:
        t += dt
        # Update mass flow rate with sinusoidal perturbation
        mfc.mass_flow_rate = mdot_base * (1.0 + drive_amplitude * np.sin(2.0 * np.pi * drive_freq * t))
        network.advance(t)
        states.append(cstr.phase.state, t=t)

    time_array = states.t
    temp_array = states.T
    Y_array = states.Y

    species_names = gas.species_names
    try:
        h2o_idx = species_names.index("H2O")
    except ValueError:
        h2o_idx = species_names.index("H2O(L)") if "H2O(L)" in species_names else 0
    h2o_array = Y_array[:, h2o_idx]

    print(f"Generated {len(temp_array)} time points")
    print(f"Temperature:  {temp_array.min():.1f} – {temp_array.max():.1f} K")
    print(f"H2O mass frac: {h2o_array.min():.4f} – {h2o_array.max():.4f}")

    # Periodicity analysis
    h2o_ac = np.correlate(h2o_array - h2o_array.mean(),
                          h2o_array - h2o_array.mean(), mode="full")
    h2o_ac = h2o_ac[len(h2o_ac)//2:] / (h2o_ac[len(h2o_ac)//2] + 1e-10)
    periodicity = float(h2o_ac[20:400].max()) if len(h2o_ac) > 20 else 1.0
    print(f"Periodicity score: {periodicity:.4f}  (cf. unforced: ~0.952)")

    return time_array, temp_array, h2o_array, species_names


def save_pkl(series, output_path, label=""):
    col = torch.tensor(series, dtype=torch.float64).unsqueeze(1)
    tensor = torch.cat((col, col.clone()), dim=1)
    with open(output_path, "wb") as f:
        pickle.dump(tensor, f)
    print(f"  {label}: {output_path}  shape={tensor.shape}  "
          f"range=[{series.min():.4f}, {series.max():.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Generate forced CSTR data")
    parser.add_argument("--amplitude", type=float, default=0.3,
                        help="Drive amplitude A (default: 0.3 = ±30%%)")
    parser.add_argument("--freq", type=float, default=0.05,
                        help="Drive frequency in Hz (default: 0.05)")
    parser.add_argument("--t_end", type=float, default=600.0,
                        help="Simulation time in seconds (default: 600)")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Time step (default: 0.1)")
    parser.add_argument("--U", type=float, default=0.02,
                        help="Heat transfer coefficient (default: 0.02)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 55)
    print("  CSTR with External Periodic Forcing")
    print("=" * 55)
    print()

    time_array, temp_array, h2o_array, species = simulate_forced_cstr(
        heat_transfer_coeff=args.U,
        t_end=args.t_end,
        dt=args.dt,
        drive_amplitude=args.amplitude,
        drive_freq=args.freq,
        random_seed=args.seed,
    )

    print(f"\nSpecies: {species}\n")

    suffix = f"_A{args.amplitude}_f{args.freq}"
    save_pkl(temp_array,
             os.path.join(OUTPUT_DIR, f"data_forced_temp{suffix}.pkl"),
             "Temperature (forced)")
    save_pkl(h2o_array,
             os.path.join(OUTPUT_DIR, f"data_forced_h2o{suffix}.pkl"),
             "H2O mass frac (forced)")

    print(f"\nDone! Data saved with suffix '{suffix}'")


if __name__ == "__main__":
    main()
