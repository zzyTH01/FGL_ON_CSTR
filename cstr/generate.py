#!/usr/bin/env python
"""
Generate CSTR (Continuously Stirred Tank Reactor) oscillatory time series dataset.

Simulates a stoichiometric H2/O2 reaction in a CSTR that exhibits periodic
oscillations due to water's role as a third-body chain terminator.

Outputs two .pkl files:
  - data.pkl          : Temperature (K) — suitable for spike-detection tasks
  - data_h2o.pkl      : H2O mass fraction — smooth continuous oscillation, best for FGL

Reference: Cantera CSTR example — h2o2.yaml mechanism
Requires: cantera >= 3.2.0
"""

import os
import sys
import pickle
import numpy as np
import torch

try:
    import cantera as ct
except ImportError:
    print("Error: cantera is required. Install with: uv add cantera")
    sys.exit(1)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Locate cantera's h2o2.yaml (avoids encoding issues with Chinese path)
_H2O2_PATH = os.path.join(os.path.dirname(ct.__file__), "data", "h2o2.yaml")
if not os.path.exists(_H2O2_PATH):
    print(f"Error: h2o2.yaml not found at {_H2O2_PATH}")
    sys.exit(1)


def simulate_cstr(
    temperature=770.0,
    pressure=60.0 * 133.3,
    composition="H2:2, O2:1",
    reactor_volume=10.0e-6,
    mass_flow_sccm=12.0,
    wall_area=1.0,
    heat_transfer_coeff=0.02,
    valve_coeff=1.0e-9,
    t_end=300.0,
    dt=0.1,
    random_seed=None,
):
    """Simulate the CSTR and return time, temperature, and mass fraction arrays."""
    gas = ct.Solution(_H2O2_PATH)
    gas.TPX = temperature, pressure, composition

    upstream = ct.Reservoir(gas, clone=True)
    cstr = ct.IdealGasReactor(gas, clone=True)
    cstr.volume = reactor_volume

    env = ct.Reservoir(gas, clone=True)
    ct.Wall(cstr, env, A=wall_area, U=heat_transfer_coeff)

    sccm = mass_flow_sccm
    vdot = sccm * 1.0e-6 / 60.0 * ((ct.one_atm / gas.P) * (gas.T / 273.15))
    mdot = gas.density * vdot
    ct.MassFlowController(upstream, cstr, mdot=mdot)

    downstream = ct.Reservoir(gas, clone=True)
    ct.Valve(cstr, downstream, K=valve_coeff)

    network = ct.ReactorNet([cstr])

    if random_seed is not None:
        np.random.seed(random_seed)

    states = ct.SolutionArray(gas, extra=["t"])
    t = 0.0

    print(f"Simulating CSTR for {t_end}s with dt={dt}s ...")
    while t < t_end:
        t += dt
        network.advance(t)
        states.append(cstr.phase.state, t=t)

    time_array = states.t
    temp_array = states.T           # Temperature (K)
    Y_array = states.Y              # Mass fractions: [H2, H2O, O2, ...]

    # Find H2O mass fraction index
    species_names = gas.species_names
    try:
        h2o_idx = species_names.index("H2O")
    except ValueError:
        h2o_idx = species_names.index("H2O(L)") if "H2O(L)" in species_names else 0
    h2o_array = Y_array[:, h2o_idx]

    print(f"Generated {len(temp_array)} time points")
    print(f"Temperature:  {temp_array.min():.1f} – {temp_array.max():.1f} K")
    print(f"H2O mass frac: {h2o_array.min():.4f} – {h2o_array.max():.4f}")

    return time_array, temp_array, h2o_array, species_names


def save_pkl(series, output_path, label=""):
    """Save a 1D series as two-column float64 tensor."""
    col = torch.tensor(series, dtype=torch.float64).unsqueeze(1)
    tensor = torch.cat((col, col.clone()), dim=1)

    with open(output_path, "wb") as f:
        pickle.dump(tensor, f)

    print(f"  {label}: {output_path}  shape={tensor.shape}  range=[{series.min():.4f}, {series.max():.4f}]")


def main():
    print("=" * 50)
    print("  CSTR Oscillatory Time Series Generator")
    print("=" * 50 + "\n")

    time_array, temp_array, h2o_array, species = simulate_cstr(
        temperature=770.0,
        pressure=60.0 * 133.3,
        composition="H2:2, O2:1",
        reactor_volume=10.0e-6,
        mass_flow_sccm=12.0,
        wall_area=1.0,
        heat_transfer_coeff=0.02,
        valve_coeff=1.0e-9,
        t_end=300.0,
        dt=0.1,
        random_seed=42,
    )

    print(f"\nSpecies: {species}\n")

    save_pkl(temp_array, os.path.join(OUTPUT_DIR, "data.pkl"), "Temperature")
    save_pkl(h2o_array, os.path.join(OUTPUT_DIR, "data_h2o.pkl"), "H2O mass frac")

    # Quick verification
    with open(os.path.join(OUTPUT_DIR, "data_h2o.pkl"), "rb") as f:
        d = pickle.load(f)
    flat = (d[:, 0].numpy() < 0.01).sum()
    print(f"\n  H2O baseline (<0.01): {flat}/{len(d)} ({flat/len(d)*100:.1f}%)")

    print("\nDone! Use 'data.pkl' for temperature, 'data_h2o.pkl' for H2O mass fraction.")


if __name__ == "__main__":
    main()
