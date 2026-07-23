#!/usr/bin/env python
"""
Dual CSTR with recycle — physical delayed feedback via a second reactor.

Reactor A: main CSTR where combustion occurs
Reactor B: "delay tank" — receives a side stream from A, recycles back to A

The residence time in Reactor B creates a physical delay (analogous to MG's τ).
By varying the recycle ratio and B's volume, we control the "effective τ".

  Upstream ──→ [Reactor A] ──→ downstream
                  ↑    │
                  │    └──→ [Reactor B] ──→ downstream
                  └───────────┘ (recycle)

Usage:
  uv run python cstr/generate_dual_cstr.py --sweep
  uv run python cstr/generate_dual_cstr.py --volume_B 5e-6 --recycle 0.3
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
    print("Error: cantera is required.")
    sys.exit(1)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
_H2O2_PATH = os.path.join(os.path.dirname(ct.__file__), "data", "h2o2.yaml")


def simulate_dual_cstr(
    temperature=770.0,
    pressure=60.0 * 133.3,
    composition="H2:2, O2:1",
    volume_A=10.0e-6,
    volume_B=5.0e-6,
    mass_flow_sccm=12.0,
    recycle_fraction=0.3,
    wall_area=1.0,
    heat_transfer_coeff=0.02,
    valve_coeff_A=1.0e-9,
    valve_coeff_B=1.0e-9,
    t_end=600.0,
    dt=0.1,
    random_seed=42,
):
    """
    Dual CSTR with recycle from B to A.

    Args:
        volume_B: volume of delay reactor B (m³). Larger = longer delay.
        recycle_fraction: fraction of A's outflow diverted to B (0 to 1).
    """
    gas = ct.Solution(_H2O2_PATH)
    gas.TPX = temperature, pressure, composition

    # ---- Reactor A (main) ----
    cstr_A = ct.IdealGasReactor(gas, clone=True)
    cstr_A.volume = volume_A

    # ---- Reactor B (delay tank, same initial state) ----
    cstr_B = ct.IdealGasReactor(gas, clone=True)
    cstr_B.volume = volume_B

    # ---- Upstream reservoir ----
    upstream = ct.Reservoir(gas, clone=True)

    # ---- Environment (heat loss for A) ----
    env_A = ct.Reservoir(gas, clone=True)
    ct.Wall(cstr_A, env_A, A=wall_area, U=heat_transfer_coeff)

    # ---- Environment (heat loss for B, same U) ----
    env_B = ct.Reservoir(gas, clone=True)
    ct.Wall(cstr_B, env_B, A=wall_area, U=heat_transfer_coeff)

    # ---- Downstream ----
    downstream = ct.Reservoir(gas, clone=True)

    # ---- Flow network ----
    # Fresh feed → A
    sccm = mass_flow_sccm
    vdot = sccm * 1.0e-6 / 60.0 * ((ct.one_atm / gas.P) * (gas.T / 273.15))
    mdot_fresh = gas.density * vdot
    mfc_feed = ct.MassFlowController(upstream, cstr_A, mdot=mdot_fresh)

    # A → downstream (exhaust valve)
    ct.Valve(cstr_A, downstream, K=valve_coeff_A)

    # A → B (recycle side stream, MassFlowController for controlled split)
    mdot_AB = mdot_fresh * recycle_fraction
    mfc_AB = ct.MassFlowController(cstr_A, cstr_B, mdot=mdot_AB)

    # B → A (recycle return — same flow rate to maintain mass balance in B)
    mfc_BA = ct.MassFlowController(cstr_B, cstr_A, mdot=mdot_AB)

    # B → downstream (excess from B)
    ct.Valve(cstr_B, downstream, K=valve_coeff_B)

    # ---- Reactor network ----
    network = ct.ReactorNet([cstr_A, cstr_B])

    species_names = gas.species_names
    try:
        h2o_idx = species_names.index("H2O")
    except ValueError:
        h2o_idx = species_names.index("H2O(L)") if "H2O(L)" in species_names else 0

    # Track reactor A state
    states_A = ct.SolutionArray(gas, extra=["t"])
    t = 0.0

    residence_time = volume_B / (mdot_AB / gas.density) if mdot_AB > 0 else 0
    print(f"Simulating Dual CSTR:")
    print(f"  Volume A = {volume_A:.1e} m³, Volume B = {volume_B:.1e} m³")
    print(f"  Recycle fraction = {recycle_fraction}")
    print(f"  B residence time ≈ {residence_time:.3f}s")
    print(f"  Natural period ≈ 7.15s")

    while t < t_end:
        t += dt
        network.advance(t)
        states_A.append(cstr_A.phase.state, t=t)

    time_array = states_A.t
    temp_array = states_A.T
    Y_array = states_A.Y
    h2o_array = Y_array[:, h2o_idx]

    print(f"Generated {len(temp_array)} time points")
    print(f"Temperature:  {temp_array.min():.1f} – {temp_array.max():.1f} K")
    print(f"H2O mass frac: {h2o_array.min():.4f} – {h2o_array.max():.4f}")

    ac = np.correlate(h2o_array - h2o_array.mean(),
                      h2o_array - h2o_array.mean(), mode="full")
    ac = ac[len(ac)//2:] / (ac[len(ac)//2] + 1e-10)
    periodicity = float(ac[20:400].max()) if len(ac) > 20 else 1.0
    print(f"Periodicity score: {periodicity:.4f}  (single CSTR: ~0.952)")

    return time_array, temp_array, h2o_array


def save_pkl(series, output_path, label=""):
    col = torch.tensor(series, dtype=torch.float64).unsqueeze(1)
    tensor = torch.cat((col, col.clone()), dim=1)
    with open(output_path, "wb") as f:
        pickle.dump(tensor, f)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Dual CSTR with recycle delay")
    parser.add_argument("--volume_B", type=float, default=5.0e-6,
                        help="Volume of delay reactor B in m³ (default: 5e-6)")
    parser.add_argument("--recycle", type=float, default=0.3,
                        help="Recycle fraction (default: 0.3)")
    parser.add_argument("--t_end", type=float, default=600.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep over volume_B and recycle fraction")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.sweep:
        # Sweep over volume_B (delay) and recycle_fraction (coupling strength)
        volumes = [1.0e-6, 2.0e-6, 5.0e-6, 10.0e-6, 20.0e-6, 50.0e-6]
        recycles = [0.1, 0.2, 0.3, 0.5]

        print("=" * 75)
        print("  Dual CSTR Parameter Sweep")
        print("=" * 75)
        print(f"  {'V_B (m³)':>10s}  {'Recycle':>8s}  {'Periodicity':>12s}  "
              f"{'T_min':>8s}  {'T_max':>8s}  {'#Spikes':>8s}  {'Note'}")
        print(f"  {'─'*10}  {'─'*8}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*25}")

        results = []
        for vb in volumes:
            for rc in recycles:
                try:
                    _, temp, h2o = simulate_dual_cstr(
                        volume_B=vb, recycle_fraction=rc,
                        t_end=args.t_end, dt=args.dt, random_seed=args.seed,
                    )
                    ac = np.correlate(h2o - h2o.mean(), h2o - h2o.mean(), mode="full")
                    ac = ac[len(ac)//2:] / (ac[len(ac)//2] + 1e-10)
                    score = float(ac[20:400].max()) if len(ac) > 20 else 1.0
                    spikes = int((temp > 1000).sum())
                    t_min, t_max = temp.min(), temp.max()

                    note = ""
                    if score < 0.6:
                        note = "★ VERY LOW PERIODICITY"
                    elif score < 0.8:
                        note = "low periodicity"
                    elif score < 0.85:
                        note = "moderately low"

                    print(f"  {vb:10.1e}  {rc:8.2f}  {score:12.4f}  "
                          f"{t_min:8.1f}  {t_max:8.1f}  {spikes:8d}  {note}")

                    results.append({
                        "volume_B": vb, "recycle": rc, "periodicity": score,
                        "t_min": t_min, "t_max": t_max, "spikes": spikes,
                    })

                    # Auto-save promising candidates
                    if score < 0.85:
                        suffix = f"_Vb{vb:.1e}_rc{rc}"
                        save_pkl(h2o,
                                 os.path.join(OUTPUT_DIR, f"data_dual_h2o{suffix}.pkl"),
                                 f"H2O (dual CSTR, Vb={vb:.1e}, rc={rc})")
                except Exception as e:
                    print(f"  {vb:10.1e}  {rc:8.2f}  {'─':>12s}  FAILED: {str(e)[:60]}")

        print()
        print("Best candidates (lowest periodicity):")
        results.sort(key=lambda r: r["periodicity"])
        for r in results[:8]:
            print(f"  V_B={r['volume_B']:.1e}  recycle={r['recycle']}  "
                  f"periodicity={r['periodicity']:.4f}  spikes={r['spikes']}")

    else:
        print("=" * 55)
        print("  Dual CSTR with Recycle Delay")
        print("=" * 55)
        print()
        time_arr, temp_arr, h2o_arr = simulate_dual_cstr(
            volume_B=args.volume_B, recycle_fraction=args.recycle,
            t_end=args.t_end, dt=args.dt, random_seed=args.seed,
        )
        suffix = f"_Vb{args.volume_B:.1e}_rc{args.recycle}"
        save_pkl(h2o_arr,
                 os.path.join(OUTPUT_DIR, f"data_dual_h2o{suffix}.pkl"),
                 "H2O (dual CSTR)")


if __name__ == "__main__":
    main()
