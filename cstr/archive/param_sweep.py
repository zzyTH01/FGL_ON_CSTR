#!/usr/bin/env python
"""
CSTR parameter sweep — search for non-periodic (chaotic / quasi-periodic) regimes.

Sweeps over heat transfer coefficient U and valve coefficient K,
analyzes autocorrelation strength as a proxy for "periodicity".

A truly periodic signal has autocorrelation peaks near 1.0 at multiples
of the period. Chaotic signals have rapidly decaying autocorrelation.

Usage:
  uv run python cstr/param_sweep.py
"""

import os
import sys
import pickle
import numpy as np

try:
    import cantera as ct
except ImportError:
    print("Error: cantera is required.")
    sys.exit(1)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
_H2O2_PATH = os.path.join(os.path.dirname(ct.__file__), "data", "h2o2.yaml")


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
):
    """Simulate CSTR. Returns (time, temperature, h2o_mass_frac)."""
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
    states = ct.SolutionArray(gas, extra=["t"])
    t = 0.0

    while t < t_end:
        t += dt
        network.advance(t)
        states.append(cstr.phase.state, t=t)

    temp = states.T
    Y = states.Y
    species_names = gas.species_names
    try:
        h2o_idx = species_names.index("H2O")
    except ValueError:
        h2o_idx = species_names.index("H2O(L)") if "H2O(L)" in species_names else 0
    h2o = Y[:, h2o_idx]

    return states.t, temp, h2o


def periodicity_score(series):
    """
    Measure how periodic a series is.
    Returns the maximum autocorrelation peak (excluding lag 0).
    Closer to 1.0 = strongly periodic. Lower = less periodic / chaotic.
    """
    series = series - series.mean()
    ac = np.correlate(series, series, mode="full")
    ac = ac[len(ac) // 2:] / (ac[len(ac) // 2] + 1e-10)
    # Find max peak after lag 10 (skip lag 0 and immediate neighbors)
    if len(ac) > 20:
        return float(ac[20:200].max())
    return 1.0


def main():
    print("=" * 70)
    print("  CSTR Parameter Sweep — Searching for non-periodic regimes")
    print("=" * 70)
    print()

    # ---- Parameter grids ----
    # Primary: U (heat transfer coefficient)
    # Secondary: K (valve coefficient)
    # Tertiary: mass_flow_sccm

    U_values = [0.010, 0.015, 0.018, 0.020, 0.022, 0.025, 0.030, 0.040, 0.050]
    K_values = [5.0e-10, 1.0e-9, 2.0e-9, 5.0e-9]
    flow_values = [6.0, 12.0, 18.0]

    results = []

    # ---- Sweep over U (primary) ----
    print("--- Sweeping U (heat transfer coefficient) ---")
    print(f"{'U':>8s}  {'K':>10s}  {'Flow':>6s}  {'Periodicity':>12s}  "
          f"{'T_min':>8s}  {'T_max':>8s}  {'#Spikes':>8s}  {'Note'}")
    print("-" * 80)

    for U in U_values:
        try:
            t_arr, temp, h2o = simulate_cstr(
                heat_transfer_coeff=U,
                valve_coeff=1.0e-9,
                mass_flow_sccm=12.0,
                t_end=300.0,
            )
            score = periodicity_score(h2o)
            spikes = int((temp > 1000).sum())
            t_min, t_max = temp.min(), temp.max()
            note = ""
            if spikes == 0 and t_max < 800:
                note = "NO OSCILLATION (too cold)"
            elif spikes > 50:
                note = "TOO MANY SPIKES"
            elif score < 0.5:
                note = "★ LOW PERIODICITY — candidate!"
            elif score < 0.8:
                note = "quasi-periodic"

            print(f"{U:8.4f}  {1.0e-9:10.1e}  {12.0:6.1f}  "
                  f"{score:12.4f}  {t_min:8.1f}  {t_max:8.1f}  "
                  f"{spikes:8d}  {note}")

            results.append({
                "U": U, "K": 1.0e-9, "flow": 12.0,
                "periodicity": score, "spikes": spikes,
                "t_min": t_min, "t_max": t_max, "n_points": len(temp),
            })
        except Exception as e:
            print(f"{U:8.4f}  {'—':>10s}  {'—':>6s}  SIMULATION FAILED: {e}")

    print()

    # ---- Sweep over K at best U candidates ----
    print("--- Sweeping K at U=0.018 and U=0.022 ---")
    for U in [0.018, 0.022]:
        for K in K_values:
            try:
                t_arr, temp, h2o = simulate_cstr(
                    heat_transfer_coeff=U,
                    valve_coeff=K,
                    mass_flow_sccm=12.0,
                    t_end=300.0,
                )
                score = periodicity_score(h2o)
                spikes = int((temp > 1000).sum())
                note = ""
                if score < 0.5:
                    note = "★ LOW PERIODICITY"
                elif score < 0.8:
                    note = "quasi-periodic"

                print(f"{U:8.4f}  {K:10.1e}  {12.0:6.1f}  "
                      f"{score:12.4f}  {temp.min():8.1f}  {temp.max():8.1f}  "
                      f"{spikes:8d}  {note}")

                results.append({
                    "U": U, "K": K, "flow": 12.0,
                    "periodicity": score, "spikes": spikes,
                    "t_min": temp.min(), "t_max": temp.max(),
                    "n_points": len(temp),
                })
            except Exception as e:
                print(f"{U:8.4f}  {K:10.1e}  {'—':>6s}  FAILED: {e}")

    print()

    # ---- Sweep over flow rate ----
    print("--- Sweeping flow rate at U=0.018 ---")
    for flow in flow_values:
        try:
            t_arr, temp, h2o = simulate_cstr(
                heat_transfer_coeff=0.018,
                valve_coeff=1.0e-9,
                mass_flow_sccm=flow,
                t_end=300.0,
            )
            score = periodicity_score(h2o)
            spikes = int((temp > 1000).sum())
            note = ""
            if score < 0.5:
                note = "★ LOW PERIODICITY"
            elif score < 0.8:
                note = "quasi-periodic"

            print(f"{0.018:8.4f}  {1.0e-9:10.1e}  {flow:6.1f}  "
                  f"{score:12.4f}  {temp.min():8.1f}  {temp.max():8.1f}  "
                  f"{spikes:8d}  {note}")

            results.append({
                "U": 0.018, "K": 1.0e-9, "flow": flow,
                "periodicity": score, "spikes": spikes,
                "t_min": temp.min(), "t_max": temp.max(),
                "n_points": len(temp),
            })
        except Exception as e:
            print(f"{0.018:8.4f}  {'—':>10s}  {flow:6.1f}  FAILED: {e}")

    # ---- Summary ----
    print()
    print("=" * 70)
    print("  SUMMARY — Top candidates (lowest periodicity)")
    print("=" * 70)
    results.sort(key=lambda r: r["periodicity"])
    for r in results[:10]:
        print(f"  U={r['U']:.4f}  K={r['K']:.1e}  flow={r['flow']:.1f}  "
              f"periodicity={r['periodicity']:.4f}  spikes={r['spikes']}  "
              f"T=[{r['t_min']:.0f}, {r['t_max']:.0f}]")

    # Save results
    path = os.path.join(OUTPUT_DIR, "param_sweep_results.txt")
    with open(path, "w") as f:
        for r in sorted(results, key=lambda x: x["periodicity"]):
            f.write(f"U={r['U']:.4f} K={r['K']:.1e} flow={r['flow']:.1f} "
                    f"periodicity={r['periodicity']:.4f} spikes={r['spikes']} "
                    f"T=[{r['t_min']:.0f},{r['t_max']:.0f}] n={r['n_points']}\n")
    print(f"\nFull results saved to {path}")


if __name__ == "__main__":
    main()
