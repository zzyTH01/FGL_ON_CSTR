#!/usr/bin/env python
"""
Generate CSTR dataset with NONLINEAR DELAYED FEEDBACK.

This introduces a second feedback loop in direct analogy to Mackey-Glass τ.
The inlet mass flow rate is modulated by the H2O concentration from τ_delay
steps ago, creating a delayed negative feedback:

  mdot(t) = mdot_0 · (1 + A · (H2O[t-τ_delay] / max_H2O - 0.5))

When τ_delay is small → instantaneous feedback → period-1 (like MG τ<10)
When τ_delay matches natural oscillation → period-doubling → "sweet spot"
When τ_delay is large → complex/chaotic dynamics → chaos zone

This creates a one-parameter sweep (τ_delay) analogous to MG's τ sweep.

Usage:
  uv run python cstr/generate_delayed_feedback.py --tau 50 --amplitude 0.5
  uv run python cstr/generate_delayed_feedback.py --sweep  # sweep τ_delay values
"""

import os
import sys
import pickle
import argparse
from collections import deque
import numpy as np
import torch

try:
    import cantera as ct
except ImportError:
    print("Error: cantera is required.")
    sys.exit(1)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
_H2O2_PATH = os.path.join(os.path.dirname(ct.__file__), "data", "h2o2.yaml")
if not os.path.exists(_H2O2_PATH):
    print(f"Error: h2o2.yaml not found at {_H2O2_PATH}")
    sys.exit(1)


def simulate_delayed_feedback(
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
    feedback_amplitude=0.5,
    random_seed=42,
):
    """
    CSTR with nonlinear delayed feedback on inlet flow.

    Args:
        tau_delay: delay in STEPS (e.g. 50 steps = 5.0s)
        feedback_amplitude: strength of delayed feedback (A)
    """
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

    # Find H2O index
    species_names = gas.species_names
    try:
        h2o_idx = species_names.index("H2O")
    except ValueError:
        h2o_idx = species_names.index("H2O(L)") if "H2O(L)" in species_names else 0

    states = ct.SolutionArray(gas, extra=["t"])
    t = 0.0
    h2o_buffer = deque(maxlen=tau_delay)  # delay line

    print(f"Simulating CSTR with delayed feedback:")
    print(f"  τ_delay = {tau_delay} steps ({tau_delay*dt:.1f}s)")
    print(f"  A = {feedback_amplitude}")
    print(f"  Natural period ≈ 71.5 steps (7.15s)")

    while t < t_end:
        t += dt
        current_h2o = cstr.thermo.Y[h2o_idx]
        h2o_buffer.append(current_h2o)

        if len(h2o_buffer) == tau_delay:
            delayed_h2o = h2o_buffer[0]
            # Nonlinear feedback: modulate inlet based on delayed H2O
            mdot = mdot_base * (1.0 + feedback_amplitude * (delayed_h2o / 0.96 - 0.5))
        else:
            mdot = mdot_base

        mfc.mass_flow_rate = mdot
        network.advance(t)
        states.append(cstr.phase.state, t=t)

    time_array = states.t
    temp_array = states.T
    Y_array = states.Y
    h2o_array = Y_array[:, h2o_idx]

    print(f"Generated {len(temp_array)} time points")
    print(f"Temperature:  {temp_array.min():.1f} – {temp_array.max():.1f} K")
    print(f"H2O mass frac: {h2o_array.min():.4f} – {h2o_array.max():.4f}")

    # Periodicity analysis
    ac = np.correlate(h2o_array - h2o_array.mean(),
                      h2o_array - h2o_array.mean(), mode="full")
    ac = ac[len(ac)//2:] / (ac[len(ac)//2] + 1e-10)
    periodicity = float(ac[20:400].max()) if len(ac) > 20 else 1.0
    print(f"Periodicity score: {periodicity:.4f}  (unforced: ~0.952)")

    return time_array, temp_array, h2o_array, species_names


def save_pkl(series, output_path, label=""):
    col = torch.tensor(series, dtype=torch.float64).unsqueeze(1)
    tensor = torch.cat((col, col.clone()), dim=1)
    with open(output_path, "wb") as f:
        pickle.dump(tensor, f)
    print(f"  Saved: {output_path}  shape={tensor.shape}")


def main():
    parser = argparse.ArgumentParser(description="CSTR with delayed feedback")
    parser.add_argument("--tau", type=int, default=50,
                        help="Delay in steps (default: 50 = 5.0s)")
    parser.add_argument("--amplitude", type=float, default=0.5,
                        help="Feedback amplitude A")
    parser.add_argument("--t_end", type=float, default=600.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep over τ_delay values")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.sweep:
        # Sweep τ_delay from 5 to 150 steps
        tau_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150]
        print("=" * 70)
        print("  CSTR Delayed Feedback — τ_delay Sweep")
        print("=" * 70)
        print(f"  {'τ_delay':>8s}  {'Period(s)':>10s}  {'Periodicity':>12s}  "
              f"{'T_min':>8s}  {'T_max':>8s}  {'#Spikes':>8s}  {'Note'}")
        print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*20}")

        results = []
        for tau in tau_values:
            try:
                _, temp, h2o, _ = simulate_delayed_feedback(
                    tau_delay=tau,
                    feedback_amplitude=args.amplitude,
                    t_end=args.t_end,
                    dt=args.dt,
                    random_seed=args.seed,
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
                elif score < 0.9:
                    note = "moderate"

                print(f"  {tau:8d}  {tau*0.1:10.1f}  {score:12.4f}  "
                      f"{t_min:8.1f}  {t_max:8.1f}  {spikes:8d}  {note}")

                results.append({
                    "tau": tau, "periodicity": score,
                    "t_min": t_min, "t_max": t_max, "spikes": spikes,
                })

                # Auto-save candidates with periodicity < 0.85
                if score < 0.85:
                    suffix = f"_tau{tau}_A{args.amplitude}"
                    save_pkl(h2o,
                             os.path.join(OUTPUT_DIR, f"data_delayed_h2o{suffix}.pkl"),
                             f"H2O (τ_delay={tau})")
            except Exception as e:
                print(f"  {tau:8d}  {'─':>10s}  SIM FAILED: {e}")

        print()
        print("Best candidates (lowest periodicity):")
        results.sort(key=lambda r: r["periodicity"])
        for r in results[:5]:
            print(f"  τ_delay={r['tau']}  periodicity={r['periodicity']:.4f}  "
                  f"spikes={r['spikes']}")

    else:
        print("=" * 55)
        print("  CSTR with Nonlinear Delayed Feedback")
        print("=" * 55)
        print()
        time_arr, temp_arr, h2o_arr, species = simulate_delayed_feedback(
            tau_delay=args.tau,
            feedback_amplitude=args.amplitude,
            t_end=args.t_end,
            dt=args.dt,
            random_seed=args.seed,
        )
        print(f"\nSpecies: {species}\n")
        suffix = f"_tau{args.tau}_A{args.amplitude}"
        save_pkl(h2o_arr,
                 os.path.join(OUTPUT_DIR, f"data_delayed_h2o{suffix}.pkl"),
                 "H2O (delayed feedback)")


if __name__ == "__main__":
    main()
