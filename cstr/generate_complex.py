"""Complex multichannel CSTR generator and FGL-compatible channel exporter."""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

try:
    import cantera as ct
except ImportError:
    ct = None

from cstr.generate_delayed_stable import simulate_delayed_feedback_stable

_DEFAULT_OUTDIR = Path(__file__).resolve().parent / "data" / "complex"
_REQUIRED = ("temperature", "H2O")


def _require_cantera():
    if ct is None or not hasattr(ct, "__version__"):
        raise RuntimeError("Cantera is required (>= 3.2.0)")
    version = tuple(int(x) for x in ct.__version__.split(".")[:2])
    if version < (3, 2):
        raise RuntimeError(f"Cantera >= 3.2.0 required, found {ct.__version__}")


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
            tensor = torch.tensor(series, dtype=torch.float64).unsqueeze(1)
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


def _species_states(result, mechanism_path):
    import cantera as local_ct

    gas = local_ct.Solution(mechanism_path)
    gas.TPX = 770.0, 60.0 * 133.3, "H2:2, O2:1"
    names = gas.species_names
    return names


def generate(tau, outdir=_DEFAULT_OUTDIR, t_end=300.0, dt=0.1, k=6, overwrite=False):
    """Generate one multichannel NPZ and export compatible channel PKLs."""
    _require_cantera()
    outdir = Path(outdir)
    npz_path = outdir / f"data_complex_cstr_multichannel_tau{tau}.npz"
    if npz_path.exists() and not overwrite:
        return npz_path, export_single_channel_pkls(npz_path, outdir)
    result = simulate_complex_cstr(tau=tau, t_end=t_end, dt=dt)
    mechanism_path = os.path.join(os.path.dirname(ct.__file__), "data", "h2o2.yaml")
    species_names = _species_states(result, mechanism_path)
    mass_fractions = result.get("mass_fraction_matrix")
    if mass_fractions is None:
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


def main():
    parser = argparse.ArgumentParser(description="Generate complex multichannel CSTR")
    parser.add_argument("--tau", type=int, default=50)
    parser.add_argument("--t-end", type=float, default=300.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--outdir", type=str, default=str(_DEFAULT_OUTDIR))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    npz_path, pkls = generate(
        args.tau,
        outdir=args.outdir,
        t_end=args.t_end,
        dt=args.dt,
        k=args.k,
        overwrite=args.overwrite,
    )
    print(f"NPZ: {npz_path}")
    for path in pkls:
        print(f"PKL: {path}")


if __name__ == "__main__":
    main()
