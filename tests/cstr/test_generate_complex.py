import numpy as np
import pytest
import pickle
import torch

from cstr.generate_complex import export_single_channel_pkls, save_multichannel, select_channels


def test_select_channels_includes_temperature_and_h2o():
    names = ["O2", "H2", "H2O", "OH"]
    arrays = [np.zeros(10), np.ones(10) * 0.1, np.linspace(0, 2, 10), np.ones(10) * 0.01]
    selected = select_channels(names, arrays, np.linspace(300, 400, 10), k=3)
    assert "H2O" in selected and "temperature" in selected


def test_save_and_export(tmp_path):
    n = 20
    metadata = {"tau": 50, "mechanism": "h2o2.yaml"}
    save_multichannel(
        tmp_path / "complex.npz",
        np.arange(n),
        np.linspace(300, 400, n),
        ["H2O"],
        [np.linspace(0, 1, n)],
        metadata,
    )
    files = export_single_channel_pkls(tmp_path / "complex.npz", tmp_path)
    assert len(files) == 2
    with open(files[-1], "rb") as f:
        obj = pickle.load(f)
    assert obj.shape == (n, 2)


def test_generate_complex_import_error(monkeypatch):
    import cstr.generate_complex as g

    monkeypatch.setattr(g, "ct", None)
    with pytest.raises(RuntimeError, match="Cantera"):
        g.simulate_complex_cstr(tau=50, t_end=1.0, dt=0.1)
