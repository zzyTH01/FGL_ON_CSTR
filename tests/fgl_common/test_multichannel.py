import pickle
import torch

from fgl_common import multichannel
from fgl_common.multichannel import run_channel_grid


def test_channel_grid(tmp_path, monkeypatch):
    N = 240
    path = tmp_path / "data_complex_cstr_H2O_tau50.pkl"
    data = torch.stack([torch.linspace(0, 1, N), torch.sin(torch.linspace(0, 6, N))], dim=1)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    monkeypatch.setattr(
        multichannel,
        "run_fgl_experiment",
        lambda *a, **k: {"lookback": k.get("lookback_window"), "horizon": k.get("forecasting_horizon"), "teacher": 1.0, "baseline": 2.0, "student": 1.5, "improvement": 25.0},
    )
    rows = run_channel_grid(str(tmp_path), [(4, 2)], [0], str(tmp_path / "out.csv"), quick=True)
    assert rows[0]["channel"] == "H2O"
    assert rows[0]["fgl_delta"] == 25.0
