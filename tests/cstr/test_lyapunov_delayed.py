import sys, pathlib
_CSTR = pathlib.Path(__file__).resolve().parents[2] / "cstr"
sys.path.insert(0, str(_CSTR))

import numpy as np
from lyapunov_delayed import largest_lyapunov_rosenstein


def test_constant_series_near_zero():
    lyap, ks, S = largest_lyapunov_rosenstein(np.ones(300), emb_dim=3, emb_lag=1)
    assert abs(lyap) < 1e-6


def test_lorenz_positive():
    # λ1 ≈ 0.9 / time-unit; at dt=0.02 that's ≈ 0.018 / sample. Rosenstein gives
    # the per-sample slope (k is in samples) — exactly what H3 needs (H in samples).
    from scipy.integrate import solve_ivp
    def lorenz(t, v, s=10.0, r=28.0, b=8.0/3.0):
        return [s*(v[1]-v[0]), v[0]*(r-v[2])-v[1], v[0]*v[1]-b*v[2]]
    sol = solve_ivp(lorenz, (0, 80), [1.0, 1.0, 1.0],
                    t_eval=np.arange(0, 80, 0.02), rtol=1e-9, atol=1e-9)
    x = sol.y[0, 500:]  # long enough + drop transient
    lyap, ks, S = largest_lyapunov_rosenstein(x, emb_dim=5, emb_lag=5)
    assert lyap > 0.01  # clearly positive chaotic divergence (per sample)
