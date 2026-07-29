# Stable Delayed-Feedback CSTR τ-Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the crashing `generate_delayed_feedback.py` with a numerically stable delayed-feedback CSTR generator (bounded + smooth-onset control law, same mdot topology) and sweep the delay τ to produce a periodicity-vs-τ transition curve.

**Architecture:** One new standalone script `cstr/generate_delayed_stable.py` with pure, unit-tested helper functions (`control_law`, `onset_factor`, `periodicity_score`) plus a Cantera `simulate_delayed_feedback_stable` integration and a `--sweep` CLI. Pure functions are TDD'd with pytest; the Cantera integration gets a short smoke test; the sweep orchestration is tested with an injected fake simulate (no Cantera) so the suite stays fast. The original `cstr/generate_delayed_feedback.py` is left untouched for traceability.

**Tech Stack:** Python 3.11, PyTorch 2.1.1, cantera 3.2.0, numpy, matplotlib, pytest 9.1.1, `uv` package manager. Run everything from the repo root via `uv run ...`.

## Global Constraints

(Spec: `docs/superpowers/specs/2026-07-29-cstr-delayed-feedback-stable-design.md`. Every task implicitly includes these.)

- cantera >= 3.2.0 (verified installed: 3.2.0); locate mechanism via `os.path.join(os.path.dirname(cantera.__file__), "data", "h2o2.yaml")`.
- Output `.pkl` MUST be a 2-column float64 torch tensor `torch.cat((col, col.clone()), dim=1)` saved with `pickle.dump` — identical to the other `cstr/generate*.py` so `run.py`'s `_load_data` can read it.
- `periodicity_score` MUST match the existing definition exactly: mean-removed `np.correlate(..., mode="full")`, take the right half, normalize by the zero-lag value `(ac[0] + 1e-10)`, take `max` over lag `[20, 400]`. This is the only way results stay comparable to `param_sweep_results.txt` / the other generators.
- `mdot` is bounded by construction via `tanh`: `|mdot/mdot₀ − 1| ≤ amplitude` for all inputs. The boundedness is a unit-test on `control_law`, not a runtime check.
- Do NOT modify `cstr/generate_delayed_feedback.py` (kept for traceability).
- Run/test from repo root: `uv run pytest tests/cstr/test_delayed_stable.py -v`, `uv run python cstr/generate_delayed_stable.py --sweep ...`.
- Keep the module import side-effect-free: compute `_H2O2_PATH` at top level but do NOT `sys.exit` there — check existence inside `simulate_*` / `main()` only, so `import generate_delayed_stable` is safe for tests.
- Tests live in `tests/cstr/test_delayed_stable.py` (new dir); match the plain-function pytest style of `tests/fgl_common/test_iterative_distillation.py`.

---

## File Structure

- **Create** `cstr/generate_delayed_stable.py` — single responsibility: stable delayed-feedback CSTR + τ-sweep. Contains, in order:
  - module imports + `_H2O2_PATH` (path only, no exit)
  - `control_law(delayed_h2o, mdot_base, sign, amplitude, center, width)` — pure
  - `onset_factor(t, t_onset)` — pure
  - `periodicity_score(series, lag_min=20, lag_max=400)` — pure, returns `(score, dom_lag)`
  - `simulate_delayed_feedback_stable(...)` — Cantera integration, returns dict
  - `save_pkl(series, output_path, label)` — pkl writer
  - `build_tau_grid(fine_around=None)` — pure
  - `run_sweep(simulate_fn, grid, burn_steps, dt)` — orchestration, injectable simulate_fn
  - `write_sweep_csv(rows, path)` — pure I/O helper
  - `plot_sweep(rows, path, dt)` — matplotlib
  - `main()` — argparse + wiring
- **Create** `tests/cstr/test_delayed_stable.py` — unit tests for all pure helpers, a Cantera smoke test, and a sweep-orchestration test with a fake simulate_fn.

Each task produces one self-contained, testable slice.

---

### Task 1: Control-law and onset helpers (pure, TDD)

**Files:**
- Create: `cstr/generate_delayed_stable.py`
- Create: `tests/cstr/test_delayed_stable.py`

**Interfaces:**
- Produces:
  - `control_law(delayed_h2o, mdot_base, sign=-1, amplitude=0.3, center=0.48, width=0.1) -> float`
    returns `mdot_base * (1.0 + sign * amplitude * np.tanh((delayed_h2o - center) / width))`.
  - `onset_factor(t, t_onset) -> float` returns `min(1.0, t / t_onset)` (1.0 if `t_onset <= 0`).

- [ ] **Step 1: Write the failing tests**

Create `tests/cstr/test_delayed_stable.py`:

```python
import sys, pathlib
_CSTR = pathlib.Path(__file__).resolve().parents[2] / "cstr"
sys.path.insert(0, str(_CSTR))

import numpy as np
from generate_delayed_stable import control_law, onset_factor


def test_control_law_zero_at_center():
    # delayed_h2o == center  =>  feedback term is 0  =>  mdot == mdot_base
    assert control_law(0.48, mdot_base=2.0) == 2.0


def test_control_law_bounded_for_extreme_inputs():
    # tanh saturates: |mdot/mdot_base - 1| <= amplitude for ANY delayed_h2o
    mdot_base = 3.0
    A = 0.3
    for x in [-1e6, -10.0, 0.0, 0.48, 1.0, 10.0, 1e6]:
        for s in (-1, 1):
            mdot = control_law(x, mdot_base, sign=s, amplitude=A)
            assert abs(mdot / mdot_base - 1.0) <= A + 1e-9, (x, s, mdot)


def test_control_law_sign_monotonicity():
    # at delayed_h2o above center, sign=+1 increases mdot, sign=-1 decreases it
    hi = control_law(0.9, 1.0, sign=+1)
    lo = control_law(0.9, 1.0, sign=-1)
    assert hi > 1.0 > lo


def test_onset_factor_ramp():
    assert onset_factor(0.0, 50.0) == 0.0
    assert onset_factor(25.0, 50.0) == 0.5
    assert onset_factor(50.0, 50.0) == 1.0
    assert onset_factor(999.0, 50.0) == 1.0  # capped


def test_onset_factor_zero_onset_is_full():
    assert onset_factor(0.001, 0.0) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/cstr/test_delayed_stable.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'generate_delayed_stable'`.

- [ ] **Step 3: Write minimal implementation**

Create `cstr/generate_delayed_stable.py`:

```python
#!/usr/bin/env python
"""
Stable delayed-feedback CSTR + tau-sweep.

Same physical topology as generate_delayed_feedback.py (single CSTR, H2O sensor,
delay line, inlet-mdot actuator) but with a BOUNDED, SMOOTH-ONSET control law so
the Cantera integrator no longer diverges at the ignition spike. See
docs/superpowers/specs/2026-07-29-cstr-delayed-feedback-stable-design.md.
"""
import os
import numpy as np

try:
    import cantera as ct
except ImportError:
    ct = None

_H2O2_PATH = (
    os.path.join(os.path.dirname(ct.__file__), "data", "h2o2.yaml")
    if ct is not None else ""
)


def control_law(delayed_h2o, mdot_base, sign=-1, amplitude=0.3, center=0.48, width=0.1):
    """Bounded feedback: mdot = mdot_base * (1 + sign*A*tanh((h2o-center)/width)).

    tanh keeps the feedback term in [-A, +A], so mdot is guaranteed in
    [mdot_base*(1-A), mdot_base*(1+A)] for any input.
    """
    return mdot_base * (1.0 + sign * amplitude * np.tanh((delayed_h2o - center) / width))


def onset_factor(t, t_onset):
    """Linear ramp of feedback strength from 0 to 1 over t_onset seconds."""
    if t_onset <= 0.0:
        return 1.0
    return float(min(1.0, t / t_onset))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/cstr/test_delayed_stable.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add cstr/generate_delayed_stable.py tests/cstr/test_delayed_stable.py
git commit -m "feat(cstr): bounded control_law + onset_factor for stable delayed feedback"
```

---

### Task 2: Periodicity metric (pure, TDD)

**Files:**
- Modify: `cstr/generate_delayed_stable.py` (add `periodicity_score`)
- Modify: `tests/cstr/test_delayed_stable.py` (add metric tests)

**Interfaces:**
- Produces: `periodicity_score(series, lag_min=20, lag_max=400) -> tuple[float, int]`
  returns `(score, dom_lag)` where `score` is normalized autocorrelation peak over
  `[lag_min, lag_max)`, and `dom_lag` is the lag (in samples) at that peak.
  `dom_lag * dt` is the dominant period in seconds.

- [ ] **Step 1: Write the failing tests**

Append to `tests/cstr/test_delayed_stable.py`:

```python
from generate_delayed_stable import periodicity_score


def test_periodicity_pure_sine_is_high_with_correct_period():
    # sine with period 50 samples -> score ~1, dom_lag == 50
    t = np.arange(4000)
    sig = np.sin(2 * np.pi * t / 50.0)
    score, dom = periodicity_score(sig)
    assert score > 0.99
    assert dom == 50


def test_periodicity_white_noise_is_low():
    rng = np.random.RandomState(0)
    sig = rng.randn(4000)
    score, _ = periodicity_score(sig)
    assert score < 0.3


def test_periodicity_detects_period_doubling():
    # period-50 + half-frequency (period-100) component -> dominant period 100
    t = np.arange(4000)
    sig = np.sin(2 * np.pi * t / 50.0) + 0.5 * np.sin(2 * np.pi * t / 100.0)
    _, dom = periodicity_score(sig)
    assert dom == 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/cstr/test_delayed_stable.py -v`
Expected: FAIL with `ImportError: cannot import name 'periodicity_score'`.

- [ ] **Step 3: Write minimal implementation**

Add to `cstr/generate_delayed_stable.py` (after `onset_factor`):

```python
def periodicity_score(series, lag_min=20, lag_max=400):
    """Normalized autocorrelation peak over lag [lag_min, lag_max).

    Matches the metric used by the other cstr/generate*.py scripts: mean-remove,
    full autocorrelation, right half, normalize by the zero-lag value, then take
    the max over the lag window. Returns (score, dom_lag).
    """
    s = np.asarray(series, dtype=float)
    s = s - s.mean()
    ac = np.correlate(s, s, mode="full")
    ac = ac[len(ac) // 2:]            # non-negative lags
    ac = ac / (ac[0] + 1e-10)         # normalize by zero-lag
    hi = min(lag_max, len(ac))
    if hi <= lag_min:
        return 1.0, lag_min
    window = ac[lag_min:hi]
    idx = int(np.argmax(window))
    return float(window[idx]), lag_min + idx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/cstr/test_delayed_stable.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add cstr/generate_delayed_stable.py tests/cstr/test_delayed_stable.py
git commit -m "feat(cstr): periodicity_score metric (matches existing generators)"
```

---

### Task 3: Stable Cantera integration + smoke test

**Files:**
- Modify: `cstr/generate_delayed_stable.py` (add `simulate_delayed_feedback_stable`, `save_pkl`)
- Modify: `tests/cstr/test_delayed_stable.py` (add smoke test)

**Interfaces:**
- Consumes: `control_law`, `onset_factor` from Task 1.
- Produces:
  - `simulate_delayed_feedback_stable(temperature=770.0, pressure=60.0*133.3, composition="H2:2, O2:1", reactor_volume=10.0e-6, mass_flow_sccm=12.0, wall_area=1.0, heat_transfer_coeff=0.02, valve_coeff=1.0e-9, t_end=600.0, dt=0.1, tau_delay=50, sign=-1, amplitude=0.3, center=0.48, width=0.1, t_onset=50.0, random_seed=42) -> dict`
    with keys `"t"`, `"T"`, `"h2o"` (1-D numpy arrays) and `"mdot_min"`, `"mdot_max"`, `"mdot_base"` (floats).
  - `save_pkl(series, output_path, label="")` — writes the 2-col float64 tensor pkl.

- [ ] **Step 1: Write the failing test**

Append to `tests/cstr/test_delayed_stable.py`:

```python
import pytest
from generate_delayed_stable import simulate_delayed_feedback_stable


@pytest.mark.slow
def test_simulate_short_run_is_stable_and_sane():
    # short run: must not crash, must stay bounded
    res = simulate_delayed_feedback_stable(
        t_end=30.0, dt=0.1, tau_delay=20,
        sign=-1, amplitude=0.3, t_onset=10.0,
    )
    assert res["h2o"].shape[0] > 0
    assert np.all(np.isfinite(res["T"]))
    assert np.all(np.isfinite(res["h2o"]))
    assert res["h2o"].min() >= 0.0 and res["h2o"].max() <= 1.0      # mass fraction
    assert res["T"].max() < 5000.0                                   # no thermal runaway
    # bounded mdot by construction: |mdot/mdot_base - 1| <= amplitude (=0.3 here)
    base = res["mdot_base"]
    assert res["mdot_max"] <= base * (1 + 0.3) + 1e-12
    assert res["mdot_min"] >= base * (1 - 0.3) - 1e-12
```

(The `@pytest.mark.slow` marker is informational; if pytest warns it is unregistered, the test still runs.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cstr/test_delayed_stable.py::test_simulate_short_run_is_stable_and_sane -v`
Expected: FAIL with `ImportError: cannot import name 'simulate_delayed_feedback_stable'`.

- [ ] **Step 3: Write minimal implementation**

Add to `cstr/generate_delayed_stable.py` (after `periodicity_score`). Note: top-level `_H2O2_PATH` is computed without `sys.exit`; the existence check lives here:

```python
from collections import deque
import pickle
import torch


def simulate_delayed_feedback_stable(
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
    sign=-1,
    amplitude=0.3,
    center=0.48,
    width=0.1,
    t_onset=50.0,
    random_seed=42,
):
    """Single CSTR with bounded delayed feedback on inlet mdot.

    Same topology as generate_delayed_feedback.simulate_delayed_feedback, but the
    feedback is bounded (tanh) and ramps in smoothly (onset_factor), so the
    integrator no longer diverges when the delay line first engages at the
    ignition spike.
    """
    if ct is None:
        raise RuntimeError("cantera is required")
    if not os.path.exists(_H2O2_PATH):
        raise FileNotFoundError(f"h2o2.yaml not found at {_H2O2_PATH}")

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

    species_names = gas.species_names
    try:
        h2o_idx = species_names.index("H2O")
    except ValueError:
        h2o_idx = species_names.index("H2O(L)") if "H2O(L)" in species_names else 0

    states = ct.SolutionArray(gas, extra=["t"])
    t = 0.0
    h2o_buffer = deque(maxlen=tau_delay) if tau_delay > 0 else None
    mdot_min, mdot_max = mdot_base, mdot_base

    while t < t_end:
        t += dt
        current_h2o = cstr.thermo.Y[h2o_idx]
        if h2o_buffer is not None:
            h2o_buffer.append(current_h2o)

        if h2o_buffer is not None and len(h2o_buffer) == tau_delay:
            delayed_h2o = h2o_buffer[0]
            a_eff = onset_factor(t, t_onset)
            mdot_full = control_law(delayed_h2o, mdot_base, sign, amplitude, center, width)
            mdot = mdot_base + a_eff * (mdot_full - mdot_base)   # onset interpolates
        else:
            mdot = mdot_base

        mdot_min = min(mdot_min, mdot)
        mdot_max = max(mdot_max, mdot)
        mfc.mass_flow_rate = mdot
        network.advance(t)
        states.append(cstr.phase.state, t=t)

    return {
        "t": states.t,
        "T": states.T,
        "h2o": states.Y[:, h2o_idx],
        "mdot_base": float(mdot_base),
        "mdot_min": float(mdot_min),
        "mdot_max": float(mdot_max),
    }


def save_pkl(series, output_path, label=""):
    """Save a 1-D series as a 2-column float64 tensor (same format as other generators)."""
    col = torch.tensor(np.asarray(series), dtype=torch.float64).unsqueeze(1)
    tensor = torch.cat((col, col.clone()), dim=1)
    with open(output_path, "wb") as f:
        pickle.dump(tensor, f)
    print(f"  {label}: {output_path}  shape={tuple(tensor.shape)}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cstr/test_delayed_stable.py::test_simulate_short_run_is_stable_and_sane -v`
Expected: PASS (one short Cantera run, a few seconds). Then run the whole file:
`uv run pytest tests/cstr/test_delayed_stable.py -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add cstr/generate_delayed_stable.py tests/cstr/test_delayed_stable.py
git commit -m "feat(cstr): stable delayed-feedback Cantera integration (bounded + onset)"
```

---

### Task 4: τ-sweep orchestration, CSV + plot outputs (TDD with fake simulate)

**Files:**
- Modify: `cstr/generate_delayed_stable.py` (add `build_tau_grid`, `run_sweep`, `write_sweep_csv`, `plot_sweep`, `main`)
- Modify: `tests/cstr/test_delayed_stable.py` (add grid/csv/sweep tests)

**Interfaces:**
- Consumes: `simulate_delayed_feedback_stable`, `periodicity_score`, `save_pkl`.
- Produces:
  - `build_tau_grid(fine_around=None) -> list[int]`
  - `run_sweep(simulate_fn, grid, burn_steps, dt) -> list[dict]` — each row has keys
    `tau, periodicity, dom_period, dom_period_s, T_min, T_max, status`.
    `simulate_fn` is `(tau) -> dict` with keys `"h2o"`, `"T"` (injectable for tests).
  - `write_sweep_csv(rows, path) -> None`
  - `plot_sweep(rows, path) -> None`
  - `main()` — argparse CLI.

- [ ] **Step 1: Write the failing tests**

Append to `tests/cstr/test_delayed_stable.py`:

```python
import csv
from generate_delayed_stable import build_tau_grid, run_sweep, write_sweep_csv


def test_build_tau_grid_default():
    assert build_tau_grid() == [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150]


def test_build_tau_grid_fine_around_merges():
    grid = build_tau_grid(fine_around=50)
    assert 35 in grid and 45 in grid and 55 in grid and 65 in grid
    assert grid == sorted(set(grid))


def test_run_sweep_with_fake_simulate():
    # fake simulate: returns a clean sine (period 50 samples) for every tau
    def fake(tau):
        t = np.arange(3000)
        return {"h2o": np.sin(2 * np.pi * t / 50.0), "T": np.full(3000, 800.0)}
    rows = run_sweep(fake, grid=[5, 50], burn_steps=100, dt=0.1)
    assert len(rows) == 2
    assert all(r["status"] == "ok" for r in rows)
    assert all(r["periodicity"] > 0.99 for r in rows)
    assert all(r["dom_period"] == 50 for r in rows)
    assert all(r["dom_period_s"] == 5.0 for r in rows)


def test_run_sweep_records_failure():
    def fake(tau):
        if tau == 50:
            raise RuntimeError("boom")
        t = np.arange(3000)
        return {"h2o": np.sin(2 * np.pi * t / 50.0), "T": np.full(3000, 800.0)}
    rows = run_sweep(fake, grid=[5, 50], burn_steps=100, dt=0.1)
    assert rows[0]["status"] == "ok"
    assert rows[1]["status"].startswith("fail")


def test_write_sweep_csv_roundtrip(tmp_path):
    rows = [{"tau": 5, "periodicity": 0.99, "dom_period": 50, "dom_period_s": 5.0,
             "T_min": 770.0, "T_max": 2900.0, "status": "ok"}]
    p = tmp_path / "out.csv"
    write_sweep_csv(rows, str(p))
    with open(p) as f:
        data = list(csv.DictReader(f))
    assert len(data) == 1
    assert float(data[0]["periodicity"]) == 0.99
    assert data[0]["status"] == "ok"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/cstr/test_delayed_stable.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_tau_grid'` (and the rest).

- [ ] **Step 3: Write minimal implementation**

Add to `cstr/generate_delayed_stable.py` (after `save_pkl`):

```python
def build_tau_grid(fine_around=None):
    grid = [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150]
    if fine_around is not None:
        lo = max(5, fine_around - 20)
        hi = min(150, fine_around + 20)
        grid = sorted(set(grid) | set(range(lo, hi + 1, 5)))
    return grid


def run_sweep(simulate_fn, grid, burn_steps, dt):
    """Run simulate_fn(tau) over the grid; compute periodicity on the post-burn segment.

    simulate_fn is injectable so tests can avoid Cantera. A failure on one tau is
    recorded, not raised.
    """
    rows = []
    for tau in grid:
        try:
            res = simulate_fn(tau)
            h2o = np.asarray(res["h2o"])
            temp = np.asarray(res["T"])
            seg = h2o[burn_steps:] if burn_steps < len(h2o) else h2o
            score, dom = periodicity_score(seg)
            rows.append({
                "tau": tau,
                "periodicity": score,
                "dom_period": dom,
                "dom_period_s": dom * dt,
                "T_min": float(np.min(temp)),
                "T_max": float(np.max(temp)),
                "status": "ok",
            })
        except Exception as e:
            rows.append({
                "tau": tau, "periodicity": float("nan"), "dom_period": -1,
                "dom_period_s": float("nan"),
                "T_min": float("nan"), "T_max": float("nan"),
                "status": f"fail:{type(e).__name__}:{str(e)[:60]}",
            })
    return rows


def write_sweep_csv(rows, path):
    cols = ["tau", "periodicity", "dom_period", "dom_period_s", "T_min", "T_max", "status"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"  wrote {path}  ({len(rows)} rows)")


def plot_sweep(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in rows if r["status"] == "ok"]
    taus = [r["tau"] for r in ok]
    pers = [r["periodicity"] for r in ok]
    doms = [r["dom_period_s"] for r in ok]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(taus, pers, "o-", color="C0", label="periodicity")
    ax1.set_xlabel("τ_delay (steps)")
    ax1.set_ylabel("periodicity score", color="C0")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(0.85, ls="--", color="grey", lw=0.8)
    ax2 = ax1.twinx()
    ax2.plot(taus, doms, "s--", color="C1", label="dominant period (s)")
    ax2.set_ylabel("dominant period (s)", color="C1")
    fig.suptitle("Delayed-feedback CSTR: periodicity vs τ")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  wrote {path}")
    plt.close(fig)
```

Then add the CLI at the bottom of the file:

```python
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stable delayed-feedback CSTR tau-sweep")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--tau", type=int, default=50)
    parser.add_argument("--sign", type=int, default=-1, choices=(-1, 1))
    parser.add_argument("--amplitude", type=float, default=0.3)
    parser.add_argument("--center", type=float, default=0.48)
    parser.add_argument("--width", type=float, default=0.1)
    parser.add_argument("--onset", type=float, default=50.0)
    parser.add_argument("--t_end", type=float, default=600.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--burn_in", type=float, default=100.0)
    parser.add_argument("--fine_around", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    out_default = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--outdir", type=str, default=out_default)
    args = parser.parse_args()

    results_dir = os.path.join(args.outdir, "results")
    data_dir = os.path.join(args.outdir, "data")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    def simulate_fn(tau):
        return simulate_delayed_feedback_stable(
            tau_delay=tau, sign=args.sign, amplitude=args.amplitude,
            center=args.center, width=args.width, t_onset=args.onset,
            t_end=args.t_end, dt=args.dt, random_seed=args.seed,
        )

    if args.sweep:
        grid = build_tau_grid(fine_around=args.fine_around)
        burn_steps = int(args.burn_in / args.dt)
        print(f"Sweeping tau in {grid} (sign={args.sign}, A={args.amplitude}, "
              f"onset={args.onset}s, burn={args.burn_in}s)")
        rows = run_sweep(simulate_fn, grid, burn_steps, args.dt)

        tag = f"s{args.sign if args.sign == 1 else -1}_A{args.amplitude}"
        csv_path = os.path.join(results_dir, f"delayed_tau_sweep_{tag}.csv")
        png_path = os.path.join(results_dir, f"delayed_tau_sweep_{tag}.png")
        write_sweep_csv(rows, csv_path)
        plot_sweep(rows, png_path)

        print("\n  tau  periodicity  dom_period(s)  status")
        for r in rows:
            print(f"  {r['tau']:4d}  {r['periodicity']:12.4f}  "
                  f"{r['dom_period_s']:12.2f}  {r['status']}")

        # auto-save low-periodicity (transition-band) datasets
        for r in rows:
            if r["status"] == "ok" and r["periodicity"] < 0.85:
                res = simulate_fn(r["tau"])
                fn = f"data_delayed_stable_h2o_tau{r['tau']}_{tag}.pkl"
                save_pkl(res["h2o"], os.path.join(data_dir, fn), f"H2O tau={r['tau']}")
    else:
        res = simulate_delayed_feedback_stable(
            tau_delay=args.tau, sign=args.sign, amplitude=args.amplitude,
            center=args.center, width=args.width, t_onset=args.onset,
            t_end=args.t_end, dt=args.dt, random_seed=args.seed,
        )
        score, dom = periodicity_score(res["h2o"][int(args.burn_in / args.dt):])
        print(f"tau={args.tau} periodicity={score:.4f} dom_period={dom*args.dt:.2f}s "
              f"T=[{res['T'].min():.0f},{res['T'].max():.0f}]")
        tag = f"tau{args.tau}_s{args.sign}_A{args.amplitude}"
        save_pkl(res["h2o"], os.path.join(data_dir, f"data_delayed_stable_h2o_{tag}.pkl"),
                 f"H2O tau={args.tau}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/cstr/test_delayed_stable.py -v`
Expected: PASS (all tests, including the new grid/sweep/csv ones — these use the fake simulate, no Cantera).

- [ ] **Step 5: Commit**

```bash
git add cstr/generate_delayed_stable.py tests/cstr/test_delayed_stable.py
git commit -m "feat(cstr): tau-sweep orchestration + CSV/plot outputs + CLI"
```

---

### Task 5: Run the coarse sweep, interpret, and follow up (fine pass / fallback sign)

**Files:**
- Read/run only: `cstr/generate_delayed_stable.py`
- Produces: `cstr/results/delayed_tau_sweep_s-1_A0.3.csv`, `.png`, and `cstr/results/delayed_tau_sweep_report.md`

This task is the scientific payoff — no new unit tests; verification is by inspection of the outputs.

- [ ] **Step 1: Run the coarse sweep with the default negative sign**

Run: `uv run python cstr/generate_delayed_stable.py --sweep`
Expected: completes without aborting (per-τ failures are caught and logged as `fail:...`).
Verify: `cstr/results/delayed_tau_sweep_s-1_A0.3.csv` and `.png` exist.

- [ ] **Step 2: Inspect the periodicity-vs-τ curve and classify**

Read the CSV. Classify the outcome:
- **Transition band found:** periodicity drops below ~0.85 for a contiguous mid-range of τ (and dom_period jumps ×2/×4) → the hypothesis is confirmed. Proceed to Step 3.
- **Phase-locked (all ~1.0):** negative feedback locked the oscillator → fallback. Proceed to Step 4.
- **All crashed:** integrator still unstable at this sign/gain → proceed to Step 4.

- [ ] **Step 3 (only if transition found): fine pass around the band**

Pick the center τ of the drop and re-run, e.g. if the drop centers near τ=50:
Run: `uv run python cstr/generate_delayed_stable.py --sweep --fine_around 50`
Verify: a finer CSV/PNG is produced and the transition is resolved more sharply.

- [ ] **Step 4 (fallback, only if locked/crashed): flip sign and/or raise gain**

If Step 2 was locked or crashed, try positive feedback first:
Run: `uv run python cstr/generate_delayed_stable.py --sweep --sign 1`
If still locked, raise amplitude: `--sign 1 --amplitude 0.5` (then `--sign -1 --amplitude 0.5`).
Verify each run produces a CSV; re-classify per Step 2.

- [ ] **Step 5: Write the interpretation report**

Create `cstr/results/delayed_tau_sweep_report.md` recording:
- which (sign, A) configuration was used;
- the periodicity-vs-τ table (paste from the winning CSV);
- the verdict (transition band present? at which τ? dom_period behavior?);
- any per-τ failures;
- one-line conclusion: does the CSTR DDE show the predicted periodic→non-periodic transition at intermediate τ, yes/no.

- [ ] **Step 6: Commit**

```bash
git add cstr/results/delayed_tau_sweep_*.csv cstr/results/delayed_tau_sweep_*.png \
        cstr/results/delayed_tau_sweep_report.md
git commit -m "exp(cstr): delayed-feedback tau-sweep results + interpretation"
```

(Also `git add` any auto-saved `cstr/data/data_delayed_stable_h2o_*.pkl` if produced.)

---

## Self-Review (run before handoff)

**1. Spec coverage:**
- §4 physical structure preserved → Task 3 uses identical reactor/valve/wall/sensor/mdot actuator; only `control_law` differs. ✓
- §5 bounded control law → Task 1 `control_law` (tanh), Task 3 applies it. ✓
- §6 onset ramp → Task 1 `onset_factor`, Task 3 applies it. ✓
- §7 τ sweep (coarse + fine, burn-in, deterministic, per-τ try/except) → Task 4 `build_tau_grid`/`run_sweep`/`main`, Task 5 runs it. ✓
- §8 metrics (periodicity + dom_period) → Task 2 `periodicity_score`. ✓
- §9 outputs (script, CSV, PNG, auto-save pkl) → Task 4 CLI + Task 5 execution. ✓
- §10 safety net (bounded mdot, per-τ try/except, fallback sign) → Task 3 + Task 4 + Task 5 Step 4. ✓
- §12 locked decisions (mdot actuator, sign=−1 default, curve-only scope) → reflected throughout. ✓

**2. Placeholder scan:** no TBD/TODO; every code step contains complete code. The Task 3 smoke test asserts real mdot boundedness against the returned `mdot_base`. ✓

**3. Type consistency:** `simulate_delayed_feedback_stable` returns a dict with keys `"t","T","h2o","mdot_min","mdot_max"`; `run_sweep`'s `simulate_fn` contract requires `"h2o"` and `"T"` — matches. `periodicity_score` returns `(score, dom_lag)` tuple — consumed consistently in Task 3-test, Task 4 `run_sweep` (`score, dom`), and `main`. `control_law` signature identical in Task 1 definition and Task 3 call. ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-29-cstr-delayed-feedback-stable.md`.
