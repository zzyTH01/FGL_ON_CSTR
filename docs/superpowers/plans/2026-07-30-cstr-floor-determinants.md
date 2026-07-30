# CSTR Floor Determinants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the experimental harness that tests what determines the minimum achievable student MSE on delayed-feedback CSTR (four hypotheses H1–H4), plus its analysis pipeline.

**Architecture:** Add one helper (`run_baseline_converged`) to `fgl_common`. Add three new scripts under `cstr/`: a Lyapunov estimator (Rosenstein, hand-rolled), a floor-sweep driver that records `{baseline, baseline_converged, teacher, fgl_student, A_iter, E_iter}` per `(dataset, L, H, seed)` into one CSV, and an analyzer that turns the CSV into four hypothesis verdicts + figures + a conclusion doc. Wire the sweep into `cstr/run.py` as an off-by-default experiment.

**Tech Stack:** Python 3.11, PyTorch 2.1.1 (MPS/CPU), numpy, scipy 1.11.3 (`scipy.signal.correlate`, `scipy.stats.linregress`, `scipy.stats.ttest_rel`, `scipy.integrate.solve_ivp`), matplotlib (Agg). No new dependencies.

## Global Constraints

- Package manager `uv`; run Python via `uv run python ...`, tests via `uv run pytest ...`.
- Device: CUDA > MPS > CPU (auto in `fgl_common.training`). Tests force CPU via `tests/conftest.py` (`FGL_DEVICE=cpu`).
- `cstr/` is NOT a Python package (no `__init__.py`). Scripts import siblings by inserting the cstr dir onto `sys.path`; tests do the same via `pathlib.Path(__file__).resolve().parents[2] / "cstr"`.
- Do NOT change existing `fgl_common` function signatures (backward compat). Only add new code.
- All MSE values are bin-index² (50-bin discretization) — consistent project-wide; relative/regression analysis only.
- Commit messages end with the trailer:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  ```

---

## Task 1: `run_baseline_converged` helper

A single-function addition to `fgl_common/training.py`: train only the baseline (H-step, no teacher) with more epochs + larger patience to find the converged "data floor". Needed by H4 (compare continuous-distillation floor vs matched-compute baseline floor).

**Files:**
- Modify: `fgl_common/training.py` (append new function after `run_fgl_experiment`, ~line 437)
- Modify: `fgl_common/__init__.py:23-29` (add to import + `__all__` at line 44)
- Test: `tests/fgl_common/test_run_baseline_converged.py`

**Interfaces:**
- Consumes: `RNN`, `compute_shared_bin_edges`, `create_time_series_dataset`, `EarlyStopper`, `evaluate_with_ph`, `device` (all already in `fgl_common.training`).
- Produces: `run_baseline_converged(data, lookback_window, forecasting_horizon, ...) -> {"lookback", "horizon", "epochs_run", "baseline_mse"}`. Later tasks call it as `from fgl_common import run_baseline_converged`.

- [ ] **Step 1: Write the failing test**

Create `tests/fgl_common/test_run_baseline_converged.py`:

```python
import math
from fgl_common import run_baseline_converged


def _series(n=600, seed=0):
    import numpy as np
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    s = np.sin(t * 0.2) * 50.0 + 100.0 + 1.0 * rng.randn(n)
    return [(float(s[i]), float(s[i])) for i in range(n)]


def test_returns_finite_baseline_mse():
    r = run_baseline_converged(_series(), lookback_window=20, forecasting_horizon=15,
                               num_bins=20, epochs=5, patience=3, seed=0, verbose=False)
    assert "baseline_mse" in r and "epochs_run" in r
    assert math.isfinite(r["baseline_mse"]) and r["baseline_mse"] >= 0
    assert r["epochs_run"] >= 1


def test_more_epochs_no_worse_than_one():
    # same seed => same init; 20 epochs should not be worse than 1 epoch
    data = _series()
    r_long = run_baseline_converged(data, 20, 15, num_bins=20, epochs=20,
                                    patience=5, seed=0, verbose=False)
    r_short = run_baseline_converged(data, 20, 15, num_bins=20, epochs=1,
                                     patience=5, seed=0, verbose=False)
    assert r_long["baseline_mse"] <= r_short["baseline_mse"] + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/fgl_common/test_run_baseline_converged.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_baseline_converged'`.

- [ ] **Step 3: Implement `run_baseline_converged`**

Append to `fgl_common/training.py` (after `run_fgl_experiment`'s return, before the `run_adaptive_weight` section):

```python
def run_baseline_converged(data, lookback_window, forecasting_horizon,
                           num_bins=50, val_size=0.2, test_size=0.2,
                           epochs=100, batch_size=64, patience=10, lr=1e-4,
                           hidden=128, num_layers=2, seed=42, verbose=True, label=""):
    """只训 baseline(H 步,无教师),用更多 epoch + 更大 patience 找"真·数据地板"。

    与 :func:`run_fgl_experiment` 的 baseline 段同口径(分类/bin-index MSE,offset=0,
    forecasting_horizon=H),但不训教师/学生。用于 H4:把连续蒸馏地板与"匹配算力的
    收敛 baseline 地板"对比,判断蒸馏能否压破数据地板。

    Returns:
        ``{"lookback", "horizon", "epochs_run", "baseline_mse"}``.
    """
    torch.manual_seed(seed)
    L, H = lookback_window, forecasting_horizon
    tag = f"[{label}] " if label else ""

    bin_edges, _, _ = compute_shared_bin_edges(data, L, num_bins)
    student_train, student_val, student_test, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H, num_bins=num_bins,
        val_size=val_size, test_size=test_size, offset=0, batch_size=batch_size,
        bin_edges=bin_edges)

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"{tag}baseline_converged  H={H:2d}  Epochs={epochs}  "
              f"Lookback={L}  patience={patience}")
        print(f"{'=' * 50}")

    ce = nn.CrossEntropyLoss()
    baseline = RNN(L, hidden, num_bins, num_layers).to(device)
    opt = optim.Adam(baseline.parameters(), lr=lr)
    stop = EarlyStopper(patience=patience)
    epochs_run = 0
    for epoch in range(epochs):
        epochs_run = epoch + 1
        baseline.train()
        for _, x, y in student_train:
            x = x.float().to(device).view(-1, 1, L)
            y = y.long().to(device)
            opt.zero_grad()
            ce(baseline(x), y).backward()
            opt.step()
        baseline.eval()
        with torch.no_grad():
            vl = sum(ce(baseline(x.float().to(device).view(-1, 1, L)),
                        y.long().to(device)).item()
                     for _, x, y in student_val) / len(student_val)
        if stop.step(vl, baseline):
            break
    stop.restore(baseline)

    baseline_mse = evaluate_with_ph(baseline, student_test, lookback_window=L)
    if verbose:
        print(f"  Baseline(converged): {baseline_mse:.4f}  (epochs_run={epochs_run})")
    return {"lookback": L, "horizon": H, "epochs_run": epochs_run,
            "baseline_mse": baseline_mse}
```

- [ ] **Step 4: Export from `fgl_common/__init__.py`**

In `fgl_common/__init__.py`, edit the `from .training import (...)` block (line 23-29) to add `run_baseline_converged` (keep alphabetical-ish, after `run_adaptive_weight`):

Change line 28 from:
```python
    run_fgl_experiment, run_iterative_distillation, run_adaptive_weight, run_adaptive_inference, run_seq2seq,
```
to:
```python
    run_fgl_experiment, run_iterative_distillation, run_adaptive_weight, run_adaptive_inference, run_seq2seq, run_baseline_converged,
```

And in the `__all__` list (line 44), change:
```python
    "run_fgl_experiment", "run_iterative_distillation", "run_adaptive_weight", "run_adaptive_inference", "run_seq2seq",
```
to:
```python
    "run_fgl_experiment", "run_iterative_distillation", "run_adaptive_weight", "run_adaptive_inference", "run_seq2seq", "run_baseline_converged",
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/fgl_common/test_run_baseline_converged.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add fgl_common/training.py fgl_common/__init__.py tests/fgl_common/test_run_baseline_converged.py
git commit -m "feat(fgl_common): run_baseline_converged — matched-compute data floor

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: Lyapunov estimator (`cstr/lyapunov_delayed.py`)

Hand-rolled Rosenstein largest-Lyapunov estimator (no `nolds` dep). Per-dataset batch over the delayed-feedback `.pkl` files. Resolves the project's "低周期性≠严格混沌,未算 Lyapunov" gap and supplies λ for H3.

**Files:**
- Create: `cstr/lyapunov_delayed.py`
- Test: `tests/cstr/test_lyapunov_delayed.py`

**Interfaces:**
- Produces: `largest_lyapunov_rosenstein(series, emb_dim=5, emb_lag=None, min_tsep=1, max_k=None, fit_range=None) -> (lyap, ks, S)`; `estimate_all(glob_pattern, out_csv) -> list[dict]`; `main()` (CLI).

- [ ] **Step 1: Write the failing test**

Create `tests/cstr/test_lyapunov_delayed.py`:

```python
import sys, pathlib
_CSTR = pathlib.Path(__file__).resolve().parents[2] / "cstr"
sys.path.insert(0, str(_CSTR))

import numpy as np
from lyapunov_delayed import largest_lyapunov_rosenstein


def test_constant_series_near_zero():
    lyap, ks, S = largest_lyapunov_rosenstein(np.ones(300), emb_dim=3, emb_lag=1)
    assert abs(lyap) < 1e-6


def test_lorenz_positive():
    from scipy.integrate import solve_ivp
    def lorenz(t, v, s=10.0, r=28.0, b=8.0/3.0):
        return [s*(v[1]-v[0]), v[0]*(r-v[2])-v[1], v[0]*v[1]-b*v[2]]
    sol = solve_ivp(lorenz, (0, 60), [1.0, 1.0, 1.0],
                    t_eval=np.arange(0, 60, 0.05), rtol=1e-9, atol=1e-9)
    x = sol.y[0, 1000:]  # drop transient
    lyap, ks, S = largest_lyapunov_rosenstein(x, emb_dim=5, emb_lag=10)
    assert lyap > 0.2  # true lambda1 ~ 0.9; Rosenstein underestimates but stays clearly positive
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cstr/test_lyapunov_delayed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lyapunov_delayed'`.

- [ ] **Step 3: Implement `cstr/lyapunov_delayed.py`**

```python
#!/usr/bin/env python
"""Rosenstein 最大 Lyapunov 指数估计(纯 numpy/scipy,无 nolds 依赖)。

对延迟反馈 CSTR 数据集批量估计 λ_max,补"低周期性≠严格混沌,未算 Lyapunov"缺口,
并为 floor_determinants 战役 H3(floor ~ teacher_mse * exp(λ·H))提供 λ。

用法::

    uv run python cstr/lyapunov_delayed.py
    uv run python cstr/lyapunov_delayed.py --glob 'cstr/data/data_delayed_stable_h2o_tau*_s1_A0.9_b0.03.pkl'
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
from scipy.signal import correlate


def _autocorr_zero_lag(series, max_lag=None):
    """首个自相关过零的 lag(嵌入延迟估计)。"""
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 2, 300)
    full = correlate(x, x, mode="full")[n - 1:]  # autocorr for lag>=0
    denom = full[0]
    if abs(denom) < 1e-12:
        return 1
    full = full / denom
    lag = 1
    while lag < max_lag and full[lag] > 0:
        lag += 1
    return max(lag, 1)


def largest_lyapunov_rosenstein(series, emb_dim=5, emb_lag=None,
                                min_tsep=1, max_k=None, fit_range=None):
    """Rosenstein 最大 Lyapunov 指数(每样本)。

    Args:
        series: 1D array.
        emb_dim: 嵌入维数。
        emb_lag: 嵌入延迟;None 则由自相关过零估计。
        min_tsep: 最近邻搜索时排除的时间邻近点数(避免时间相关)。
        max_k: 跟踪发散的最大步数;None 则 min(M//2, 100)。
        fit_range: (a, b) 线性拟合区(1-indexed 步数);None 则 (1, max_k//2)。
    Returns:
        ``(lyap, ks, S)``。lyap = <log d(k)> 线性区斜率;正=>混沌发散。
        退化输入(常数/太短)返回 ``(0.0, ks, S)``。
    """
    s = np.asarray(series, dtype=float)
    s = s - s.mean()
    n = len(s)
    if emb_lag is None:
        emb_lag = _autocorr_zero_lag(s)
    M = n - (emb_dim - 1) * emb_lag          # 嵌入向量数
    if M <= emb_dim + 2:
        return 0.0, np.array([]), np.array([])
    Y = np.empty((M, emb_dim))
    for j in range(emb_dim):
        Y[:, j] = s[j * emb_lag: j * emb_lag + M]
    if max_k is None:
        max_k = min(M // 2, 100)
    if max_k < 2:
        return 0.0, np.array([]), np.array([])

    # 每个点 i 找最近邻 j(排除时间邻近)
    upper = M - max_k
    pairs = []
    for i in range(upper):
        diff = Y[:upper] - Y[i]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        dist2[i] = np.inf
        lo = max(0, i - min_tsep); hi = min(upper, i + min_tsep + 1)
        dist2[lo:hi] = np.inf
        j = int(np.argmin(dist2))
        d0 = dist2[j]
        if np.isfinite(d0) and d0 > 0:
            pairs.append((i, j))

    if not pairs:
        return 0.0, np.arange(1, max_k + 1), np.full(max_k, np.nan)

    S = np.zeros(max_k)
    counts = np.zeros(max_k)
    for (i, j) in pairs:
        for k in range(1, max_k + 1):
            if i + k < M and j + k < M:
                dk = np.linalg.norm(Y[i + k] - Y[j + k])
                if dk > 0:
                    S[k - 1] += np.log(dk)
                    counts[k - 1] += 1
    valid = counts > 0
    S[valid] /= counts[valid]
    S[~valid] = np.nan
    ks = np.arange(1, max_k + 1)

    if fit_range is None:
        fit_range = (1, max(2, max_k // 2))
    a, b = fit_range
    seg_k = ks[a - 1:b]
    seg_S = S[a - 1:b]
    mask = np.isfinite(seg_S)
    if mask.sum() < 2:
        return 0.0, ks, S
    lyap = float(np.polyfit(seg_k[mask], seg_S[mask], 1)[0])
    return lyap, ks, S


def _tau_from_name(fn):
    m = re.search(r"tau(\d+)_", os.path.basename(fn))
    return int(m.group(1)) if m else ""


def estimate_all(glob_pattern, out_csv, burn_frac=0.2):
    """对 glob 匹配的所有延迟数据集估计 λ,写 CSV。

    Returns 列表 of ``{"file", "tau", "lyap", "N"}``。
    """
    import pickle
    rows = []
    for fn in sorted(glob.glob(glob_pattern)):
        with open(fn, "rb") as f:
            d = pickle.load(f)
        series = np.asarray(d[:, 0], dtype=float)
        burn = min(1000, max(50, int(len(series) * burn_frac)))
        lyap, _, _ = largest_lyapunov_rosenstein(series[burn:])
        rows.append({"file": os.path.basename(fn), "tau": _tau_from_name(fn),
                     "lyap": lyap, "N": len(series)})
        print(f"  {os.path.basename(fn)}  tau={_tau_from_name(fn)}  "
              f"λ={lyap:+.4f}  N={len(series)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "tau", "lyap", "N"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}  ({len(rows)} rows)")
    return rows


def main():
    p = argparse.ArgumentParser(description="Rosenstein λ for delayed-feedback CSTR datasets")
    p.add_argument("--glob", default="cstr/data/data_delayed_stable_h2o_tau*_s1_A0.9_b0.03.pkl")
    p.add_argument("--out", default="cstr/results/lyapunov_tau.csv")
    args = p.parse_args()
    estimate_all(args.glob, args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/cstr/test_lyapunov_delayed.py -v`
Expected: PASS (constant→≈0, Lorenz→>0.2).

- [ ] **Step 5: Commit**

```bash
git add cstr/lyapunov_delayed.py tests/cstr/test_lyapunov_delayed.py
git commit -m "feat(cstr): Rosenstein largest-Lyapunov estimator + τ-sweep batch

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: Floor-sweep driver (`cstr/run_floor_sweep.py`)

The orchestrator. For each `(dataset, L, H, seed)` call `run_fgl_experiment` + `run_iterative_distillation` + `run_baseline_converged`, write one row to `floor_sweep.csv`. CLI exposes the τ=100 deep grid + anchors.

**Files:**
- Create: `cstr/run_floor_sweep.py`
- Test: `tests/cstr/test_run_floor_sweep.py`

**Interfaces:**
- Consumes (from Task 1 + existing fgl_common): `run_fgl_experiment` (returns `teacher/baseline/student`), `run_iterative_distillation` (returns `A_iter/E_iter` each with `student_mse`), `run_baseline_converged` (returns `baseline_mse`); `periodicity_score` from `generate_delayed_stable`.
- Produces: `run(entries, cells_by_dataset, seeds, *, ..., outdir) -> list[dict]`; `default_tau100_grid()`, `default_anchor_cells()`, `load_entries(names)`, `COLUMNS`; `main()` CLI.
- CSV `COLUMNS` (exact order): `["dataset","tau","periodicity","L","H","LplusH_minus_1","seed","baseline_mse","baseline_converged_mse","teacher_mse","fgl_student_mse","A_iter_mse","E_iter_mse"]`.

- [ ] **Step 1: Write the failing test**

Create `tests/cstr/test_run_floor_sweep.py`:

```python
import sys, pathlib, math, csv as _csv
_CSTR = pathlib.Path(__file__).resolve().parents[2] / "cstr"
sys.path.insert(0, str(_CSTR))

import numpy as np
import run_floor_sweep
from run_floor_sweep import run, COLUMNS


def _series(n=400, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    s = np.sin(t * 0.2) * 50.0 + 100.0 + rng.randn(n)
    return [(float(s[i]), float(s[i])) for i in range(n)]


def test_default_grids():
    cells = run_floor_sweep.default_tau100_grid()
    # 9 surface cells + 4 extra H=15 cells = 13 unique
    assert (20, 15) in cells and (100, 30) in cells and (85, 15) in cells and (120, 15) in cells
    assert len(cells) == 13
    assert len(set(cells)) == 13  # no duplicates


def test_run_writes_csv_with_all_columns(tmp_path):
    data = _series()
    entry = {"label": "tiny", "data": data, "tau": 100, "periodicity": 0.5}
    rows = run(entries=[entry], cells_by_dataset={"tiny": [(20, 15)]},
               seeds=[0], alpha=0.5, temperature=4.0, bins=20, epochs=3,
               round_epochs=2, batch_size=32, patience=2, conv_epochs=3,
               conv_patience=2, K=2, outdir=str(tmp_path), verbose=False)
    csv_path = tmp_path / "floor_sweep.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        reader = _csv.DictReader(f)
        header = reader.fieldnames
        row = next(reader)
    for col in COLUMNS:
        assert col in header, f"missing column {col}"
    for vc in ("baseline_mse", "baseline_converged_mse", "teacher_mse",
               "fgl_student_mse", "A_iter_mse", "E_iter_mse"):
        v = float(row[vc])
        assert math.isfinite(v) and v >= 0, f"{vc}={v}"
    assert int(row["LplusH_minus_1"]) == 20 + 15 - 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cstr/test_run_floor_sweep.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_floor_sweep'`.

- [ ] **Step 3: Implement `cstr/run_floor_sweep.py`**

```python
#!/usr/bin/env python
"""地板成因战役主驱动:在延迟 CSTR 上对每个 (dataset, L, H, seed) 同时记录
{baseline, baseline_converged, teacher, fgl_student, A_iter, E_iter} MSE,
写入一张 floor_sweep.csv,供 analyze_floor.py 检验 H1-H4。

用法::

    uv run python cstr/run_floor_sweep.py --datasets tau100 --seeds 3 --K 5
    uv run python cstr/run_floor_sweep.py --anchors
"""
import argparse
import csv
import os
import pickle
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for generate_delayed_stable

from fgl_common import (  # noqa: E402
    run_fgl_experiment, run_iterative_distillation, run_baseline_converged,
)
from generate_delayed_stable import periodicity_score  # noqa: E402

CSTR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(CSTR, "data")
TAG = "s1_A0.9_b0.03"

COLUMNS = ["dataset", "tau", "periodicity", "L", "H", "LplusH_minus_1", "seed",
           "baseline_mse", "baseline_converged_mse", "teacher_mse",
           "fgl_student_mse", "A_iter_mse", "E_iter_mse"]

DATASET_REGISTRY = {
    "base":   ("data_h2o.pkl", None),
    "tau50":  (f"data_delayed_stable_h2o_tau50_{TAG}.pkl", 50),
    "tau100": (f"data_delayed_stable_h2o_tau100_{TAG}.pkl", 100),
    "tau150": (f"data_delayed_stable_h2o_tau150_{TAG}.pkl", 150),
}


def default_tau100_grid():
    """9 surface cells (L×H) + 4 extra H=15 cells = 13 unique cells straddling τ=100."""
    surface = [(L, H) for L in (20, 50, 100) for H in (5, 15, 30)]
    extra_h15 = [(L, 15) for L in (40, 70, 85, 120)]
    seen, cells = set(), []
    for c in surface + extra_h15:
        if c not in seen:
            seen.add(c)
            cells.append(c)
    return cells


def default_anchor_cells():
    return [(20, 15)]


def load_entries(names):
    """names: list of registry keys (e.g. ['tau100']). Returns list of entry dicts."""
    entries = []
    for name in names:
        fn, tau = DATASET_REGISTRY[name]
        path = os.path.join(DATA, fn)
        if not os.path.exists(path):
            print(f"  skip {name}: {fn} not found", flush=True)
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        series = np.asarray(data[:, 0], dtype=float)
        burn = min(1000, len(series) // 5)
        per, _ = periodicity_score(series[burn:])
        entries.append({"label": name, "data": data, "tau": tau if tau is not None else "",
                        "periodicity": per})
    return entries


def run(entries, cells_by_dataset, seeds, *, alpha=0.5, temperature=4.0, bins=50,
        epochs=30, round_epochs=15, batch_size=64, patience=5,
        conv_epochs=100, conv_patience=10, K=5, outdir="cstr/results", verbose=True):
    """Run the floor sweep. entries: list of {label,data,tau,periodicity}.
    cells_by_dataset: {label: [(L,H), ...]}. Returns list of row dicts.
    """
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for entry in entries:
        label = entry["label"]
        data = entry["data"]
        tau = entry["tau"]
        per = entry["periodicity"]
        cells = cells_by_dataset.get(label, [])
        if verbose:
            print(f"\n=== {label}  (tau={tau}, per={per:.3f}, {len(cells)} cells) ===",
                  flush=True)
        for (L, H) in cells:
            for s in seeds:
                r_fgl = run_fgl_experiment(
                    data, lookback_window=L, forecasting_horizon=H,
                    alpha=alpha, temperature=temperature, num_bins=bins, epochs=epochs,
                    batch_size=batch_size, patience=patience, seed=s, verbose=False,
                    label=f"{label}/L{L}_H{H}/s{s}")
                r_conv = run_baseline_converged(
                    data, lookback_window=L, forecasting_horizon=H, num_bins=bins,
                    epochs=conv_epochs, patience=conv_patience, batch_size=batch_size,
                    seed=s, verbose=False, label=f"{label}/L{L}_H{H}/s{s}")
                r_it = run_iterative_distillation(
                    data, L=L, H=H, alpha=alpha, temperature=temperature, num_bins=bins,
                    epochs=epochs, round_epochs=round_epochs, batch_size=batch_size,
                    patience=patience, K=K, seed=s, variant="E", verbose=False)
                row = {"dataset": label, "tau": tau, "periodicity": per,
                       "L": L, "H": H, "LplusH_minus_1": L + H - 1, "seed": s,
                       "baseline_mse": r_fgl["baseline"],
                       "baseline_converged_mse": r_conv["baseline_mse"],
                       "teacher_mse": r_fgl["teacher"],
                       "fgl_student_mse": r_fgl["student"],
                       "A_iter_mse": r_it["A_iter"]["student_mse"],
                       "E_iter_mse": r_it["E_iter"]["student_mse"]}
                rows.append(row)
                if verbose:
                    print(f"  L={L:3d} H={H:2d} s{s}: base={row['baseline_mse']:.1f} "
                          f"baseC={row['baseline_converged_mse']:.1f} "
                          f"tch={row['teacher_mse']:.1f} "
                          f"E_iter={row['E_iter_mse']:.1f}", flush=True)

    out = os.path.join(outdir, "floor_sweep.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if verbose:
        print(f"\nwrote {out}  ({len(rows)} rows)")
    return rows


def main():
    p = argparse.ArgumentParser(description="Floor-determinants sweep on delayed CSTR")
    p.add_argument("--datasets", type=str, default="tau100",
                   help="逗号分隔的 registry 键(默认 tau100)")
    p.add_argument("--anchors", action="store_true",
                   help="跑横向锚点 tau50/tau150 @ L20H15(忽略 --datasets)")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("-T", "--temperature", type=float, default=4.0, dest="temperature")
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--round_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--conv_epochs", type=int, default=100)
    p.add_argument("--conv_patience", type=int, default=10)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--outdir", type=str, default=os.path.join(CSTR, "results"))
    args = p.parse_args()

    seeds = list(range(args.seeds))
    if args.anchors:
        entries = load_entries(["tau50", "tau150"])
        cells_by_dataset = {e["label"]: default_anchor_cells() for e in entries}
    else:
        names = [n.strip() for n in args.datasets.split(",")]
        entries = load_entries(names)
        # tau100 用深挖网格;其余数据集默认只跑 L20H15 锚
        cells_by_dataset = {}
        for e in entries:
            cells_by_dataset[e["label"]] = (default_tau100_grid()
                                            if e["label"] == "tau100"
                                            else default_anchor_cells())

    run(entries, cells_by_dataset, seeds,
        alpha=args.alpha, temperature=args.temperature, bins=args.bins,
        epochs=args.epochs, round_epochs=args.round_epochs,
        batch_size=args.batch_size, patience=args.patience,
        conv_epochs=args.conv_epochs, conv_patience=args.conv_patience,
        K=args.K, outdir=args.outdir, verbose=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/cstr/test_run_floor_sweep.py -v`
Expected: PASS (grid has 13 unique cells; CSV has all 13 columns with finite non-negative values).

- [ ] **Step 5: Commit**

```bash
git add cstr/run_floor_sweep.py tests/cstr/test_run_floor_sweep.py
git commit -m "feat(cstr): floor-determinants sweep driver (records teacher/baseline/E_iter per cell)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: Analysis pipeline (`cstr/analyze_floor.py`)

Read `floor_sweep.csv` + `lyapunov_tau.csv`, produce the four hypothesis verdicts, figures, and `conclusion/floor_determinants.md`.

**Files:**
- Create: `cstr/analyze_floor.py`
- Test: `tests/cstr/test_analyze_floor.py`

**Interfaces:**
- Consumes: `cstr/results/floor_sweep.csv` (Task 3 columns), `cstr/results/lyapunov_tau.csv` (Task 2 columns `tau,lyap`).
- Produces: `analyze(csv_path, lyap_path, outdir, conclusion_dir, deep_label="tau100") -> stats_dict`; figures `floor_h1_*.png`, `floor_h2_*.png`, `floor_h3_*.png`, `floor_h4_*.png`; `conclusion/floor_determinants.md`.
- `stats_dict` keys (used by the test): `h1_transition_drop`, `h2_c_mean`, `h2_c_cv`, `h3_floor_vs_teacher_r2`, `h3_logfloor_vs_H_slope`, `h3_lambda`, `h4_paired_p`, `h4_mean_diff`.

- [ ] **Step 1: Write the failing test**

Create `tests/cstr/test_analyze_floor.py`:

```python
import sys, pathlib, csv as _csv
_CSTR = pathlib.Path(__file__).resolve().parents[2] / "cstr"
sys.path.insert(0, str(_CSTR))

from analyze_floor import analyze


def _write_floor_csv(path):
    rows = []
    for L in (20, 50, 100):
        for H in (5, 15, 30):
            for seed in (0, 1, 2):
                tm = 10.0 + L * 0.1 + H * 0.5  # teacher_mse
                rows.append({"dataset": "tau100", "tau": "100", "periodicity": 0.5,
                             "L": L, "H": H, "LplusH_minus_1": L + H - 1, "seed": seed,
                             "baseline_mse": tm * 3.0,
                             "baseline_converged_mse": tm * 2.0,
                             "teacher_mse": tm,
                             "fgl_student_mse": tm * 2.5,
                             "A_iter_mse": tm * 2.2,
                             "E_iter_mse": tm * 2.0})
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _write_lyap_csv(path):
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["file", "tau", "lyap", "N"])
        w.writeheader()
        w.writerow({"file": "tau100", "tau": "100", "lyap": "0.02", "N": "6000"})


def test_analyze_finds_known_relation(tmp_path):
    csv_path = tmp_path / "floor_sweep.csv"
    lyap_path = tmp_path / "lyapunov_tau.csv"
    _write_floor_csv(csv_path)
    _write_lyap_csv(lyap_path)

    stats = analyze(csv_path=str(csv_path), lyap_path=str(lyap_path),
                    outdir=str(tmp_path), conclusion_dir=str(tmp_path),
                    deep_label="tau100")
    # floor = 2 * teacher exactly => R^2 ~ 1
    assert stats["h3_floor_vs_teacher_r2"] > 0.99
    md = (tmp_path / "floor_determinants.md").read_text()
    for h in ("H1", "H2", "H3", "H4"):
        assert h in md
    # at least the H3 figure written
    assert (tmp_path / "floor_h3_floor_vs_teacher.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cstr/test_analyze_floor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analyze_floor'`.

- [ ] **Step 3: Implement `cstr/analyze_floor.py`**

```python
#!/usr/bin/env python
"""读 floor_sweep.csv + lyapunov_tau.csv,出 H1-H4 的判决、图、与 conclusion/floor_determinants.md。

H1: baseline 地板 vs (L,H),且固定 H 扫 L 时 L+H-1≈τ_data 处相变?
H2: c = E_iter/baseline 在 (L,H) 上近似常数?
H3: floor ∝ teacher_mse 且 log(floor) vs H 斜率≈λ?
H4: E_iter ≈ baseline_converged?(蒸馏是否压破数据地板)

用法::

    uv run python cstr/analyze_floor.py
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sstats


def _load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _to_float(rows, key):
    return np.array([float(r[key]) for r in rows], dtype=float)


def _agg(rows, group_keys, value_key):
    """mean over seeds for each group."""
    buckets = defaultdict(list)
    for r in rows:
        k = tuple(r[gk] for gk in group_keys)
        buckets[k].append(float(r[value_key]))
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def analyze(csv_path="cstr/results/floor_sweep.csv",
            lyap_path="cstr/results/lyapunov_tau.csv",
            outdir="cstr/results", conclusion_dir="conclusion",
            deep_label="tau100"):
    rows = _load_csv(csv_path)
    for r in rows:
        r["L"] = int(r["L"]); r["H"] = int(r["H"])
        r["LplusH_minus_1"] = int(r["LplusH_minus_1"])
    deep = [r for r in rows if r["dataset"] == deep_label]

    # ---- H1: baseline floor vs (L,H) + transition on H=15 column ----
    base_grid = _agg(deep, ["L", "H"], "baseline_mse")
    Ls = sorted({r["L"] for r in deep})
    Hs = sorted({r["H"] for r in deep})
    h1_heat = np.full((len(Ls), len(Hs)), np.nan)
    for i, L in enumerate(Ls):
        for j, H in enumerate(Hs):
            h1_heat[i, j] = base_grid.get((str(L), str(H)), np.nan)
    # transition along H=15 (or nearest available H)
    h15 = sorted([(r["LplusH_minus_1"], float(np.mean(
        [float(x["baseline_mse"]) for x in deep if x["L"] == L and x["H"] == 15])))
        for L in {r["L"] for r in deep if r["H"] == 15}])
    h1_transition_drop = 0.0
    h1_transition_at = None
    if len(h15) >= 2:
        xs = np.array([a for a, _ in h15]); ys = np.array([b for _, b in h15])
        ratios = ys[:-1] / np.maximum(ys[1:], 1e-9)
        idx = int(np.argmax(ratios))
        h1_transition_drop = float(ratios[idx])
        h1_transition_at = float(xs[idx + 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(h1_heat, aspect="auto", origin="lower", cmap="YlOrRd",
                   extent=[Hs[0] - 0.5, Hs[-1] + 0.5, Ls[0] - 0.5, Ls[-1] + 0.5])
    ax.set_xticks(Hs); ax.set_yticks(Ls); ax.set_xlabel("H"); ax.set_ylabel("L")
    ax.set_title(f"H1: baseline_mse floor ({deep_label})")
    plt.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "floor_h1_baseline_heatmap.png"), dpi=150)
    plt.close(fig)

    if h15:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([a for a, _ in h15], [b for _, b in h15], "o-")
        ax.axvline(100, color="r", ls="--", label="τ_data=100")
        ax.set_xlabel("L+H-1"); ax.set_ylabel("baseline_mse")
        ax.set_title("H1: baseline floor vs L+H-1 (H=15)"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "floor_h1_transition.png"), dpi=150)
        plt.close(fig)

    # ---- H2: c = E_iter / baseline, constant across (L,H)? ----
    cs = []
    for r in deep:
        b = float(r["baseline_mse"])
        if b > 1e-9:
            cs.append(float(r["E_iter_mse"]) / b)
    cs = np.array(cs)
    h2_c_mean = float(np.mean(cs)) if len(cs) else float("nan")
    h2_c_cv = float(np.std(cs, ddof=1) / np.abs(h2_c_mean)) if len(cs) > 1 else float("nan")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(cs); ax.set_ylabel("c = E_iter / baseline")
    ax.set_title(f"H2: c constancy (mean={h2_c_mean:.2f}, CV={h2_c_cv:.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "floor_h2_c_ratio.png"), dpi=150)
    plt.close(fig)

    # ---- H3: floor vs teacher_mse + log(floor) vs H ----
    teacher = _to_float(deep, "teacher_mse")
    floor_conv = _to_float(deep, "baseline_converged_mse")
    lr = sstats.linregress(teacher, floor_conv)
    h3_r2 = float(lr.rvalue ** 2)
    h3_slope = float(lr.slope)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(teacher, floor_conv)
    xs = np.linspace(teacher.min(), teacher.max(), 50)
    ax.plot(xs, lr.intercept + lr.slope * xs, "r-", label=f"R²={h3_r2:.2f}")
    ax.set_xlabel("teacher_mse"); ax.set_ylabel("floor (baseline_converged)")
    ax.set_title("H3: floor vs teacher_mse"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "floor_h3_floor_vs_teacher.png"), dpi=150)
    plt.close(fig)

    logfloor_vs_H = _agg(deep, ["H"], "baseline_converged_mse")
    Hs_h3 = sorted(float(k[0]) for k in logfloor_vs_H)
    floor_H = np.array([logfloor_vs_H[(str(int(h)),)] for h in Hs_h3])
    mask = floor_H > 0
    h3_logfloor_slope = float("nan")
    if mask.sum() >= 2:
        h3_logfloor_slope = float(sstats.linregress(Hs_h3[mask], np.log(floor_H[mask])).slope)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Hs_h3, np.log(np.maximum(floor_H, 1e-9)), "o-")
    ax.set_xlabel("H"); ax.set_ylabel("log(floor)"); ax.set_title("H3: log(floor) vs H")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "floor_h3_logfloor_vs_H.png"), dpi=150)
    plt.close(fig)

    h3_lambda = float("nan")
    if os.path.exists(lyap_path):
        for lr_row in _load_csv(lyap_path):
            if str(lr_row.get("tau")) == "100" or lr_row.get("file", "").find("tau100") >= 0:
                h3_lambda = float(lr_row["lyap"])
                break

    # ---- H4: E_iter vs baseline_converged (paired) ----
    e_iter = _to_float(deep, "E_iter_mse")
    base_c = _to_float(deep, "baseline_converged_mse")
    if len(e_iter) >= 2:
        t = sstats.ttest_rel(e_iter, base_c)
        h4_paired_p = float(t.pvalue)
    else:
        h4_paired_p = float("nan")
    h4_mean_diff = float(np.mean(e_iter - base_c)) if len(e_iter) else float("nan")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(base_c, e_iter)
    lo = min(np.nanmin(base_c), np.nanmin(e_iter))
    hi = max(np.nanmax(base_c), np.nanmax(e_iter))
    ax.plot([lo, hi], [lo, hi], "r--", label="y=x")
    ax.set_xlabel("baseline_converged_mse"); ax.set_ylabel("E_iter_mse")
    ax.set_title(f"H4: paired p={h4_paired_p:.3g}, mean(E_iter-baseC)={h4_mean_diff:+.1f}")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "floor_h4_paired.png"), dpi=150)
    plt.close(fig)

    # ---- verdicts ----
    h1_verdict = ("支持" if h1_transition_drop > 1.5 else "弱/无明显相变") + \
                 f"(最大连续降幅 {h1_transition_drop:.2f}× @ L+H-1≈{h1_transition_at})"
    h2_verdict = ("支持常数 c" if (np.isfinite(h2_c_cv) and h2_c_cv < 0.25) else "c 有漂移") + \
                 f"(c={h2_c_mean:.2f}±CV={h2_c_cv:.2f})"
    h3_verdict = ("支持 floor∝teacher" if h3_r2 > 0.7 else "弱关系") + \
                 f"(R²={h3_r2:.2f}, slope={h3_slope:.2f}, log(floor)/H={h3_logfloor_slope:.4f}, λ={h3_lambda:.4f})"
    if not np.isfinite(h4_paired_p):
        h4_verdict = "样本不足"
    elif h4_paired_p > 0.05:
        h4_verdict = f"支持:蒸馏≈数据地板(p={h4_paired_p:.3g}, 无法压破)"
    else:
        win = "压破" if h4_mean_diff < 0 else "反而更高"
        h4_verdict = f"证伪:蒸馏{win}了地板(p={h4_paired_p:.3g}, Δ={h4_mean_diff:+.1f})"

    os.makedirs(conclusion_dir, exist_ok=True)
    md_path = os.path.join(conclusion_dir, "floor_determinants.md")
    with open(md_path, "w") as f:
        f.write(f"# 延迟 CSTR 地板成因结论\n\n")
        f.write(f"**数据:** `{csv_path}` + `{lyap_path}`\n")
        f.write(f"**深挖数据集:** {deep_label}\n\n")
        f.write("## 四假设判决\n\n")
        f.write(f"- **H1**(L+H-1≥τ 相变): {h1_verdict}\n")
        f.write(f"- **H2**(c=E_iter/baseline 常数): {h2_verdict}\n")
        f.write(f"- **H3**(floor∝teacher_mse, Lyapunov 标定): {h3_verdict}\n")
        f.write(f"- **H4**(蒸馏 vs 匹配算力 baseline 地板): {h4_verdict}\n\n")
        f.write("## 如何达到最低 MSE\n\n")
        if np.isfinite(h4_paired_p) and h4_paired_p > 0.05:
            f.write("H4 成立 ⇒ **最好的 MSE 就是数据地板**:把 baseline 训到收敛即可达到,"
                    "连续蒸馏不额外加分。压低地板的关键在 (L, H) 与数据本身的可预测性。\n")
        else:
            f.write("H4 证伪 ⇒ 连续蒸馏确实改变了地板;达到最低 MSE 需用连续蒸馏(配合 H2/H3 的 (L,H,teacher) 规律选点)。\n")
    print(f"wrote {md_path}")

    return {"h1_transition_drop": h1_transition_drop,
            "h1_transition_at": h1_transition_at,
            "h2_c_mean": h2_c_mean, "h2_c_cv": h2_c_cv,
            "h3_floor_vs_teacher_r2": h3_r2, "h3_floor_vs_teacher_slope": h3_slope,
            "h3_logfloor_vs_H_slope": h3_logfloor_slope, "h3_lambda": h3_lambda,
            "h4_paired_p": h4_paired_p, "h4_mean_diff": h4_mean_diff}


def main():
    p = argparse.ArgumentParser(description="Analyze floor_sweep.csv → H1-H4 verdicts")
    p.add_argument("--csv", default="cstr/results/floor_sweep.csv")
    p.add_argument("--lyap", default="cstr/results/lyapunov_tau.csv")
    p.add_argument("--outdir", default="cstr/results")
    p.add_argument("--conclusion_dir", default="conclusion")
    p.add_argument("--deep_label", default="tau100")
    args = p.parse_args()
    analyze(args.csv, args.lyap, args.outdir, args.conclusion_dir, args.deep_label)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/cstr/test_analyze_floor.py -v`
Expected: PASS (synthetic floor=2·teacher → R²>0.99; md has H1–H4; figure written).

- [ ] **Step 5: Commit**

```bash
git add cstr/analyze_floor.py tests/cstr/test_analyze_floor.py
git commit -m "feat(cstr): floor analysis pipeline (H1-H4 verdicts + figures + conclusion)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 5: Wire into `cstr/run.py`

Register `floor_sweep` as an off-by-default experiment that launches the τ=100 deep sweep.

**Files:**
- Modify: `cstr/run.py` (add `run_floor_sweep_exp` wrapper near `run_lh_sweep_exp` ~line 188; add to `EXPERIMENTS` dict ~line 211; the `--list`/CLI already pick it up generically)
- Test: `tests/cstr/test_run_py_floor_sweep.py`

**Interfaces:**
- Consumes: `run_floor_sweep.run`, `run_floor_sweep.default_tau100_grid`, `run_floor_sweep.load_entries` (from Task 3).
- Produces: `EXPERIMENTS["floor_sweep"] = dict(fn=run_floor_sweep_exp, enabled=False, note=...)`.

- [ ] **Step 1: Write the failing test**

Create `tests/cstr/test_run_py_floor_sweep.py`:

```python
import os, subprocess, sys, pathlib

_REPO = pathlib.Path(__file__).resolve().parents[2]


def test_floor_sweep_listed_and_off():
    out = subprocess.check_output(
        [sys.executable, str(_REPO / "cstr" / "run.py"), "--list"], text=True)
    assert "floor_sweep" in out
    # the line for floor_sweep must show it as off (enabled=False)
    line = [ln for ln in out.splitlines() if ln.strip().startswith("floor_sweep")][0]
    assert "off" in line


def test_experiments_dict_registered():
    # import the module directly via file path (cstr is not a package)
    import importlib.util
    spec = importlib.util.spec_from_file_location("cstr_run_mod", _REPO / "cstr" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "floor_sweep" in mod.EXPERIMENTS
    assert mod.EXPERIMENTS["floor_sweep"]["enabled"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cstr/test_run_py_floor_sweep.py -v`
Expected: FAIL (`floor_sweep` not in `--list` output / not in EXPERIMENTS).

- [ ] **Step 3: Add the wrapper + EXPERIMENTS entry**

In `cstr/run.py`, add this wrapper function immediately after `run_lh_sweep_exp` (before the `EXPERIMENTS = {...}` block):

```python
def run_floor_sweep_exp(args):
    """对应 cstr/run_floor_sweep.py:地板成因战役(τ=100 深挖 L×H 网格)。

    记录 {baseline, baseline_converged, teacher, fgl_student, A_iter, E_iter} 每
    (dataset, L, H, seed),写 cstr/results/floor_sweep.csv,供 H1-H4 检验。
    """
    sys.path.insert(0, _CSTR_DIR)
    import run_floor_sweep
    entries = run_floor_sweep.load_entries(["tau100"])
    cells = {"tau100": run_floor_sweep.default_tau100_grid()}
    seeds = list(range(args.seeds if args.seeds else 3))
    run_floor_sweep.run(
        entries, cells, seeds,
        alpha=args.alpha, temperature=args.temperature, bins=args.bins,
        epochs=args.epochs, round_epochs=args.round_epochs,
        batch_size=args.batch_size, patience=args.patience, K=args.K,
        conv_epochs=100, conv_patience=10,
        outdir=os.path.join(_CSTR_DIR, "results"), verbose=True)
```

Then add to the `EXPERIMENTS` dict (after the `lh_sweep` entry):

```python
    "floor_sweep":      dict(fn=run_floor_sweep_exp, enabled=False, note="地板成因战役:τ=100 深挖 L×H,记录 baseline/teacher/E_iter 等地板量(H1-H4)"),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/cstr/test_run_py_floor_sweep.py -v`
Expected: PASS (floor_sweep listed as off; EXPERIMENTS has it with enabled=False).

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run: `uv run pytest -v`
Expected: All tests PASS (prior tests + the new ones added in Tasks 1–5).

- [ ] **Step 6: Commit**

```bash
git add cstr/run.py tests/cstr/test_run_py_floor_sweep.py
git commit -m "feat(cstr): register floor_sweep experiment (off by default)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 6: Run the campaign + write the conclusion

Execute the full pipeline for real on τ=100 + anchors, then run the analyzer. This is the "produce the actual research result" step.

**Files:**
- Produces: `cstr/results/floor_sweep.csv`, `cstr/results/lyapunov_tau.csv`, `cstr/results/floor_h*.png`, `conclusion/floor_determinants.md`.

**Interfaces:** All from Tasks 1–4.

- [ ] **Step 1: Estimate Lyapunov exponents for all delayed datasets**

Run: `uv run python cstr/lyapunov_delayed.py`
Expected: writes `cstr/results/lyapunov_tau.csv`, prints λ per dataset; τ=100 should be clearly positive (chaotic).

- [ ] **Step 2: Run the combined sweep (τ=100 deep + τ50/τ150 anchors in one CSV)**

Run: `uv run python cstr/run_floor_sweep.py --datasets tau100,tau50,tau150 --seeds 3 --K 5`
Expected: writes `cstr/results/floor_sweep.csv` in **one pass** — τ=100 uses the 13-cell deep grid (39 rows), τ50/τ150 fall back to the L20H15 anchor (3 rows each), all in the same file. ~2h on M5 MPS. `main()` writes `floor_sweep.csv` once at the end (overwrites), so the three datasets must run in a single invocation — do not run them as separate commands or earlier rows are lost. Spot-check that `teacher_mse` and `E_iter_mse` columns are populated and finite, and that `dataset` takes values `tau100 / tau50 / tau150`.

- [ ] **Step 3: Run the analyzer**

Run: `uv run python cstr/analyze_floor.py --deep_label tau100`
Expected: writes 6 figures to `cstr/results/` and `conclusion/floor_determinants.md` with the four verdicts.

- [ ] **Step 4: Read the verdicts and reconcile**

Open `conclusion/floor_determinants.md`. Cross-check against `conclusion/项目汇报总结.md` §6.3/§7.2 and `conclusion/chaotic_cstr_fgl_exploration.md` §4.2. If a verdict contradicts prior findings, note it explicitly in the conclusion file (do not paper over contradictions).

- [ ] **Step 5: Commit results**

```bash
git add cstr/results/floor_sweep.csv cstr/results/lyapunov_tau.csv \
        cstr/results/floor_h*.png conclusion/floor_determinants.md
git commit -m "exp(cstr): floor-determinants results — H1-H4 verdicts on tau=100

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
