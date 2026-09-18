# CSTR External Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fair continuous-MSE external baseline suite for CSTR and run the approved pilot at L=20/H=15 and L=20/H=12.

**Architecture:** Add reusable sliding-window regression models and training/evaluation in `fgl_common.baselines`; add a CSTR-only driver that generates one CSV row per dataset/method/L/H/seed and adds an off-by-default `external_baselines` switch in `cstr/run.py`. Inputs, splits, normalization, early stopping, and test metrics remain identical across methods.

**Tech Stack:** Python 3.11, PyTorch 2.1.1, NumPy closed-form ridge, no new runtime dependency.

**Spec:** Confirmed brainstorm proposal: first-tier baselines are Ridge, DLinear, PatchTST, GRU, and TCN; continuous physical-value MSE is the primary metric; preprocessing uses train-only statistics; 5 seeds.

## Global Constraints

- Do not compare against historical bin-index MSE directly.
- Normalize input and target using train-only mean/std.
- Preserve chronological train/val/test splits.
- Keep `external_baselines` disabled by default.
- CSV path: `cstr/results/cstr_external_baselines.csv`.
- Use existing project style and no new dependencies.
- Minimum pilot seeds: 5.

---

### Task 1: Continuous dataset and Ridge baseline

**Files:**
- Create: `fgl_common/baselines.py`
- Test: `tests/fgl_common/test_baselines.py`

**Interfaces:**
- Produces: `build_continuous_windows(data, lookback_window, forecasting_horizon, val_size=0.2, test_size=0.2) -> dict`, `run_ridge_baseline(...) -> dict`, `MODEL_BUILDERS`, `run_forecasting_baseline(...) -> dict`.
- Data contract: arrays shaped `(N, 1, L)`; target shaped `(N,)`; split key `train`, `val`, `test`.

- [x] Step 1: Write failing tests for train-only scaling, chronological windows, ridge fitting, and shape contract.
- [ ] Step 2: Implement continuous window preparation and ridge closed-form regression.
- [ ] Step 3: Run focused tests.

### Task 2: Deep forecasting baselines

**Files:**
- Modify: `fgl_common/baselines.py`
- Test: `tests/fgl_common/test_baselines.py`

**Interfaces:**
- Produces: builders `dlinear`, `patchtst`, `gru`, `tcn`; all accept `(lookback_window, output_size=1)` and return `nn.Module`.
- Produces: `run_forecasting_baseline(data, method, lookback_window, forecasting_horizon, seed, epochs, ...)` returning val/test MSE, prediction count, and epochs.

- [x] Step 1: Write failing shape and one-step optimization/smoke tests.
- [ ] Step 2: Implement DLinear, PatchTST, GRU, and TCN with generic early-stopped training.
- [ ] Step 3: Run focused tests.

### Task 3: CSTR integration

**Files:**
- Modify: `cstr/run.py`
- Create: `cstr/baselines_driver.py`
- Test: `tests/cstr/test_baselines_driver.py`

**Interfaces:**
- Produces: `cstr.baselines_driver.run_all(args)`; EXPERIMENTS key `external_baselines` disabled by default.
- CLI: `--baseline_methods`, `--baseline_epochs`, reusing common `L/H/seeds/dataset` arguments.
- Output columns: `dataset,L,H,method,seed,val_mse,test_mse,n_test,epochs`.

- [x] Step 1: Write failing driver/test-entry smoke tests using tiny data.
- [ ] Step 2: Implement CSV driver, EXPERIMENTS entry, CLI parsing.
- [ ] Step 3: Run focused tests.

### Task 4: Documentation and verification

**Files:**
- Modify: `cstr/results/INDEX.md`, `README.md` only if a natural entry point exists.

- [x] Step 1: Document command, metric, methods, and pilot CSV attribution.
- [ ] Step 2: Run full focused test suite.
- [ ] Step 3: Run pilot commands with 5 seeds:
  `uv run python cstr/run.py -e external_baselines --L 20 --H 15 --seeds 5`
  `uv run python cstr/run.py -e external_baselines --L 20 --H 12 --seeds 5`
- [ ] Step 4: Aggregate means/std and record whether continuous adaptive distillation must be rerun under the same continuous metric.
