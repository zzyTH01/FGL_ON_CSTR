# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Future-Guided Learning (FGL) — implementation for the Nature Communications 2025 paper *"A predictive approach to enhance time-series forecasting"*. FGL enhances time-series forecasting via knowledge distillation: a **teacher** model sees near-future data (short horizon), a **student** model predicts far-future data (long horizon), and the teacher's insights are distilled into the student during training.

The repo covers five domains. The first three are from the original paper; **CSTR** and **Lorenz-63** are user extensions studying when FGL helps (conclusion: FGL effectiveness scales with the number of feedback loops — MG τ=13 and Lorenz ρ=60 are good benchmarks; CSTR's single-loop periodic-1 oscillation caps FGL gain). Full research findings live in `conclusion/`.

| Domain | Directory | Task type | Model | Metric |
|--------|-----------|-----------|-------|--------|
| AES (EEG seizure) | `AES/` | Binary classification | CNN-LSTM (3D conv + LSTM) | AUC-ROC, Sensitivity, FPR |
| CHB-MIT (EEG seizure) | `CHBMIT/` | Binary classification | CNN-LSTM or MViT | AUC-ROC, Sensitivity, FPR |
| Mackey-Glass (chaotic) | `mackey_glass/` | Regression (discretized) | 2-layer RNN | MSE |
| CSTR (chemical reactor) | `cstr/` | Regression (discretized) | 2-layer RNN | MSE |
| **Lorenz-63 (chaotic)** | `lorenz/` | Regression (discretized) | 2-layer RNN | MSE |

## Shared core: `fgl_common/` (root-level package)

All three regression domains (cstr / mackey_glass / lorenz) share one library — **`fgl_common/`** — which consolidates what used to be duplicated across ~18 scripts:

- **`models.py`** — `RNN`, `LSTMModel`, `RNNRegression`, `SeqRNN` (all take `(batch, 1, lookback_window)`).
- **`data.py`** — `create_time_series_dataset(data, lookback_window, forecasting_horizon, num_bins, val_size, test_size, offset=0, MSE=False, batch_size=1, bin_edges=None)` — sliding-window + discretization + DataLoaders. The `offset` param shifts indices to align teacher/student streams. Also `create_seq_dataset` (multi-step targets).
- **`distillation.py`** — `KL(student_logits, teacher_logits, temperature, alpha)` = `(1-α)·T²·KL(softmax(teacher/T)‖log_softmax(student/T))`; plus `KL_weighted`, `seq_KL`, `compute_weights`.
- **`training.py`** — `device`, `EarlyStopper`, `evaluate*` (classification/regression/seq, optional Page-Hinkley drift retrain), and the **unified** `run_fgl_experiment(data, lookback_window, forecasting_horizon, model_fn=RNN, alpha, temperature, num_bins, epochs, regression=False, use_ph=False, ...)` — the 3-stage teacher→baseline→student loop. Variant differences collapse into parameters (`model_fn` / `regression` / `use_ph`). Also `run_adaptive_weight`, `run_adaptive_inference`, `run_seq2seq`.
- **`sweep.py`** — `run_lh_sweep(run_fn, data, L_values, H_values, seeds, outdir, ...)` — generic L×H grid sweep with CSV + heatmap + report.

`mackey_glass/utils/utils.py` now **re-exports** `RNN`/`KL`/`create_time_series_dataset` from `fgl_common` (and keeps the MG-specific `MackeyGlass` jitcdde dataset class), so old `from utils.utils import ...` imports still work.

## Per-domain unified entry: `run.py` (experiment switches)

Each regression domain has ONE `run.py` with an `EXPERIMENTS` config dict (`enabled` flag = on/off switch) + CLI override:

```bash
python cstr/run.py                       # run all enabled=True experiments
python cstr/run.py -e baseline,lh_sweep  # run specific ones
python cstr/run.py --list                # list all + switch state + notes
```

- **`cstr/run.py`** — baseline / lstm / regression / seq2seq / adaptive / adaptive_weight / **lh_sweep** (enabled: baseline, lh_sweep). Failed CSTR optimizations default off.
- **`mackey_glass/run.py`** — base / drift / **lh_sweep** / tau_sweep / l_threshold / h_threshold / geometry (enabled: base, lh_sweep). Threshold/geometry tests verify the L+H-1≥τ formula.
- **`lorenz/run.py`** — **generate** (ρ sweep) / **lh_sweep** (ρ=60 strong chaos).

Old single-purpose scripts are in each domain's `archive/` (kept for traceability; the active code is `run.py` + `fgl_common`).

## Three-stage training pipeline (all domains)

1. **Train Teacher** — short-horizon (1-step) task with `offset=H-1`, so its window is shifted H-1 steps into the "future" relative to the student.
2. **Train Baseline** — long-horizon (H-step) task WITHOUT teacher guidance (control).
3. **Train Student (FGL)** — long-horizon WITH distillation from the frozen teacher. Loss = `α·CE(output,label) + (1-α)·T²·KL(teacher‖student)`.

### Time alignment trick (regression domains)

Teacher dataset uses `offset=H-1`, `forecasting_horizon=1`; student uses `offset=0`, `forecasting_horizon=H`. During training, `zip(student_train, teacher_train)` pairs aligned batches. Shared `bin_edges` (from all H=1 targets) keep teacher and student on the same discretization.

### Key hyperparameters
- **α (alpha)**: CE weight (0 = full distillation, 1 = baseline-equivalent)
- **T (temperature)**: softens distributions for KL. Higher T → softer → more generalization transferred.

## EEG domains (AES / CHBMIT)

EEG is a **separate implementation** (not using `fgl_common`): real-valued EEG → 3D CNN or ViT → binary classification. Teacher/student have different architectures but same input shape; distillation uses `F.softmax/F.kl_div` inline. Three stages run in order with teacher models saved to disk:

```bash
python -m CHBMIT.exp.seizure_detection --patient 1 --epochs 50 --patience 5   # Stage 1: teacher
python -m CHBMIT.exp.seizure_prediction --patient 1 --model MViT --trials 3   # Stage 2: baseline
python -m CHBMIT.exp.FGL_CHBMIT --patient 1 --epochs 30 --alpha 0.5 --temperature 4  # Stage 3: student
# AES: python AES/exp/create_teacher.py / seizure_detection.py / seizure_prediction.py / FGL_AES.py
```

## CSTR data generation (kept as standalone scripts)

`cstr/generate.py` (original H₂/O₂ combustion → `data/data.pkl`, `data/data_h2o.pkl`), `generate_dual_cstr.py`, `generate_forced.py`, `generate_delayed_feedback.py` are kept standalone (each has its own argparse + `--sweep`). They write to `cstr/`; move outputs into `cstr/data/` if regenerated. Requires `cantera >= 3.2.0`.

## Directory layout (regression domains)

```
cstr/  mackey_glass/  lorenz/
├── run.py            # unified entry (EXPERIMENTS switches)
├── data/             # *.pkl datasets (consolidated)
├── results/          # sweep CSVs + heatmaps + reports (+ plots/ subdir)
├── archive/          # old single-purpose scripts (traceability)
└── (mackey_glass only) utils/utils.py  # re-export fgl_common + MackeyGlass class
fgl_common/           # shared library (models/data/distillation/training/sweep)
conclusion/           # research summaries (final_conclusions.md is the key one)
docs/                 # references: paper PDF, Cantera notebook, notes
```

## Data conventions
- Mackey-Glass: generated on-the-fly via `MackeyGlass` class (jitcdde) in `mackey_glass/utils/utils.py`; `data.pkl` is a pre-generated snapshot.
- CSTR: pre-generated by `cstr/generate*.py` (Cantera), saved as 2-column float64 tensors in `cstr/data/`.
- Lorenz: generated on-the-fly by `lorenz/run.py` (`solve_ivp`); per-ρ snapshots in `lorenz/data/`.
- EEG: must be downloaded externally into `Dataset/` (not in repo).
- All `.pkl` are PyTorch tensors saved with `pickle.dump()`.

## Environment
- Python 3.11, PyTorch 2.1.1, package manager `uv` (`.venv/` at repo root).
- Device: CUDA > MPS > CPU (automatic, defined once in `fgl_common.training`).
- EEG extras: `mne`, `scikit-learn`. CSTR: `cantera>=3.2.0`. MG: `jitcdde`, `sympy`. Lorenz: `scipy`.
