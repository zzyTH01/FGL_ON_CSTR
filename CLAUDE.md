# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Future-Guided Learning (FGL) — official implementation for the Nature Communications 2025 paper *"A predictive approach to enhance time-series forecasting"*. FGL enhances time-series forecasting via knowledge distillation: a **teacher** model sees near-future data (short horizon), a **student** model predicts far-future data (long horizon), and the teacher's insights are distilled into the student during training.

## Domains & experiment structure

The repo covers four domains. The first three are from the original paper; the fourth (CSTR) is a user extension.

| Domain | Directory | Task type | Model architecture | Metric |
|--------|-----------|-----------|-------------------|--------|
| AES (EEG seizure) | `AES/` | Binary classification | CNN-LSTM (3D conv + LSTM) | AUC-ROC, Sensitivity, FPR |
| CHB-MIT (EEG seizure) | `CHBMIT/` | Binary classification | CNN-LSTM or MViT (Vision Transformer) | AUC-ROC, Sensitivity, FPR |
| Mackey-Glass (chaotic system) | `mackey_glass/` | Regression (discretized) | 2-layer RNN | MSE |
| **CSTR (chemical reactor)** | `cstr/` | Regression (discretized) | 2-layer RNN (shared with MG) | MSE |

## Shared core (`mackey_glass/utils/utils.py`)

Three reusable components imported by both Mackey-Glass and CSTR experiments:

- **`RNN(input_size, hidden_size, output_size, num_layers=2)`** — 2-layer RNN with dropout, FC-ReLU-FC head. Takes `(batch, 1, lookback_window)` input.
- **`KL(student_logits, teacher_logits, temperature, alpha)`** — Knowledge distillation loss: `(1-α) × T² × KL(softmax(teacher/T) || log_softmax(student/T))`.
- **`create_time_series_dataset(data, lookback_window, forecasting_horizon, num_bins, val_size, test_size, offset=0, MSE=False, batch_size=1)`** — Sliding-window construction, discretization into `num_bins` bins, train/val/test split, returns DataLoaders. The `offset` parameter shifts data indices to align teacher and student time streams.

## Three-stage training pipeline (all domains)

Every domain follows the same pattern:

1. **Train Teacher** — short-horizon task (1-step prediction or seizure detection on near-future data)
2. **Train Baseline** — long-horizon task WITHOUT teacher guidance (control group)
3. **Train Student (FGL)** — long-horizon task WITH knowledge distillation from the frozen teacher. Loss = `α × CrossEntropy(output, label) + (1-α) × KL(output, teacher_logits, T)`

### Time alignment trick (Mackey-Glass / CSTR)

The teacher dataset uses `offset = H-1` and `forecasting_horizon = 1`, while the student uses `offset = 0` and `forecasting_horizon = H`. This means teacher's input window is shifted `H-1` steps forward — it sees "the future" relative to the student. During training, `zip(student_train, teacher_train)` pairs aligned batches.

### Key hyperparameters

- **α (alpha)**: weight of cross-entropy loss (0 = full distillation, 1 = no distillation / baseline-equivalent)
- **T (temperature)**: softens probability distributions for KL divergence. Higher T → softer → more generalized patterns transferred

## Running experiments

All commands run from repo root. The project uses `uv` for package management with a `.venv` at the repo root.

### EEG domains (CHBMIT / AES)

Three stages must run in order, with teacher models saved to disk:

```bash
# Stage 1: Train teacher (seizure detection)
python -m CHBMIT.exp.seizure_detection --patient 1 --epochs 50 --patience 5
# or for AES universal teacher:
python AES/exp/create_teacher.py --epochs 50

# Stage 2: Train baseline (seizure prediction, no FGL)
python -m CHBMIT.exp.seizure_prediction --patient 1 --model MViT --trials 3

# Stage 3: Train student with FGL
python -m CHBMIT.exp.FGL_CHBMIT --patient 1 --epochs 30 --trials 3 --alpha 0.5 --temperature 4
```

### Mackey-Glass

```bash
# Single run
python mackey_glass/exp/base_exp.py --horizon 5 --alpha 0.5 --num_bins 50 --epochs 20

# With Page-Hinkley drift detection
python mackey_glass/exp/drift_exp.py --horizon 5 --alpha 0.5 --num_bins 50 --epochs 20 --use_ph

# Hyperparameter analysis & plots
python mackey_glass/exp/analysis.py --mode all --epochs 30
```

### CSTR (user extension)

```bash
# Single run
python cstr/exp/fgl_cstr.py --horizon 5 --alpha 0.5 --epochs 30

# Horizon sweep
python cstr/exp/fgl_cstr.py --sweep --alpha 0.5 --epochs 30

# Regenerate CSTR data (requires cantera >= 3.2.0)
python cstr/generate.py
```

## EEG vs. regression code differences

The EEG and regression (MG/CSTR) codebases are **independent implementations** of the same FGL idea — they do not share code:

- **EEG**: Real-valued EEG signals → 3D CNN or ViT encoder → binary classification (ictal/interictal). Teacher and student have **different architectures** but the same input data shape. Knowledge distillation uses `F.softmax/F.log_softmax/F.kl_div` directly (inline), not the shared `KL()` utility.
- **MG/CSTR**: Scalar time series → discretized into `num_bins` bins → RNN classification over bins. Teacher and student share the `RNN` architecture. Knowledge distillation uses the shared `KL()` function from `mackey_glass/utils/utils.py`.

## Data conventions

- Mackey-Glass data is generated on-the-fly via `jitcdde` (delay differential equation solver)
- CSTR data is pre-generated by `cstr/generate.py` using Cantera, saved as `.pkl` files (2-column float64 tensors)
- EEG data must be downloaded externally and placed in `Dataset/` directories (not included in the repo)
- All `.pkl` files contain PyTorch tensors saved with `pickle.dump()`

## Environment

- Python 3.11, PyTorch 2.1.1
- Package manager: `uv` (see `pyproject.toml` and `uv.lock`)
- Virtual environment: `.venv/` at repo root
- Device selection: CUDA > MPS (Apple Silicon) > CPU (automatic in all scripts)
- Additional EEG dependencies: `mne` (EEG processing), `scikit-learn` (metrics)
- Additional CSTR dependency: `cantera >= 3.2.0`
- Additional MG dependency: `jitcdde`, `jitcxde_common`, `sympy`
