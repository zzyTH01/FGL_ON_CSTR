# A Predictive Approach to Enhance Time-Series Forecasting

[![Paper](https://img.shields.io/badge/paper-nature_communications-B31B1B.svg)](https://doi.org/10.1038/s41467-025-63786-4)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is a research study of **Future-Guided Learning (FGL)** (Nature Communications 2025, Gunasekaran et al.). FGL enhances time-series forecasting via teacher–student knowledge distillation: a **teacher** model sees near-future data, a **student** model predicts the far future, and the teacher's insight is distilled into the student by minimizing the discrepancy between their probability distributions.

This fork focuses on **three regression / nonlinear-dynamical-system domains** — Mackey-Glass, CSTR, and Lorenz-63 — and studies *when and why* FGL helps. The original EEG (AES / CHB-MIT) experiments are **not part of this study and have been removed**. Research findings are in [`conclusion/final_conclusions.md`](conclusion/final_conclusions.md).

---

## Key finding

FGL effectiveness scales with the **number of feedback loops** in the system (not raw "chaos vs periodicity"):

* **Mackey-Glass τ=13** (period-doubling, 2 loops) — best FGL gain (**+79%**).
* **Lorenz-63 ρ=60** (strong chaos, ∞ loops) — most robust (96% of L×H configs positive).
* **CSTR** (period-1 limit cycle, 1 loop) — single feedback loop caps FGL gain (~+12%); seven optimization directions all failed, which is a dynamical-structure limit rather than a code issue.

![Overview of FGL](fig3.png)
<details>
<summary><b>Figure 3: Overview of FGL and its applications. (Click to expand)</b></summary>
<b>A</b> In the FGL framework, a teacher model operates in the relative future of a student model that focuses on long-term forecasting. After training the teacher on its future-oriented task, both models perform inference during the student’s training phase. The probability distributions from the teacher and student are extracted, and a loss is computed based on Eq. (1). <b>A1</b> Knowledge distillation transfers information via the Kullback–Leibler (KL) divergence between class distributions. <b>C</b> In a regression forecasting scenario, the teacher and student perform short-term and long-term forecasting, respectively. The student gains insights from the teacher during training, enhancing its ability to predict further into the future.
</details>

---

## 1. Setup

Python 3.11 + `uv` (a `.venv/` lives at the repo root).

```bash
uv sync          # or: pip install -r requirements.txt
```

## 2. Domains & data

| Domain | Directory | System | Data |
|--------|-----------|--------|------|
| Mackey-Glass | `mackey_glass/` | delay-DDE chaotic (τ=13) | generated on-the-fly via `jitcdde` |
| CSTR | `cstr/` | H₂/O₂ reactor (periodic oscillation) | `cstr/generate*.py` (Cantera) → `cstr/data/` |
| Lorenz-63 | `lorenz/` | 3D ODE chaotic (ρ=60) | generated on-the-fly via `scipy.integrate` |

All three share one library — **`fgl_common/`** (RNN model, KL distillation, sliding-window discretization, the 3-stage teacher→baseline→student training loop, and an L×H sweep helper).

## 3. Running experiments

Each domain has a single `run.py` entry with an `EXPERIMENTS` on/off switch dict:

```bash
uv run python cstr/run.py --list                       # list experiments + switch state
uv run python cstr/run.py -e baseline -H 5 --alpha 0.5 # one experiment
uv run python mackey_glass/run.py -e lh_sweep          # L×H grid sweep
uv run python lorenz/run.py -e generate --sweep        # sweep ρ, generate data
```

The three-stage FGL pipeline (teacher: 1-step, offset=H-1 → baseline: H-step → student: H-step + KL distillation) is implemented once in `fgl_common.run_fgl_experiment` and reused by all domains. See [`CLAUDE.md`](CLAUDE.md) for the full API and the per-domain experiment list.

## 4. Hyperparameters

* **α (alpha)** — CE vs KL weight (0 = full distillation, 1 = baseline-equivalent).
* **T (temperature)** — softens teacher/student distributions before KL.
* **L / H** — lookback window / forecast horizon; the dominant variables for FGL effectiveness (the L+H-1 ≥ τ threshold).

## 5. Citation

```
@article{Gunasekaran2025,
  author = {Gunasekaran, Skye and Kembay, Assel and Ladret, Hugo and Zhu, Rui-Jie and Perrinet, Laurent and Kavehei, Omid and Eshraghian, Jason},
  title = {A predictive approach to enhance time-series forecasting},
  journal = {Nature Communications},
  year = {2025},
  volume = {16},
  number = {8645},
  pages = {1--7},
  doi = {10.1038/s41467-025-63786-4}
}
```
