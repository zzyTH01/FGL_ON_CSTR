# FGL on Delayed-Feedback CSTR Datasets — Results & Interpretation

**Date:** 2026-07-30
**Driver:** `cstr/run_fgl_delayed.py`
**Data:** `cstr/results/fgl_delayed_summary.csv`
**Question:** Does FGL gain recover on aperiodic (chaotic) CSTR, breaking the period-1 "floor"?

## Setup

Standard 3-stage FGL (`run_fgl_experiment`: teacher 1-step → baseline H-step → student H-step + distillation), same configuration as the mainline CSTR baseline so the comparison to the period-1 floor is apples-to-apples:

- Classification mode (50 bins), `model_fn=RNN`, **L=20, H=15**, α=0.5, T=4.0, 30 epochs, patience 5, batch 64.
- **3 seeds** per dataset → mean ± std.
- Datasets: base period-1 `data_h2o.pkl` (control) + representative τ from the delayed-feedback sweep (`s1_A0.9_b0.03`).

## Results

| dataset | periodicity | N | baseline MSE | student MSE | FGL Δ% |
|---|---|---|---|---|---|
| base_h2o (period-1) | 0.940 | 3001 | 127.5 ±8.7 | 112.5 ±6.4 | **+11.6%** ±7.0 |
| τ=30 (mild) | 0.789 | 6000 | 247.5 ±1.3 | 245.3 ±4.6 | +0.9% ±1.4 |
| τ=50 (aperiodic) | 0.557 | 6000 | 104.0 ±0.3 | 104.7 ±0.9 | **−0.7%** ±1.0 |
| τ=80 (quasi-periodic) | 0.785 | 6000 | 160.7 ±1.2 | 149.5 ±3.1 | +7.0% ±2.6 |
| τ=100 (aperiodic) | 0.486 | 6000 | 172.5 ±1.2 | 160.5 ±0.7 | +6.9% ±1.0 |
| τ=150 (strongly aperiodic) | 0.468 | 6000 | 159.3 ±3.8 | 150.9 ±2.5 | +5.2% ±3.8 |

## Interpretation

1. **FGL trains successfully on every dataset** (3-stage pipeline converged, no crashes) and **helps on 5/6** (+0.9% to +11.6%), including several aperiodic ones (τ=80/100/150). **Aperiodicity does not prevent FGL from working.**

2. **No clean "more aperiodic → more FGL gain" trend.** The periodic base (+11.6%) gains the most; the cleanest aperiodic point τ=50 (per 0.56) shows ~0 / slightly negative (−0.7%); the more aperiodic τ=100 (per 0.49) gains +6.9%. The relationship is **non-monotone**.

3. **n=3 is underpowered** given the variance (base ±7.0, τ=150 ±3.8) — several confidence intervals overlap zero. Need n≥5 + a significance test before calling any single Δ% real.

4. **Confound:** baseline MSE varies ~5× across datasets (104–247) — inherent predictability differs wildly, so cross-dataset Δ% comparison is muddied by difficulty, not just periodicity.

## Bottom line

Making CSTR aperiodic does **not** cleanly recover or boost FGL gain. FGL helps on a subset of aperiodic datasets but not consistently, and the periodic baseline benefits about as much. This **reinforces** the project's broader finding: the CSTR domain is hard for FGL regardless of periodicity — the "floor" is not simply a periodicity artifact.

## Caveats & next steps (if a rigorous answer is wanted)

- **Power:** rerun at n=5–10 seeds, add a paired significance test (baseline vs student MSE per seed).
- **Confound control:** the datasets differ in inherent difficulty; a cleaner test would normalize or match difficulty, or sweep L/H on a single aperiodic dataset to apply the L+H-1≥τ threshold finding.
- **Mode:** this used classification/bins (mainline default); a regression-mode run is worth comparing.
- **τ=50 anomaly:** the deepest first-transition point shows no gain while τ=100 (more aperiodic) does — worth understanding (teacher quality? regime-specific transfer?).

## Reproduce

```bash
uv run python cstr/run_fgl_delayed.py --seeds 3      # this run
uv run python cstr/run_fgl_delayed.py --seeds 5      # tighter, recommended for significance
```
