# Design: Stable Delayed-Feedback CSTR for τ-Bifurcation Sweep

**Date:** 2026-07-29
**Status:** Approved (pending spec review)
**Owner:** zzyTH01
**Related code:** `cstr/generate_delayed_feedback.py` (broken original), `cstr/generate.py` (base CSTR)

---

## 1. Background & motivation

The base CSTR (`cstr/generate.py`, H₂/O₂ in a single `IdealGasReactor`) is a stiff relaxation
oscillator with a clean period-1 limit cycle (periodicity ≈ 0.95, natural period ≈ 7.15 s).

Empirical finding (this session, cantera 3.2.0): **no existing generator can make this CSTR
chaotic by tuning a single parameter.**

| Generator | Tunable param | Result |
|-----------|---------------|--------|
| base `generate.py` | U / K / flow | periodicity 0.93–0.98 (`results/param_sweep_results.txt`) |
| `generate_delayed_feedback.py` | `tau_delay` (5–120), A (0.1–0.5) | **CVODES diverges for every τ and A** |
| `generate_dual_cstr.py` | `volume_B` (@ recycle 0.3) | periodicity 1.0000 (phase-locked) |
| `generate_forced.py` | amplitude (0.3–0.8) × freq | periodicity 1.0000 (phase-locked) |

Additionally, the 5 archived `data_dual_*.pkl` "low-periodicity candidates" recompute to
**periodicity 1.0000** (whole and 2nd-half) — they are false positives from an earlier sweep,
not chaotic data.

**Crash root cause** (for `generate_delayed_feedback.py`): the feedback law
`mdot = mdot₀·(1 + A·(H₂O[t−τ]/0.96 − 0.5))` is **positive × unbounded × hard-engaged**.
At the ignition spike, positive feedback amplifies the spike into thermal runaway and CVODES
hits `hmin`. Crash time for τ=50 is t≈4.9 s (≈ step 49, exactly when the delay line first fills
and feedback switches on). Reducing A to 0.1 does not help — the instability is structural, not
amplitude-limited.

**User's hypothesis (theoretically sound):** a delay-differential system should show a
periodic → non-periodic transition at intermediate delay τ (the classic DDE / Mackey-Glass
bifurcation picture). The blocker is that the current generator dies for every τ, so the
predicted transition curve is unobservable.

## 2. Goal & success criterion

Produce a **periodicity-vs-τ curve** from a *stable* delayed-feedback CSTR sweep that reveals
(or rules out) an intermediate-τ transition band between periodic and non-periodic regimes.

Success = the sweep runs to completion across the τ grid without integrator divergence, and the
resulting curve is plotted and interpreted. Whether a transition band actually appears is the
scientific question under test, not a precondition.

## 3. Scope

**In:** a numerically stable delayed-feedback generator; a τ sweep; periodicity + dominant-period
metrics; CSV + plot; auto-save of transition-band datasets.

**Out (explicitly):** rigorous chaos proof (Lyapunov exponent); `run.py` integration; FGL L×H
experiments; changing the actuator to inlet-temperature or composition; changing the chemical
mechanism.

## 4. Physical structure (preserved — locked by user)

The topology is identical to the existing `generate_delayed_feedback.py`. **Only the control law
changes.**

- Reactor: single `IdealGasReactor`, H₂/O₂ via `h2o2.yaml`, same `T/P/composition/volume/valve/wall`
  params as `generate.py`.
- Sensor: H₂O mass fraction `cstr.thermo.Y[h2o_idx]`.
- Delay line: `deque(maxlen=τ_delay)`; delayed sample = `buffer[0]` = `H₂O((t−τ)·dt)`.
- Actuator: inlet `MassFlowController` **mdot** — unchanged (NOT temperature, NOT composition).

User-locked constraint: *"保留 mdot 反馈拓扑,控制律可调"* (keep the mdot feedback topology;
the control law may be adjusted).

## 5. Control law (the only change)

```
mdot(t) = mdot₀ · (1 + s · A_eff(t) · tanh((H₂O[t−τ] − c) / w))
```

| Symbol | Meaning | Default | CLI flag |
|--------|---------|---------|----------|
| `s` | sign (+1 / −1) | **−1** | `--sign` |
| `A` | feedback amplitude | 0.3 | `--amplitude` |
| `c` | center (H₂O midpoint) | 0.48 | `--center` |
| `w` | saturation width | 0.1 | `--width` |
| `t_onset` | onset ramp duration (s) | 50 | `--onset` |

Properties:

- `tanh` bounds the feedback term to `[−A, +A]` ⇒ `mdot ∈ [mdot₀(1−A), mdot₀(1+A)]`,
  **guaranteed non-divergent** — removes the runaway that crashes CVODES.
- Small-signal limit (|H₂O−c| ≪ w) reduces to a linear law ⇒ this is a *bounded version of the
  original law*, not a new structure.
- Default sign `s = −1` (negative): high past-H₂O → reduce inlet → suppress the spike.
  Stabilizing at the spike, and delayed negative feedback is the canonical DDE route to
  oscillation/chaos (best bet to avoid phase-locking). `+1` retained as a flag for fallback.

## 6. Onset (removes the hard-engagement crash)

```
A_eff(t) = A · min(1, t / t_onset)
```

Linear ramp from 0 to A over `t_onset` = 50 s (500 steps, ≈ 7 natural periods). Feedback engages
gradually instead of a step at step τ.

**Delay-line interaction:** feedback contributes only once the buffer is full
(`len(buffer) == τ_delay`); before that `mdot = mdot₀`. Since `t_onset` (50 s) ≫ max τ (15 s),
the buffer is full long before the onset ramp completes, so the two transients do not collide.

## 7. τ sweep protocol

- **Coarse pass:** τ ∈ {5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150}.
- **Fine pass:** if the coarse pass shows a transition band (a sustained drop in periodicity),
  refine within ±20 steps of its center at step 5 (e.g. τ ∈ {35, 40, 45, 50, 55, 60}).
- **Per τ:** `t_end = 600 s`, `dt = 0.1`; discard the first 100 s (onset + transient); compute
  metrics on the stationary segment [100, 600] s (5000 points).
- **Deterministic:** `seed = 42`, no in-loop randomness → 1 run per τ.
- **Robustness:** per-τ `try/except`; a single failure is logged and does not abort the sweep.

## 8. Metrics

- **Primary — periodicity:** max of the mean-removed autocorrelation over lag ∈ [20, 400] steps,
  computed on the post-transient segment. Clean limit cycle ≈ 1; chaotic/broadband → low.
  (Same definition as the existing scripts, for comparability.)
- **Secondary — dominant period:** the lag in [20, 400] at which the autocorrelation peaks
  (×dt gives the period in seconds). A sudden ×2 / ×4 is a period-doubling
  bifurcation signature.
- **Per-τ record:** `τ, sign, A, periodicity, dom_period, T_min, T_max, status`.

## 9. Outputs

- **Script:** `cstr/generate_delayed_stable.py` (the original `generate_delayed_feedback.py` is
  kept for traceability, per repo convention).
- **CLI:** `--sweep --sign -1 --amplitude 0.3 --onset 50 --width 0.1 --center 0.48
  [--fine_around TAU] [--t_end 600]`.
- **`cstr/results/delayed_tau_sweep.csv`** — per-τ records.
- **`cstr/results/delayed_tau_sweep.png`** — periodicity-vs-τ (primary axis) with dominant-period
  (secondary axis).
- **Auto-saved datasets** for transition-band τ (periodicity < 0.85):
  `cstr/data/data_delayed_stable_h2o_tau{T}_s{sign}_A{A}.pkl` (same 2-column float64 tensor format
  as the other generators, so it is loadable by `run.py`'s `_load_data` if later desired).

## 10. Numerical safety net

1. Bounded `mdot` (tanh) + smooth onset ⇒ primary defense against CVODES divergence.
2. Per-τ `try/except` with `hmin`-failure logging.
3. **Fallback sequence (decided from sweep result, not pre-implemented):** if `s = −1` phase-locks
   (periodicity ≈ 1 across all τ) → flip to `s = +1`; if still locked → widen τ range; if still
   locked → raise `A`. Each is a one-flag change.

## 11. Risks (honest)

- **Phase-locking:** even negative saturating delayed feedback may lock the oscillator to
  periodicity ≈ 1 (this is what forced/dual did). The transition band may not appear — that *is*
  the scientific question.
- **Narrow transition band** may require fine τ resolution to resolve.
- **Negative sign** is a control-law choice (allowed by the user's locked constraint), defaulted to
  for stability / anti-locking; positive is retained as a flag.

## 12. Decisions locked

- Actuator = mdot (user: preserve physical topology).
- Default sign = −1 (user accepted the recommended default).
- Scope = periodicity-vs-τ curve only (no Lyapunov, no FGL pipeline).
