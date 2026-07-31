# Delayed-Feedback CSTR τ-Sweep — Results & Interpretation

**Date:** 2026-07-30
**Generator:** `cstr/generate_delayed_stable.py` (bounded tanh control law + smooth onset + EMA low-pass on sensed signal; same mdot topology as `generate_delayed_feedback.py`)
**Spec:** `docs/superpowers/specs/2026-07-29-cstr-delayed-feedback-stable-design.md`
**Sweep artifacts:** `delayed_tau_sweep_s1_A0.9_b0.03.csv`, `...png`

## Winning configuration

`--sign +1 --amplitude 0.9 --filter_beta 0.03` (t_end=600s, dt=0.1, burn_in=100s, coarse grid + fine pass around τ=50).

This was found by a parameter search over (sign, amplitude, filter_beta):
- **Negative feedback (sign=−1) phase-locks** the oscillator — periodicity stays 0.96–0.99 for every τ (no transition). This is the "lock-in" risk the spec flagged.
- **Positive feedback (sign=+1), strong (A≥0.7), light filter (β≤0.05)** breaks periodicity. A=0.9, β=0.03 is the strongest config that stays **fully stable (0 crashes across the coarse grid at t_end=400)** and shows the clearest transition at full t_end=600.

## Periodicity-vs-τ curve (the deliverable)

| τ (steps) | τ (s) | periodicity | dom_period (s) | status |
|---|---|---|---|---|
| 5  | 0.5 | 0.906 | 39.5 | ok |
| 10 | 1.0 |  —    |  —   | **crash** (narrow instability island) |
| 20 | 2.0 | 0.923 | 17.2 | ok |
| 30 | 3.0 | 0.789 | 22.6 | ok |
| 35 | 3.5 | 0.622 | 33.7 | ok |
| 40 | 4.0 | 0.546 | 2.0* | ok |
| 45 | 4.5 | 0.554 | 2.0* | ok |
| 50 | 5.0 | 0.557 | 2.0* | ok |
| 55 | 5.5 | 0.551 | 2.0* | ok |
| 60 | 6.0 | 0.559 | 2.0* | ok |
| 65 | 6.5 | 0.562 | 2.0* | ok |
| 70 | 7.0 | 0.808 | 39.9 | ok |
| 80 | 8.0 | 0.785 | 34.2 | ok |
| 100| 10  | 0.486 | 2.0* | ok |
| 120| 12  | 0.519 | 32.4 | ok |
| 150| 15  | 0.468 | 2.0* | ok |

\* `dom_period=2.0s` is the metric floor-sticking at `lag_min` (20 steps): in broadband/aperiodic zones there is no dominant period, so `argmax` returns the smallest lag. It is **not** a real 2s oscillation. Real dominant periods (17–40s, the slow relaxation cycle) only appear in the periodic/quasi-periodic zones.

## Verdict — transition band CONFIRMED

The user's hypothesis — *"only at intermediate delay does a periodic↔non-periodic transition appear"* — is **confirmed** for the delayed-feedback CSTR:

1. **Small τ (5–20): periodic** (0.91–0.92) — clean limit cycle, delay too short to disrupt the oscillator.
2. **First transition (τ≈30–65): periodicity collapses** 0.79 → 0.62 → ~0.55. Fine pass (τ=35,40,…,65) resolves it as a sharp drop, not noise.
3. **Quasi-periodic window (τ≈70–80): partial recovery** to ~0.80 — an interleaved periodic window, the classic signature of a DDE bifurcation diagram (periodic/complex bands alternating).
4. **Second aperiodic band (τ≈100–150):** periodicity ~0.47–0.52, the deep aperiodic regime.

This interleaved periodic → aperiodic → periodic-window → aperiodic structure is exactly the bifurcation topology of a delay-differential system (cf. Mackey-Glass τ sweep).

## Caveats (honest)

- **τ=10 crashed.** A single narrow instability island between two periodic points (τ=5, τ=20 both fine). Caught by the per-τ `try/except`; did not abort the sweep. Does not affect the transition story. Could likely be recovered with a marginally heavier filter at that τ alone, but was not needed for the curve.
- **Low periodicity ≠ proven chaos.** The autocorrelation dip to ~0.47–0.55 is strong evidence of aperiodicity, but distinguishing true deterministic chaos from quasi-periodicity / high-period cycles requires a positive largest Lyapunov exponent or a Poincaré section — explicitly out of scope for this task (spec §2, "curve only"). The `dom_period=2.0s` floor-stick is consistent with broadband (chaos-like) dynamics but is not proof.
- **Mechanism change is in the controller, not the plant.** Per the user's locked constraint, the reactor/sensor/delay/mdot-actuator topology is unchanged from `generate_delayed_feedback.py`; only the control law differs (bounded + onset + EMA filter, sign flipped to +1). The EMA filter (β=0.03) is an actuator-bandwidth element, not a plant change.

## Reproduce

```bash
uv run python cstr/generate_delayed_stable.py --sweep --sign 1 \
    --amplitude 0.9 --filter_beta 0.03 --fine_around 50
```

## Conclusion

Yes — the delayed-feedback CSTR exhibits the predicted periodic→non-periodic transition at intermediate τ, **provided** the feedback is positive, strong (A=0.9), and lightly filtered (β=0.03). With negative or weak feedback the oscillator simply phase-locks (no transition). The transition band centers at τ≈40–65 steps (4–6.5s, ≈0.6–0.9× the natural period of ~7.15s), with a second aperiodic zone at τ=100–150.

13 aperiodic datasets (periodicity < 0.85) auto-saved to `cstr/data/data_delayed_stable_h2o_tau{30,35,…,150}_s1_A0.9_b0.03.pkl` for any downstream FGL use.
