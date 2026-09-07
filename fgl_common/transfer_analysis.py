"""Summaries for transfer and teacher-to-student gain experiments."""

import numpy as np


def _bootstrap_ci(values, n_boot=1000, seed=0):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if values.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def summarize_transfer(rows, value_key="teacher_mse", group_keys=("path", "condition")):
    """Group rows and compute mean/std and bootstrap 95% CI."""
    groups = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row.get(value_key))
    out = []
    for key, values in groups.items():
        clean = [v for v in values if v is not None]
        ci_low, ci_high = _bootstrap_ci(clean)
        out.append(
            {
                **dict(zip(group_keys, key)),
                "n": len(clean),
                "mean": float(np.mean(clean)) if clean else None,
                "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return out


def compute_transfer_ratios(rows):
    """Aggregate paired gains, then compute protected ratios."""
    teacher_gain = float(np.mean([r["teacher_gain"] for r in rows if r["teacher_gain"] is not None]))
    student_gain = float(np.mean([r["student_gain"] for r in rows if r["student_gain"] is not None]))
    ratio = student_gain / teacher_gain if abs(teacher_gain) > 1e-8 else None
    return [{"teacher_gain": teacher_gain, "student_gain": student_gain, "transfer_ratio": ratio}]
