"""FGL 共享蒸馏损失。

- ``KL``             —— 标准 KL(batchmean)。迁自 ``mackey_glass/utils/utils.py``。
- ``KL_weighted``    —— 逐样本加权 KL。收自 ``cstr/exp/adaptive_weight_exp.py``。
- ``seq_KL``         —— 序列步级 KL(前 K 步)。收自 ``cstr/exp/fgl_cstr_seq2seq.py``。
- ``compute_weights``—— 自适应蒸馏权重 A/B/C/D(判定标准 = teacher−student 逐样本 MSE 差距)。
"""
import numpy as np
import torch.nn.functional as F


def KL(student_logits, teacher_logits, temperature, alpha):
    """``(1-α) T² KL( softmax(teacher/T) ∥ log_softmax(student/T) )``。

    Migrated verbatim from ``mackey_glass/utils/utils.py``.
    """
    log_p_s = F.log_softmax(student_logits / temperature, dim=1)
    p_t = F.softmax(teacher_logits / temperature, dim=1)
    kd = F.kl_div(log_p_s, p_t, reduction='batchmean') * (temperature ** 2)
    return (1.0 - alpha) * kd


def KL_weighted(student_logits, teacher_logits, temperature, alpha, sample_weights):
    """Per-sample weighted KL divergence.

    Args:
        sample_weights: (batch,) tensor, mean ≈ 1.0.
    Returns:
        ``(1-α) T² mean(w_i · KL_i)``.

    Collected from ``cstr/exp/adaptive_weight_exp.py``.
    """
    log_p_s = F.log_softmax(student_logits / temperature, dim=1)
    p_t = F.softmax(teacher_logits / temperature, dim=1)
    kl_per_sample = F.kl_div(log_p_s, p_t, reduction='none').sum(dim=1)  # (batch,)
    weighted_kl = (sample_weights * kl_per_sample).mean()
    return (1.0 - alpha) * (temperature ** 2) * weighted_kl


def seq_KL(student_logits, teacher_logits, temperature, alpha, num_steps):
    """KL averaged over the first ``num_steps`` timesteps.

    Args:
        student_logits / teacher_logits: ``(batch, H, num_bins)`` — only the
            first ``num_steps`` are used.
    Returns:
        ``(1-α) T² mean_KL`` over the first K steps.

    Collected from ``cstr/exp/fgl_cstr_seq2seq.py``.
    """
    s = student_logits[:, :num_steps, :]   # (batch, K, num_bins)
    t = teacher_logits[:, :num_steps, :]

    B, K, C = s.shape
    s_flat = s.reshape(B * K, C)
    t_flat = t.reshape(B * K, C)

    log_p_s = F.log_softmax(s_flat / temperature, dim=1)
    p_t = F.softmax(t_flat / temperature, dim=1)
    kd = F.kl_div(log_p_s, p_t, reduction='batchmean') * (temperature ** 2)
    return (1.0 - alpha) * kd


def compute_weights(variant, student_errors, teacher_errors, student_train_indices):
    """Compute per-sample weights for the KL term.

    The weighting criterion is the **teacher–student MSE gap**: samples where the
    student still lags the teacher get higher distillation weight.

    Args:
        variant: 'A' (uniform control), 'B' (∝ student MSE),
            'C'/'D' (∝ max(0, se_student − se_teacher), normalized to [0.2, 2.0]),
            'E' (same gap, but **amplified + zero-floored**: samples the student
            already nails get 0 distillation weight, the hardest get up to W_MAX —
            concentrates the entire distillation budget on the student's weak points).
        student_errors / teacher_errors: dict {student_idx: per-sample MSE}, both
            keyed in the *same* (student) index space and computed on the aligned
            target. ``teacher_errors`` must already be offset-aligned by the caller
            (teacher loader uses offset=H-1, so its raw idx j maps to student idx
            j-(H-1) on the same target).
        student_train_indices: list of sample indices in the student loader.
    Returns:
        ``(weights_dict, raw, normalized)`` where ``weights_dict`` maps idx→weight.

    Evolved from ``cstr/exp/adaptive_weight_exp.py`` (criterion switched from the
    baseline-vs-teacher CE gap to the student-vs-teacher MSE gap).
    """
    n = len(student_train_indices)
    raw = np.zeros(n)

    if variant == 'A':
        raw[:] = 1.0
    elif variant == 'B':
        for i, idx in enumerate(student_train_indices):
            raw[i] = student_errors.get(idx, 0.0)
    elif variant in ('C', 'D', 'E'):
        for i, idx in enumerate(student_train_indices):
            se = student_errors.get(idx, 0.0)
            te = teacher_errors.get(idx, 0.0)
            raw[i] = max(0.0, se - te)
    elif variant == 'E-soft':
        # signed gap(可为负:学生优于老师)→ 喂给 sigmoid 做软地板
        for i, idx in enumerate(student_train_indices):
            se = student_errors.get(idx, 0.0)
            te = teacher_errors.get(idx, 0.0)
            raw[i] = se - te
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if variant == 'A':
        normalized = raw.copy()
    elif variant == 'E':
        # Amplified, zero-floored: gap=0 → weight 0 (pure CE, student already
        # correct), gap≥p95 → W_MAX. Concentrates distillation on weak points.
        W_MAX = 4.0
        p95 = np.percentile(raw, 95)
        normalized = np.clip(raw, 0.0, p95) / p95 * W_MAX if p95 > 1e-8 else np.ones(n)
    elif variant == 'E-soft':
        # Sigmoid 软地板激活:w = w_floor + (W_MAX - w_floor)·σ((gap - c)/s)。
        # 大正 gap → 饱和到 W_MAX(满档蒸馏,同 E);负 gap → 软地板 w_floor
        # (非零,老师信号不断流 → 不干涸);中间 S 形过渡。中心 c=中位数,
        # 尺度 s=(p75-p25)/2。
        w_floor, W_MAX = 0.2, 4.0
        if raw.std() < 1e-8:
            normalized = np.full(n, (w_floor + W_MAX) / 2.0)
        else:
            c = float(np.median(raw))
            p25, p75 = np.percentile(raw, 25), np.percentile(raw, 75)
            s = (p75 - p25) / 2.0
            if s < 1e-8:
                s = float(raw.std())
            sig = 1.0 / (1.0 + np.exp(-(raw - c) / s))
            normalized = w_floor + (W_MAX - w_floor) * sig
    else:  # B / C / D — gentle [0.2, 2.0] mapping
        p5, p95 = np.percentile(raw, 5), np.percentile(raw, 95)
        if p95 - p5 < 1e-8:
            normalized = np.ones(n)
        else:
            clipped = np.clip(raw, p5, p95)
            normalized = 0.2 + 1.8 * (clipped - p5) / (p95 - p5)

    weights = {idx: float(normalized[i]) for i, idx in enumerate(student_train_indices)}
    return weights, raw, normalized
