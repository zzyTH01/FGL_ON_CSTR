"""FGL 共享蒸馏损失。

- ``KL``             —— 标准 KL(batchmean)。迁自 ``mackey_glass/utils/utils.py``。
- ``KL_weighted``    —— 逐样本加权 KL。收自 ``cstr/exp/adaptive_weight_exp.py``。
- ``seq_KL``         —— 序列步级 KL(前 K 步)。收自 ``cstr/exp/fgl_cstr_seq2seq.py``。
- ``compute_weights``—— 自适应蒸馏权重 A/B/C/D。收自 ``cstr/exp/adaptive_weight_exp.py``。
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


def compute_weights(variant, baseline_errors, teacher_errors, student_train_indices):
    """Compute per-sample weights for the KL term.

    Args:
        variant: 'A' (uniform), 'B' (∝ baseline error), 'C' (∝ max(0, base−teacher)).
        baseline_errors / teacher_errors: dict {idx: error}.
        student_train_indices: list of sample indices in the student loader.
    Returns:
        ``(weights_dict, raw, normalized)`` where ``weights_dict`` maps idx→weight.

    Collected from ``cstr/exp/adaptive_weight_exp.py``.
    """
    n = len(student_train_indices)
    raw = np.zeros(n)

    if variant == 'A':
        raw[:] = 1.0
    elif variant == 'B':
        for i, idx in enumerate(student_train_indices):
            raw[i] = baseline_errors.get(idx, 0.0)
    elif variant in ('C', 'D'):
        for i, idx in enumerate(student_train_indices):
            be = baseline_errors.get(idx, 0.0)
            te = teacher_errors.get(idx, 0.0)
            raw[i] = max(0.0, be - te)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if variant == 'A':
        normalized = raw.copy()
    else:
        p5, p95 = np.percentile(raw, 5), np.percentile(raw, 95)
        if p95 - p5 < 1e-8:
            normalized = np.ones(n)
        else:
            clipped = np.clip(raw, p5, p95)
            normalized = 0.2 + 1.8 * (clipped - p5) / (p95 - p5)

    weights = {idx: float(normalized[i]) for i, idx in enumerate(student_train_indices)}
    return weights, raw, normalized
