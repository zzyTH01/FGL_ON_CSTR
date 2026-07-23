#!/usr/bin/env python
"""
Experiment 7: Adaptive Distillation Weights.

Three variants:
  A (no weight):       standard FGL, w_i = 1.0 for all samples
  B (naive baseline):  w_i ∝ err_baseline,i   (weight high-error samples more)
  C (difference):      w_i ∝ max(0, err_baseline,i - err_teacher,i)
                       (weight samples where teacher has genuine advantage)

Usage:
  uv run python cstr/exp/adaptive_weight_exp.py --L 20 --H 15 --epochs 30 --seeds 5
"""

import argparse, pickle, sys, os, csv
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MG_UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "mackey_glass")
sys.path.insert(0, MG_UTILS_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.utils import RNN, create_time_series_dataset, KL as KL_orig

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

_AVAILABLE_DATASETS = {
    "h2o": os.path.join(os.path.dirname(SCRIPT_DIR), "data_h2o.pkl"),
}

RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ================================================================
#  Weighted KL divergence
# ================================================================
def KL_weighted(student_logits, teacher_logits, temperature, alpha, sample_weights):
    """
    Per-sample weighted KL divergence.

    Args:
        sample_weights: (batch,) tensor of per-sample weights, mean ≈ 1.0
    Returns:
        scalar loss = (1-alpha) * T^2 * mean(w_i * KL_i)
    """
    log_p_s = F.log_softmax(student_logits / temperature, dim=1)
    p_t = F.softmax(teacher_logits / temperature, dim=1)
    kl_per_sample = F.kl_div(log_p_s, p_t, reduction='none').sum(dim=1)  # (batch,)
    weighted_kl = (sample_weights * kl_per_sample).mean()
    return (1.0 - alpha) * (temperature ** 2) * weighted_kl


# ================================================================
#  Early Stopping
# ================================================================
class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best_loss, self.counter, self.best_state = float("inf"), 0, None

    def step(self, loss, model):
        if loss + self.min_delta < self.best_loss:
            self.best_loss, self.counter = loss, 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


# ================================================================
#  Evaluation
# ================================================================
def evaluate(model, loader, L):
    mse = nn.MSELoss()
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _, x, y in loader:
            x = x.float().to(device).view(-1, 1, L)
            y_int = y.long().to(device).squeeze(-1)
            pred = model(x).argmax(dim=1).float()
            total += mse(pred, y_int.float()).item()
    return total / len(loader)


# ================================================================
#  Per-sample error computation (no drop_last, to cover all samples)
# ================================================================
def compute_per_sample_errors(model, loader, L, is_teacher_loader=False):
    """
    Compute per-sample cross-entropy loss.
    Returns: dict mapping sample_idx -> error (float)
    """
    celoss = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    errors = {}
    with torch.no_grad():
        for indices, x, y in loader:
            x = x.float().to(device).view(-1, 1, L)
            y_int = y.long().to(device)
            logits = model(x)
            per_sample = celoss(logits, y_int)  # (batch,)
            for j, idx in enumerate(indices):
                errors[idx.item()] = per_sample[j].item()
    return errors


# ================================================================
#  Weight computation
# ================================================================
def compute_weights(variant, baseline_errors, teacher_errors, student_train_indices):
    """
    Compute per-sample weights for KL divergence term.

    Args:
        variant: 'A', 'B', or 'C'
        baseline_errors: dict {idx: err} for baseline on student training set
        teacher_errors:  dict {idx: err} for teacher on aligned teacher training set
        student_train_indices: list of sample indices in the student training loader

    Returns:
        weights: dict {idx: weight} for each training sample
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

    # Normalize: clip to [p5, p95], then map to [0.2, 2.0]
    if variant == 'A':
        # All 1.0, no normalization needed
        normalized = raw.copy()
    else:
        p5, p95 = np.percentile(raw, 5), np.percentile(raw, 95)
        if p95 - p5 < 1e-8:
            normalized = np.ones(n)
        else:
            clipped = np.clip(raw, p5, p95)
            # Map [p5, p95] → [0.2, 2.0]
            normalized = 0.2 + 1.8 * (clipped - p5) / (p95 - p5)

    weights = {idx: float(normalized[i]) for i, idx in enumerate(student_train_indices)}
    return weights, raw, normalized


# ================================================================
#  Core experiment
# ================================================================
def run_experiment(L=20, H=15, alpha=0.5, temperature=4, num_bins=50,
                   epochs=30, batch_size=64, patience=5, seed=42,
                   variant='A', verbose=True):
    """
    Train FGL with variant A/B/C at (L, H).
    Returns dict with metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    hidden_size, output_size, num_layers = 128, num_bins, 2
    lr = 1e-4

    # Load data
    with open(_AVAILABLE_DATASETS["h2o"], "rb") as f:
        data = pickle.load(f)

    x_raw = np.array([float(pt[0]) for pt in data])
    y_raw = np.array([float(pt[1]) for pt in data])

    # Shared bin edges
    all_y = []
    for i in range(len(x_raw) - L - 1 + 1):
        all_y.append(y_raw[i + L + 1 - 1])
    shared_bin_edges = np.linspace(np.array(all_y).min(), np.array(all_y).max(), num_bins - 1)

    if verbose:
        print(f"  [{variant}] L={L} H={H} α={alpha} T={temperature} seed={seed}")

    # Datasets
    teacher_train, teacher_val, teacher_test, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=1,
        num_bins=num_bins, val_size=0.2, test_size=0.2,
        offset=H - 1, batch_size=batch_size, bin_edges=shared_bin_edges,
    )
    student_train, student_val, student_test, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H,
        num_bins=num_bins, val_size=0.2, test_size=0.2,
        offset=0, batch_size=batch_size, bin_edges=shared_bin_edges,
    )

    # Also create no-drop_last loaders for error computation
    student_train_full, _, _, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H,
        num_bins=num_bins, val_size=0.2, test_size=0.2,
        offset=0, batch_size=1, bin_edges=shared_bin_edges,
    )
    teacher_train_full, _, _, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=1,
        num_bins=num_bins, val_size=0.2, test_size=0.2,
        offset=H - 1, batch_size=1, bin_edges=shared_bin_edges,
    )

    celoss = nn.CrossEntropyLoss()

    # ---------- Teacher ----------
    teacher = RNN(L, hidden_size, output_size, num_layers).to(device)
    opt_t = optim.Adam(teacher.parameters(), lr=lr)
    stop_t = EarlyStopper(patience=patience)
    for ep in range(epochs):
        teacher.train()
        for _, x, y in teacher_train:
            x = x.float().to(device).view(-1, 1, L)
            opt_t.zero_grad()
            celoss(teacher(x), y.long().to(device)).backward()
            opt_t.step()
        teacher.eval()
        with torch.no_grad():
            vl = sum(celoss(teacher(x.float().to(device).view(-1, 1, L)), y.long().to(device)).item()
                     for _, x, y in teacher_val) / len(teacher_val)
        if stop_t.step(vl, teacher): break
    stop_t.restore(teacher)
    teacher.eval()

    # ---------- Baseline ----------
    baseline = RNN(L, hidden_size, output_size, num_layers).to(device)
    opt_b = optim.Adam(baseline.parameters(), lr=lr)
    stop_b = EarlyStopper(patience=patience)
    for ep in range(epochs):
        baseline.train()
        for _, x, y in student_train:
            x = x.float().to(device).view(-1, 1, L)
            opt_b.zero_grad()
            celoss(baseline(x), y.long().to(device)).backward()
            opt_b.step()
        baseline.eval()
        with torch.no_grad():
            vl = sum(celoss(baseline(x.float().to(device).view(-1, 1, L)), y.long().to(device)).item()
                     for _, x, y in student_val) / len(student_val)
        if stop_b.step(vl, baseline): break
    stop_b.restore(baseline)
    baseline.eval()

    # ---------- Compute per-sample weights ----------
    # Get training sample indices from the (no-drop_last) loader
    student_train_indices = []
    for indices, _, _ in student_train_full:
        student_train_indices.append(indices[0].item())

    baseline_errors = compute_per_sample_errors(baseline, student_train_full, L)
    teacher_errors = compute_per_sample_errors(teacher, teacher_train_full, L, is_teacher_loader=True)

    weights_dict, raw_weights, normalized_weights = compute_weights(
        variant, baseline_errors, teacher_errors, student_train_indices)

    if verbose:
        print(f"    Raw weights: mean={raw_weights.mean():.3f} "
              f"std={raw_weights.std():.3f} "
              f"min={raw_weights.min():.3f} max={raw_weights.max():.3f}")
        if variant != 'A':
            print(f"    Normalized:   mean={normalized_weights.mean():.3f} "
                  f"std={normalized_weights.std():.3f} "
                  f"min={normalized_weights.min():.3f} max={normalized_weights.max():.3f}")

    # ---------- Student (FGL with adaptive weighting) ----------
    student = RNN(L, hidden_size, output_size, num_layers).to(device)
    opt_s = optim.Adam(student.parameters(), lr=lr)
    stop_s = EarlyStopper(patience=patience)

    for ep in range(epochs):
        student.train()
        for (indices_s, x_s, y_s), (_, x_t, _) in zip(student_train, teacher_train):
            x_s = x_s.float().to(device).view(-1, 1, L)
            targets = y_s.long().to(device)
            outputs = student(x_s)
            x_t = x_t.float().to(device).view(-1, 1, L)
            with torch.no_grad():
                logits = teacher(x_t)

            # Per-sample weights for this batch
            batch_weights = torch.tensor(
                [weights_dict.get(idx.item(), 1.0) for idx in indices_s],
                dtype=torch.float32, device=device
            )

            if variant in ('A', 'B', 'C'):
                # Weight on KL term (original approach)
                loss = alpha * celoss(outputs, targets) + \
                       KL_weighted(outputs, logits, temperature, alpha, batch_weights)
            elif variant == 'D':
                # Weight on α: modulate distillation STRENGTH per sample
                # High weight → low α → more distillation
                alpha_i = torch.clamp(alpha / batch_weights, 0.01, 0.99)
                ce_per_sample = F.cross_entropy(outputs, targets, reduction='none')
                log_p_s = F.log_softmax(outputs / temperature, dim=1)
                p_t = F.softmax(logits / temperature, dim=1)
                kl_per_sample = F.kl_div(log_p_s, p_t, reduction='none').sum(dim=1)
                loss = (alpha_i * ce_per_sample + (1 - alpha_i) * (temperature ** 2) * kl_per_sample).mean()
            else:
                raise ValueError(f"Unknown variant: {variant}")

            opt_s.zero_grad()
            loss.backward()
            opt_s.step()

        student.eval()
        with torch.no_grad():
            vl = sum(celoss(student(x.float().to(device).view(-1, 1, L)), y.long().to(device)).item()
                     for _, x, y in student_val) / len(student_val)
        if stop_s.step(vl, student): break
    stop_s.restore(student)
    student.eval()

    # ---------- Evaluation ----------
    t_mse = evaluate(teacher, teacher_test, L)
    b_mse = evaluate(baseline, student_test, L)
    s_mse = evaluate(student, student_test, L)
    improvement = (b_mse - s_mse) / b_mse * 100 if b_mse > 0 else 0
    abs_imp = b_mse - s_mse

    return {
        "variant": variant, "L": L, "H": H, "seed": seed,
        "teacher_mse": t_mse, "baseline_mse": b_mse, "student_mse": s_mse,
        "abs_improvement": abs_imp, "fgl_delta": improvement,
        "weights": weights_dict, "raw_weights": raw_weights,
    }


# ================================================================
#  Main
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Weight Experiment")
    parser.add_argument("--L", type=int, default=20)
    parser.add_argument("--H", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds per variant")
    parser.add_argument("--variants", type=str, default="A,B,C", help="Comma-separated variants to run")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    seeds = list(range(args.seeds))
    n_total = len(variants) * len(seeds)

    print(f"Adaptive Weight Experiment")
    print(f"  L={args.L} H={args.H} α={args.alpha} T={args.temperature}")
    print(f"  Variants: {variants}  Seeds: {seeds}")
    print(f"  Total runs: {n_total}")
    print()

    csv_path = os.path.join(RESULTS_DIR, "adaptive_weight_results.csv")
    fieldnames = ["variant", "L", "H", "seed", "baseline_mse", "teacher_mse",
                  "student_mse", "abs_improvement", "fgl_delta"]

    # Check existing
    existing = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                existing.add((row["variant"], int(row["seed"])))

    all_results = []
    for variant in variants:
        for seed in seeds:
            if (variant, seed) in existing:
                print(f"  [{variant}] seed={seed} — already done, skipping")
                continue
            r = run_experiment(L=args.L, H=args.H, alpha=args.alpha,
                               temperature=args.temperature, epochs=args.epochs,
                               seed=seed, variant=variant)
            row = {k: r[k] for k in fieldnames}
            all_results.append(row)
            print(f"  [{variant}] seed={seed}: Baseline={r['baseline_mse']:.1f} "
                  f"Student={r['student_mse']:.1f} Δ={r['fgl_delta']:+.1f}%")

    if all_results:
        # Append to CSV
        mode = "a" if os.path.exists(csv_path) else "w"
        with open(csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            for row in all_results:
                writer.writerow(row)

    # Load all results for summary
    all_rows = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            all_rows.append(row)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Adaptive Weight Experiment (L={args.L}, H={args.H})")
    print(f"{'='*70}")
    print(f"{'Variant':<20} {'n':>3} {'Baseline':>10} {'Student':>10} {'Abs Imp':>10} {'Δ%':>10}")
    print("-" * 65)

    from scipy import stats as st
    agg = defaultdict(list)
    for r in all_rows:
        agg[r["variant"]].append(r)

    for v in ["A", "B", "C"]:
        rows_v = agg.get(v, [])
        if not rows_v:
            continue
        deltas = [float(r["fgl_delta"]) for r in rows_v]
        abs_imps = [float(r["abs_improvement"]) for r in rows_v]
        b_mse = [float(r["baseline_mse"]) for r in rows_v]
        s_mse = [float(r["student_mse"]) for r in rows_v]
        print(f"{v:<20} {len(deltas):>3} {np.mean(b_mse):>10.1f} {np.mean(s_mse):>10.1f} "
              f"{np.mean(abs_imps):>10.1f} {np.mean(deltas):>+9.1f}% ± {np.std(deltas, ddof=1):.1f}%")

    # Statistical tests
    print(f"\nStatistical Tests:")
    for v1, v2, test_name in [("C", "A", "C vs A (difference weight vs baseline)"),
                               ("C", "B", "C vs B (difference vs naive)"),
                               ("B", "A", "B vs A (naive vs baseline)")]:
        d1 = [float(r["fgl_delta"]) for r in agg.get(v1, [])]
        d2 = [float(r["fgl_delta"]) for r in agg.get(v2, [])]
        if len(d1) >= 3 and len(d2) >= 3:
            t, p = st.ttest_ind(d1, d2, equal_var=False)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            print(f"  {test_name}: t={t:+.3f}, p={p:.4f} {sig}")

    # One-sample tests
    print(f"\nOne-sample t-tests (Δ > 0?):")
    for v in ["A", "B", "C"]:
        deltas = [float(r["fgl_delta"]) for r in agg.get(v, [])]
        if len(deltas) >= 3:
            t, p = st.ttest_1samp(deltas, 0)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            print(f"  {v}: mean={np.mean(deltas):+.1f}% t={t:+.3f} p={p:.4f} {sig}")

    print(f"\nResults saved to: {csv_path}")
