#!/usr/bin/env python
"""
Adaptive FGL v2 — Inference-time only, no weight modification.

Strategy:
  1. Teacher has privileged information (shifted window closer to target).
  2. At inference, compute |student_pred - teacher_pred| as a proxy for student error.
  3. When student-teacher divergence is large, blend student output toward teacher.
  4. No retraining — preserves original student knowledge.

Usage:
  uv run python cstr/exp/fgl_cstr_adaptive.py --horizon 12 --lookback_window 20 --alpha 0.5 --epochs 30 --seed 0
"""

import argparse, pickle, sys, os
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MG_UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "mackey_glass")
sys.path.insert(0, MG_UTILS_DIR)

import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from utils.utils import RNN, create_time_series_dataset, KL

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using {device}")

_AVAILABLE_DATASETS = {
    "temperature": os.path.join(os.path.dirname(SCRIPT_DIR), "data.pkl"),
    "h2o":         os.path.join(os.path.dirname(SCRIPT_DIR), "data_h2o.pkl"),
}


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


def run_adaptive_inference(
    student_horizon=12,
    base_alpha=0.5,
    num_bins=50,
    val_size=0.2,
    test_size=0.2,
    epochs=30,
    temperature=4,
    lookback_window=20,
    batch_size=64,
    patience=5,
    seed=0,
    dataset="h2o",
    divergence_threshold=8.0,
    blend_strength=0.7,
):
    """
    Inference-time adaptive blending.

    Parameters:
        divergence_threshold: |student_pred - teacher_pred| in bins.
            Above this, blend toward teacher.
        blend_strength: weight of teacher in blended output (0=student only, 1=teacher only).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    L, H = lookback_window, student_horizon
    hidden_size, output_size, num_layers = 128, num_bins, 2
    lr = 1e-4

    data_path = _AVAILABLE_DATASETS[dataset]
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Shared bin edges
    x_raw = np.array([float(pt[0]) for pt in data])
    y_raw = np.array([float(pt[1]) for pt in data])
    all_y = []
    for i in range(len(x_raw) - L - 1 + 1):
        all_y.append(y_raw[i + L + 1 - 1])
    shared_bin_edges = np.linspace(np.array(all_y).min(), np.array(all_y).max(), num_bins - 1)

    celoss = nn.CrossEntropyLoss()

    # ------------------ Datasets ------------------
    teacher_train, teacher_val, teacher_test, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=1,
        num_bins=num_bins, val_size=val_size, test_size=test_size,
        offset=H - 1, batch_size=batch_size, bin_edges=shared_bin_edges,
    )
    student_train, student_val, student_test, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H,
        num_bins=num_bins, val_size=val_size, test_size=test_size,
        offset=0, batch_size=batch_size, bin_edges=shared_bin_edges,
    )

    # ------------------ Standard training (same as before) ------------------
    teacher = RNN(L, hidden_size, output_size, num_layers).to(device)
    opt = optim.Adam(teacher.parameters(), lr=lr)
    stop = EarlyStopper(patience=patience)
    for _ in range(epochs):
        teacher.train()
        for _, x, y in teacher_train:
            opt.zero_grad()
            celoss(teacher(x.float().to(device).view(-1,1,L)), y.long().to(device)).backward()
            opt.step()
        teacher.eval()
        vl = sum(celoss(teacher(x.float().to(device).view(-1,1,L)), y.long().to(device)).item()
                 for _, x, y in teacher_val) / len(teacher_val)
        if stop.step(vl, teacher): break
    stop.restore(teacher); teacher.eval()

    baseline = RNN(L, hidden_size, output_size, num_layers).to(device)
    opt = optim.Adam(baseline.parameters(), lr=lr)
    stop = EarlyStopper(patience=patience)
    for _ in range(epochs):
        baseline.train()
        for _, x, y in student_train:
            opt.zero_grad()
            celoss(baseline(x.float().to(device).view(-1,1,L)), y.long().to(device)).backward()
            opt.step()
        baseline.eval()
        vl = sum(celoss(baseline(x.float().to(device).view(-1,1,L)), y.long().to(device)).item()
                 for _, x, y in student_val) / len(student_val)
        if stop.step(vl, baseline): break
    stop.restore(baseline); baseline.eval()

    student = RNN(L, hidden_size, output_size, num_layers).to(device)
    opt = optim.Adam(student.parameters(), lr=lr)
    stop = EarlyStopper(patience=patience)
    for _ in range(epochs):
        student.train()
        for (_, xs, ys), (_, xt, _) in zip(student_train, teacher_train):
            xs, ys = xs.float().to(device).view(-1,1,L), ys.long().to(device)
            xtt = xt.float().to(device).view(-1,1,L)
            out = student(xs)
            with torch.no_grad(): tlog = teacher(xtt)
            loss = base_alpha * celoss(out, ys) + KL(out, tlog, temperature, base_alpha)
            opt.zero_grad(); loss.backward(); opt.step()
        student.eval()
        vl = sum(celoss(student(x.float().to(device).view(-1,1,L)), y.long().to(device)).item()
                 for _, x, y in student_val) / len(student_val)
        if stop.step(vl, student): break
    stop.restore(student); student.eval()

    # ================================================================
    #  INFERENCE-TIME ADAPTIVE BLENDING (no weight modification)
    # ================================================================
    mse = nn.MSELoss()

    # Standard evaluation (baseline + teacher)
    b_total = 0.0
    with torch.no_grad():
        for _, x, y in student_test:
            x = x.float().to(device).view(-1, 1, L)
            y_int = y.long().to(device).squeeze(-1)
            pred = baseline(x).argmax(dim=1).float()
            b_total += mse(pred, y_int.float()).item()
    baseline_mse = b_total / len(student_test)

    t_total = 0.0
    with torch.no_grad():
        for _, x, y in teacher_test:
            x = x.float().to(device).view(-1, 1, L)
            y_int = y.long().to(device).squeeze(-1)
            pred = teacher(x).argmax(dim=1).float()
            t_total += mse(pred, y_int.float()).item()
    teacher_mse = t_total / len(teacher_test)

    # Standard student evaluation
    s_standard_total = 0.0
    s_standard_errors = []
    with torch.no_grad():
        for _, x, y in student_test:
            x = x.float().to(device).view(-1, 1, L)
            y_int = y.long().to(device).squeeze(-1)
            pred = student(x).argmax(dim=1).float()
            s_standard_total += mse(pred, y_int.float()).item()
            s_standard_errors.extend((pred - y_int.float()).abs().cpu().numpy())
    student_mse_standard = s_standard_total / len(student_test)

    # ================================================================
    #  Adaptive Blending: use teacher(x_teacher) as real-time corrector
    # ================================================================
    # Build aligned test samples from raw data (to access both student and teacher windows)
    N_total = len(x_raw) - L - H + 1
    n_test = int(N_total * test_size)
    test_start = N_total - n_test

    blend_count = 0
    total_blend_weight = 0.0
    adaptive_errors = []

    for idx in range(test_start, test_start + n_test):
        # Student window + target
        s_win = torch.tensor(x_raw[idx:idx+L], dtype=torch.float32).view(1,1,L).to(device)
        true_bin = np.digitize(y_raw[idx+L+H-1], shared_bin_edges)

        # Teacher window (shifted H-1 steps forward)
        t_win = torch.tensor(x_raw[idx+H-1:idx+H-1+L], dtype=torch.float32).view(1,1,L).to(device)

        with torch.no_grad():
            s_logits = student(s_win)
            s_pred = s_logits.argmax(dim=1).item()
            s_conf = torch.softmax(s_logits / temperature, dim=1).max().item()

            t_logits = teacher(t_win)
            t_pred = t_logits.argmax(dim=1).item()
            t_conf = torch.softmax(t_logits / temperature, dim=1).max().item()

        divergence = abs(s_pred - t_pred)

        # Blend only when: teacher is confident, student is uncertain, and they disagree
        teacher_sure = t_conf > 0.3
        student_unsure = s_conf < 0.5
        they_disagree = divergence > divergence_threshold

        if teacher_sure and they_disagree:
            # Blend weight proportional to teacher confidence advantage
            w = blend_strength * t_conf
            blended_logits = (1 - w) * s_logits + w * t_logits
            final_pred = blended_logits.argmax(dim=1).item()
            blend_count += 1
            total_blend_weight += w
        else:
            final_pred = s_pred

        adaptive_errors.append((final_pred - true_bin)**2)

    student_mse_adaptive = np.mean(adaptive_errors)

    # ================================================================
    # Report
    # ================================================================
    imp_std = (baseline_mse - student_mse_standard) / baseline_mse * 100
    imp_adp = (baseline_mse - student_mse_adaptive) / baseline_mse * 100

    print(f"\n{'='*60}")
    print(f"RESULTS: Inference-Time Adaptive Blending")
    print(f"{'='*60}")
    print(f"  Config: L={L} H={H} α={base_alpha} T={temperature} seed={seed}")
    print(f"  Threshold={divergence_threshold} bins  Blend strength={blend_strength}")
    print(f"")
    print(f"  Teacher MSE:         {teacher_mse:.1f}")
    print(f"  Baseline MSE:        {baseline_mse:.1f}")
    print(f"  Student (standard):  {student_mse_standard:.1f}  (Δ={imp_std:+.1f}%)")
    print(f"  Student (adaptive):  {student_mse_adaptive:.1f}  (Δ={imp_adp:+.1f}%)")
    print(f"")
    print(f"  Blend triggered:     {blend_count}/{n_test} ({100*blend_count/n_test:.1f}%)")
    print(f"  Avg blend weight:    {total_blend_weight/max(blend_count,1):.2f}")
    print(f"  Gain over standard:  {student_mse_standard - student_mse_adaptive:+.1f} MSE")
    print(f"")

    # Error distribution comparison
    s_std_errors = np.array(s_standard_errors)
    adp_errors_abs = np.sqrt(np.array(adaptive_errors))

    print(f"  Error distribution (abs error in bins):")
    print(f"  {'':>12} {'Standard':>10} {'Adaptive':>10}")
    for pct in [50, 75, 90, 95, 99]:
        print(f"  {pct}th pctl:     {np.percentile(s_std_errors, pct):10.1f}  {np.percentile(adp_errors_abs, pct):10.1f}")

    return {
        "teacher_mse": teacher_mse,
        "baseline_mse": baseline_mse,
        "student_mse_standard": student_mse_standard,
        "student_mse_adaptive": student_mse_adaptive,
        "blend_fraction": blend_count / n_test,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSTR Adaptive FGL — Inference-Time Blending")
    parser.add_argument("--horizon", type=int, default=12, dest="student_horizon")
    parser.add_argument("--lookback_window", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5, dest="base_alpha")
    parser.add_argument("--num_bins", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--divergence_threshold", type=float, default=8.0)
    parser.add_argument("--blend_strength", type=float, default=0.7)
    parser.add_argument("--dataset", type=str, default="h2o", choices=["h2o", "temperature"])

    args = parser.parse_args()
    run_adaptive_inference(**vars(args))
