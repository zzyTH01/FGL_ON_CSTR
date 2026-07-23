#!/usr/bin/env python
"""
FGL (Future-Guided Learning) on CSTR — Sequence-to-Sequence Mode.

Teacher predicts K steps (short horizon, high accuracy).
Student predicts full 72 steps (one sub-cycle).
Distillation: KL divergence on first K steps where Teacher is most reliable.

Key insight: Teacher's advantage comes from predicting fewer steps (K << 72),
not from temporal offset. Both models see the same input window.

Usage:
  uv run python cstr/exp/fgl_cstr_seq2seq.py --horizon 72 --teacher_steps 10 --alpha 0.5 --epochs 50
  uv run python cstr/exp/fgl_cstr_seq2seq.py --sweep --teacher_steps 10 --alpha 0.5 --epochs 50
"""

import argparse
import pickle
import sys
import os
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MG_UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "mackey_glass")
sys.path.insert(0, MG_UTILS_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

# -------------------- Device setup --------------------
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


# ==================== Seq2Seq Model ====================
class SeqRNN(nn.Module):
    """
    Encoder RNN → multi-output head that predicts `output_steps` values,
    each discretized into `num_bins` bins.

    Output shape: (batch, output_steps, num_bins)
    """
    def __init__(self, input_size, hidden_size, output_steps, num_bins, num_layers=2):
        super().__init__()
        self.output_steps = output_steps
        self.num_bins = num_bins
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_steps * num_bins)

    def forward(self, x):
        # x: (batch, 1, lookback_window)
        h0 = torch.zeros(self.rnn.num_layers, x.size(0),
                         self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]           # (batch, hidden_size)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)           # (batch, output_steps * num_bins)
        return out.view(-1, self.output_steps, self.num_bins)


# -------------------- Sequence-aware data loading --------------------
def create_seq_dataset(data, lookback_window, forecasting_horizon,
                       num_bins, val_size, test_size,
                       batch_size=64, bin_edges=None):
    """
    Build train/val/test DataLoaders for sequence prediction.

    Each sample: input = window of L past values,
                 target = next `forecasting_horizon` values (a sequence).

    Returns:
        train_loader, val_loader, test_loader, bin_edges
    """
    x_raw = np.array([float(pt[0]) for pt in data])
    y_raw = np.array([float(pt[1]) for pt in data])

    L = lookback_window
    H = forecasting_horizon

    X_windows, Y_windows = [], []
    for i in range(len(x_raw) - L - H + 1):
        X_windows.append(x_raw[i : i + L])
        Y_windows.append(y_raw[i + L : i + L + H])

    X = np.stack(X_windows)   # (N, L)
    Y = np.stack(Y_windows)   # (N, H)

    N = X.shape[0]
    assert 0 < val_size + test_size < 1

    n_test  = int(N * test_size)
    n_val   = int(N * val_size)
    n_train = N - n_val - n_test

    X_train, X_val,   X_test   = X[:n_train], X[n_train:n_train+n_val], X[-n_test:]
    Y_train, Y_val,   Y_test   = Y[:n_train], Y[n_train:n_train+n_val], Y[-n_test:]

    # Discretize
    if bin_edges is None:
        bin_edges = np.linspace(Y_train.min(), Y_train.max(), num_bins - 1)

    X_train_b = np.digitize(X_train, bin_edges).clip(0, num_bins - 1)
    X_val_b   = np.digitize(X_val,   bin_edges).clip(0, num_bins - 1)
    X_test_b  = np.digitize(X_test,  bin_edges).clip(0, num_bins - 1)
    Y_train_b = np.digitize(Y_train, bin_edges).clip(0, num_bins - 1)
    Y_val_b   = np.digitize(Y_val,   bin_edges).clip(0, num_bins - 1)
    Y_test_b  = np.digitize(Y_test,  bin_edges).clip(0, num_bins - 1)

    def to_loader(X_arr, Y_arr):
        X_t = torch.tensor(X_arr, dtype=torch.float32)
        Y_t = torch.tensor(Y_arr, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(X_t, Y_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

    return (to_loader(X_train_b, Y_train_b),
            to_loader(X_val_b,   Y_val_b),
            to_loader(X_test_b,  Y_test_b),
            bin_edges)


# -------------------- Early Stopping --------------------
class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, current_loss, model):
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


# -------------------- Evaluation --------------------
def evaluate_seq(model, loader, output_steps):
    """MSE between predicted bin indices and true bin indices, averaged over steps."""
    mse_loss = nn.MSELoss()
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device).view(-1, 1, x.shape[1])
            y = y.long().to(device)  # (batch, output_steps)
            pred = model(x)  # (batch, output_steps, num_bins)
            pred_bins = pred.argmax(dim=-1).float()  # (batch, output_steps)
            total += mse_loss(pred_bins, y.float()).item()
            count += 1
    return total / count


# -------------------- Step-wise KL divergence --------------------
def seq_KL(student_logits, teacher_logits, temperature, alpha, num_steps):
    """
    KL divergence averaged over the first `num_steps` timesteps.

    Args:
        student_logits: (batch, H, num_bins)
        teacher_logits: (batch, H, num_bins) — only first num_steps used
        temperature: softmax temperature
        alpha: CE weight (return (1-alpha) * KL)
        num_steps: K — number of teacher-guided steps

    Returns:
        scalar loss = (1-α) × T² × mean_KL over first K steps
    """
    s = student_logits[:, :num_steps, :]  # (batch, K, num_bins)
    t = teacher_logits[:, :num_steps, :]  # (batch, K, num_bins)

    # Flatten batch and time dimensions
    B, K, C = s.shape
    s_flat = s.reshape(B * K, C)
    t_flat = t.reshape(B * K, C)

    log_p_s = F.log_softmax(s_flat / temperature, dim=1)
    p_t     = F.softmax(t_flat / temperature, dim=1)
    kd = F.kl_div(log_p_s, p_t, reduction='batchmean') * (temperature ** 2)
    return (1.0 - alpha) * kd


# ==================== Core FGL Seq2Seq Training ====================
def run_fgl_seq2seq(
    student_horizon=72,
    teacher_steps=10,
    alpha=0.5,
    num_bins=50,
    val_size=0.2,
    test_size=0.2,
    epochs=50,
    temperature=4,
    lookback_window=8,
    batch_size=64,
    patience=5,
    verbose=True,
    dataset="h2o",
):
    """
    Seq2Seq FGL: Teacher predicts K steps, Student predicts H steps (full cycle).

    Teacher advantage: K << H → Teacher's short-horizon predictions are more accurate.
    Distillation: KL on first K student outputs vs teacher outputs.
    """
    torch.manual_seed(42)
    hidden_size = 128
    num_layers = 2
    lr = 1e-4
    H = student_horizon  # full prediction length (e.g., 72)
    K = min(teacher_steps, H)  # teacher-guided steps

    # Load data
    data_path = _AVAILABLE_DATASETS[dataset]
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Compute shared bin edges from full data
    x_raw = np.array([float(pt[0]) for pt in data])
    y_raw = np.array([float(pt[1]) for pt in data])
    L = lookback_window
    all_y = []
    for i in range(len(x_raw) - L - H + 1):
        all_y.extend(y_raw[i + L : i + L + H])
    all_y = np.array(all_y)
    shared_bin_edges = np.linspace(all_y.min(), all_y.max(), num_bins - 1)

    if verbose:
        print(f"\n{'='*55}")
        print(f"[Seq2Seq] Dataset={dataset}  H={H}  K={K}  α={alpha:.2f}  "
              f"T={temperature:.1f}  Bins={num_bins}  Lookback={L}  Epochs={epochs}")
        print(f"Bin edges: [{all_y.min():.4f}, {all_y.max():.4f}] → {num_bins} bins")
        print(f"{'='*55}")

    # Teacher data: predicts K steps (easier)
    teacher_train, teacher_val, teacher_test, _ = create_seq_dataset(
        data=data, lookback_window=L, forecasting_horizon=K,
        num_bins=num_bins, val_size=val_size, test_size=test_size,
        batch_size=batch_size, bin_edges=shared_bin_edges,
    )
    # Student data: predicts H steps (harder)
    student_train, student_val, student_test, _ = create_seq_dataset(
        data=data, lookback_window=L, forecasting_horizon=H,
        num_bins=num_bins, val_size=val_size, test_size=test_size,
        batch_size=batch_size, bin_edges=shared_bin_edges,
    )

    celoss = nn.CrossEntropyLoss()

    # ────────── Teacher training (K-step prediction) ──────────
    teacher = SeqRNN(L, hidden_size, K, num_bins, num_layers).to(device)
    opt_t = optim.Adam(teacher.parameters(), lr=lr)
    stop_t = EarlyStopper(patience=patience)

    for epoch in range(epochs):
        teacher.train()
        for x, y in teacher_train:
            x = x.float().to(device).view(-1, 1, L)
            y = y.long().to(device)  # (batch, K)
            out = teacher(x)         # (batch, K, num_bins)
            loss = celoss(out.reshape(-1, num_bins), y.reshape(-1))
            opt_t.zero_grad()
            loss.backward()
            opt_t.step()
        teacher.eval()
        with torch.no_grad():
            val_loss = sum(
                celoss(teacher(x.float().to(device).view(-1, 1, L)).reshape(-1, num_bins),
                       y.long().to(device).reshape(-1)).item()
                for x, y in teacher_val
            ) / len(teacher_val)
        if stop_t.step(val_loss, teacher):
            break
    stop_t.restore(teacher)

    # ────────── Baseline training (H-step prediction, no distillation) ──────────
    baseline = SeqRNN(L, hidden_size, H, num_bins, num_layers).to(device)
    opt_b = optim.Adam(baseline.parameters(), lr=lr)
    stop_b = EarlyStopper(patience=patience)

    for epoch in range(epochs):
        baseline.train()
        for x, y in student_train:
            x = x.float().to(device).view(-1, 1, L)
            y = y.long().to(device)  # (batch, H)
            out = baseline(x)        # (batch, H, num_bins)
            loss = celoss(out.reshape(-1, num_bins), y.reshape(-1))
            opt_b.zero_grad()
            loss.backward()
            opt_b.step()
        baseline.eval()
        with torch.no_grad():
            val_loss = sum(
                celoss(baseline(x.float().to(device).view(-1, 1, L)).reshape(-1, num_bins),
                       y.long().to(device).reshape(-1)).item()
                for x, y in student_val
            ) / len(student_val)
        if stop_b.step(val_loss, baseline):
            break
    stop_b.restore(baseline)

    # ────────── Student training (H-step + KL distillation on first K) ──────────
    # Student outputs H steps. Teacher outputs K steps.
    # We run teacher on student's input to get K-step predictions for distillation.
    student = SeqRNN(L, hidden_size, H, num_bins, num_layers).to(device)
    opt_s = optim.Adam(student.parameters(), lr=lr)
    stop_s = EarlyStopper(patience=patience)

    for epoch in range(epochs):
        student.train()
        for (x_s, y_s), (x_t, _) in zip(student_train, teacher_train):
            x_s = x_s.float().to(device).view(-1, 1, L)
            y_s = y_s.long().to(device)  # (batch, H)
            s_out = student(x_s)          # (batch, H, num_bins)

            # Teacher predicts K steps from student's input
            x_t = x_t.float().to(device).view(-1, 1, L)
            with torch.no_grad():
                t_out = teacher(x_t)      # (batch, K, num_bins)

            # CE loss on all H steps
            ce = celoss(s_out.reshape(-1, num_bins), y_s.reshape(-1))
            # KL distillation on first K steps
            kl = seq_KL(s_out, t_out, temperature, alpha, K)
            loss = alpha * ce + kl

            opt_s.zero_grad()
            loss.backward()
            opt_s.step()

        student.eval()
        with torch.no_grad():
            val_loss = sum(
                celoss(student(x.float().to(device).view(-1, 1, L)).reshape(-1, num_bins),
                       y.long().to(device).reshape(-1)).item()
                for x, y in student_val
            ) / len(student_val)
        if stop_s.step(val_loss, student):
            break
    stop_s.restore(student)

    # ────────── Final evaluation ──────────
    t_mse = evaluate_seq(teacher, teacher_test, K)
    b_mse = evaluate_seq(baseline, student_test, H)
    s_mse = evaluate_seq(student, student_test, H)
    improvement = (b_mse - s_mse) / b_mse * 100 if b_mse > 0 else 0

    print(f"  Teacher (K={K}):  {t_mse:.4f}")
    print(f"  Baseline (H={H}): {b_mse:.4f}")
    print(f"  Student (H={H}):  {s_mse:.4f}  (Δ={improvement:+.1f}%)")

    return {
        "horizon": H,
        "teacher_steps": K,
        "teacher": t_mse,
        "baseline": b_mse,
        "student": s_mse,
        "improvement": improvement,
    }


# -------------------- Sweep --------------------
def sweep_params(args):
    results = []

    # Sweep over teacher_steps if specified
    if args.sweep_K:
        ks = list(range(args.k_min, args.k_max + 1, args.k_step))
        print(f"\n{'='*60}")
        print(f"CSTR Seq2Seq K-Sweep: K={ks[0]}→{ks[-1]}  H={args.horizon}  α={args.alpha}")
        print(f"{'='*60}")
        for k in ks:
            r = run_fgl_seq2seq(
                student_horizon=args.horizon,
                teacher_steps=k,
                alpha=args.alpha,
                num_bins=args.num_bins,
                epochs=args.epochs,
                temperature=args.temperature,
                lookback_window=args.lookback_window,
                batch_size=args.batch_size,
                patience=args.patience,
                dataset=args.dataset,
            )
            results.append(r)
    else:
        # Sweep over horizons
        if args.sweep_range is None:
            start, end = 4, min(72, args.horizon + 1)
        else:
            start, end = map(int, args.sweep_range.split(","))
        horizons = list(range(start, end))

        print(f"\n{'='*60}")
        print(f"CSTR Seq2Seq Horizon Sweep: H={start}→{end-1}  "
              f"K={args.teacher_steps}  α={args.alpha}")
        print(f"{'='*60}")
        for h in horizons:
            r = run_fgl_seq2seq(
                student_horizon=h,
                teacher_steps=args.teacher_steps,
                alpha=args.alpha,
                num_bins=args.num_bins,
                epochs=args.epochs,
                temperature=args.temperature,
                lookback_window=args.lookback_window,
                batch_size=args.batch_size,
                patience=args.patience,
                dataset=args.dataset,
            )
            results.append(r)

    improvements = [r["improvement"] for r in results]
    positive = sum(1 for v in improvements if v > 0)
    avg_imp = np.mean(improvements)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  FGL wins:        {positive}/{len(results)}")
    print(f"  Avg improvement: {avg_imp:+.1f}%")
    print(f"  Max improvement: {max(improvements):+.1f}%")
    print(f"  Min improvement: {min(improvements):+.1f}%")

    return results


# ==================== Main ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSTR FGL Seq2Seq Experiment")
    parser.add_argument("--horizon", type=int, default=72,
                        help="Student prediction length H (default: 72 = one sub-cycle)")
    parser.add_argument("--teacher_steps", type=int, default=10,
                        help="Teacher prediction length K (K << H)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="CE weight (0=full distillation)")
    parser.add_argument("--num_bins", type=int, default=50, help="Discretization bins")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--temperature", type=float, default=4,
                        help="Distillation temperature")
    parser.add_argument("--lookback_window", type=int, default=8, help="History length")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test fraction")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--sweep", action="store_true", help="Run horizon sweep")
    parser.add_argument("--sweep_K", action="store_true", help="Run K (teacher steps) sweep")
    parser.add_argument("--sweep_range", type=str, default=None)
    parser.add_argument("--k_min", type=int, default=3, help="Min K for sweep")
    parser.add_argument("--k_max", type=int, default=20, help="Max K for sweep")
    parser.add_argument("--k_step", type=int, default=2, help="K sweep step")
    parser.add_argument("--dataset", type=str, default="h2o",
                        choices=["h2o", "temperature"])

    args = parser.parse_args()

    if args.sweep or args.sweep_K:
        sweep_params(args)
    else:
        run_fgl_seq2seq(
            student_horizon=args.horizon,
            teacher_steps=args.teacher_steps,
            alpha=args.alpha,
            num_bins=args.num_bins,
            val_size=args.val_size,
            test_size=args.test_size,
            epochs=args.epochs,
            temperature=args.temperature,
            lookback_window=args.lookback_window,
            batch_size=args.batch_size,
            patience=args.patience,
            dataset=args.dataset,
        )
