#!/usr/bin/env python
"""
Quick experiment: LSTM vs RNN for CSTR single-point FGL prediction.

Hypothesis: A stronger model (LSTM) will improve both Baseline and Student,
but the FGL improvement (Δ) will remain near zero or negative — because the
problem is the deterministic periodic nature of CSTR data, not model capacity.

Usage:
  uv run python cstr/exp/fgl_cstr_lstm.py --horizon 5 --alpha 0.5 --epochs 30
  uv run python cstr/exp/fgl_cstr_lstm.py --sweep --alpha 0.5 --epochs 30
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
from utils.utils import create_time_series_dataset, KL

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


# ==================== LSTM Model ====================
class LSTMModel(nn.Module):
    """2-layer LSTM replacing the original RNN. Same hidden_size, same structure."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, 1, lookback_window)
        out, _ = self.lstm(x)          # (batch, lookback, hidden)
        out = out[:, -1, :]            # last timestep
        out = F.relu(self.fc1(out))
        out = self.fc2(out)            # (batch, num_bins)
        return out


# ==================== Early Stopping ====================
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


# ==================== Evaluation ====================
def evaluate(model, loader, lookback_window=8):
    mse_loss = nn.MSELoss()
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _, x, y in loader:
            x = x.float().to(device).view(-1, 1, lookback_window)
            y_int = y.long().to(device).squeeze(-1)
            pred = model(x).argmax(dim=1).float()
            total += mse_loss(pred, y_int.float()).item()
    return total / len(loader)


# ==================== Core FGL Training ====================
def run_fgl_lstm(
    student_horizon=5,
    alpha=0.5,
    num_bins=50,
    val_size=0.2,
    test_size=0.2,
    epochs=30,
    temperature=4,
    lookback_window=8,
    batch_size=64,
    patience=5,
    verbose=True,
    dataset="h2o",
    seed=42,
):
    torch.manual_seed(seed)
    hidden_size = 128
    output_size = num_bins
    num_layers = 2
    lr = 1e-4
    H = student_horizon
    L = lookback_window

    # Load data
    data_path = _AVAILABLE_DATASETS[dataset]
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Shared bin edges (same fix as fgl_cstr.py)
    x_raw = np.array([float(pt[0]) for pt in data])
    y_raw = np.array([float(pt[1]) for pt in data])
    all_y = []
    for i in range(len(x_raw) - L - 1 + 1):
        all_y.append(y_raw[i + L + 1 - 1])
    shared_bin_edges = np.linspace(np.array(all_y).min(), np.array(all_y).max(), num_bins - 1)

    if verbose:
        print(f"\n{'='*50}")
        print(f"[LSTM] Dataset={dataset}  H={H:2d}  α={alpha:.2f}  T={temperature:.1f}  "
              f"Bins={num_bins}  Epochs={epochs}  Lookback={L}")
        print(f"Bin edges: [{np.array(all_y).min():.4f}, {np.array(all_y).max():.4f}] → {num_bins} bins")
        print(f"{'='*50}")

    # Datasets
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

    celoss = nn.CrossEntropyLoss()

    # ────────── Teacher ──────────
    teacher = LSTMModel(L, hidden_size, output_size, num_layers).to(device)
    opt_t = optim.Adam(teacher.parameters(), lr=lr)
    stop_t = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        teacher.train()
        for _, x, y in teacher_train:
            x = x.float().to(device).view(-1, 1, L)
            y = y.long().to(device)
            opt_t.zero_grad()
            celoss(teacher(x), y).backward()
            opt_t.step()
        teacher.eval()
        with torch.no_grad():
            val_loss = sum(
                celoss(teacher(x.float().to(device).view(-1, 1, L)), y.long().to(device)).item()
                for _, x, y in teacher_val
            ) / len(teacher_val)
        if stop_t.step(val_loss, teacher):
            break
    stop_t.restore(teacher)

    # ────────── Baseline ──────────
    baseline = LSTMModel(L, hidden_size, output_size, num_layers).to(device)
    opt_b = optim.Adam(baseline.parameters(), lr=lr)
    stop_b = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        baseline.train()
        for _, x, y in student_train:
            x = x.float().to(device).view(-1, 1, L)
            y = y.long().to(device)
            opt_b.zero_grad()
            celoss(baseline(x), y).backward()
            opt_b.step()
        baseline.eval()
        with torch.no_grad():
            val_loss = sum(
                celoss(baseline(x.float().to(device).view(-1, 1, L)), y.long().to(device)).item()
                for _, x, y in student_val
            ) / len(student_val)
        if stop_b.step(val_loss, baseline):
            break
    stop_b.restore(baseline)

    # ────────── Student (FGL) ──────────
    student = LSTMModel(L, hidden_size, output_size, num_layers).to(device)
    opt_s = optim.Adam(student.parameters(), lr=lr)
    stop_s = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        student.train()
        for (_, x_s, y_s), (_, x_t, _) in zip(student_train, teacher_train):
            x_s = x_s.float().to(device).view(-1, 1, L)
            targets = y_s.long().to(device)
            outputs = student(x_s)
            x_t = x_t.float().to(device).view(-1, 1, L)
            with torch.no_grad():
                logits = teacher(x_t)
            loss = alpha * celoss(outputs, targets) + KL(outputs, logits, temperature, alpha)
            opt_s.zero_grad()
            loss.backward()
            opt_s.step()
        student.eval()
        with torch.no_grad():
            val_loss = sum(
                celoss(student(x.float().to(device).view(-1, 1, L)), y.long().to(device)).item()
                for _, x, y in student_val
            ) / len(student_val)
        if stop_s.step(val_loss, student):
            break
    stop_s.restore(student)

    # ────────── Evaluation ──────────
    t_mse = evaluate(teacher, teacher_test, L)
    b_mse = evaluate(baseline, student_test, L)
    s_mse = evaluate(student, student_test, L)
    improvement = (b_mse - s_mse) / b_mse * 100 if b_mse > 0 else 0

    print(f"  Teacher:  {t_mse:.4f}")
    print(f"  Baseline: {b_mse:.4f}")
    print(f"  Student:  {s_mse:.4f}  (Δ={improvement:+.1f}%)")

    return {"teacher": t_mse, "baseline": b_mse, "student": s_mse, "improvement": improvement}


# ==================== Main ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSTR FGL — LSTM Experiment")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num_bins", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=4)
    parser.add_argument("--lookback_window", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset", type=str, default="h2o",
                        choices=["h2o", "temperature"])
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep_range", type=str, default="3,10")

    args = parser.parse_args()

    if args.sweep:
        start, end = map(int, args.sweep_range.split(","))
        print(f"\n{'='*60}")
        print(f"CSTR FGL LSTM Sweep: H={start}→{end-1}  α={args.alpha}  "
              f"T={args.temperature}  Dataset={args.dataset}")
        print(f"{'='*60}")
        results = []
        for h in range(start, end):
            r = run_fgl_lstm(
                student_horizon=h, alpha=args.alpha,
                num_bins=args.num_bins, epochs=args.epochs,
                temperature=args.temperature, lookback_window=args.lookback_window,
                batch_size=args.batch_size, patience=args.patience,
                dataset=args.dataset, seed=args.seed,
            )
            results.append(r)
        improvements = [r["improvement"] for r in results]
        print(f"\n{'='*60}")
        print(f"SUMMARY (LSTM)")
        print(f"{'='*60}")
        print(f"  FGL wins:        {sum(1 for v in improvements if v > 0)}/{len(results)}")
        print(f"  Avg improvement: {np.mean(improvements):+.1f}%")
        print(f"  Max improvement: {max(improvements):+.1f}%")
        print(f"  Min improvement: {min(improvements):+.1f}%")
    else:
        run_fgl_lstm(
            student_horizon=args.horizon, alpha=args.alpha,
            num_bins=args.num_bins, val_size=args.val_size,
            test_size=args.test_size, epochs=args.epochs,
            temperature=args.temperature, lookback_window=args.lookback_window,
            batch_size=args.batch_size, patience=args.patience,
            dataset=args.dataset, seed=args.seed,
        )
