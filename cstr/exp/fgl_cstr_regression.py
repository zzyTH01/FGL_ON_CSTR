#!/usr/bin/env python
"""
FGL (Future-Guided Learning) experiment on CSTR — Regression Mode.

Unlike the original classification-based approach (discretize into bins),
this version predicts continuous values directly using MSE loss.

Usage:
  uv run python cstr/exp/fgl_cstr_regression.py --horizon 5 --alpha 0.5 --epochs 30
  uv run python cstr/exp/fgl_cstr_regression.py --sweep --alpha 0.5 --epochs 30
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
import torch.optim as optim
import numpy as np
from utils.utils import create_time_series_dataset

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


# -------------------- Regression RNN --------------------
class RNNRegression(nn.Module):
    """RNN for continuous value prediction (single scalar output)."""
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc1(out[:, -1, :])
        out = torch.relu(out)
        out = self.fc2(out)
        return out.squeeze(-1)  # (batch,)


# -------------------- Early Stopping --------------------
class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-6):
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


# -------------------- Page-Hinkley drift detection --------------------
def page_hinkley_update(error, state, delta=0.005):
    state["t"] += 1
    m_prev = state["m"]
    state["m"] += (error - state["m"]) / state["t"]
    state["PH"] = max(0.0, state["PH"] + (error - m_prev - delta))
    return state


# -------------------- Evaluation --------------------
def evaluate(model, loader, use_ph=False,
             delta=0.005, lambda_thr=1.0,
             window_size=50, retrain_epochs=3,
             lr=1e-4, lookback_window=8):
    mse_loss = nn.MSELoss()
    model.eval()

    if not use_ph:
        total = 0.0
        with torch.no_grad():
            for _, x, y in loader:
                x = x.float().to(device).view(-1, 1, lookback_window)
                y = y.float().to(device)
                pred = model(x)
                total += mse_loss(pred, y).item()
        return total / len(loader)

    # Page–Hinkley–driven evaluation
    optimizer = optim.Adam(model.parameters(), lr=lr)
    state = {"m": 0.0, "PH": 0.0, "t": 0}
    window = deque(maxlen=window_size)
    errors = []

    for _, x, y in loader:
        x = x.float().to(device).view(-1, 1, lookback_window)
        y_float = y.float().to(device)
        with torch.no_grad():
            pred = model(x)
        err = mse_loss(pred, y_float).item()
        errors.append(err)
        state = page_hinkley_update(err, state, delta)
        window.append((x.cpu(), y_float.cpu()))

        if state["PH"] > lambda_thr and len(window) == window_size:
            model.train()
            for _ in range(retrain_epochs):
                for wx, wy in window:
                    wx, wy = wx.to(device), wy.to(device)
                    out = model(wx)
                    loss = mse_loss(out, wy)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            model.eval()
            state = {"m": 0.0, "PH": 0.0, "t": 0}
            window.clear()

    return sum(errors) / len(errors)


# -------------------- Core FGL Training (Regression) --------------------
def run_fgl_regression(
    student_horizon,
    alpha,
    val_size=0.2,
    test_size=0.2,
    epochs=50,
    lookback_window=8,
    batch_size=64,
    patience=5,
    use_ph=False,
    ph_delta=0.005,
    ph_lambda=1.0,
    ph_window=50,
    ph_retrain_epochs=3,
    verbose=True,
    dataset="h2o",
    seed=42,
):
    """
    FGL with regression (continuous value prediction, no discretization).

    Loss: α × MSE(output, target) + (1-α) × MSE(output, teacher_output)
    """
    torch.manual_seed(seed)
    hidden_size = 128
    num_layers = 2
    lr = 1e-4

    # Load CSTR data
    data_path = _AVAILABLE_DATASETS[dataset]
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    if verbose:
        print(f"\n{'='*50}")
        print(f"[Regression] Dataset={dataset}  H={student_horizon:2d}  α={alpha:.2f}  "
              f"Epochs={epochs}  Lookback={lookback_window}")
        print(f"{'='*50}")

    # Teacher: predicts 1 step ahead (offset=H-1 to align with Student target)
    # MSE=True → raw continuous values (no discretization), num_bins is ignored
    teacher_train, teacher_val, teacher_test, _, _ = create_time_series_dataset(
        data=data,
        lookback_window=lookback_window,
        forecasting_horizon=1,
        num_bins=50,  # ignored when MSE=True
        val_size=val_size,
        test_size=test_size,
        offset=student_horizon - 1,
        batch_size=batch_size,
        MSE=True,
    )
    student_train, student_val, student_test, _, _ = create_time_series_dataset(
        data=data,
        lookback_window=lookback_window,
        forecasting_horizon=student_horizon,
        num_bins=50,  # ignored when MSE=True
        val_size=val_size,
        test_size=test_size,
        offset=0,
        batch_size=batch_size,
        MSE=True,
    )

    mse_loss = nn.MSELoss()

    # ────────── Teacher training ──────────
    teacher = RNNRegression(lookback_window, hidden_size, num_layers).to(device)
    opt_t = optim.Adam(teacher.parameters(), lr=lr)
    stop_t = EarlyStopper(patience=patience)

    for epoch in range(epochs):
        teacher.train()
        for _, x, y in teacher_train:
            x = x.float().to(device).view(-1, 1, lookback_window)
            y = y.float().to(device)
            opt_t.zero_grad()
            mse_loss(teacher(x), y).backward()
            opt_t.step()
        teacher.eval()
        with torch.no_grad():
            val_loss = sum(
                mse_loss(teacher(x.float().to(device).view(-1, 1, lookback_window)),
                         y.float().to(device)).item()
                for _, x, y in teacher_val
            ) / len(teacher_val)
        if stop_t.step(val_loss, teacher):
            break
    stop_t.restore(teacher)

    # ────────── Baseline training ──────────
    baseline = RNNRegression(lookback_window, hidden_size, num_layers).to(device)
    opt_b = optim.Adam(baseline.parameters(), lr=lr)
    stop_b = EarlyStopper(patience=patience)

    for epoch in range(epochs):
        baseline.train()
        for _, x, y in student_train:
            x = x.float().to(device).view(-1, 1, lookback_window)
            y = y.float().to(device)
            opt_b.zero_grad()
            mse_loss(baseline(x), y).backward()
            opt_b.step()
        baseline.eval()
        with torch.no_grad():
            val_loss = sum(
                mse_loss(baseline(x.float().to(device).view(-1, 1, lookback_window)),
                         y.float().to(device)).item()
                for _, x, y in student_val
            ) / len(student_val)
        if stop_b.step(val_loss, baseline):
            break
    stop_b.restore(baseline)

    # ────────── Student training (FGL distillation) ──────────
    student = RNNRegression(lookback_window, hidden_size, num_layers).to(device)
    opt_s = optim.Adam(student.parameters(), lr=lr)
    stop_s = EarlyStopper(patience=patience)

    for epoch in range(epochs):
        student.train()
        for (_, x_s, y_s), (_, x_t, _) in zip(student_train, teacher_train):
            x_s = x_s.float().to(device).view(-1, 1, lookback_window)
            y_s = y_s.float().to(device)
            pred_s = student(x_s)

            x_t = x_t.float().to(device).view(-1, 1, lookback_window)
            with torch.no_grad():
                pred_t = teacher(x_t)

            # α × MSE(pred, truth) + (1-α) × MSE(pred, teacher_pred)
            loss = alpha * mse_loss(pred_s, y_s) + (1.0 - alpha) * mse_loss(pred_s, pred_t)
            opt_s.zero_grad()
            loss.backward()
            opt_s.step()

        student.eval()
        with torch.no_grad():
            val_loss = sum(
                mse_loss(student(x.float().to(device).view(-1, 1, lookback_window)),
                         y.float().to(device)).item()
                for _, x, y in student_val
            ) / len(student_val)
        if stop_s.step(val_loss, student):
            break
    stop_s.restore(student)

    # ────────── Final evaluation ──────────
    teacher_mse = evaluate(teacher, teacher_test, use_ph, ph_delta, ph_lambda,
                           ph_window, ph_retrain_epochs, lr, lookback_window)
    baseline_mse = evaluate(baseline, student_test, use_ph, ph_delta, ph_lambda,
                            ph_window, ph_retrain_epochs, lr, lookback_window)
    student_mse = evaluate(student, student_test, use_ph, ph_delta, ph_lambda,
                           ph_window, ph_retrain_epochs, lr, lookback_window)

    improvement = (baseline_mse - student_mse) / baseline_mse * 100 if baseline_mse > 0 else 0

    print(f"  Teacher:  {teacher_mse:.6f}")
    print(f"  Baseline: {baseline_mse:.6f}")
    print(f"  Student:  {student_mse:.6f}  (Δ={improvement:+.1f}%)")

    return {
        "horizon": student_horizon,
        "teacher": teacher_mse,
        "baseline": baseline_mse,
        "student": student_mse,
        "improvement": improvement,
    }


# -------------------- Sweep over horizons --------------------
def sweep_horizons(args):
    results = []
    horizon_range = args.sweep_range
    if horizon_range is None:
        horizon_range = "2,31"
    start, end = map(int, horizon_range.split(","))
    horizons = list(range(start, end))

    print(f"\n{'='*60}")
    print(f"CSTR FGL Regression Sweep: H={start}→{end-1}  α={args.alpha}  "
          f"Lookback={args.lookback_window}  Dataset={args.dataset}")
    print(f"{'='*60}")

    for h in horizons:
        r = run_fgl_regression(
            student_horizon=h,
            alpha=args.alpha,
            epochs=args.epochs,
            lookback_window=args.lookback_window,
            batch_size=args.batch_size,
            patience=args.patience,
            use_ph=args.use_ph,
            ph_delta=args.ph_delta,
            ph_lambda=args.ph_lambda,
            dataset=args.dataset,
            seed=args.seed,
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
    print(f"  Max improvement: {max(improvements):+.1f}% "
          f"(H={results[improvements.index(max(improvements))]['horizon']})")
    print(f"  Min improvement: {min(improvements):+.1f}% "
          f"(H={results[improvements.index(min(improvements))]['horizon']})")

    return results


# ==================== Main ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSTR FGL Regression Experiment")
    parser.add_argument("--horizon", type=int, default=5, help="Student horizon H")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight of ground-truth MSE (0=full distillation)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lookback_window", type=int, default=8, help="History length")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test fraction")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--sweep", action="store_true", help="Run horizon sweep")
    parser.add_argument("--sweep_range", type=str, default=None,
                        help="Horizon range for sweep, e.g. '2,31'")
    parser.add_argument("--use_ph", action="store_true", help="Enable Page-Hinkley evaluation")
    parser.add_argument("--ph_delta", type=float, default=0.005, help="PH delta")
    parser.add_argument("--ph_lambda", type=float, default=1.0, help="PH lambda threshold")
    parser.add_argument("--ph_window", type=int, default=50, help="PH retrain window")
    parser.add_argument("--ph_retrain_epochs", type=int, default=3, help="PH retrain epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset", type=str, default="h2o",
                        choices=["h2o", "temperature"],
                        help="Dataset: 'h2o' (smooth oscillation) or 'temperature' (spike detection)")

    args = parser.parse_args()

    if args.sweep:
        sweep_horizons(args)
    else:
        run_fgl_regression(
            student_horizon=args.horizon,
            alpha=args.alpha,
            val_size=args.val_size,
            test_size=args.test_size,
            epochs=args.epochs,
            lookback_window=args.lookback_window,
            batch_size=args.batch_size,
            patience=args.patience,
            use_ph=args.use_ph,
            ph_delta=args.ph_delta,
            ph_lambda=args.ph_lambda,
            dataset=args.dataset,
            seed=args.seed,
        )
