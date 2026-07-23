#!/usr/bin/env python
"""
FGL (Future-Guided Learning) experiment on CSTR oscillatory time series.

CSTR = Continuously Stirred Tank Reactor
Periodic H2/O2 combustion with temperature oscillations (770~2000 K).

Usage:
  # Single run (H=5, default params)
  uv run cstr/exp/fgl_cstr.py --horizon 5 --alpha 0.5

  # Horizon sweep
  uv run cstr/exp/fgl_cstr.py --sweep --alpha 0.5 --epochs 30

  # With Page-Hinkley drift detection
  uv run cstr/exp/fgl_cstr.py --sweep --use_ph --alpha 0.0
"""

import argparse
import pickle
import sys
import os
from collections import deque

# Add mackey_glass/ to path so we can import RNN, KL, create_time_series_dataset
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MG_UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "mackey_glass")
sys.path.insert(0, MG_UTILS_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from utils.utils import RNN, create_time_series_dataset, KL

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
DATA_PATH = _AVAILABLE_DATASETS["h2o"]   # Default: H2O mass fraction (continuous oscillation)

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
        else:
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
    celoss = nn.CrossEntropyLoss()
    model.eval()

    if not use_ph:
        total = 0.0
        with torch.no_grad():
            for _, x, y in loader:
                x = x.float().to(device).view(-1, 1, lookback_window)
                y_int = y.long().to(device).squeeze(-1)
                pred = model(x).argmax(dim=1).float()
                total += mse_loss(pred, y_int.float()).item()
        return total / len(loader)

    # Page–Hinkley–driven evaluation
    optimizer = optim.Adam(model.parameters(), lr=lr)
    state = {"m": 0.0, "PH": 0.0, "t": 0}
    window = deque(maxlen=window_size)
    errors = []

    for _, x, y in loader:
        x = x.float().to(device).view(-1, 1, lookback_window)
        y_int = y.long().to(device).squeeze(-1)
        y_float = y_int.float()

        with torch.no_grad():
            pred_class = model(x).argmax(dim=1)
        err = mse_loss(pred_class.float(), y_float).item()
        errors.append(err)
        state = page_hinkley_update(err, state, delta)
        window.append((x.cpu(), y_int.cpu()))

        if state["PH"] > lambda_thr and len(window) == window_size:
            model.train()
            for _ in range(retrain_epochs):
                for wx, wy_int in window:
                    wx = wx.to(device)
                    wy = wy_int.to(device)
                    out = model(wx)
                    loss = celoss(out, wy)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            model.eval()
            state = {"m": 0.0, "PH": 0.0, "t": 0}
            window.clear()

    return sum(errors) / len(errors)


# -------------------- Core FGL Training --------------------
def run_fgl(
    student_horizon,
    alpha,
    num_bins=50,
    val_size=0.2,
    test_size=0.2,
    epochs=50,
    temperature=4,
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
    torch.manual_seed(seed)
    hidden_size = 128
    output_size = num_bins
    num_layers = 2
    lr = 1e-4

    # Load CSTR data
    data_path = _AVAILABLE_DATASETS[dataset]
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Compute shared bin edges from full y_windows (teacher H=1 has most windows,
    # giving the widest coverage of target values). This fixes the bug where
    # teacher and student used different bin edges due to different train splits.
    x_raw = np.array([float(pt[0]) for pt in data])
    y_raw = np.array([float(pt[1]) for pt in data])
    all_y_windows = []
    for i in range(len(x_raw) - lookback_window - 1 + 1):
        all_y_windows.append(y_raw[i + lookback_window + 1 - 1])
    all_y = np.array(all_y_windows)
    shared_bin_edges = np.linspace(all_y.min(), all_y.max(), num_bins - 1)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Dataset={dataset}  H={student_horizon:2d}  α={alpha:.2f}  T={temperature:.1f}  "
              f"Bins={num_bins}  Epochs={epochs}  Lookback={lookback_window}")
        print(f"Bin edges: [{all_y.min():.4f}, {all_y.max():.4f}] → {num_bins} bins")
        print(f"{'='*50}")

    # Teacher: predicts 1 step ahead (offset=H-1 to align with Student target)
    teacher_train, teacher_val, teacher_test, _, _ = create_time_series_dataset(
        data=data,
        lookback_window=lookback_window,
        forecasting_horizon=1,
        num_bins=num_bins,
        val_size=val_size,
        test_size=test_size,
        offset=student_horizon - 1,
        batch_size=batch_size,
        bin_edges=shared_bin_edges,
    )
    # Student: predicts H steps ahead
    student_train, student_val, student_test, _, _ = create_time_series_dataset(
        data=data,
        lookback_window=lookback_window,
        forecasting_horizon=student_horizon,
        num_bins=num_bins,
        val_size=val_size,
        test_size=test_size,
        offset=0,
        batch_size=batch_size,
        bin_edges=shared_bin_edges,
    )

    mse = nn.MSELoss()
    celoss = nn.CrossEntropyLoss()

    # ────────── Teacher training ──────────
    teacher = RNN(lookback_window, hidden_size, output_size, num_layers).to(device)
    opt_t = optim.Adam(teacher.parameters(), lr=lr)
    stop_t = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        teacher.train()
        for _, x, y in teacher_train:
            x = x.float().to(device).view(-1, 1, lookback_window)
            y = y.long().to(device)
            opt_t.zero_grad()
            celoss(teacher(x), y).backward()
            opt_t.step()
        teacher.eval()
        with torch.no_grad():
            val_loss = sum(
                celoss(teacher(x.float().to(device).view(-1, 1, lookback_window)), y.long().to(device)).item()
                for _, x, y in teacher_val
            ) / len(teacher_val)
        if stop_t.step(val_loss, teacher):
            break
    stop_t.restore(teacher)

    # ────────── Baseline training ──────────
    baseline = RNN(lookback_window, hidden_size, output_size, num_layers).to(device)
    opt_b = optim.Adam(baseline.parameters(), lr=lr)
    stop_b = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        baseline.train()
        for _, x, y in student_train:
            x = x.float().to(device).view(-1, 1, lookback_window)
            y = y.long().to(device)
            opt_b.zero_grad()
            celoss(baseline(x), y).backward()
            opt_b.step()
        baseline.eval()
        with torch.no_grad():
            val_loss = sum(
                celoss(baseline(x.float().to(device).view(-1, 1, lookback_window)), y.long().to(device)).item()
                for _, x, y in student_val
            ) / len(student_val)
        if stop_b.step(val_loss, baseline):
            break
    stop_b.restore(baseline)

    # ────────── Student training (FGL distillation) ──────────
    student = RNN(lookback_window, hidden_size, output_size, num_layers).to(device)
    opt_s = optim.Adam(student.parameters(), lr=lr)
    stop_s = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        student.train()
        for (_, x_s, y_s), (_, x_t, _) in zip(student_train, teacher_train):
            x_s = x_s.float().to(device).view(-1, 1, lookback_window)
            targets = y_s.long().to(device)
            outputs = student(x_s)
            x_t = x_t.float().to(device).view(-1, 1, lookback_window)
            with torch.no_grad():
                logits = teacher(x_t)
            loss = alpha * celoss(outputs, targets) + KL(outputs, logits, temperature, alpha)
            opt_s.zero_grad()
            loss.backward()
            opt_s.step()
        student.eval()
        with torch.no_grad():
            val_loss = sum(
                celoss(student(x.float().to(device).view(-1, 1, lookback_window)), y.long().to(device)).item()
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

    print(f"  Teacher:  {teacher_mse:.4f}")
    print(f"  Baseline: {baseline_mse:.4f}")
    print(f"  Student:  {student_mse:.4f}  (Δ={improvement:+.1f}%)")

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
        # Default: 2 to ~30, but CSTR has only 3000 points
        # With lookback=8, max H ≈ 3000-8-2 ≈ 2990, but 2~30 is fine
        horizon_range = "2,31"

    start, end = map(int, horizon_range.split(","))
    horizons = list(range(start, end))

    print(f"\n{'='*60}")
    print(f"CSTR FGL Horizon Sweep: H={start}→{end-1}  α={args.alpha}  T={args.temperature}")
    print(f"{'='*60}")

    for h in horizons:
        r = run_fgl(
            student_horizon=h,
            alpha=args.alpha,
            num_bins=args.num_bins,
            epochs=args.epochs,
            temperature=args.temperature,
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

    # Summary
    improvements = [r["improvement"] for r in results]
    positive = sum(1 for v in improvements if v > 0)
    avg_imp = np.mean(improvements)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  FGL wins:        {positive}/{len(results)}")
    print(f"  Avg improvement: {avg_imp:+.1f}%")
    print(f"  Max improvement: {max(improvements):+.1f}% (H={results[improvements.index(max(improvements))]['horizon']})")
    print(f"  Min improvement: {min(improvements):+.1f}% (H={results[improvements.index(min(improvements))]['horizon']})")

    return results


# ==================== Main ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSTR FGL Experiment")
    parser.add_argument("--horizon", type=int, default=5, help="Student horizon H")
    parser.add_argument("--alpha", type=float, default=0.5, help="CE weight α (0=full distillation)")
    parser.add_argument("--num_bins", type=int, default=50, help="Discretization bins")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--temperature", type=float, default=4, help="Distillation temperature")
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
    parser.add_argument("--dataset", type=str, default="h2o", choices=["h2o", "temperature"],
                        help="Dataset: 'h2o' (smooth oscillation) or 'temperature' (spike detection)")

    args = parser.parse_args()

    if args.sweep:
        sweep_horizons(args)
    else:
        run_fgl(
            student_horizon=args.horizon,
            alpha=args.alpha,
            num_bins=args.num_bins,
            val_size=args.val_size,
            test_size=args.test_size,
            epochs=args.epochs,
            temperature=args.temperature,
            lookback_window=args.lookback_window,
            batch_size=args.batch_size,
            patience=args.patience,
            use_ph=args.use_ph,
            ph_delta=args.ph_delta,
            ph_lambda=args.ph_lambda,
            dataset=args.dataset,
            seed=args.seed,
        )
