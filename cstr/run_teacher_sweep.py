#!/usr/bin/env python
"""Teacher-quality sweep + distillation validation for continuous PatchTST FGL.

Stage 0 (sweep): vary teacher lookback L_t and architecture; record teacher MSE.
Stage 1 (distill): train a PatchTST student (L_s=20) with distillation from the
best teacher and compare against the PatchTST baseline.

Run on GPU server:
    cd ~/FutureGuidedLearning && uv run python cstr/run_teacher_sweep.py --stage sweep
    cd ~/FutureGuidedLearning && uv run python cstr/run_teacher_sweep.py --stage distill
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fgl_common.baselines import PatchTST  # noqa: E402
from fgl_common.training import device, EarlyStopper  # noqa: E402

_CSTR_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_CSTR_DIR, "results")
_SWEEP_CSV = os.path.join(_RESULTS_DIR, "cstr_teacher_sweep.csv")
_DISTILL_CSV = os.path.join(_RESULTS_DIR, "cstr_strong_teacher_distill.csv")


def _load_data(dataset: str = "h2o") -> np.ndarray:
    filename = "data.pkl" if dataset == "temperature" else "data_h2o.pkl"
    with open(os.path.join(_CSTR_DIR, "data", filename), "rb") as f:
        return np.asarray(pickle.load(f), dtype=np.float64)


# ---------------------------------------------------------------- window utils

def _standardize(x_tr: np.ndarray, y_tr: np.ndarray):
    x_mean = x_tr.mean(axis=0)
    x_std = x_tr.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std())
    if y_std == 0.0:
        y_std = 1.0
    return x_mean, x_std, y_mean, y_std


def _to_tensor(x: np.ndarray, y: np.ndarray, x_mean, x_std, y_mean, y_std):
    xn = (x - x_mean) / x_std
    yn = (y - y_mean) / y_std
    return (torch.tensor(xn, dtype=torch.float32)[:, None, :],
            torch.tensor(yn, dtype=torch.float32))


def build_teacher_windows(arr: np.ndarray, l_student: int, horizon: int, l_teacher: int):
    """Teacher windows that end exactly one step before the student target.

    Student target for window i is y[i + L_s + H - 1].
    Teacher window is x[target - l_t : target].
    """
    x_raw, y_raw = arr[:, 0], arr[:, 1]
    target_idx = np.arange(l_student + horizon - 1, len(y_raw))
    starts = target_idx - l_teacher
    valid = starts >= 0
    target_idx = target_idx[valid]
    starts = starts[valid]
    X = np.stack([x_raw[s:s + l_teacher] for s in starts])
    Y = y_raw[target_idx]
    return X, Y


def build_student_windows(arr: np.ndarray, l_student: int, horizon: int):
    x_raw, y_raw = arr[:, 0], arr[:, 1]
    target_idx = np.arange(l_student + horizon - 1, len(y_raw))
    starts = target_idx - (l_student + horizon - 1)
    X = np.stack([x_raw[s:s + l_student] for s in starts])
    Y = y_raw[target_idx]
    return X, Y


def build_student_teacher_pairs(arr, l_student, horizon, l_teacher):
    """Return (student_X, teacher_X, Y) aligned by target index."""
    x_raw, y_raw = arr[:, 0], arr[:, 1]
    target_idx = np.arange(l_student + horizon - 1, len(y_raw))
    s_starts = target_idx - (l_student + horizon - 1)
    t_starts = target_idx - l_teacher
    valid = (t_starts >= 0) & (s_starts >= 0)
    target_idx = target_idx[valid]
    s_starts = s_starts[valid]
    t_starts = t_starts[valid]
    SX = np.stack([x_raw[s:s + l_student] for s in s_starts])
    TX = np.stack([x_raw[t:t + l_teacher] for t in t_starts])
    Y = y_raw[target_idx]
    return SX, TX, Y


# ---------------------------------------------------------------- training

def _train_regressor(model, x_tr, y_tr, x_val, y_val, *, epochs, batch_size,
                     patience, lr, seed, teacher_x=None, teacher=None, alpha=1.0):
    generator = torch.Generator().manual_seed(seed)
    if teacher_x is None:
        teacher_x = x_tr
    loader = DataLoader(TensorDataset(x_tr, teacher_x, y_tr),
                        batch_size=batch_size, shuffle=True, generator=generator)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopper(patience=patience, min_delta=1e-6)
    best_val, best_state = float("inf"), None
    mse = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xs, xt, yb in loader:
            opt.zero_grad()
            out = model(xs.to(device)).squeeze(-1)
            loss = alpha * mse(out, yb.to(device))
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(xt.to(device)).squeeze(-1)
                loss = loss + (1.0 - alpha) * mse(out, t_out)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = mse(model(x_val.to(device)).squeeze(-1), y_val.to(device)).item()
        if v + 1e-8 < best_val:
            best_val = v
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if stopper.step(v, model):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"val_mse_normalized": best_val}


def _eval_mse(model, x, y, y_std):
    model.eval()
    with torch.no_grad():
        pred = model(x.to(device)).squeeze(-1)
        return float(torch.mean((pred - y.to(device)) ** 2).item() * y_std ** 2)


# ---------------------------------------------------------------- stage 0: sweep

def run_teacher_sweep(args) -> None:
    arr = _load_data(args.dataset)
    rows = []
    print(f"=== Stage 0: teacher sweep on {args.dataset} "
          f"(L_s={args.lookback}, H={args.horizon}) ===", flush=True)

    for l_t in args.teacher_lookbacks:
        for d_model in args.d_models:
            for nhead in args.nheads:
                for lr in args.learning_rates:
                    seed_best = None
                    seed_vals = []
                    for seed in range(args.seeds):
                        TX, TY = build_teacher_windows(arr, args.lookback, args.horizon, l_t)
                        n = len(TX)
                        n_tr = int(n * 0.6)
                        n_val = int(n * 0.2)
                        Xm, Xs, Ym, Ys = _standardize(TX[:n_tr], TY[:n_tr])
                        x_tr, y_tr = _to_tensor(TX[:n_tr], TY[:n_tr], Xm, Xs, Ym, Ys)
                        x_val, y_val = _to_tensor(TX[n_tr:n_tr+n_val], TY[n_tr:n_tr+n_val], Xm, Xs, Ym, Ys)
                        x_te, y_te = _to_tensor(TX[n_tr+n_val:], TY[n_tr+n_val:], Xm, Xs, Ym, Ys)
                        torch.manual_seed(seed)
                        model = PatchTST(l_t, output_size=1, d_model=d_model,
                                         nhead=nhead, dim_feedforward=args.dim_feedforward,
                                         dropout=args.dropout).to(device)
                        info = _train_regressor(model, x_tr, y_tr, x_val, y_val,
                                                epochs=args.epochs, batch_size=args.batch_size,
                                                patience=args.patience, lr=lr, seed=seed)
                        test_mse = _eval_mse(model, x_te, y_te, Ys)
                        seed_vals.append(test_mse)
                    mean_mse = float(np.mean(seed_vals))
                    row = {"dataset": args.dataset, "L_s": args.lookback, "H": args.horizon,
                           "L_teacher": l_t, "d_model": d_model, "nhead": nhead,
                           "dim_feedforward": args.dim_feedforward, "lr": lr,
                           "dropout": args.dropout, "seeds": args.seeds,
                           "teacher_mse_mean": mean_mse,
                           "teacher_mse_sd": float(np.std(seed_vals, ddof=1)) if len(seed_vals) > 1 else 0.0}
                    rows.append(row)
                    print(f"  L_t={l_t:>3} d={d_model:>3} h={nhead} lr={lr:.0e} "
                          f"-> teacher MSE={mean_mse:.4e}", flush=True)

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    fields = list(rows[0].keys())
    tmp = f"{_SWEEP_CSV}.tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, _SWEEP_CSV)
    best = min(rows, key=lambda r: r["teacher_mse_mean"])
    print(f"\nBest teacher config: L_t={best['L_teacher']} d={best['d_model']} "
          f"h={best['nhead']} lr={best['lr']:.0e} "
          f"MSE={best['teacher_mse_mean']:.4e}")
    print(f"-> {_SWEEP_CSV}")


# ---------------------------------------------------------------- stage 1: distill

def _train_baseline_and_student(args, arr, teacher_cfg, alpha, seed):
    L_s, H, L_t = args.lookback, args.horizon, teacher_cfg["L_teacher"]
    SX, TX, Y = build_student_teacher_pairs(arr, L_s, H, L_t)
    n = len(SX)
    n_tr = int(n * 0.6)
    n_val = int(n * 0.2)
    # Student and teacher standardize independently but train-only.
    sx_mean, sx_std, sy_mean, sy_std = _standardize(SX[:n_tr], Y[:n_tr])
    tx_mean, tx_std, _, _ = _standardize(TX[:n_tr], Y[:n_tr])

    sx_tr, y_tr = _to_tensor(SX[:n_tr], Y[:n_tr], sx_mean, sx_std, sy_mean, sy_std)
    tx_tr, _ = _to_tensor(TX[:n_tr], Y[:n_tr], tx_mean, tx_std, sy_mean, sy_std)
    sx_val, y_val = _to_tensor(SX[n_tr:n_tr+n_val], Y[n_tr:n_tr+n_val], sx_mean, sx_std, sy_mean, sy_std)
    sx_te, y_te = _to_tensor(SX[n_tr+n_val:], Y[n_tr+n_val:], sx_mean, sx_std, sy_mean, sy_std)
    tx_te, _ = _to_tensor(TX[n_tr+n_val:], Y[n_tr+n_val:], tx_mean, tx_std, sy_mean, sy_std)

    # --- teacher (frozen) ---
    torch.manual_seed(seed)
    teacher = PatchTST(L_t, output_size=1, d_model=teacher_cfg["d_model"],
                       nhead=teacher_cfg["nhead"],
                       dim_feedforward=teacher_cfg["dim_feedforward"],
                       dropout=teacher_cfg["dropout"]).to(device)
    _train_regressor(teacher, tx_tr, y_tr, tx_tr, y_tr,
                     epochs=args.epochs, batch_size=args.batch_size,
                     patience=args.patience, lr=teacher_cfg["lr"], seed=seed)

    # --- baseline (no distillation) ---
    torch.manual_seed(seed)
    baseline = PatchTST(L_s, output_size=1, d_model=args.student_d_model,
                        nhead=args.student_nhead,
                        dim_feedforward=args.student_dim_feedforward,
                        dropout=args.student_dropout).to(device)
    _train_regressor(baseline, sx_tr, y_tr, sx_val, y_val,
                     epochs=args.epochs, batch_size=args.batch_size,
                     patience=args.patience, lr=args.student_lr, seed=seed)

    # --- student (distilled from teacher) ---
    torch.manual_seed(seed)
    student = PatchTST(L_s, output_size=1, d_model=args.student_d_model,
                       nhead=args.student_nhead,
                       dim_feedforward=args.student_dim_feedforward,
                       dropout=args.student_dropout).to(device)
    _train_regressor(student, sx_tr, y_tr, sx_val, y_val,
                     epochs=args.epochs, batch_size=args.batch_size,
                     patience=args.patience, lr=args.student_lr, seed=seed,
                     teacher_x=tx_tr, teacher=teacher, alpha=alpha)

    baseline_mse = _eval_mse(baseline, sx_te, y_te, sy_std)
    student_mse = _eval_mse(student, sx_te, y_te, sy_std)
    teacher_x_eval_mse = _eval_mse(teacher, tx_te, y_te, sy_std)
    return {
        "alpha": alpha, "seed": seed,
        "teacher_mse_on_teacher_window": teacher_x_eval_mse,
        "baseline_mse": baseline_mse,
        "student_mse": student_mse,
        "improvement_percent": (baseline_mse - student_mse) / baseline_mse * 100.0 if baseline_mse > 0 else 0.0,
    }


def run_distill(args, teacher_cfg) -> None:
    arr = _load_data(args.dataset)
    rows = []
    print(f"\n=== Stage 1: distillation with best teacher "
          f"(L_t={teacher_cfg['L_teacher']}, d={teacher_cfg['d_model']}, "
          f"h={teacher_cfg['nhead']}) ===", flush=True)
    for alpha in args.alphas:
        for seed in range(args.seeds):
            row = _train_baseline_and_student(args, arr, teacher_cfg, alpha, seed)
            row.update({"dataset": args.dataset, "L_s": args.lookback, "H": args.horizon,
                        "L_teacher": teacher_cfg["L_teacher"],
                        "d_model_teacher": teacher_cfg["d_model"],
                        "nhead_teacher": teacher_cfg["nhead"],
                        "dim_feedforward_teacher": teacher_cfg["dim_feedforward"],
                        "dropout_teacher": teacher_cfg["dropout"],
                        "lr_teacher": teacher_cfg["lr"],
                        "student_d_model": args.student_d_model,
                        "student_nhead": args.student_nhead,
                        "student_dim_feedforward": args.student_dim_feedforward,
                        "student_dropout": args.student_dropout,
                        "student_lr": args.student_lr,
                        "epochs": args.epochs, "batch_size": args.batch_size,
                        "patience": args.patience})
            rows.append(row)
            print(f"  alpha={alpha:.2f} seed={seed}: teacher={row['teacher_mse_on_teacher_window']:.4e} "
                  f"baseline={row['baseline_mse']:.4e} student={row['student_mse']:.4e} "
                  f"({row['improvement_percent']:+.2f}%)", flush=True)

    # summary
    summary = []
    groups = defaultdict(list)
    for row in rows:
        groups[row["alpha"]].append(row)
    for alpha, group in sorted(groups.items()):
        base = np.mean([r["baseline_mse"] for r in group])
        stud = np.mean([r["student_mse"] for r in group])
        teach = np.mean([r["teacher_mse_on_teacher_window"] for r in group])
        summary.append({"alpha": alpha, "n_seeds": len(group),
                        "teacher_mse_mean": teach,
                        "baseline_mse_mean": base,
                        "student_mse_mean": stud,
                        "student_over_baseline": stud / base if base > 0 else float("nan"),
                        "improvement_percent": (base - stud) / base * 100.0 if base > 0 else 0.0})

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    fields = list(rows[0].keys())
    tmp = f"{_DISTILL_CSV}.tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, _DISTILL_CSV)
    print("\nSummary (physical test MSE)")
    for item in summary:
        print(f"  alpha={item['alpha']:.2f}: teacher={item['teacher_mse_mean']:.4e} "
              f"baseline={item['baseline_mse_mean']:.4e} student={item['student_mse_mean']:.4e} "
              f"S/B={item['student_over_baseline']:.3f} ({item['improvement_percent']:+.2f}%)")
    print(f"-> {_DISTILL_CSV}")


# ---------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sweep", "distill", "both"), default="both")
    parser.add_argument("--dataset", choices=("h2o", "temperature"), default="h2o")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--teacher_lookbacks", type=int, nargs="+", default=[20, 25, 31, 40, 60])
    parser.add_argument("--d_models", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--nheads", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--learning_rates", type=float, nargs="+", default=[5e-4, 1e-3])
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    # student hyperparameters (default: strong enough to match baseline arm)
    parser.add_argument("--student_d_model", type=int, default=32)
    parser.add_argument("--student_nhead", type=int, default=4)
    parser.add_argument("--student_dim_feedforward", type=int, default=64)
    parser.add_argument("--student_dropout", type=float, default=0.1)
    parser.add_argument("--student_lr", type=float, default=5e-4)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    args = parser.parse_args()

    if args.stage in ("sweep", "both"):
        run_teacher_sweep(args)

    if args.stage in ("distill", "both"):
        # load the best sweep result if the CSV exists, otherwise default
        if os.path.exists(_SWEEP_CSV):
            with open(_SWEEP_CSV, newline="") as f:
                sweep_rows = list(csv.DictReader(f))
            best = min(sweep_rows, key=lambda r: float(r["teacher_mse_mean"]))
            teacher_cfg = {
                "L_teacher": int(best["L_teacher"]),
                "d_model": int(best["d_model"]),
                "nhead": int(best["nhead"]),
                "dim_feedforward": int(best["dim_feedforward"]),
                "dropout": float(best["dropout"]),
                "lr": float(best["lr"]),
            }
        else:
            teacher_cfg = {"L_teacher": 31, "d_model": 128, "nhead": 8,
                           "dim_feedforward": 256, "dropout": 0.05, "lr": 1e-3}
        run_distill(args, teacher_cfg)


if __name__ == "__main__":
    main()
