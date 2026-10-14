#!/usr/bin/env python
"""Student architecture sweep for continuous FGL distillation.

Fixes a strong teacher (PatchTST, L_t=60) and sweeps student architecture,
capacity, and distillation weight α to identify the best student configuration.

Phases:
  teacher  — train and save the strong teacher once
  arch     — architecture × capacity × α sweep
  ls       — student lookback sweep with best architecture
  all      — run everything

Run on GPU server:
    cd ~/FutureGuidedLearning && uv run python cstr/run_student_sweep.py --phase all
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
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
_CSV_PATH = os.path.join(_RESULTS_DIR, "cstr_student_sweep.csv")
_TEACHER_PATH = os.path.join(_RESULTS_DIR, "strong_teacher_state.pt")

# Fixed strong teacher config (from cstr_teacher_sweep.csv best result)
TEACHER_CFG = {
    "L_teacher": 60, "d_model": 64, "nhead": 8,
    "dim_feedforward": 256, "dropout": 0.05, "lr": 1e-3,
}

# ============================================================ student models

class RNNRegressor(nn.Module):
    """Vanilla RNN for univariate regression. Input: (batch, 1, L)."""
    def __init__(self, lookback: int, hidden: int = 128, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.RNN(1, hidden, num_layers, batch_first=True,
                          dropout=0.1 if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x.transpose(1, 2))       # (batch, L, hidden)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMRegressor(nn.Module):
    """LSTM for univariate regression. Input: (batch, 1, L)."""
    def __init__(self, lookback: int, hidden: int = 128, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden, num_layers, batch_first=True,
                           dropout=0.1 if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x.transpose(1, 2))
        return self.head(out[:, -1, :]).squeeze(-1)


class TransformerRegressor(nn.Module):
    """Light Transformer for univariate regression. Input: (batch, 1, L)."""
    def __init__(self, lookback: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_ff: int = 128):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)                     # (batch, L, 1)
        tokens = self.input_proj(x) + self.pos_embed
        encoded = self.encoder(tokens)
        return self.head(encoded.mean(dim=1)).squeeze(-1)


# Capacity presets: (label -> kwargs per architecture)
STUDENT_CONFIGS = {
    "rnn": {
        "small":  {"hidden": 64,  "num_layers": 2},
        "medium": {"hidden": 128, "num_layers": 2},
    },
    "lstm": {
        "small":  {"hidden": 64,  "num_layers": 2},
        "medium": {"hidden": 128, "num_layers": 2},
    },
    "transformer": {
        "small":  {"d_model": 32, "nhead": 4, "num_layers": 2, "dim_ff": 64},
        "medium": {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_ff": 128},
    },
    "patchtst": {
        "small":  {"d_model": 32, "nhead": 4, "dim_feedforward": 64},
        "medium": {"d_model": 64, "nhead": 4, "dim_feedforward": 128},
    },
}

STUDENT_LR = {"rnn": 1e-3, "lstm": 1e-3, "transformer": 5e-4, "patchtst": 5e-4}


def make_student(arch: str, lookback: int, capacity: str) -> nn.Module:
    if arch == "patchtst":
        return PatchTST(lookback, output_size=1, **STUDENT_CONFIGS[arch][capacity]).to(device)
    if arch == "rnn":
        return RNNRegressor(lookback, **STUDENT_CONFIGS[arch][capacity]).to(device)
    if arch == "lstm":
        return LSTMRegressor(lookback, **STUDENT_CONFIGS[arch][capacity]).to(device)
    if arch == "transformer":
        return TransformerRegressor(lookback, **STUDENT_CONFIGS[arch][capacity]).to(device)
    raise ValueError(f"unknown arch {arch}")


# ============================================================ data pipeline

def _load_data(dataset: str = "h2o") -> np.ndarray:
    filename = "data.pkl" if dataset == "temperature" else "data_h2o.pkl"
    with open(os.path.join(_CSTR_DIR, "data", filename), "rb") as f:
        return np.asarray(pickle.load(f), dtype=np.float64)


def _standardize(x: np.ndarray, y: np.ndarray):
    xm, xs = x.mean(axis=0), x.std(axis=0)
    xs[xs == 0] = 1.0
    ym, ys = float(y.mean()), float(y.std())
    if ys == 0:
        ys = 1.0
    return xm, xs, ym, ys


def _tensor(x, y, xm, xs, ym, ys):
    return (torch.tensor((x - xm) / xs, dtype=torch.float32)[:, None, :],
            torch.tensor((y - ym) / ys, dtype=torch.float32))


def build_windows(arr, l_student, horizon, l_teacher):
    """Build aligned (student_x, teacher_x, y) for all valid targets."""
    x_raw, y_raw = arr[:, 0], arr[:, 1]
    target_idx = np.arange(l_student + horizon - 1, len(y_raw))
    s_start = target_idx - (l_student + horizon - 1)
    t_start = target_idx - l_teacher
    valid = (s_start >= 0) & (t_start >= 0) & (target_idx < len(y_raw))
    target_idx = target_idx[valid]
    s_start = s_start[valid]
    t_start = t_start[valid]
    SX = np.stack([x_raw[s:s + l_student] for s in s_start])
    TX = np.stack([x_raw[t:t + l_teacher] for t in t_start])
    Y = y_raw[target_idx]
    return SX, TX, Y


# ============================================================ training

def _train(model, sx_tr, y_tr, sx_val, y_val, *, tx_tr=None, teacher=None,
           alpha=1.0, epochs=200, batch_size=64, patience=15, lr=5e-4, seed=42):
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(sx_tr, y_tr), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    stop = EarlyStopper(patience=patience, min_delta=1e-6)
    mse = nn.MSELoss()
    best_val, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for xs, yb in loader:
            opt.zero_grad()
            out = model(xs.to(device))
            loss = alpha * mse(out, yb.to(device))
            if teacher is not None and tx_tr is not None:
                # Teacher batch: same targets, teacher windows
                # (we shuffle identically by using the same permutation)
                # Simpler: for each batch of student x, find matching teacher x
                # by index — but DataLoader loses indices. Instead, precompute
                # teacher predictions for ALL training samples at epoch start.
                pass  # handled below
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = mse(model(sx_val.to(device)), y_val.to(device)).item()
        if v + 1e-8 < best_val:
            best_val = v
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if stop.step(v, model):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _train_with_distill(model, sx_tr, tx_tr, y_tr, sx_val, y_val, teacher,
                        alpha, *, epochs=200, batch_size=64, patience=15,
                        lr=5e-4, seed=42):
    """Train student with MSE distillation from frozen teacher."""
    torch.manual_seed(seed)
    dataset = TensorDataset(sx_tr, tx_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    stop = EarlyStopper(patience=patience, min_delta=1e-6)
    mse = nn.MSELoss()
    best_val, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for xs, xt, yb in loader:
            opt.zero_grad()
            out = model(xs.to(device))
            with torch.no_grad():
                t_out = teacher(xt.to(device))
            loss = alpha * mse(out, yb.to(device)) + (1 - alpha) * mse(out, t_out)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = mse(model(sx_val.to(device)), y_val.to(device)).item()
        if v + 1e-8 < best_val:
            best_val = v
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if stop.step(v, model):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _eval(model, x, y, y_std):
    model.eval()
    with torch.no_grad():
        pred = model(x.to(device))
        return float(torch.mean((pred - y.to(device)) ** 2).item() * y_std ** 2)


# ============================================================ teacher

def train_or_load_teacher(arr, l_student, horizon, seed=42, force=False):
    """Train the strong teacher once and cache its state_dict."""
    L_t = TEACHER_CFG["L_teacher"]
    if os.path.exists(_TEACHER_PATH) and not force:
        ckpt = torch.load(_TEACHER_PATH, map_location=device)
        teacher = PatchTST(L_t, output_size=1, d_model=ckpt["d_model"],
                           nhead=ckpt["nhead"], dim_feedforward=ckpt["dim_ff"],
                           dropout=ckpt["dropout"]).to(device)
        teacher.load_state_dict(ckpt["state_dict"])
        teacher.eval()
        return teacher, ckpt["teacher_mse"]
    # Build teacher windows and train
    SX, TX, Y = build_windows(arr, l_student, horizon, L_t)
    n = len(SX)
    n_tr = int(n * 0.6)
    n_val = int(n * 0.2)
    txm, txs, ym, ys = _standardize(TX[:n_tr], Y[:n_tr])
    tx_tr, ty_tr = _tensor(TX[:n_tr], Y[:n_tr], txm, txs, ym, ys)
    tx_val, ty_val = _tensor(TX[n_tr:n_tr+n_val], Y[n_tr:n_tr+n_val], txm, txs, ym, ys)
    tx_te, ty_te = _tensor(TX[n_tr+n_val:], Y[n_tr+n_val:], txm, txs, ym, ys)
    torch.manual_seed(seed)
    teacher = PatchTST(L_t, output_size=1, d_model=TEACHER_CFG["d_model"],
                       nhead=TEACHER_CFG["nhead"],
                       dim_feedforward=TEACHER_CFG["dim_feedforward"],
                       dropout=TEACHER_CFG["dropout"]).to(device)
    _train(teacher, tx_tr, ty_tr, tx_val, ty_val,
           epochs=200, batch_size=64, patience=15, lr=TEACHER_CFG["lr"], seed=seed)
    teacher_mse = _eval(teacher, tx_te, ty_te, ys)
    os.makedirs(os.path.dirname(_TEACHER_PATH), exist_ok=True)
    torch.save({"state_dict": teacher.state_dict(),
                "d_model": TEACHER_CFG["d_model"], "nhead": TEACHER_CFG["nhead"],
                "dim_ff": TEACHER_CFG["dim_feedforward"], "dropout": TEACHER_CFG["dropout"],
                "teacher_mse": teacher_mse}, _TEACHER_PATH)
    return teacher, teacher_mse


# ============================================================ phase: arch sweep

def run_arch_sweep(args, arr, teacher, teacher_mse):
    l_s, H = args.lookback, args.horizon
    L_t = TEACHER_CFG["L_teacher"]
    SX, TX, Y = build_windows(arr, l_s, H, L_t)
    n = len(SX)
    n_tr = int(n * 0.6)
    n_val = int(n * 0.2)
    sxm, sxs, ym, ys = _standardize(SX[:n_tr], Y[:n_tr])
    txm, txs, _, _ = _standardize(TX[:n_tr], Y[:n_tr])
    sx_tr, y_tr = _tensor(SX[:n_tr], Y[:n_tr], sxm, sxs, ym, ys)
    tx_tr, _ = _tensor(TX[:n_tr], Y[:n_tr], txm, txs, ym, ys)
    sx_val, y_val = _tensor(SX[n_tr:n_tr+n_val], Y[n_tr:n_tr+n_val], sxm, sxs, ym, ys)
    sx_te, y_te = _tensor(SX[n_tr+n_val:], Y[n_tr+n_val:], sxm, sxs, ym, ys)

    rows = []
    total = len(STUDENT_CONFIGS) * 2 * len(args.alphas) * args.seeds
    done = 0
    print(f"=== Student architecture sweep ===")
    print(f"Teacher: PatchTST L_t={L_t} d={TEACHER_CFG['d_model']} h={TEACHER_CFG['nhead']} "
          f"MSE={teacher_mse:.4e}")
    print(f"Student task: L_s={l_s} H={H}, {total} runs total\n")

    for arch in args.architectures:
        for cap in args.capacities:
            lr = STUDENT_LR[arch]
            for seed in range(args.seeds):
                # Baseline (no distillation)
                torch.manual_seed(seed)
                baseline = make_student(arch, l_s, cap)
                _train(baseline, sx_tr, y_tr, sx_val, y_val,
                       epochs=args.epochs, batch_size=args.batch_size,
                       patience=args.patience, lr=lr, seed=seed)
                baseline_mse = _eval(baseline, sx_te, y_te, ys)

                # Students with distillation at each α
                for alpha in args.alphas:
                    torch.manual_seed(seed)
                    student = make_student(arch, l_s, cap)
                    _train_with_distill(student, sx_tr, tx_tr, y_tr, sx_val, y_val,
                                        teacher, alpha,
                                        epochs=args.epochs, batch_size=args.batch_size,
                                        patience=args.patience, lr=lr, seed=seed)
                    student_mse = _eval(student, sx_te, y_te, ys)
                    imp = (baseline_mse - student_mse) / baseline_mse * 100 if baseline_mse > 0 else 0
                    row = {"phase": "arch", "student_arch": arch, "capacity": cap,
                           "L_s": l_s, "H": H, "alpha": alpha, "seed": seed,
                           "teacher_mse": teacher_mse, "baseline_mse": baseline_mse,
                           "student_mse": student_mse, "improvement_percent": imp}
                    rows.append(row)
                    done += 1
                    print(f"  [{done:>3}/{total}] {arch:>12}/{cap:<7} α={alpha:.2f} seed={seed}: "
                          f"baseline={baseline_mse:.4e} student={student_mse:.4e} ({imp:+.1f}%)",
                          flush=True)
    _save_rows(rows, args)
    return rows


# ============================================================ phase: L_s sweep

def run_ls_sweep(args, arr, teacher, teacher_mse):
    """Sweep student lookback with the best architecture from arch phase."""
    best_arch, best_cap = _best_from_csv()
    if best_arch is None:
        best_arch, best_cap = "patchtst", "small"
    L_t = TEACHER_CFG["L_teacher"]
    H = args.horizon
    print(f"\n=== Student lookback sweep (arch={best_arch}/{best_cap}) ===")
    rows = []
    total = len(args.student_lookbacks) * args.seeds
    done = 0
    for l_s in args.student_lookbacks:
        if l_s + H - 1 > L_t:
            print(f"  skipping L_s={l_s}: L_s+H-1={l_s+H-1} > L_t={L_t} (teacher no longer superset)")
            continue
        SX, TX, Y = build_windows(arr, l_s, H, L_t)
        n = len(SX)
        n_tr = int(n * 0.6)
        n_val = int(n * 0.2)
        sxm, sxs, ym, ys = _standardize(SX[:n_tr], Y[:n_tr])
        txm, txs, _, _ = _standardize(TX[:n_tr], Y[:n_tr])
        sx_tr, y_tr = _tensor(SX[:n_tr], Y[:n_tr], sxm, sxs, ym, ys)
        tx_tr, _ = _tensor(TX[:n_tr], Y[:n_tr], txm, txs, ym, ys)
        sx_val, y_val = _tensor(SX[n_tr:n_tr+n_val], Y[n_tr:n_tr+n_val], sxm, sxs, ym, ys)
        sx_te, y_te = _tensor(SX[n_tr+n_val:], Y[n_tr+n_val:], sxm, sxs, ym, ys)
        lr = STUDENT_LR[best_arch]
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            baseline = make_student(best_arch, l_s, best_cap)
            _train(baseline, sx_tr, y_tr, sx_val, y_val,
                   epochs=args.epochs, batch_size=args.batch_size,
                   patience=args.patience, lr=lr, seed=seed)
            baseline_mse = _eval(baseline, sx_te, y_te, ys)
            torch.manual_seed(seed)
            student = make_student(best_arch, l_s, best_cap)
            _train_with_distill(student, sx_tr, tx_tr, y_tr, sx_val, y_val,
                                teacher, args.alpha,
                                epochs=args.epochs, batch_size=args.batch_size,
                                patience=args.patience, lr=lr, seed=seed)
            student_mse = _eval(student, sx_te, y_te, ys)
            imp = (baseline_mse - student_mse) / baseline_mse * 100 if baseline_mse > 0 else 0
            row = {"phase": "ls", "student_arch": best_arch, "capacity": best_cap,
                   "L_s": l_s, "H": H, "alpha": args.alpha, "seed": seed,
                   "teacher_mse": teacher_mse, "baseline_mse": baseline_mse,
                   "student_mse": student_mse, "improvement_percent": imp}
            rows.append(row)
            done += 1
            print(f"  [{done:>3}/{total}] L_s={l_s:>3} seed={seed}: "
                  f"baseline={baseline_mse:.4e} student={student_mse:.4e} ({imp:+.1f}%)",
                  flush=True)
    _save_rows(rows, args)
    return rows


def _best_from_csv():
    """Read the arch sweep results and find the best (arch, cap) by student MSE at α=0.9."""
    if not os.path.exists(_CSV_PATH):
        return None, None
    with open(_CSV_PATH, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["phase"] == "arch" and float(r["alpha"]) == 0.9]
    if not rows:
        return None, None
    groups = defaultdict(list)
    for r in rows:
        groups[(r["student_arch"], r["capacity"])].append(float(r["student_mse"]))
    best = min(groups, key=lambda k: sum(groups[k]) / len(groups[k]))
    return best


# ============================================================ output

def _save_rows(new_rows, args):
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    existing = []
    if os.path.exists(_CSV_PATH) and not args.fresh:
        with open(_CSV_PATH, newline="") as f:
            existing = list(csv.DictReader(f))
    all_rows = existing + new_rows
    fields = ["phase", "student_arch", "capacity", "L_s", "H", "alpha", "seed",
              "teacher_mse", "baseline_mse", "student_mse", "improvement_percent"]
    tmp = f"{_CSV_PATH}.tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(tmp, _CSV_PATH)
    print(f"  -> {_CSV_PATH}")


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["phase"], r["student_arch"], r["capacity"], r["alpha"])].append(r)
    print(f"\n{'='*100}")
    print(f"SUMMARY (mean over seeds)")
    print(f"{'='*100}")
    print(f"{'phase':>6} {'arch':>12} {'cap':>8} {'α':>5} {'baseline':>10} {'student':>10} {'S/B':>6} {'Δ%':>8} {'std':>9}")
    print("-" * 100)
    for key in sorted(groups):
        g = groups[key]
        b = np.mean([r["baseline_mse"] for r in g])
        s = np.mean([r["student_mse"] for r in g])
        sd = np.std([r["student_mse"] for r in g])
        imp = (b - s) / b * 100 if b > 0 else 0
        ratio = s / b if b > 0 else float("nan")
        print(f"{key[0]:>6} {key[1]:>12} {key[2]:>8} {key[3]:>5.2f} "
              f"{b:>10.4e} {s:>10.4e} {ratio:>6.3f} {imp:>+7.1f}% {sd:>9.2e}")


# ============================================================ CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("teacher", "arch", "ls", "all"), default="all")
    parser.add_argument("--dataset", choices=("h2o", "temperature"), default="h2o")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--architectures", nargs="+",
                        default=["rnn", "lstm", "transformer", "patchtst"])
    parser.add_argument("--capacities", nargs="+", default=["small", "medium"])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--student_lookbacks", type=int, nargs="+", default=[20, 25, 31, 40])
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--fresh", action="store_true", help="overwrite CSV instead of appending")
    args = parser.parse_args()

    arr = _load_data(args.dataset)

    if args.phase in ("teacher", "arch", "ls", "all"):
        teacher, teacher_mse = train_or_load_teacher(arr, args.lookback, args.horizon,
                                                     force=(args.phase == "teacher"))
        print(f"Teacher loaded/trained: MSE={teacher_mse:.4e}")

    if args.phase in ("arch", "all"):
        rows = run_arch_sweep(args, arr, teacher, teacher_mse)
        summarize(rows)

    if args.phase in ("ls", "all"):
        rows = run_ls_sweep(args, arr, teacher, teacher_mse)
        summarize(rows)


if __name__ == "__main__":
    main()
