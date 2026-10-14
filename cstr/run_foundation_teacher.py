#!/usr/bin/env python
"""Foundation-model teacher for CSTR: pretrain on multiple datasets, finetune per-dataset.

Task 1: Test the strong-teacher FGL method (L_t >= L_s+H-1, weak distillation)
        across multiple CSTR dataset families (periodic, forced, dual, delayed).
Task 2: Test whether a foundation teacher pretrained on all h2o datasets,
        then finetuned on a single dataset, produces a stronger per-dataset
        teacher than training from scratch.

Pipeline (per target dataset):
  1. Pretrain a foundation PatchTST teacher on ALL other h2o datasets
     (leave-one-dataset-out).
  2. Finetune it on the target dataset (teacher task: 1-step, L_t=L_s+H-1).
  3. Compare against a from-scratch teacher on the same target.
  4. Distill the better teacher into a PatchTST student (L_s, alpha=0.9).
  5. Compare: pretrained-finetuned teacher vs scratch teacher,
     and student(better teacher) vs baseline.

Run on GPU:
    cd ~/FutureGuidedLearning && uv run python cstr/run_foundation_teacher.py --phase all
"""
from __future__ import annotations

import argparse
import csv
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
_CSV_PATH = os.path.join(_RESULTS_DIR, "cstr_foundation_teacher.csv")

# Best teacher config from cstr_teacher_sweep.csv
TEACHER_CFG = {"L_t": 31, "d_model": 128, "nhead": 8, "dim_ff": 256, "dropout": 0.05, "lr": 1e-3}
# Best student config from cstr_student_sweep.csv
STUDENT_CFG = {"d_model": 32, "nhead": 4, "dim_ff": 64, "dropout": 0.1, "lr": 5e-4}
ALPHA = 0.9  # weak distillation (best from teacher sweep)
EPOCHS_PRETRAIN = 100
EPOCHS_FINETUNE = 100
EPOCHS_STUDENT = 200
PATIENCE = 15
BATCH = 64

FAMILIES = {
    "periodic": ["data_h2o.pkl"],
    "forced": [f"data_forced_h2o_A{a}_f0.05.pkl" for a in (0.3, 0.5, 0.7)],
    "dual": ["data_dual_h2o_Vb2.0e-05_rc0.1.pkl", "data_dual_h2o_Vb5.0e-05_rc0.1.pkl",
             "data_dual_h2o_Vb5.0e-05_rc0.2.pkl", "data_dual_h2o_Vb5.0e-05_rc0.3.pkl",
             "data_dual_h2o_Vb5.0e-05_rc0.5.pkl"],
    "delayed": [f"data_delayed_stable_h2o_tau{t}_s1_A0.9_b0.03.pkl"
                for t in (30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 100, 120, 150)],
    "multi_delay": [f"data_complex_multidelay_tau{t}.pkl"
                    for t in ("30_70", "50_120", "30_70_120")],
    "forced_delay": [f"data_complex_forceddelay_tau{t}_A{a}_f{f}.pkl"
                     for (t, a, f) in ((50, "0.05", "0.03"), (50, "0.1", "0.05"),
                                       (80, "0.08", "0.02"), (100, "0.05", "0.07"))],
    "param_mod": [f"data_complex_parammod_tau{t}_V{vs}-{ve}_T{mp}.pkl"
                  for (t, vs, ve, mp) in (("50", "8e-06", "1e-05", "200"),
                                          ("80", "8e-06", "1e-05", "150"),
                                          ("50", "6e-06", "2e-05", "300"))],
    "noisy": [f"data_complex_noisy_tau{t}_s{n}.pkl"
              for (t, n) in ((50, "0.01"), (50, "0.02"), (100, "0.02"))],
}

ALL_H2O = [f for files in FAMILIES.values() for f in files]


def _load(name):
    with open(os.path.join(_CSTR_DIR, "data", name), "rb") as f:
        return np.asarray(pickle.load(f), dtype=np.float64)


def _windows(arr, l_w, horizon, teacher_offset):
    """Build (input_x, y) windows: input x[t:t+l_w], target y[t+l_w+horizon-1].
    If teacher_offset>0, shift the window forward by that many steps."""
    x_raw, y_raw = arr[:, 0], arr[:, 1]
    n = len(y_raw) - l_w - horizon + 1
    starts = np.arange(n)
    if teacher_offset > 0:
        starts = starts + teacher_offset
        valid = starts + l_w <= len(x_raw)
        starts = starts[valid]
        # target index stays the same (aligned)
        targets = np.arange(n)[valid] + l_w + horizon - 1
    else:
        targets = starts + l_w + horizon - 1
    X = np.stack([x_raw[s:s + l_w] for s in starts])
    Y = y_raw[targets]
    return X, Y


def _standardize(x, y):
    xm, xs = x.mean(axis=0), x.std(axis=0)
    xs[xs == 0] = 1.0
    ym, ys = float(y.mean()), float(y.std())
    if ys == 0:
        ys = 1.0
    return xm, xs, ym, ys


def _t(x, y, xm, xs, ym, ys):
    return (torch.tensor((x - xm) / xs, dtype=torch.float32)[:, None, :],
            torch.tensor((y - ym) / ys, dtype=torch.float32))


def _train(model, x_tr, y_tr, x_val, y_val, *, epochs, batch, patience, lr, seed):
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    stop = EarlyStopper(patience=patience, min_delta=1e-6)
    mse = nn.MSELoss()
    best_val, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.mse_loss(model(xb.to(device)).squeeze(-1), yb.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = mse(model(x_val.to(device)).squeeze(-1), y_val.to(device)).item()
        if v + 1e-8 < best_val:
            best_val = v
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if stop.step(v, model):
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _eval(model, x, y, ys):
    model.eval()
    with torch.no_grad():
        pred = model(x.to(device)).squeeze(-1)
        target = y.to(device)
        if pred.ndim > target.ndim:
            pred = pred.squeeze(-1)
        return float(torch.mean((pred - target) ** 2).item() * ys ** 2)


def _distill(student, teacher, sx_tr, tx_tr, y_tr, sx_val, y_val, alpha,
             *, epochs, batch, patience, lr, seed):
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(sx_tr, tx_tr, y_tr), batch_size=batch,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    stop = EarlyStopper(patience=patience, min_delta=1e-6)
    mse = nn.MSELoss()
    best_val, best_state = float("inf"), None
    for _ in range(epochs):
        student.train()
        for xs, xt, yb in loader:
            opt.zero_grad()
            out = student(xs.to(device)).squeeze(-1)
            with torch.no_grad():
                t_out = teacher(xt.to(device)).squeeze(-1)
            loss = alpha * mse(out, yb.to(device)) + (1 - alpha) * mse(out, t_out)
            loss.backward()
            opt.step()
        student.eval()
        with torch.no_grad():
            v = mse(student(sx_val.to(device)).squeeze(-1), y_val.to(device)).item()
        if v + 1e-8 < best_val:
            best_val = v
            best_state = {k: t.detach().cpu().clone() for k, t in student.state_dict().items()}
        if stop.step(v, student):
            break
    if best_state:
        student.load_state_dict(best_state)
    student.eval()
    return student


def _make_teacher(l_t):
    return PatchTST(l_t, output_size=1, d_model=TEACHER_CFG["d_model"],
                    nhead=TEACHER_CFG["nhead"], dim_feedforward=TEACHER_CFG["dim_ff"],
                    dropout=TEACHER_CFG["dropout"]).to(device)


def _make_student(l_s):
    return PatchTST(l_s, output_size=1, d_model=STUDENT_CFG["d_model"],
                    nhead=STUDENT_CFG["nhead"], dim_feedforward=STUDENT_CFG["dim_ff"],
                    dropout=STUDENT_CFG["dropout"]).to(device)


def _teacher_windows(arr, l_s, h, l_t):
    """Teacher window ends 1 step before student target."""
    x_raw, y_raw = arr[:, 0], arr[:, 1]
    target_idx = np.arange(l_s + h - 1, len(y_raw))
    starts = target_idx - l_t
    valid = starts >= 0
    target_idx = target_idx[valid]
    starts = starts[valid]
    X = np.stack([x_raw[s:s + l_t] for s in starts])
    Y = y_raw[target_idx]
    return X, Y


def _student_windows(arr, l_s, h):
    x_raw, y_raw = arr[:, 0], arr[:, 1]
    target_idx = np.arange(l_s + h - 1, len(y_raw))
    starts = target_idx - (l_s + h - 1)
    X = np.stack([x_raw[s:s + l_s] for s in starts])
    Y = y_raw[target_idx]
    return X, Y


def _pairs(arr, l_s, h, l_t):
    x_raw, y_raw = arr[:, 0], arr[:, 1]
    target_idx = np.arange(l_s + h - 1, len(y_raw))
    s_start = target_idx - (l_s + h - 1)
    t_start = target_idx - l_t
    valid = (s_start >= 0) & (t_start >= 0)
    target_idx = target_idx[valid]
    s_start = s_start[valid]
    t_start = t_start[valid]
    SX = np.stack([x_raw[s:s + l_s] for s in s_start])
    TX = np.stack([x_raw[t:t + l_t] for t in t_start])
    Y = y_raw[target_idx]
    return SX, TX, Y


# ==============================================================================

def pretrain_foundation(target_name, args, seed):
    """Leave-one-out pretrain: train teacher on all h2o datasets except target."""
    pretrain_files = [f for f in ALL_H2O if f != target_name]
    l_t = TEACHER_CFG["L_t"]
    all_x, all_y = [], []
    for fname in pretrain_files:
        arr = _load(fname)
        tx, ty = _teacher_windows(arr, args.lookback, args.horizon, l_t)
        # Trim each dataset's windows to n_tr portion only (no target leakage)
        n = len(tx)
        n_tr = int(n * 0.6)  # only the train split of each source dataset
        all_x.append(tx[:n_tr])
        all_y.append(ty[:n_tr])
    X_pre = np.concatenate(all_x)
    Y_pre = np.concatenate(all_y)
    # Shuffle
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X_pre))
    X_pre, Y_pre = X_pre[idx], Y_pre[idx]

    # Split into train/val
    n = len(X_pre)
    n_tr = int(n * 0.9)
    n_val = n - n_tr
    xm, xs, ym, ys = _standardize(X_pre[:n_tr], Y_pre[:n_tr])
    x_tr, y_tr = _t(X_pre[:n_tr], Y_pre[:n_tr], xm, xs, ym, ys)
    x_val, y_val = _t(X_pre[n_tr:], Y_pre[n_tr:], xm, xs, ym, ys)

    torch.manual_seed(seed)
    model = _make_teacher(l_t)
    _train(model, x_tr, y_tr, x_val, y_val,
           epochs=EPOCHS_PRETRAIN, batch=BATCH, patience=PATIENCE,
           lr=TEACHER_CFG["lr"], seed=seed)
    return model


def finetune_teacher(pretrained, arr, l_s, h, l_t, seed):
    """Finetune the pretrained teacher on the target dataset."""
    TX, TY = _teacher_windows(arr, l_s, h, l_t)
    n = len(TX)
    n_tr = int(n * 0.6)
    n_val = int(n * 0.2)
    xm, xs, ym, ys = _standardize(TX[:n_tr], TY[:n_tr])
    x_tr, y_tr = _t(TX[:n_tr], TY[:n_tr], xm, xs, ym, ys)
    x_val, y_val = _t(TX[n_tr:n_tr+n_val], TY[n_tr:n_tr+n_val], xm, xs, ym, ys)
    x_te, y_te = _t(TX[n_tr+n_val:], TY[n_tr+n_val:], xm, xs, ym, ys)

    # Zero-shot evaluate first
    # Need to re-standardize with target's statistics
    zs_model = _make_teacher(l_t)
    zs_model.load_state_dict(pretrained.state_dict())
    zero_shot_mse = _eval(zs_model, x_te, y_te, ys)

    # Finetune (copy pretrained weights, lower lr)
    finetuned = _make_teacher(l_t)
    finetuned.load_state_dict(pretrained.state_dict())
    _train(finetuned, x_tr, y_tr, x_val, y_val,
           epochs=EPOCHS_FINETUNE, batch=BATCH, patience=PATIENCE,
           lr=TEACHER_CFG["lr"] * 0.1, seed=seed)  # lower lr for finetuning
    ft_mse = _eval(finetuned, x_te, y_te, ys)
    return finetuned, zero_shot_mse, ft_mse


def scratch_teacher(arr, l_s, h, l_t, seed):
    """Train a teacher from scratch on the target dataset."""
    TX, TY = _teacher_windows(arr, l_s, h, l_t)
    n = len(TX)
    n_tr = int(n * 0.6)
    n_val = int(n * 0.2)
    xm, xs, ym, ys = _standardize(TX[:n_tr], TY[:n_tr])
    x_tr, y_tr = _t(TX[:n_tr], TY[:n_tr], xm, xs, ym, ys)
    x_val, y_val = _t(TX[n_tr:n_tr+n_val], TY[n_tr:n_tr+n_val], xm, xs, ym, ys)
    x_te, y_te = _t(TX[n_tr+n_val:], TY[n_tr+n_val:], xm, xs, ym, ys)
    torch.manual_seed(seed)
    model = _make_teacher(l_t)
    _train(model, x_tr, y_tr, x_val, y_val,
           epochs=EPOCHS_FINETUNE, batch=BATCH, patience=PATIENCE,
           lr=TEACHER_CFG["lr"], seed=seed)
    return model, _eval(model, x_te, y_te, ys)


def run_single_dataset(target, args, seed):
    """Full pipeline for one target dataset."""
    l_s, h, l_t = args.lookback, args.horizon, TEACHER_CFG["L_t"]
    arr = _load(target)

    # 1) From-scratch teacher
    t_scratch, t_scratch_mse = scratch_teacher(arr, l_s, h, l_t, seed)

    # 2) Foundation teacher (pretrain + finetune)
    pretrained = pretrain_foundation(target, args, seed)
    t_finetuned, t_zero_shot, t_ft_mse = finetune_teacher(pretrained, arr, l_s, h, l_t, seed)

    # Pick the better teacher
    t_best, t_best_mse, t_source = (
        (t_finetuned, t_ft_mse, "finetuned") if t_ft_mse < t_scratch_mse
        else (t_scratch, t_scratch_mse, "scratch")
    )

    # 3) Student baseline (no distillation)
    SX, TX, Y = _pairs(arr, l_s, h, l_t)
    n = len(SX)
    n_tr = int(n * 0.6)
    n_val = int(n * 0.2)
    sxm, sxs, ym, ys = _standardize(SX[:n_tr], Y[:n_tr])
    txm, txs, _, _ = _standardize(TX[:n_tr], Y[:n_tr])
    sx_tr, y_tr = _t(SX[:n_tr], Y[:n_tr], sxm, sxs, ym, ys)
    tx_tr, _ = _t(TX[:n_tr], Y[:n_tr], txm, txs, ym, ys)
    sx_val, y_val = _t(SX[n_tr:n_tr+n_val], Y[n_tr:n_tr+n_val], sxm, sxs, ym, ys)
    sx_te, y_te = _t(SX[n_tr+n_val:], Y[n_tr+n_val:], sxm, sxs, ym, ys)

    torch.manual_seed(seed)
    baseline = _make_student(l_s)
    _train(baseline, sx_tr, y_tr, sx_val, y_val,
           epochs=EPOCHS_STUDENT, batch=BATCH, patience=PATIENCE,
           lr=STUDENT_CFG["lr"], seed=seed)
    baseline_mse = _eval(baseline, sx_te, y_te, ys)

    # 4) Student with distillation from best teacher
    torch.manual_seed(seed)
    student = _make_student(l_s)
    _distill(student, t_best, sx_tr, tx_tr, y_tr, sx_val, y_val, ALPHA,
             epochs=EPOCHS_STUDENT, batch=BATCH, patience=PATIENCE,
             lr=STUDENT_CFG["lr"], seed=seed)
    student_mse = _eval(student, sx_te, y_te, ys)

    return {
        "target": target, "family": _family(target), "seed": seed,
        "l_s": l_s, "h": h, "l_t": l_t,
        "teacher_scratch_mse": t_scratch_mse,
        "teacher_zero_shot_mse": t_zero_shot,
        "teacher_finetune_mse": t_ft_mse,
        "teacher_best_mse": t_best_mse,
        "teacher_best_source": t_source,
        "baseline_mse": baseline_mse,
        "student_mse": student_mse,
        "improvement_percent": (baseline_mse - student_mse) / baseline_mse * 100 if baseline_mse > 0 else 0,
        "alpha": ALPHA,
    }


def _family(name):
    for fam, files in FAMILIES.items():
        if name in files:
            return fam
    return "unknown"


# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("test", "foundation", "all"), default="all")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="specific dataset filenames (default: one per family)")
    args = parser.parse_args()

    if args.datasets:
        targets = args.datasets
    else:
        # One representative per family
        targets = [
            "data_h2o.pkl",                              # periodic
            "data_forced_h2o_A0.5_f0.05.pkl",            # forced
            "data_dual_h2o_Vb5.0e-05_rc0.3.pkl",         # dual
            "data_delayed_stable_h2o_tau50_s1_A0.9_b0.03.pkl",   # delayed (quasi-periodic)
            "data_delayed_stable_h2o_tau100_s1_A0.9_b0.03.pkl",  # delayed (chaotic)
            "data_delayed_stable_h2o_tau150_s1_A0.9_b0.03.pkl",  # delayed (strongly chaotic)
            "data_complex_multidelay_tau30_70.pkl",      # multi-delay
            "data_complex_forceddelay_tau50_A0.05_f0.03.pkl",  # forced+delay
            "data_complex_parammod_tau50_V8e-06-1e-05_T200.pkl",  # param mod
            "data_complex_noisy_tau50_s0.02.pkl",        # noisy control
        ]

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    all_rows = []
    print(f"=== CSTR foundation teacher pipeline ===")
    print(f"L_s={args.lookback} H={args.horizon} L_t={TEACHER_CFG['L_t']} α={ALPHA}")
    print(f"Targets: {targets}")
    print(f"Seeds: {args.seeds}\n")

    for target in targets:
        for seed in range(args.seeds):
            t0 = time.time()
            row = run_single_dataset(target, args, seed)
            all_rows.append(row)
            dt = time.time() - t0
            print(f"  {target} seed={seed} ({dt:.0f}s): "
                  f"t_scratch={row['teacher_scratch_mse']:.4e} "
                  f"t_zeroshot={row['teacher_zero_shot_mse']:.4e} "
                  f"t_finetune={row['teacher_finetune_mse']:.4e} "
                  f"→ best={row['teacher_best_source']} ({row['teacher_best_mse']:.4e}) | "
                  f"baseline={row['baseline_mse']:.4e} student={row['student_mse']:.4e} "
                  f"({row['improvement_percent']:+.1f}%)", flush=True)

    # Save CSV
    fields = list(all_rows[0].keys())
    tmp = f"{_CSV_PATH}.tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(tmp, _CSV_PATH)

    # Summary
    print(f"\n{'='*110}")
    print("SUMMARY")
    print(f"{'='*110}")
    groups = defaultdict(list)
    for r in all_rows:
        groups[(r["family"], r["target"])].append(r)
    for (fam, tgt), g in sorted(groups.items()):
        ts = np.mean([r["teacher_scratch_mse"] for r in g])
        tz = np.mean([r["teacher_zero_shot_mse"] for r in g])
        tf = np.mean([r["teacher_finetune_mse"] for r in g])
        tb = np.mean([r["baseline_mse"] for r in g])
        st = np.mean([r["student_mse"] for r in g])
        imp = np.mean([r["improvement_percent"] for r in g])
        src = g[0]["teacher_best_source"]
        ft_wins = sum(1 for r in g if r["teacher_finetune_mse"] < r["teacher_scratch_mse"])
        print(f"{fam:>10} {tgt.split('data_')[1][:30]:>30}: "
              f"t_scratch={ts:.3e} t_zeroshot={tz:.3e} t_ft={tf:.3e} ({src}, {ft_wins}/{len(g)} ft wins) | "
              f"baseline={tb:.3e} student={st:.3e} ({imp:+.1f}%)")
    print(f"-> {_CSV_PATH}")


if __name__ == "__main__":
    main()
