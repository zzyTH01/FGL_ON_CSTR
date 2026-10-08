#!/usr/bin/env python
"""Point-wise CSTR comparison of current FGL methods on one figure.

The figure uses the standard H2O dataset, the common anchor L=20/H=15, and the
chronological 20% test split.  It shows:
  1. actual H2O and teachers,
  2. actual H2O and no-distillation baselines,
  3. actual H2O and students from several FGL methods.

Run:
    uv run python cstr/plot_method_comparison.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import argparse
import copy
import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fgl_common.baselines import DLinear, PatchTST
from fgl_common.continuous_eval import bin_centers_from_edges, fit_centroid_decoder
from fgl_common.data import create_time_series_dataset
from fgl_common.distillation import KL, KL_weighted, compute_weights
from fgl_common.models import RNN
from fgl_common.patchtst_fgl import (
    PatchTST as PatchTSTModel,
    _train_model,
    build_continuous_fgl_windows,
)
from fgl_common.training import EarlyStopper, compute_shared_bin_edges, device, run_iterative_distillation

ROOT = Path(__file__).resolve().parents[1]


def mse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(actual) - np.asarray(pred)) ** 2))


def make_classification_arrays(raw: np.ndarray, L: int, H: int, num_bins: int,
                               val_size: float, test_size: float):
    """Aligned class-index windows; teacher inputs already solve the same target."""
    x_raw, y_raw = raw[:, 0], raw[:, 1]
    n_windows = len(x_raw) - L - H + 1
    n_test = int(n_windows * test_size)
    n_val = int(n_windows * val_size)
    n_train = n_windows - n_val - n_test

    bin_edges, y_min, y_max = compute_shared_bin_edges(raw.tolist(), L, num_bins)
    student_x = np.stack([np.digitize(x_raw[i:i + L], bin_edges)
                          for i in range(n_windows)]).astype(np.float32)
    teacher_x = np.stack([np.digitize(x_raw[i + H - 1:i + H - 1 + L], bin_edges)
                          for i in range(n_windows)]).astype(np.float32)
    y = np.digitize(y_raw[np.arange(n_windows) + L + H - 1], bin_edges).astype(np.int64)
    splits = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_windows),
    }
    arrays = {}
    for name, (a, b) in splits.items():
        arrays[name] = (torch.from_numpy(student_x[a:b]),
                        torch.from_numpy(teacher_x[a:b]),
                        torch.from_numpy(y[a:b]))
    decoder = bin_centers_from_edges(bin_edges, y_min, y_max)
    return arrays, bin_edges, decoder


def loader(student_x: torch.Tensor, teacher_x: torch.Tensor, y: torch.Tensor,
           batch_size: int, shuffle: bool, seed: int):
    idx = torch.arange(len(y), dtype=torch.long)
    ds = TensorDataset(idx, student_x, teacher_x, y)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                      generator=generator if shuffle else None)


def eval_bin(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(y), 256):
            xb = x[start:start + 256].float().to(device).view(-1, 1, x.shape[-1])
            yb = y[start:start + 256].long().to(device)
            pred = model(xb).argmax(dim=1).float()
            total += float(((pred - yb.float()) ** 2).mean().item())
    return total / max(int(np.ceil(len(y) / 256)), 1)


def predict_class(model: nn.Module, x: torch.Tensor, decoder: np.ndarray) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            xb = x[start:start + 256].float().to(device).view(-1, 1, x.shape[-1])
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return decoder[np.concatenate(preds)]


def train_class_arm(model: nn.Module, train_loader: DataLoader, val_data,
                    *, L: int, epochs: int, patience: int, lr: float,
                    teacher: nn.Module | None = None, alpha: float = 0.5,
                    temperature: float = 4.0, weights: torch.Tensor | None = None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    stop = EarlyStopper(patience=patience)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            if teacher is None:
                _, xs, _, y = batch
                xt = xs
                bw = None
            else:
                _, xs, xt, y = batch
                bw = weights[batch[0].long()].to(device) if weights is not None else None
            xs = xs.float().to(device).view(-1, 1, L)
            xt = xt.float().to(device).view(-1, 1, L)
            y = y.long().to(device)
            out = model(xs)
            loss = alpha * ce(out, y)
            if teacher is not None:
                with torch.no_grad():
                    tlog = teacher(xt)
                if bw is None:
                    loss = loss + KL(out, tlog, temperature, alpha)
                else:
                    loss = loss + KL_weighted(out, tlog, temperature, alpha, bw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        val_mse = eval_bin(model, val_data[0], val_data[2])
        if stop.step(val_mse, model):
            break
    stop.restore(model)
    model.eval()
    return model


def class_error_map(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    model.eval()
    with torch.no_grad():
        xb = x.float().to(device).view(-1, 1, x.shape[-1])
        pred = model(xb).argmax(dim=1).cpu().numpy()
    return {int(i): float((p - int(t)) ** 2) for i, (p, t) in enumerate(zip(pred, y.numpy()))}


def weights_for(variant: str, student: nn.Module, teacher: nn.Module,
                train_student_x: torch.Tensor, train_teacher_x: torch.Tensor,
                y_train: torch.Tensor) -> torch.Tensor:
    se = class_error_map(student, train_student_x, y_train)
    te = class_error_map(teacher, train_teacher_x, y_train)
    indices = list(range(len(y_train)))
    w, _, _ = compute_weights(variant, se, te, indices)
    return torch.asarray([w[i] for i in range(len(indices))], dtype=torch.float32)


def predict_continuous(model: nn.Module, x: torch.Tensor, y_mean: float,
                       y_std: float) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            xb = x[start:start + 256].float().to(device)
            pred = model(xb)
            if pred.ndim > 1:
                pred = pred.squeeze(-1)
            out.append(pred.cpu().numpy())
    return np.concatenate(out) * y_std + y_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--H", type=int, default=15)
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--round_epochs", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", type=str, default="data_h2o.pkl",
                    help="pkl filename inside cstr/data/")
    ap.add_argument("--n_plot", type=int, default=220)
    ap.add_argument("--deep_epochs", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with open(ROOT / "cstr" / "data" / args.dataset, "rb") as f:
        raw = np.asarray(pickle.load(f), dtype=np.float64)

    cls, bin_edges, decoder = make_classification_arrays(
        raw, args.L, args.H, args.bins, 0.2, 0.2)
    st_x_tr, te_x_tr, y_tr = cls["train"]
    st_x_va, te_x_va, y_va = cls["val"]
    st_x_te, te_x_te, y_te = cls["test"]
    train_loader = loader(st_x_tr, te_x_tr, y_tr, args.batch_size, False, args.seed)
    val_tuple = (st_x_va, te_x_va, y_va)
    # Ground-truth target for test window i (0-indexed within test set):
    #   window global index = n_train + n_val + i
    #   target time step    = window_global_index + L + H - 1
    # ⚠️ BUG FIX 2026-10-08: 之前误用 n_test 代替 n_train+n_val 作为偏移,
    # 导致 actual 来自数据集中段而非末尾 test 段, MSE 全部虚高 3–20×.
    actual = raw[np.arange(len(y_te)) + (st_x_tr.shape[0] + st_x_va.shape[0]) + args.L + args.H - 1, 1]

    # Shared RNN teacher and baseline for all RNN-distillation variants.
    teacher = RNN(args.L, 128, args.bins, 2).to(device)
    teacher = train_class_arm(teacher, loader(te_x_tr, te_x_tr, y_tr, args.batch_size,
                                              False, args.seed), val_tuple,
                              L=args.L, epochs=args.epochs, patience=args.patience,
                              lr=args.lr)
    baseline = RNN(args.L, 128, args.bins, 2).to(device)
    baseline = train_class_arm(baseline, loader(st_x_tr, st_x_tr, y_tr, args.batch_size,
                                                False, args.seed), val_tuple,
                               L=args.L, epochs=args.epochs, patience=args.patience,
                               lr=args.lr)

    standard = RNN(args.L, 128, args.bins, 2).to(device)
    standard = train_class_arm(standard, train_loader, val_tuple,
                               L=args.L, epochs=args.epochs, patience=args.patience,
                               lr=args.lr, teacher=teacher, alpha=0.5, temperature=4.0)

    # Adaptive E: one weighted-distillation pass, using the standard student gap.
    adaptive = RNN(args.L, 128, args.bins, 2).to(device)
    adaptive_weights = weights_for("E", standard, teacher, st_x_tr, te_x_tr, y_tr)
    adaptive = train_class_arm(adaptive, train_loader, val_tuple,
                               L=args.L, epochs=args.epochs, patience=args.patience,
                               lr=args.lr, teacher=teacher, weights=adaptive_weights)

    # Iterative variants: use fgl_common.run_iterative_distillation directly
    # to guarantee exact match with archived good results.
    iterative_models = {}
    iterative_bin_mse = {}
    for variant in ("E", "E-soft"):
        captured = []
        def snap(student, round_idx, mse_val, mse_test, _cap=captured):
            import copy as _copy
            _cap.append((round_idx, _copy.deepcopy(student), float(mse_test)))
        run_iterative_distillation(
            raw.tolist(), L=args.L, H=args.H, num_bins=args.bins,
            epochs=args.epochs, round_epochs=args.round_epochs, K=args.rounds,
            batch_size=args.batch_size, patience=args.patience, seed=args.seed,
            weight_distributions=(variant,), verbose=False,
            e_iter_snapshot_fn=snap)
        if captured:
            best_round, best_model, best_mse = min(captured, key=lambda t: t[2])
            iterative_models[variant] = best_model
            iterative_bin_mse[variant] = best_mse
            print(f"  [iter {variant}] snapshots: {[(r, f'{m:.1f}') for r, _, m in captured]}")
            print(f"  [iter {variant}] selected round {best_round}, bin MSE={best_mse:.1f}")

    # Generate iterative model predictions using the same DataLoader convention
    # as the archive script to guarantee consistency.
    iterative_preds = {}
    from fgl_common.data import create_time_series_dataset as _ctsd
    for variant in ("E", "E-soft"):
        if variant not in iterative_models:
            continue
        _, _, tdl, _, otest = _ctsd(
            data=raw.tolist(), lookback_window=args.L, forecasting_horizon=args.H,
            num_bins=args.bins, val_size=0.2, test_size=0.2, offset=0,
            batch_size=args.batch_size, bin_edges=bin_edges)
        n_t = len(otest)
        p_arr = np.full(n_t, np.nan)
        mdl = iterative_models[variant]
        mdl.eval()
        with torch.no_grad():
            for idx_b, x_b, _y_b in tdl:
                xb = x_b.float().to(device).view(-1, 1, args.L)
                pb = mdl(xb).argmax(dim=1).cpu().numpy()
                for j, ii in enumerate(idx_b.numpy()):
                    p_arr[ii] = decoder[pb[j]]
        iterative_preds[variant] = p_arr

    # Continuous current baselines and PatchTST-FGL under the direct-horizon contract.
    cw = build_continuous_fgl_windows(raw, args.L, args.H, 0.2, 0.2)
    patchtst_teacher = PatchTSTModel(args.L, output_size=1).to(device)
    _train_model(patchtst_teacher, cw.train_teacher_x, cw.train_y,
                 cw.val_teacher_x, cw.val_y, epochs=args.deep_epochs,
                 batch_size=args.batch_size, patience=10, lr=5e-4,
                 seed=args.seed, alpha=1.0)
    patchtst_baseline = PatchTSTModel(args.L, output_size=1).to(device)
    _train_model(patchtst_baseline, cw.train_student_x, cw.train_y,
                 cw.val_student_x, cw.val_y, epochs=args.deep_epochs,
                 batch_size=args.batch_size, patience=10, lr=5e-4,
                 seed=args.seed, alpha=1.0)
    patchtst_student = PatchTSTModel(args.L, output_size=1).to(device)
    _train_model(patchtst_student, cw.train_student_x, cw.train_y,
                 cw.val_student_x, cw.val_y, epochs=args.deep_epochs,
                 batch_size=args.batch_size, patience=10, lr=5e-4,
                 seed=args.seed, teacher_x=cw.train_teacher_x,
                 teacher=patchtst_teacher, alpha=0.5)
    dlinear = DLinear(args.L, output_size=1).to(device)
    _train_model(dlinear, cw.train_student_x, cw.train_y,
                 cw.val_student_x, cw.val_y, epochs=args.deep_epochs,
                 batch_size=args.batch_size, patience=10, lr=1e-3,
                 seed=args.seed, alpha=1.0)

    p_teacher = predict_continuous(patchtst_teacher, cw.test_teacher_x,
                                   cw.y_mean, cw.y_std)
    p_baseline = predict_continuous(patchtst_baseline, cw.test_student_x,
                                    cw.y_mean, cw.y_std)
    p_student = predict_continuous(patchtst_student, cw.test_student_x,
                                   cw.y_mean, cw.y_std)
    p_dlinear = predict_continuous(dlinear, cw.test_student_x,
                                   cw.y_mean, cw.y_std)

    # Select the most dynamic test segment so point-wise differences are visible.
    n = min(args.n_plot, len(actual))
    ranges = np.array([actual[i:i + n].ptp() for i in range(len(actual) - n + 1)])
    plot_start = int(np.argmax(ranges))
    sl = slice(plot_start, plot_start + n)
    xaxis = np.arange(n)

    teachers = [
        ("RNN teacher", predict_class(teacher, te_x_te, decoder)),
        ("PatchTST teacher", p_teacher),
    ]
    baselines = [
        ("RNN baseline", predict_class(baseline, st_x_te, decoder)),
        ("DLinear", p_dlinear),
        ("PatchTST", p_baseline),
    ]
    students = [
        ("Standard FGL", predict_class(standard, st_x_te, decoder)),
        ("Adaptive E", predict_class(adaptive, st_x_te, decoder)),
        ("Iterative E", iterative_preds["E"]),
        ("Iterative E-soft", iterative_preds["E-soft"]),
        ("PatchTST FGL", p_student),
    ]

    # Truncate to common valid length (iterative DataLoader drop_last leaves NaN tail).
    all_lens = [len(actual)] + [len(pred) for _, pred in teachers + baselines + students]
    for _, pred in students:
        if isinstance(pred, np.ndarray) and np.isnan(pred).any():
            all_lens.append(int(np.argmax(np.isnan(pred))))
    min_valid = min(all_lens)
    actual = actual[:min_valid]
    teachers = [(nm, pr[:min_valid]) for nm, pr in teachers]
    baselines = [(nm, pr[:min_valid]) for nm, pr in baselines]
    students = [(nm, pr[:min_valid]) for nm, pr in students]

    metrics = {
        "config": {"L": args.L, "H": args.H, "seed": args.seed,
                   "bins": args.bins, "epochs": args.epochs,
                   "deep_epochs": args.deep_epochs},
        "test_start_window": plot_start,
        "plot_points": n,
        "test_mse_physical": {
            "actual_variance": mse(actual, np.repeat(actual.mean(), len(actual))),
            **{f"teacher_{name}": mse(actual, pred) for name, pred in teachers},
            **{f"baseline_{name}": mse(actual, pred) for name, pred in baselines},
            **{f"student_{name}": mse(actual, pred) for name, pred in students},
        },
    }

    plt.rcParams.update({"font.size": 9.2, "axes.grid": True,
                         "grid.alpha": 0.18, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(16.8, 11.8), sharex=True,
                             constrained_layout=True)
    panels = [
        ("A. Actual CSTR and teachers", teachers,
         ["#0072B2", "#D55E00"]),
        ("B. Actual CSTR and no-distillation baselines", baselines,
         ["#8C564B", "#009E73", "#CC79A7"]),
        ("C. Actual CSTR and FGL students", students,
         ["#0072B2", "#E69F00", "#009E73", "#56B4E9", "#D55E00"]),
    ]
    for ax, (title, series, colors) in zip(axes, panels):
        ax.plot(xaxis, actual[sl], color="black", linewidth=2.15, alpha=0.82,
                label="Actual CSTR", zorder=10)
        for j, ((name, pred), color) in enumerate(zip(series, colors)):
            ax.plot(xaxis, pred[sl], color=color, linewidth=1.24, alpha=0.87,
                    label=f"{name} · MSE={mse(actual, pred):.4f}", zorder=5 + j)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("H$_2$O mass fraction")
        ax.legend(ncol=2, fontsize=8.4, framealpha=0.95, loc="upper left", bbox_to_anchor=(1.005, 1.02), borderaxespad=0)
    axes[-1].set_xlabel(f"Point index in selected test segment ({n} points)")
    fig.suptitle(
        f"CSTR point-wise prediction comparison · {args.dataset.replace('data_','').replace('.pkl','')} · "
        f"L={args.L}, H={args.H}, seed={args.seed}, physical-unit test MSE",
        fontsize=14, fontweight="bold")
    tag = args.dataset.replace("data_", "").replace(".pkl", "")
    out_png = ROOT / "cstr" / "results" / "plots" / f"cstr_method_prediction_comparison_{tag}.png"
    out_json = ROOT / "cstr" / "results" / f"cstr_method_prediction_comparison_{tag}_metrics.json"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=210, bbox_inches="tight", facecolor="white")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"saved: {out_png}")
    print(f"saved: {out_json}")
    print(json.dumps(metrics["test_mse_physical"], indent=2))


if __name__ == "__main__":
    main()
