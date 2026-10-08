#!/usr/bin/env python
"""Three-curve comparison: pre-trained student_0 → converged baseline → E-iter.

Uses the corrected test split indexing (see test_split_index_regression.py).
Run on GPU:
    uv run python cstr/plot_pretrain_finetune_eiter.py
"""
from __future__ import annotations
import os, sys, pickle, copy
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
_CSTR = os.path.join(_REPO, "cstr")

from fgl_common.training import (
    run_iterative_distillation, compute_shared_bin_edges, device,
    EarlyStopper, RNN, create_time_series_dataset,
)
from fgl_common.continuous_eval import bin_centers_from_edges
import torch.nn as nn
import torch.optim as optim

L, H, NB, SEED = 20, 15, 50, 0
ALPHA, TEMP, LR = 0.5, 4.0, 1e-4
EPOCHS_INIT, EPOCHS_CONV, ROUND_EPOCHS, K = 20, 100, 10, 6

torch.manual_seed(SEED); np.random.seed(SEED)
with open(os.path.join(_CSTR, "data", "data_h2o.pkl"), "rb") as f:
    data = pickle.load(f)

edges, ymin, ymax = compute_shared_bin_edges(data, L, NB)
centers = bin_centers_from_edges(edges, ymin, ymax)
cent = lambda bins: centers[np.clip(bins, 0, NB - 1)]

def mk_loader(offset, bs=64):
    return create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=1 if offset > 0 else H,
        num_bins=NB, val_size=0.2, test_size=0.2,
        offset=offset, batch_size=bs, bin_edges=edges)

t_tr, t_va, _, _, _ = mk_loader(H - 1)
s_tr, s_va, s_te, _, orig_test = mk_loader(0)
actual = np.asarray(orig_test, dtype=float)

ce = nn.CrossEntropyLoss()

def train_simple(model, loader, vloader, epochs, patience=5):
    opt = optim.Adam(model.parameters(), lr=LR)
    stop = EarlyStopper(patience=patience)
    for _ in range(epochs):
        model.train()
        for _, x, y in loader:
            x = x.float().to(device).view(-1, 1, L)
            opt.zero_grad(); ce(model(x), y.long().to(device)).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = sum(ce(model(x.float().to(device).view(-1, 1, L)), y.long().to(device)).item()
                     for _, x, y in vloader) / len(vloader)
        if stop.step(vl, model): break
    stop.restore(model); model.eval()

def train_student(model, teacher_l, vloader, epochs, patience=5):
    opt = optim.Adam(model.parameters(), lr=LR)
    stop = EarlyStopper(patience=patience)
    for _ in range(epochs):
        model.train()
        for (_, xs, ys), (_, xt, _) in zip(s_tr, teacher_l):
            xs = xs.float().to(device).view(-1, 1, L)
            xt = xt.float().to(device).view(-1, 1, L)
            out = model(xs)
            with torch.no_grad(): tlog = teacher(xt)
            loss = ALPHA * ce(out, ys.long().to(device)) + \
                   (1 - ALPHA) * TEMP**2 * nn.functional.kl_div(
                       nn.functional.log_softmax(out / TEMP, 1),
                       nn.functional.softmax(tlog / TEMP, 1), reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = sum(ce(model(x.float().to(device).view(-1, 1, L)), y.long().to(device)).item()
                     for _, x, y in vloader) / len(vloader)
        if stop.step(vl, model): break
    stop.restore(model); model.eval()

# ---- 1. Teacher ----
teacher = RNN(L, 128, NB, 2).to(device)
train_simple(teacher, t_tr, t_va, EPOCHS_INIT)

# ---- 2. student_0 (pre-trained, standard uniform KL) ----
student_0 = RNN(L, 128, NB, 2).to(device)
train_student(student_0, t_tr, s_va, EPOCHS_INIT)

# ---- 3. Converged baseline (post-training, 100 epochs, no distillation) ----
baseline_conv = RNN(L, 128, NB, 2).to(device)
train_simple(baseline_conv, s_tr, s_va, EPOCHS_CONV, patience=10)

# ---- 4. E-iter (iterative adaptive distillation, best-by-val) ----
snaps = []
def snap(s, r, mv, mt):
    snaps.append((r, copy.deepcopy(s), float(mt)))
run_iterative_distillation(
    data, L=L, H=H, num_bins=NB, epochs=EPOCHS_INIT, round_epochs=ROUND_EPOCHS,
    K=K, seed=SEED, weight_distributions=("E",), verbose=False,
    e_iter_snapshot_fn=snap)
best_r, e_iter_model, e_iter_bin_mse = min(snaps, key=lambda t: t[2])

# ---- Predict on full test set ----
def predict(model):
    model.eval()
    preds = np.full(len(actual), np.nan)
    _, _, tdl, _, _ = mk_loader(0)
    with torch.no_grad():
        for idx, x, _ in tdl:
            xb = x.float().to(device).view(-1, 1, L)
            pb = model(xb).argmax(dim=1).cpu().numpy()
            for j, ii in enumerate(idx.numpy()):
                preds[ii] = centers[pb[j]]
    return preds

pred_s0 = predict(student_0)
pred_conv = predict(baseline_conv)
pred_eiter = predict(e_iter_model)

# Truncate to valid length (drop_last)
for p in [pred_s0, pred_conv, pred_eiter]:
    nan_idx = np.where(np.isnan(p))[0]
    valid_n = nan_idx[0] if len(nan_idx) else len(p)
act = actual[:valid_n]
pred_s0 = pred_s0[:valid_n]; pred_conv = pred_conv[:valid_n]; pred_eiter = pred_eiter[:valid_n]

def mse(a, b): return float(np.mean((a - b) ** 2))

# ---- Plot ----
n = min(220, valid_n)
ranges = np.array([act[i:i+n].ptp() for i in range(valid_n - n + 1)])
start = int(np.argmax(ranges)); sl = slice(start, start + n)
xx = np.arange(n)

plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": 0.15,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)

ax.plot(xx, act[sl], color="black", lw=2.2, alpha=0.85, label="Actual CSTR", zorder=10)
ax.plot(xx, pred_s0[sl], color="#1f77b4", lw=1.3, alpha=0.85,
        label=f"Pre-trained (student$_0$) · MSE={mse(act, pred_s0):.4f}", zorder=5)
ax.plot(xx, pred_conv[sl], color="#ff7f0e", lw=1.3, alpha=0.85,
        label=f"Converged baseline (100 ep) · MSE={mse(act, pred_conv):.4f}", zorder=6)
ax.plot(xx, pred_eiter[sl], color="#2ca02c", lw=1.5, alpha=0.9,
        label=f"E-iter (round {best_r}, bin MSE={e_iter_bin_mse:.1f}) · MSE={mse(act, pred_eiter):.4f}", zorder=7)

ax.set_xlabel("Point index in selected test segment")
ax.set_ylabel("H$_2$O mass fraction")
ax.set_title("Pre-trained vs Converged baseline vs E-iter adaptive distillation "
             f"(CSTR H$_2$O, L={L}, H={H}, seed={SEED})", fontweight="bold")
ax.legend(ncol=2, fontsize=9, framealpha=0.95, loc="upper right")
ax.set_xlim(0, n)

out = os.path.join(_CSTR, "results", "plots", "cstr_pretrain_finetune_eiter.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved: {out}")
print(f"E-iter bin MSE: {e_iter_bin_mse:.1f} (round {best_r})")
print(f"physical MSE: student_0={mse(act,pred_s0):.5f}  converged={mse(act,pred_conv):.5f}  E-iter={mse(act,pred_eiter):.5f}")
