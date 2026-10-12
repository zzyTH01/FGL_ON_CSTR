"""Continuous Future-Guided Learning with adaptive weight & iterative refinement.

Ports the discrete bin-index E / E-soft / iterative mechanisms to continuous
regression.  All errors are computed in physical units (no argmax, no bins).
"""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from .training import EarlyStopper, device


def _aligned_continuous_arrays(data, L, H, val_size=0.2, test_size=0.2):
    raw = np.asarray(data, dtype=np.float64)
    x_raw, y_raw = raw[:, 0], raw[:, 1]
    n_win = len(x_raw) - L - H + 1
    n_te = int(n_win * test_size)
    n_va = int(n_win * val_size)
    n_tr = n_win - n_va - n_te
    sx = np.stack([x_raw[i:i+L] for i in range(n_win)])
    tx = np.stack([x_raw[i+H-1:i+H-1+L] for i in range(n_win)])
    y = np.asarray([y_raw[i+L+H-1] for i in range(n_win)], dtype=np.float64)
    xm, xs_ = sx[:n_tr].mean(0), sx[:n_tr].std(0)
    xs_[xs_ == 0] = 1.0
    ym, ys_ = float(y[:n_tr].mean()), float(y[:n_tr].std())
    ys_ = max(ys_, 1e-8)
    norm = lambda a: (a - xm) / xs_
    ny = lambda a: (a - ym) / ys_
    def t(a, sl):
        return torch.as_tensor(norm(a[sl]), dtype=torch.float32)
    return {
        "train_sx": t(sx, slice(0, n_tr)), "train_tx": t(tx, slice(0, n_tr)),
        "train_y": torch.as_tensor(ny(y[:n_tr]), dtype=torch.float32),
        "val_sx": t(sx, slice(n_tr, n_tr+n_va)), "val_tx": t(tx, slice(n_tr, n_tr+n_va)),
        "val_y": torch.as_tensor(ny(y[n_tr:n_tr+n_va]), dtype=torch.float32),
        "test_sx": t(sx, slice(n_tr+n_va, n_win)), "test_tx": t(tx, slice(n_tr+n_va, n_win)),
        "test_y": torch.as_tensor(ny(y[n_tr+n_va:]), dtype=torch.float32),
        "actual": y[n_tr+n_va:],
        "y_mean": ym, "y_std": ys_,
    }


def _loader(sx, tx, y, bs, shuffle=False, seed=0):
    idx = torch.arange(len(y), dtype=torch.long)
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(TensorDataset(idx, sx, tx, y), batch_size=bs,
                      shuffle=shuffle, drop_last=True, generator=gen)


def _mse_per_sample(model, sx, y):
    model.eval()
    with torch.no_grad():
        xb = sx.to(device).view(-1, 1, sx.shape[-1])
        pred = model(xb).cpu()
    return ((pred - y) ** 2).numpy()


def _compute_e_weights(se, te, variant="E", w_floor=0.2):
    raw = np.maximum(0.0, se - te) if variant == "E" else se - te
    if variant == "E":
        p95 = np.percentile(raw, 95)
        if p95 < 1e-8:
            return np.ones_like(raw)
        return np.clip(raw, 0, p95) / p95 * 4.0
    else:
        std = raw.std()
        if std < 1e-8:
            return np.full_like(raw, (w_floor + 4.0) / 2)
        c = np.median(raw)
        p25, p75 = np.percentile(raw, 25), np.percentile(raw, 75)
        s = max((p75 - p25) / 2, std, 1e-8)
        sig = 1.0 / (1.0 + np.exp(-(raw - c) / s))
        return w_floor + (4.0 - w_floor) * sig


def _train_cont(model, loader, val_sx, val_y, epochs, patience, lr,
                teacher=None, alpha=0.5, weights=None):
    opt = optim.Adam(model.parameters(), lr=lr)
    stop = EarlyStopper(patience=patience)
    mse = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for batch in loader:
            idx, sx, tx, y = batch
            sx = sx.to(device).view(-1, 1, sx.shape[-1])
            tx = tx.to(device).view(-1, 1, tx.shape[-1]) if teacher is not None else sx
            y = y.to(device)
            out = model(sx)
            loss = alpha * mse(out, y)
            if teacher is not None:
                with torch.no_grad():
                    tout = teacher(tx)
                if weights is not None:
                    w = weights[idx.long()].to(device)
                    loss = loss + (1 - alpha) * (w * (out - tout) ** 2).mean()
                else:
                    loss = loss + (1 - alpha) * mse(out, tout)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vpred = model(val_sx.to(device).view(-1, 1, val_sx.shape[-1])).cpu()
            vl = float(((vpred - val_y) ** 2).mean())
        if stop.step(vl, model):
            break
    stop.restore(model)
    model.eval()


def _cont_mse(model, sx, y):
    model.eval()
    with torch.no_grad():
        xb = sx.to(device).view(-1, 1, sx.shape[-1])
        pred = model(xb).cpu()
    return float(((pred - y) ** 2).mean())


def run_continuous_iterative_fgl(
    data, L=20, H=15, model_fn=None, alpha=0.5,
    epochs=30, round_epochs=10, K=6, batch_size=64, patience=5,
    lr=5e-4, seed=0, variant="E", w_floor=0.2, val_size=0.2, test_size=0.2,
    snapshot_fn=None, verbose=True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = _aligned_continuous_arrays(data, L, H, val_size, test_size)
    if model_fn is None:
        raise ValueError("model_fn required")
    tr_loader = _loader(d["train_sx"], d["train_tx"], d["train_y"], batch_size, True, seed)
    val_pair = (d["val_sx"], d["val_y"])
    teacher = model_fn(L).to(device)
    _train_cont(teacher, _loader(d["train_tx"], d["train_tx"], d["train_y"],
                                 batch_size, True, seed), *val_pair, epochs, patience, lr)
    baseline = model_fn(L).to(device)
    _train_cont(baseline, _loader(d["train_sx"], d["train_sx"], d["train_y"],
                                  batch_size, True, seed), *val_pair, epochs, patience, lr)
    student_0 = model_fn(L).to(device)
    _train_cont(student_0, tr_loader, *val_pair, epochs, patience, lr,
                teacher=teacher, alpha=alpha)
    student = copy.deepcopy(student_0)
    best_val = _cont_mse(student, d["val_sx"], d["val_y"])
    best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
    arms = {"student_0": {"val": best_val,
                          "test": _cont_mse(student, d["test_sx"], d["test_y"])}}
    if snapshot_fn:
        snapshot_fn(student, 0, best_val, arms["student_0"]["test"])
    for r in range(1, K + 1):
        se = _mse_per_sample(student, d["train_sx"], d["train_y"])
        te = _mse_per_sample(teacher, d["train_tx"], d["train_y"])
        w = _compute_e_weights(se, te, variant, w_floor)
        wt = torch.as_tensor(w, dtype=torch.float32)
        _train_cont(student, tr_loader, *val_pair, round_epochs, patience, lr,
                    teacher=teacher, alpha=alpha, weights=wt)
        mv = _cont_mse(student, d["val_sx"], d["val_y"])
        mt = _cont_mse(student, d["test_sx"], d["test_y"])
        arms[f"round_{r}"] = {"val": mv, "test": mt}
        if snapshot_fn:
            snapshot_fn(student, r, mv, mt)
        if mv < best_val:
            best_val = mv
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
    student.load_state_dict(best_state)
    student.eval()
    arms["final"] = {"val": best_val,
                     "test": _cont_mse(student, d["test_sx"], d["test_y"])}
    if verbose:
        for k, v in arms.items():
            print(f"  {k:12s}  val={v['val']:.5f}  test={v['test']:.5f}")
    return {"teacher": teacher, "baseline": baseline,
            "student_0": student_0, "student_final": student,
            "arms": arms, "data": d}
