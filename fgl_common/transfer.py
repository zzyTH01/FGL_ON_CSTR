"""Leave-one-example-out teacher transfer utilities."""

import copy
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .data import create_time_series_dataset
from .models import RNN
from .training import EarlyStopper, evaluate


def list_delayed_cstr_datasets(data_dir):
    return sorted(
        str(p)
        for p in Path(data_dir).glob("data_delayed_stable_h2o_tau*_s1_A0.9_b0.03.pkl")
    )


def split_lopo(paths, target):
    return (
        [p for p in paths if Path(p).resolve() != Path(target).resolve()],
        target,
    )


def _load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return torch.as_tensor(data, dtype=torch.float32)


def _train(model, loader, L, epochs, val_loader=None):
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    ce = nn.CrossEntropyLoss()
    stopper = EarlyStopper(patience=5) if val_loader is not None else None
    for _ in range(epochs):
        model.train()
        for _, x, y in loader:
            x = x.float().to(device).view(-1, 1, L)
            y = y.long().to(device)
            optimizer.zero_grad()
            ce(model(x), y).backward()
            optimizer.step()
        if val_loader is None:
            continue
        model.eval()
        with torch.no_grad():
            val_mse = evaluate(model, val_loader, L)
        if stopper.step(val_mse, model):
            break
    if stopper is not None:
        stopper.restore(model)
    model.eval()
    return model


def compute_global_bin_edges(paths, L=20, num_bins=50,
                             val_size: float = 0.2, test_size: float = 0.2):
    """Jointly bound source *training* targets for cross-dataset transfer."""
    train_values = []
    for path in paths:
        data = _load_data(path).numpy()
        y = np.asarray([float(pt[1]) for pt in data])
        targets = y[L:]
        n_val = int(len(targets) * val_size)
        n_test = int(len(targets) * test_size)
        n_train = len(targets) - n_val - n_test
        if n_train <= 0:
            raise ValueError(f"not enough training samples in {path}")
        train_values.append(targets[:n_train])
    values = np.concatenate(train_values)
    return np.linspace(values.min(), values.max(), num_bins - 1), float(values.min()), float(values.max())


def _dataset(path, L, num_bins, batch_size=32, bin_edges=None):
    data = _load_data(path)
    if bin_edges is None:
        bin_edges, _, _ = compute_global_bin_edges([path], L, num_bins)
    train, val, test, _, _ = create_time_series_dataset(
        data,
        L,
        1,
        num_bins,
        val_size=0.2,
        test_size=0.2,
        offset=0,
        batch_size=batch_size,
        bin_edges=bin_edges,
    )
    return train, val, test


def build_pretrained_teacher(paths, H=1, hidden=128, epochs=3, seed=42, device=None,
                             L=20, num_bins=50, bin_edges=None):
    """Train one teacher by rotating through aligned source datasets."""
    del H
    torch.manual_seed(seed)
    np.random.seed(seed)
    if bin_edges is None:
        bin_edges, _, _ = compute_global_bin_edges(paths, L, num_bins)
    model = None
    for path in paths:
        train, val, _ = _dataset(path, L, num_bins, bin_edges=bin_edges)
        if model is None:
            model = RNN(L, hidden, num_bins, 2).to(device or torch.device("cpu"))
        _train(model, train, L, epochs, val_loader=val)
    return model


def evaluate_teacher_transfer(path, pretrained, L=20, H=1, hidden=128,
                              finetune_epochs=3, scratch_epochs=3,
                              converged_epochs=30,
                              num_bins=50, seed=42, bin_edges=None):
    """Compare frozen, finetuned, and both equal/converged scratch teachers."""
    del H
    torch.manual_seed(seed)
    np.random.seed(seed)
    if bin_edges is None:
        bin_edges, _, _ = compute_global_bin_edges([path], L, num_bins)
    train, val, test = _dataset(path, L, num_bins, bin_edges=bin_edges)

    frozen = copy.deepcopy(pretrained).to(next(pretrained.parameters()).device).eval()
    with torch.no_grad():
        zero_shot = evaluate(frozen, test, L)

    finetuned = copy.deepcopy(pretrained)
    _train(finetuned, train, L, finetune_epochs, val_loader=val)
    finetuned_mse = evaluate(finetuned, test, L)

    device = next(pretrained.parameters()).device
    torch.manual_seed(seed + 1)
    scratch = RNN(L, hidden, num_bins, 2).to(device)
    _train(scratch, train, L, scratch_epochs)
    scratch_mse = evaluate(scratch, test, L)

    torch.manual_seed(seed + 1)
    scratch_converged = RNN(L, hidden, num_bins, 2).to(device)
    _train(scratch_converged, train, L, converged_epochs, val_loader=val)
    scratch_converged_mse = evaluate(scratch_converged, test, L)

    return [
        {"path": path, "condition": "zero_shot", "teacher_mse": zero_shot},
        {"path": path, "condition": "finetuned", "teacher_mse": finetuned_mse},
        {"path": path, "condition": "scratch", "teacher_mse": scratch_mse},
        {"path": path, "condition": "scratch_converged",
         "teacher_mse": scratch_converged_mse},
    ]
