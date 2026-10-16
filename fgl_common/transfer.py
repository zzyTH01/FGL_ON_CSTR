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
from .training import compute_shared_bin_edges, evaluate


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


def _train(model, loader, L, epochs):
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for _, x, y in loader:
            x = x.float().to(device).view(-1, 1, L)
            y = y.long().to(device)
            optimizer.zero_grad()
            ce(model(x), y).backward()
            optimizer.step()
    model.eval()
    return model


def _dataset(path, L, num_bins, batch_size=32):
    data = _load_data(path)
    edges, _, _ = compute_shared_bin_edges(data.numpy(), L, num_bins)
    train, val, test, _, _ = create_time_series_dataset(
        data,
        L,
        1,
        num_bins,
        val_size=0.2,
        test_size=0.2,
        offset=0,
        batch_size=batch_size,
        bin_edges=edges,
    )
    return train, val, test


def build_pretrained_teacher(paths, H=1, hidden=128, epochs=3, seed=42, device=None):
    """Train one teacher by rotating through source datasets."""
    del H
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = None
    L = 20
    for path in paths:
        train, _, _ = _dataset(path, L, 50)
        if model is None:
            model = RNN(L, hidden, 50, 2).to(device or torch.device("cpu"))
        _train(model, train, L, epochs)
    return model


def evaluate_teacher_transfer(path, pretrained, L=20, H=1, hidden=128,
                              finetune_epochs=3, scratch_epochs=3,
                              num_bins=50, seed=42):
    """Compare frozen, finetuned, and scratch teachers on one target dataset."""
    del H
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, val, test = _dataset(path, L, num_bins)

    frozen = copy.deepcopy(pretrained).to(next(pretrained.parameters()).device).eval()
    with torch.no_grad():
        zero_shot = evaluate(frozen, test, L)

    finetuned = copy.deepcopy(pretrained)
    _train(finetuned, train, L, finetune_epochs)
    finetuned_mse = evaluate(finetuned, test, L)

    torch.manual_seed(seed + 1)
    scratch = RNN(L, hidden, num_bins, 2).to(next(pretrained.parameters()).device)
    _train(scratch, train, L, scratch_epochs)
    scratch_mse = evaluate(scratch, test, L)

    return [
        {"path": path, "condition": "zero_shot", "teacher_mse": zero_shot},
        {"path": path, "condition": "finetuned", "teacher_mse": finetuned_mse},
        {"path": path, "condition": "scratch", "teacher_mse": scratch_mse},
    ]
