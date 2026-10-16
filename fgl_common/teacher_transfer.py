"""Teacher-optimization transfer experiment.

Trains several teacher variants and paired FGL students to quantify how much a
teacher-side MSE improvement is transferred to the distilled student.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .models import RNN
from .data import create_time_series_dataset
from .training import compute_shared_bin_edges, EarlyStopper, evaluate, run_fgl_experiment


TEACHER_CONDITIONS = ("baseline", "more_epochs", "lower_lr", "ensemble", "regularized")


def _train_teacher(data, L, H, num_bins, epochs, lr, batch_size, patience, seed,
                   weight_decay=0.0):
    """Train a single H=1 teacher with the same data split as FGL."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    bin_edges, _, _ = compute_shared_bin_edges(data, L, num_bins)
    train, val, _, _, _ = create_time_series_dataset(
        data=data,
        lookback_window=L,
        forecasting_horizon=1,
        num_bins=num_bins,
        val_size=0.2,
        test_size=0.2,
        offset=H - 1,
        batch_size=batch_size,
        bin_edges=bin_edges,
    )
    model = RNN(L, 128, num_bins, 2).to(_model_device())
    ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    stopper = EarlyStopper(patience=patience)
    for _ in range(epochs):
        model.train()
        for _, x, y in train:
            x = x.float().to(_model_device()).view(-1, 1, L)
            optimizer.zero_grad()
            ce(model(x), y.long().to(_model_device())).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = sum(
                ce(model(x.float().to(_model_device()).view(-1, 1, L)), y.long().to(_model_device())).item()
                for _, x, y in val
            ) / len(val)
        if stopper.step(val_loss, model):
            break
    stopper.restore(model)
    model.eval()
    return model


def _model_device():
    from .training import device
    return device


def run_teacher_student_transfer(data, L=20, H=15, num_bins=50, seed=0,
                                 epochs=30, batch_size=64, patience=5,
                                 alpha=0.5, temperature=4.0, ensemble_seeds=(0, 1, 2),
                                 verbose=False):
    """Run paired teacher variants and student experiments for one dataset/seed.

    ``teacher_gain = baseline_teacher_mse - treated_teacher_mse``.
    ``student_gain = baseline_teacher_student_mse - treated_teacher_student_mse``.
    Ratios are ``None`` when the denominator is numerically zero.
    """
    del seed  # used through ensemble seed tuple; target student uses ensemble_seeds[0]
    common = dict(
        data=data,
        lookback_window=L,
        forecasting_horizon=H,
        alpha=alpha,
        temperature=temperature,
        num_bins=num_bins,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        seed=ensemble_seeds[0],
        verbose=verbose,
    )
    baseline_teacher = _train_teacher(
        data, L, H, num_bins, epochs, 1e-4, batch_size, patience, ensemble_seeds[0]
    )
    treated_teachers = {
        "baseline": baseline_teacher,
        "more_epochs": _train_teacher(
            data, L, H, num_bins, epochs * 2, 1e-4, batch_size, patience, ensemble_seeds[0]
        ),
        "lower_lr": _train_teacher(
            data, L, H, num_bins, epochs, 1e-5, batch_size, patience, ensemble_seeds[0]
        ),
        "regularized": _train_teacher(
            data, L, H, num_bins, epochs, 1e-4, batch_size, patience, ensemble_seeds[0],
            weight_decay=1e-3,
        ),
    }
    ensemble = [RNN(L, 128, num_bins, 2).to(_model_device()) for _ in range(3)]
    for i, member in enumerate(ensemble):
        member.load_state_dict(_train_teacher(
            data, L, H, num_bins, epochs, 1e-4, batch_size, patience, ensemble_seeds[i % len(ensemble_seeds)]
        ).state_dict())
    treated_teachers["ensemble"] = _EnsembleTeacher(ensemble)

    teacher_mse = {}
    for condition, teacher in treated_teachers.items():
        teacher_mse[condition] = _evaluate_teacher(
            teacher, data, L, H, num_bins, batch_size
        )

    # Distill from each teacher. run_fgl_experiment retrains its own teacher, so
    # this interface is intentionally used only to keep student-side settings
    # identical. We therefore train paired students here instead.
    rows = []
    baseline_student_mse = _train_student(baseline_teacher, **common)
    for condition, teacher in treated_teachers.items():
        student_mse = _train_student(teacher, **common)
        teacher_gain = teacher_mse["baseline"] - teacher_mse[condition]
        student_gain = baseline_student_mse - student_mse
        row = {
            "condition": condition,
            "seed": ensemble_seeds[0],
            "teacher_mse": teacher_mse[condition],
            "student_mse": student_mse,
            "teacher_gain": teacher_gain,
            "student_gain": student_gain,
        }
        if abs(teacher_gain) > 1e-8:
            row["transfer_ratio"] = student_gain / teacher_gain
        else:
            row["transfer_ratio"] = None
        if abs(teacher_mse["baseline"]) > 1e-8 and abs(student_mse) > 1e-8:
            row["teacher_relative_gain"] = teacher_gain / teacher_mse["baseline"]
            row["student_relative_gain"] = student_gain / baseline_student_mse
            row["normalized_transfer"] = (
                row["student_relative_gain"] / row["teacher_relative_gain"]
                if abs(row["teacher_relative_gain"]) > 1e-8 else None
            )
        else:
            row["teacher_relative_gain"] = None
            row["student_relative_gain"] = None
            row["normalized_transfer"] = None
        rows.append(row)
    return rows


def _evaluate_teacher(model, data, L, H, num_bins, batch_size):
    bin_edges, _, _ = compute_shared_bin_edges(data, L, num_bins)
    _, _, test, _, _ = create_time_series_dataset(
        data=data,
        lookback_window=L,
        forecasting_horizon=1,
        num_bins=num_bins,
        val_size=0.2,
        test_size=0.2,
        offset=H - 1,
        batch_size=batch_size,
        bin_edges=bin_edges,
    )
    return evaluate(model, test, L)


def _train_student(teacher, data, lookback_window, forecasting_horizon, alpha,
                   temperature, num_bins, epochs, batch_size, patience, seed,
                   verbose):
    """Train one distilled student with fixed hyperparameters."""
    L, H = lookback_window, forecasting_horizon
    torch.manual_seed(seed)
    np.random.seed(seed)
    bin_edges, _, _ = compute_shared_bin_edges(data, L, num_bins)
    student_train, student_val, student_test, _, _ = create_time_series_dataset(
        data=data,
        lookback_window=L,
        forecasting_horizon=H,
        num_bins=num_bins,
        val_size=0.2,
        test_size=0.2,
        offset=0,
        batch_size=batch_size,
        bin_edges=bin_edges,
    )
    teacher_train, _, _, _, _ = create_time_series_dataset(
        data=data,
        lookback_window=L,
        forecasting_horizon=1,
        num_bins=num_bins,
        val_size=0.2,
        test_size=0.2,
        offset=H - 1,
        batch_size=batch_size,
        bin_edges=bin_edges,
    )
    model = RNN(L, 128, num_bins, 2).to(_model_device())
    ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    stopper = EarlyStopper(patience=patience)
    for _ in range(epochs):
        model.train()
        for (_, x_s, y_s), (_, x_t, _) in zip(student_train, teacher_train):
            x_s = x_s.float().to(_model_device()).view(-1, 1, L)
            x_t = x_t.float().to(_model_device()).view(-1, 1, L)
            outputs = model(x_s)
            with torch.no_grad():
                teacher_logits = teacher(x_t)
            loss = alpha * ce(outputs, y_s.long().to(_model_device())) + _kl(
                outputs, teacher_logits, temperature, alpha
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = sum(
                ce(model(x.float().to(_model_device()).view(-1, 1, L)), y.long().to(_model_device())).item()
                for _, x, y in student_val
            ) / len(student_val)
        if stopper.step(val_loss, model):
            break
    stopper.restore(model)
    return evaluate(model, student_test, L)


def _kl(student_logits, teacher_logits, temperature, alpha):
    # Keep the exact FGL KL formula in this focused student loop.
    from .distillation import KL
    return KL(student_logits, teacher_logits, temperature, alpha)


class _EnsembleTeacher(torch.nn.Module):
    """Average softmax probabilities of member teachers."""

    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    def forward(self, x):
        probs = [torch.softmax(member(x), dim=-1) for member in self.members]
        return torch.log(torch.stack(probs).mean(dim=0) + 1e-12)
