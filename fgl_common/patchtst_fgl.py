"""Continuous-regression Future-Guided Learning with a PatchTST backbone.

This module keeps the same direct-horizon forecasting contract as the external
baselines in :mod:`fgl_common.baselines`: an ``L``-step input predicts the
scalar target ``t + L + H - 1``.  The teacher input is shifted ``H - 1`` steps
forward and therefore solves a one-step task for the same target.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .baselines import PatchTST
from .training import device


@dataclass(frozen=True)
class ContinuousFGLWindows:
    """Train-only standardized windows for one FGL regression trial.

    ``student_x`` and ``teacher_x`` are aligned per target: for target index
    ``i``, the student observes ``x[i:i+L]`` while the teacher observes
    ``x[i+H-1:i+H-1+L]``.  Both predict ``y[i+L+H-1]``.
    """

    train_student_x: torch.Tensor
    train_teacher_x: torch.Tensor
    train_y: torch.Tensor
    val_student_x: torch.Tensor
    val_teacher_x: torch.Tensor
    val_y: torch.Tensor
    test_student_x: torch.Tensor
    test_teacher_x: torch.Tensor
    test_y: torch.Tensor
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float


def build_continuous_fgl_windows(data, lookback_window: int, forecasting_horizon: int,
                                 val_size: float = 0.2, test_size: float = 0.2,
                                 ) -> ContinuousFGLWindows:
    """Build aligned student/teacher windows with train-only scaling."""
    if lookback_window < 1 or forecasting_horizon < 1:
        raise ValueError("lookback_window and forecasting_horizon must be positive")
    if not 0 < val_size + test_size < 1:
        raise ValueError("0 < val_size + test_size < 1 is required")

    raw = np.asarray(data, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("data must be an (N, 2) sequence of input,target pairs")
    x_raw = raw[:, 0]
    y_raw = raw[:, 1]
    L, H = lookback_window, forecasting_horizon
    n_windows = len(x_raw) - L - H + 1
    if n_windows <= 0:
        raise ValueError("not enough observations for lookback_window/forecasting_horizon")

    student_x = np.stack([x_raw[i:i + L] for i in range(n_windows)])
    teacher_x = np.stack([x_raw[i + H - 1:i + H - 1 + L] for i in range(n_windows)])
    y = np.asarray([y_raw[i + L + H - 1] for i in range(n_windows)], dtype=np.float64)

    n_test = int(n_windows * test_size)
    n_val = int(n_windows * val_size)
    n_train = n_windows - n_val - n_test
    if n_train <= 0:
        raise ValueError("train split is empty")

    # Fit both scalers only on the student training task, matching the external
    # baselines' train-only continuous-MSE contract.
    x_mean = student_x[:n_train].mean(axis=0)
    x_std = student_x[:n_train].std(axis=0)
    x_std[x_std == 0.0] = 1.0
    y_mean = float(y[:n_train].mean())
    y_std = float(y[:n_train].std())
    if y_std == 0.0:
        y_std = 1.0

    def norm_x(arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor((arr - x_mean) / x_std, dtype=torch.float32)[:, None, :]

    def norm_y(arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor((arr - y_mean) / y_std, dtype=torch.float32)

    return ContinuousFGLWindows(
        train_student_x=norm_x(student_x[:n_train]),
        train_teacher_x=norm_x(teacher_x[:n_train]),
        train_y=norm_y(y[:n_train]),
        val_student_x=norm_x(student_x[n_train:n_train + n_val]),
        val_teacher_x=norm_x(teacher_x[n_train:n_train + n_val]),
        val_y=norm_y(y[n_train:n_train + n_val]),
        test_student_x=norm_x(student_x[n_train + n_val:]),
        test_teacher_x=norm_x(teacher_x[n_train + n_val:]),
        test_y=norm_y(y[n_train + n_val:]),
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )


def _output(model: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    pred = model(x.to(device))
    return pred.squeeze(-1) if pred.ndim > 1 else pred


def _physical_mse(pred_norm: torch.Tensor, target_norm: torch.Tensor, y_std: float) -> float:
    return float(torch.mean((pred_norm - target_norm) ** 2).item() * y_std ** 2)


def _evaluate(model, x: torch.Tensor, y: torch.Tensor, y_std: float) -> float:
    model.eval()
    with torch.no_grad():
        return _physical_mse(_output(model, x), y.to(device), y_std)


def _train_model(model, train_x: torch.Tensor, train_y: torch.Tensor,
                 val_x: torch.Tensor, val_y: torch.Tensor, *, epochs: int,
                 batch_size: int, patience: int, lr: float, seed: int,
                 teacher_x: torch.Tensor | None = None, teacher: torch.Tensor | None = None,
                 alpha: float = 1.0):
    """Train one continuous regression arm and restore its best val state."""
    from .training import EarlyStopper

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    if teacher_x is None:
        teacher_x = train_x
    dataset = TensorDataset(train_x, teacher_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopper(patience=patience, min_delta=1e-5)
    best_state = None
    best_val = float("inf")
    epochs_used = 0
    mse = torch.nn.MSELoss()

    for _ in range(epochs):
        epochs_used += 1
        model.train()
        for xs, xt, y in loader:
            optimizer.zero_grad()
            output = _output(model, xs.to(device))
            loss = alpha * mse(output, y.to(device))
            if teacher is not None:
                with torch.no_grad():
                    teacher_output = _output(teacher, xt.to(device))
                loss = loss + (1.0 - alpha) * mse(output, teacher_output)
            loss.backward()
            optimizer.step()

        val_mse = _evaluate(model, val_x, val_y, y_std=1.0)
        if val_mse + 1e-8 < best_val:
            best_val = val_mse
            best_state = {k: value.detach().cpu().clone()
                          for k, value in model.state_dict().items()}
        if stopper.step(val_mse, model):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"epochs": epochs_used, "val_mse_normalized": best_val}


def run_patchtst_fgl(data, lookback_window: int, forecasting_horizon: int,
                     alpha: float = 0.5, val_size: float = 0.2, test_size: float = 0.2,
                     epochs: int = 50, batch_size: int = 64, patience: int = 10,
                     lr: float = 5e-4, seed: int = 42, d_model: int = 32, nhead: int = 4,
                     dim_feedforward: int = 64, dropout: float = 0.1,
                     verbose: bool = False, label: str = "") -> dict:
    """Run teacher -> baseline -> student FGL with continuous PatchTST arms."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    L, H = lookback_window, forecasting_horizon
    windows = build_continuous_fgl_windows(
        data, lookback_window=L, forecasting_horizon=H,
        val_size=val_size, test_size=test_size,
    )

    def make_model():
        return PatchTST(L, output_size=1, d_model=d_model, nhead=nhead,
                        dim_feedforward=dim_feedforward, dropout=dropout).to(device)

    torch.manual_seed(seed)
    teacher = make_model()
    teacher_info = _train_model(
        teacher, windows.train_teacher_x, windows.train_y,
        windows.val_teacher_x, windows.val_y,
        epochs=epochs, batch_size=batch_size, patience=patience, lr=lr,
        seed=seed, alpha=1.0,
    )

    torch.manual_seed(seed)
    baseline = make_model()
    baseline_info = _train_model(
        baseline, windows.train_student_x, windows.train_y,
        windows.val_student_x, windows.val_y,
        epochs=epochs, batch_size=batch_size, patience=patience, lr=lr,
        seed=seed, alpha=1.0,
    )

    torch.manual_seed(seed)
    student = make_model()
    student_info = _train_model(
        student, windows.train_student_x, windows.train_y,
        windows.val_student_x, windows.val_y,
        epochs=epochs, batch_size=batch_size, patience=patience, lr=lr,
        seed=seed, teacher_x=windows.train_teacher_x, teacher=teacher,
        alpha=alpha,
    )

    teacher_mse = _evaluate(teacher, windows.test_teacher_x, windows.test_y, windows.y_std)
    baseline_mse = _evaluate(baseline, windows.test_student_x, windows.test_y, windows.y_std)
    student_mse = _evaluate(student, windows.test_student_x, windows.test_y, windows.y_std)
    improvement = ((baseline_mse - student_mse) / baseline_mse * 100.0
                   if baseline_mse > 0.0 else 0.0)
    result = {
        "lookback": L,
        "horizon": H,
        "alpha": alpha,
        "teacher_mse": teacher_mse,
        "baseline_mse": baseline_mse,
        "student_mse": student_mse,
        "improvement": improvement,
        "teacher_epochs": teacher_info["epochs"],
        "baseline_epochs": baseline_info["epochs"],
        "student_epochs": student_info["epochs"],
        "teacher_val_mse_normalized": teacher_info["val_mse_normalized"],
        "baseline_val_mse_normalized": baseline_info["val_mse_normalized"],
        "student_val_mse_normalized": student_info["val_mse_normalized"],
        "n_test": len(windows.test_y),
    }
    if verbose:
        tag = f"[{label}] " if label else ""
        print(f"{tag}PatchTST-FGL L={L} H={H} alpha={alpha:g} seed={seed}: "
              f"teacher={teacher_mse:.6g} baseline={baseline_mse:.6g} "
              f"student={student_mse:.6g} delta={improvement:+.2f}%")
    return result
