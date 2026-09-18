"""External forecasting baselines with a continuous-MSE evaluation contract.

These helpers intentionally use the same direct-horizon task as the FGL
regression experiments: predict the scalar target at ``t + L + H - 1`` from the
input series over ``[t, t + L)``.  Unlike the historical classification setup,
all metrics are transformed back to physical units before reporting.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .training import device


# ==================== Continuous windows ====================

@dataclass(frozen=True)
class ContinuousWindows:
    """Train-only standardized sliding windows for scalar forecasting."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float


def build_continuous_windows(data, lookback_window: int, forecasting_horizon: int,
                             val_size: float = 0.2, test_size: float = 0.2,
                             batch_size: int = 64) -> ContinuousWindows:
    """Create chronological, train-only standardized windows.

    The split follows the existing FGL convention: the trailing ``int(N *`` 
    ``test_size)`` windows are test, the preceding ``int(N * val_size)`` are
    validation, and all remaining initial windows are training.
    """
    if lookback_window < 1 or forecasting_horizon < 1:
        raise ValueError("lookback_window and forecasting_horizon must be positive")
    if not 0 < val_size + test_size < 1:
        raise ValueError("0 < val_size + test_size < 1 is required")

    raw = np.asarray(data, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("data must be an (N, 2) sequence of input,target pairs")
    x_raw = raw[:, 0]
    y_raw = raw[:, 1]

    x_windows, y_windows = [], []
    for i in range(len(x_raw) - lookback_window - forecasting_horizon + 1):
        x_windows.append(x_raw[i:i + lookback_window])
        y_windows.append(y_raw[i + lookback_window + forecasting_horizon - 1])
    if not x_windows:
        raise ValueError("not enough observations for lookback_window/forecasting_horizon")

    x_all = np.stack(x_windows)
    y_all = np.asarray(y_windows, dtype=np.float64)
    n = len(x_all)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("not enough training windows after validation/test split")

    # Slice by window to ensure normalization never sees val/test observations.
    train_x = x_all[:n_train]
    val_x = x_all[n_train:n_train + n_val]
    test_x = x_all[n_train + n_val:]
    train_y = y_all[:n_train]
    val_y = y_all[n_train:n_train + n_val]
    test_y = y_all[n_train + n_val:]

    x_mean = train_x.mean(axis=0)
    x_std = train_x.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    y_mean = float(train_y.mean())
    y_std = float(train_y.std())
    if y_std == 0.0:
        y_std = 1.0

    return ContinuousWindows(
        x_train=((train_x - x_mean) / x_std)[:, None, :],
        y_train=(train_y - y_mean) / y_std,
        x_val=((val_x - x_mean) / x_std)[:, None, :],
        y_val=(val_y - y_mean) / y_std,
        x_test=((test_x - x_mean) / x_std)[:, None, :],
        y_test=(test_y - y_mean) / y_std,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )


def _physical_mse(pred_norm: np.ndarray, target_norm: np.ndarray, y_std: float) -> float:
    return float(np.mean((pred_norm - target_norm) ** 2) * y_std ** 2)


# ==================== Ridge ====================

def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge_lambda: float):
    design = np.column_stack([x.reshape(len(x), -1), np.ones(len(x))])
    n_features = design.shape[1]
    regularizer = np.eye(n_features) * ridge_lambda
    regularizer[-1, -1] = 0.0  # do not penalize the intercept
    return np.linalg.solve(design.T @ design + regularizer, design.T @ y)


def _ridge_predict(weights, x: np.ndarray) -> np.ndarray:
    return np.column_stack([x.reshape(len(x), -1), np.ones(len(x))]) @ weights


def run_ridge_baseline(data, lookback_window: int, forecasting_horizon: int,
                       val_size: float = 0.2, test_size: float = 0.2,
                       ridge_lambda: float = 1e-2) -> dict:
    """Fit closed-form ridge regression and report physical-unit MSE."""
    w = build_continuous_windows(data, lookback_window, forecasting_horizon,
                                 val_size=val_size, test_size=test_size)
    weights = _fit_ridge(w.x_train, w.y_train, ridge_lambda)
    return {
        "method": "ridge",
        "val_mse": _physical_mse(_ridge_predict(weights, w.x_val), w.y_val, w.y_std),
        "test_mse": _physical_mse(_ridge_predict(weights, w.x_test), w.y_test, w.y_std),
        "n_test": len(w.y_test),
        "epochs": 0,
    }


# ==================== Deep model implementations ====================

def _moving_average(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    edge = kernel_size // 2
    left = x[:, :, :1].repeat(1, 1, edge)
    right = x[:, :, -1:].repeat(1, 1, edge)
    padded = torch.cat([left, x, right], dim=2)
    kernel = torch.full((1, 1, kernel_size), 1.0 / kernel_size, device=x.device, dtype=x.dtype)
    return nn.functional.conv1d(padded, kernel)


class DLinear(nn.Module):
    """Channel-independent trend/seasonal linear forecaster."""

    def __init__(self, lookback_window: int, output_size: int = 1, kernel_size: int = 9):
        super().__init__()
        self.trend = nn.Linear(lookback_window, output_size)
        self.seasonal = nn.Linear(lookback_window, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0, :]
        trend = _moving_average(x.unsqueeze(1), 9)[:, 0, :]
        seasonal = x - trend
        return self.trend(trend) + self.seasonal(seasonal)


class PatchTST(nn.Module):
    """Small channel-independent patch Transformer for short univariate windows."""

    def __init__(self, lookback_window: int, output_size: int = 1,
                 d_model: int = 32, nhead: int = 4, dim_feedforward: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        if lookback_window < 4:
            raise ValueError("PatchTST requires lookback_window >= 4")
        patch_len = 5 if lookback_window >= 20 else max(2, lookback_window // 2)
        stride = patch_len
        n_patches = (lookback_window - patch_len) // stride + 1
        if n_patches < 1:
            raise ValueError("lookback_window is too short for a PatchTST patch")
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = n_patches
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0, :].unfold(dimension=1, size=self.patch_len, step=self.stride)
        tokens = self.patch_embed(x) + self.position
        encoded = self.encoder(tokens).mean(dim=1)
        return self.head(encoded)


class GRUForecast(nn.Module):
    def __init__(self, lookback_window: int, output_size: int = 1,
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x.transpose(1, 2))
        return self.head(hidden[-1])


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.norm = nn.GroupNorm(1, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = nn.functional.pad(x, (self.padding, 0))
        y = self.dropout(torch.nn.functional.gelu(self.norm(self.conv(y))))
        return torch.nn.functional.gelu(y + self.residual(x))


class TCNForecast(nn.Module):
    def __init__(self, lookback_window: int, output_size: int = 1,
                 hidden_size: int = 48, dropout: float = 0.1):
        super().__init__()
        channels = [hidden_size, hidden_size, hidden_size]
        blocks = []
        in_channels = 1
        for i, out_channels in enumerate(channels):
            blocks.append(TemporalBlock(in_channels, out_channels, dilation=2 ** i,
                                        dropout=dropout))
            in_channels = out_channels
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.network(x)[:, :, -1])


MODEL_BUILDERS = {
    "dlinear": DLinear,
    "patchtst": PatchTST,
    "gru": GRUForecast,
    "tcn": TCNForecast,
}

_LEARNING_RATES = {
    "dlinear": 1e-3,
    "patchtst": 5e-4,
    "gru": 1e-3,
    "tcn": 1e-3,
}


# ==================== Deep training / evaluation ====================

def _arrays_to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool,
                      seed: int) -> DataLoader:
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32),
                            torch.as_tensor(y, dtype=torch.float32))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device))
            total += nn.functional.mse_loss(pred, y.to(device).unsqueeze(-1)).item()
    return total / max(len(loader), 1)


def run_forecasting_baseline(data, method: str, lookback_window: int,
                             forecasting_horizon: int, seed: int = 42,
                             epochs: int = 50, batch_size: int = 64,
                             patience: int = 10, lr: float | None = None,
                             val_size: float = 0.2, test_size: float = 0.2) -> dict:
    """Train one deep baseline under the continuous forecasting contract."""
    if method == "ridge":
        return run_ridge_baseline(data, lookback_window, forecasting_horizon,
                                  val_size=val_size, test_size=test_size)
    if method not in MODEL_BUILDERS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(MODEL_BUILDERS)}")
    if epochs < 1:
        raise ValueError("epochs must be positive")

    torch.manual_seed(seed)
    np.random.seed(seed)
    w = build_continuous_windows(data, lookback_window, forecasting_horizon,
                                 val_size=val_size, test_size=test_size,
                                 batch_size=batch_size)
    train_loader = _arrays_to_loader(w.x_train, w.y_train, batch_size, True, seed)
    val_loader = _arrays_to_loader(w.x_val, w.y_val, batch_size, False, seed)
    test_loader = _arrays_to_loader(w.x_test, w.y_test, batch_size, False, seed)

    model = MODEL_BUILDERS[method](lookback_window, output_size=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr or _LEARNING_RATES[method])
    best_val = float("inf")
    best_state = None
    epochs_used = 0
    bad_epochs = 0

    for epoch in range(epochs):
        epochs_used = epoch + 1
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x.to(device)), y.to(device).unsqueeze(-1))
            loss.backward()
            optimizer.step()

        val_mse_norm = _evaluate(model, val_loader)
        if val_mse_norm + 1e-5 < best_val:
            best_val = val_mse_norm
            best_state = {k: value.detach().cpu() for k, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {
        "method": method,
        "val_mse": _physical_mse(np.array([_evaluate(model, val_loader)]), np.zeros(1), w.y_std),
        "test_mse": _physical_mse(np.array([_evaluate(model, test_loader)]), np.zeros(1), w.y_std),
        "n_test": len(w.y_test),
        "epochs": epochs_used,
    }
