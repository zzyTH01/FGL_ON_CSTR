"""FGL 共享数据集构造。

- ``create_time_series_dataset`` —— 滑动窗口 + 离散化,单步目标。
    迁自 ``mackey_glass/utils/utils.py``,逻辑原样保留(含 ``offset`` / ``MSE`` / ``bin_edges``)。
- ``create_seq_dataset``         —— 多步序列目标。
    收自 ``cstr/exp/fgl_cstr_seq2seq.py``。
"""
import numpy as np
import torch
from torch.utils.data import DataLoader


def create_time_series_dataset(data,
                               lookback_window: int,
                               forecasting_horizon: int,
                               num_bins: int,
                               val_size: float,
                               test_size: float,
                               offset: int = 0,
                               MSE: bool = False,
                               batch_size: int = 1,
                               bin_edges: np.ndarray = None):
    """Build train/val/test DataLoaders with fractional splits.

    Args:
        data: list of (input_value, target_value) tuples.
        lookback_window: number of past timesteps (L).
        forecasting_horizon: steps ahead for the target (H).
        num_bins: number of discretization bins.
        val_size / test_size: fractions reserved for val / test.
        offset: index shift to align student vs teacher streams
            (teacher uses ``offset=H-1`` so its window is shifted forward).
        MSE: if True, skip discretization (regression on raw values).
        batch_size: batch size for DataLoaders.
        bin_edges: pre-computed bin edges (length ``num_bins - 1``). When
            provided, used directly instead of computing from ``y_train`` —
            this keeps teacher and student on the *same* discretization.

    Returns:
        ``(train_loader, val_loader, test_loader,
           original_data_val, original_data_test)``
    """
    # build sliding windows
    x = np.array([pt[0] for pt in data])
    y = np.array([pt[1] for pt in data])
    X_windows, y_windows = [], []
    for i in range(len(x) - lookback_window - forecasting_horizon + 1):
        X_windows.append(x[i: i + lookback_window])
        y_windows.append(y[i + lookback_window + forecasting_horizon - 1])

    X = np.stack(X_windows)
    y = np.stack(y_windows)

    N = X.shape[0]
    assert 0 < val_size + test_size < 1, "val_size + test_size must be in (0,1)"

    # compute split indices
    n_test = int(N * test_size)
    n_val = int(N * val_size)
    n_train = N - n_val - n_test

    # slice
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train + n_val], X[-n_test:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train + n_val], y[-n_test:]

    original_data_val = y_val.copy()
    original_data_test = y_test.copy()

    # discretize if needed
    if not MSE:
        if bin_edges is None:
            bin_edges = np.linspace(y_train.min(), y_train.max(), num_bins - 1)
        X_train = np.digitize(X_train, bin_edges)
        X_val = np.digitize(X_val, bin_edges)
        X_test = np.digitize(X_test, bin_edges)
        y_train = np.digitize(y_train, bin_edges)
        y_val = np.digitize(y_val, bin_edges)
        y_test = np.digitize(y_test, bin_edges)

    # make tuples (idx, x, y) and apply offset
    def to_tuples(X_arr, y_arr):
        tup = [(i, X_arr[i], y_arr[i]) for i in range(len(X_arr))]
        return tup[offset:] if offset else tup

    train_tuples = to_tuples(X_train, y_train)
    val_tuples = to_tuples(X_val, y_val)
    test_tuples = to_tuples(X_test, y_test)

    train_loader = DataLoader(train_tuples, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_tuples, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_tuples, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader, original_data_val, original_data_test


def create_seq_dataset(data, lookback_window, forecasting_horizon,
                       num_bins, val_size, test_size,
                       batch_size=64, bin_edges=None):
    """Build train/val/test DataLoaders for multi-step sequence prediction.

    Each sample: input = window of L past values,
                 target = next ``forecasting_horizon`` values (a sequence).

    Returns: ``(train_loader, val_loader, test_loader, bin_edges)``.

    Collected from ``cstr/exp/fgl_cstr_seq2seq.py``.
    """
    x_raw = np.array([float(pt[0]) for pt in data])
    y_raw = np.array([float(pt[1]) for pt in data])

    L = lookback_window
    H = forecasting_horizon

    X_windows, Y_windows = [], []
    for i in range(len(x_raw) - L - H + 1):
        X_windows.append(x_raw[i: i + L])
        Y_windows.append(y_raw[i + L: i + L + H])

    X = np.stack(X_windows)   # (N, L)
    Y = np.stack(Y_windows)   # (N, H)

    N = X.shape[0]
    assert 0 < val_size + test_size < 1

    n_test = int(N * test_size)
    n_val = int(N * val_size)
    n_train = N - n_val - n_test

    X_train, X_val, X_test = X[:n_train], X[n_train:n_train + n_val], X[-n_test:]
    Y_train, Y_val, Y_test = Y[:n_train], Y[n_train:n_train + n_val], Y[-n_test:]

    if bin_edges is None:
        bin_edges = np.linspace(Y_train.min(), Y_train.max(), num_bins - 1)

    X_train_b = np.digitize(X_train, bin_edges).clip(0, num_bins - 1)
    X_val_b = np.digitize(X_val, bin_edges).clip(0, num_bins - 1)
    X_test_b = np.digitize(X_test, bin_edges).clip(0, num_bins - 1)
    Y_train_b = np.digitize(Y_train, bin_edges).clip(0, num_bins - 1)
    Y_val_b = np.digitize(Y_val, bin_edges).clip(0, num_bins - 1)
    Y_test_b = np.digitize(Y_test, bin_edges).clip(0, num_bins - 1)

    def to_loader(X_arr, Y_arr):
        X_t = torch.tensor(X_arr, dtype=torch.float32)
        Y_t = torch.tensor(Y_arr, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(X_t, Y_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

    return (to_loader(X_train_b, Y_train_b),
            to_loader(X_val_b, Y_val_b),
            to_loader(X_test_b, Y_test_b),
            bin_edges)
