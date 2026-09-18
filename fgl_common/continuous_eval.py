"""Post-hoc physical-value evaluation for bin-classified forecasters.

The historical FGL classifiers predict a bin index.  These helpers map that
index back to a physical target value without consulting validation or test
targets: the centroid decoder uses training targets only.
"""
from __future__ import annotations

import numpy as np
import torch


def bin_centers_from_edges(bin_edges: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    """Return one physical representative per bin.

    With ``B-1`` interior edges there are ``B`` bins.  End bins use the observed
    minimum/maximum targets; inner bins use the geometric edge midpoints.
    """
    edges = np.asarray(bin_edges, dtype=float)
    centers = np.empty(len(edges) + 1, dtype=float)
    centers[0] = y_min
    centers[-1] = y_max
    if len(edges) > 1:
        centers[1:-1] = (edges[:-1] + edges[1:]) / 2.0
    return centers


def _split_bounds(n_windows: int, val_size: float, test_size: float) -> tuple[int, int, int]:
    n_test = int(n_windows * test_size)
    n_val = int(n_windows * val_size)
    n_train = n_windows - n_test - n_val
    if n_train <= 0 or n_val < 0 or n_test < 0:
        raise ValueError("invalid validation/test split")
    return n_train, n_val, n_test


def _split_arrays(data, L: int, H: int, split: str,
                  val_size: float, test_size: float):
    raw = np.asarray(data, dtype=float)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("data must have shape (N, 2)")
    x_raw, y_raw = raw[:, 0], raw[:, 1]
    n_windows = len(x_raw) - L - H + 1
    if n_windows <= 0:
        raise ValueError("not enough observations")
    n_train, n_val, n_test = _split_bounds(n_windows, val_size, test_size)
    bounds = {"train": (0, n_train), "val": (n_train, n_train + n_val),
              "test": (n_train + n_val, n_windows)}
    if split not in bounds:
        raise ValueError("split must be train, val, or test")
    start, stop = bounds[split]
    window_starts = np.arange(start, stop)
    x_windows = np.stack([x_raw[i:i + L] for i in window_starts])
    y_targets = np.asarray([y_raw[i + L + H - 1] for i in window_starts])
    return x_windows, y_targets


def fit_centroid_decoder(data, L: int, H: int, bin_edges: np.ndarray,
                         y_min: float, y_max: float, num_bins: int | None = None,
                         val_size: float = 0.2, test_size: float = 0.2) -> np.ndarray:
    """Fit class -> training-target mean; unseen classes fall back to bin centers."""
    edges = np.asarray(bin_edges, dtype=float)
    n_bins = num_bins if num_bins is not None else len(edges) + 1
    x_train, y_train = _split_arrays(data, L, H, "train", val_size, test_size)
    classes = np.digitize(y_train, edges)
    decoder = bin_centers_from_edges(edges, y_min, y_max)
    for klass in np.unique(classes):
        decoder[klass] = y_train[classes == klass].mean()
    if len(decoder) != n_bins:
        raise ValueError("decoder size does not match num_bins")
    return decoder


def evaluate_classifier_as_continuous(model, data, L: int, H: int,
                                      bin_edges: np.ndarray, y_min: float,
                                      y_max: float, decoder="bin_center",
                                      split: str = "test", val_size: float = 0.2,
                                      test_size: float = 0.2, batch_size: int = 256) -> float:
    """Evaluate argmax class predictions against original physical targets."""
    edges = np.asarray(bin_edges, dtype=float)
    if isinstance(decoder, str):
        if decoder != "bin_center":
            raise ValueError("string decoder must be 'bin_center'")
        values = bin_centers_from_edges(edges, y_min, y_max)
    else:
        values = np.asarray(decoder, dtype=float)

    x_windows, y_targets = _split_arrays(data, L, H, split, val_size, test_size)
    classes = np.digitize(x_windows, edges)
    predictions = []
    from .training import device  # lazy import avoids module initialization cycle
    model.eval()
    with torch.no_grad():
        for start in range(0, len(classes), batch_size):
            x = torch.as_tensor(classes[start:start + batch_size], dtype=torch.float32)
            x = x.view(len(x), 1, L)
            logits = model(x.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    predictions = np.asarray(predictions, dtype=int)
    if len(predictions) != len(y_targets):
        raise RuntimeError("prediction/target count mismatch")
    return float(np.mean((values[predictions] - y_targets) ** 2))
