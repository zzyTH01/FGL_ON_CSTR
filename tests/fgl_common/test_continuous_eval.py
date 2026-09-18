import numpy as np
import torch
from torch import nn

from fgl_common.continuous_eval import (
    bin_centers_from_edges,
    evaluate_classifier_as_continuous,
    fit_centroid_decoder,
)


class ConstantModel(nn.Module):
    def __init__(self, klass, n_bins=4):
        super().__init__()
        self.klass = klass
        self.n_bins = n_bins

    def forward(self, x):
        logits = torch.full((len(x), self.n_bins), -10.0)
        logits[:, self.klass] = 10.0
        return logits


def _data():
    x = np.linspace(0.0, 1.0, 80)
    y = 1.0 + 2.0 * x
    return list(zip(x, y))


def _test_targets(data, L=2, H=2, val_size=0.2, test_size=0.2):
    raw_y = np.asarray([item[1] for item in data])
    n_windows = len(data) - L - H + 1
    n_test = int(n_windows * test_size)
    n_val = int(n_windows * val_size)
    n_train = n_windows - n_test - n_val
    starts = np.arange(n_train + n_val, n_windows)
    return raw_y[starts + L + H - 1]


def test_bin_centers_use_outer_targets_and_inner_midpoints():
    edges = np.array([0.25, 0.5, 0.75])
    centers = bin_centers_from_edges(edges, y_min=0.0, y_max=1.0)
    np.testing.assert_allclose(centers, [0.0, 0.375, 0.625, 1.0])


def test_centroid_decoder_is_fit_on_train_windows_only():
    data = _data()
    edges = np.array([1.25, 1.5, 1.75])
    decoder = fit_centroid_decoder(data, L=2, H=2, bin_edges=edges,
                                   y_min=1.0, y_max=2.0,
                                   val_size=0.2, test_size=0.2)
    assert len(decoder) == 4
    # Training windows are starts 0..46; targets are starts + L + H - 1.
    train_targets = np.asarray([data[j + 3][1] for j in range(47)])
    expected = train_targets[(train_targets >= 1.25) & (train_targets < 1.5)].mean()
    np.testing.assert_allclose(decoder[1], expected)
    assert 1.25 <= decoder[1] < 1.5


def test_constant_classifier_maps_to_physical_mse_with_bin_centers():
    data = _data()
    edges = np.array([1.25, 1.5, 1.75])
    model = ConstantModel(1)
    mse = evaluate_classifier_as_continuous(
        model, data, L=2, H=2, bin_edges=edges, y_min=1.0, y_max=2.0,
        decoder="bin_center", split="test", val_size=0.2, test_size=0.2,
    )
    test_y = _test_targets(data)
    expected = np.mean((1.375 - test_y) ** 2)
    np.testing.assert_allclose(mse, expected)


def test_constant_classifier_accepts_fitted_centroid_mapping():
    data = _data()
    edges = np.array([1.25, 1.5, 1.75])
    decoder = fit_centroid_decoder(data, L=2, H=2, bin_edges=edges,
                                   y_min=1.0, y_max=2.0,
                                   val_size=0.2, test_size=0.2)
    model = ConstantModel(1)
    mse = evaluate_classifier_as_continuous(
        model, data, L=2, H=2, bin_edges=edges, y_min=1.0, y_max=2.0,
        decoder=decoder, split="test", val_size=0.2, test_size=0.2,
    )
    test_y = _test_targets(data)
    expected = np.mean((decoder[1] - test_y) ** 2)
    np.testing.assert_allclose(mse, expected)
