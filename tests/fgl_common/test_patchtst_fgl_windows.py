import numpy as np
import pytest
import torch

from fgl_common.patchtst_fgl import build_continuous_fgl_windows


def test_windows_align_teacher_and_student_on_same_target():
    n, L, H = 40, 5, 3
    x = np.arange(n, dtype=float)
    y = np.arange(100, 100 + n, dtype=float)
    data = list(zip(x, y))
    windows = build_continuous_fgl_windows(data, L, H, val_size=0.2, test_size=0.2)
    n_windows = n - L - H + 1
    n_test = int(n_windows * 0.2)
    n_val = int(n_windows * 0.2)
    n_train = n_windows - n_val - n_test
    assert len(windows.train_y) == n_train
    # De-normalized first training target must be y[L+H-1].
    np.testing.assert_allclose(
        float(windows.train_y[0]) * windows.y_std + windows.y_mean,
        y[L + H - 1],
    )
    # Student i=0 observes x[0:L]; teacher observes x[H-1:H-1+L].
    np.testing.assert_allclose(
        windows.train_student_x[0, 0] * windows.x_std + windows.x_mean, x[:L],
        rtol=1e-6, atol=1e-6,
    )
    np.testing.assert_allclose(
        windows.train_teacher_x[0, 0] * windows.x_std + windows.x_mean,
        x[H - 1:H - 1 + L], rtol=1e-6, atol=1e-6,
    )
