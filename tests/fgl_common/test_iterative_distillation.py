from fgl_common.training import _should_stop


def test_stop_cap():
    # 4 条记录 => t=3 >= max_rounds=3
    assert _should_stop([1.0, 0.9, 0.8, 0.7], eps=0.01, N_stall=2, max_rounds=3) == (True, "cap")


def test_stop_degradation():
    # 0.8 -> 0.85 上升(退化)
    assert _should_stop([1.0, 0.9, 0.8, 0.85], eps=0.01, N_stall=2, max_rounds=5) == (True, "degradation")


def test_stop_stall():
    # 0.50->0.496 (0.8%), 0.496->0.492 (0.8%),均 < 1%
    assert _should_stop([0.50, 0.496, 0.492], eps=0.01, N_stall=2, max_rounds=5) == (True, "stall")


def test_stop_continue_big_improvement():
    assert _should_stop([1.0, 0.5], eps=0.01, N_stall=2, max_rounds=5) == (False, "continue")


def test_stop_round0_continues():
    assert _should_stop([1.0], eps=0.01, N_stall=2, max_rounds=5) == (False, "continue")


def test_stop_stall_reset_by_big_improvement():
    # 0.50->0.30 (大), 0.30->0.296 (停滞) => 只 1 次停滞,不够 N_stall=2
    assert _should_stop([0.50, 0.30, 0.296], eps=0.01, N_stall=2, max_rounds=5) == (False, "continue")


# ==================== Task 2: _compute_arm_weights ====================
import pytest
import torch
from fgl_common import RNN
from fgl_common.training import _compute_arm_weights, device


def _tiny_series(n=400, seed=0):
    """平滑周期序列 [(x, y), ...],x=y=series(自回归)。"""
    import numpy as np
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    series = np.sin(t * 0.3) * 50.0 + 100.0 + 2.0 * rng.randn(n)
    return [(float(series[i]), float(series[i])) for i in range(n)]


@pytest.fixture(scope="module")
def tiny_loaders():
    from fgl_common import compute_shared_bin_edges, create_time_series_dataset
    L, H, num_bins = 20, 15, 50
    data = _tiny_series()
    bin_edges, _, _ = compute_shared_bin_edges(data, L, num_bins)
    student_train_full, _, _, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H, num_bins=num_bins,
        val_size=0.2, test_size=0.2, offset=0, batch_size=1, bin_edges=bin_edges)
    teacher_train_full, _, _, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=1, num_bins=num_bins,
        val_size=0.2, test_size=0.2, offset=H - 1, batch_size=1, bin_edges=bin_edges)
    indices = [idx[0].item() for idx, _, _ in student_train_full]
    return dict(L=L, H=H, num_bins=num_bins,
                sf=student_train_full, tf=teacher_train_full, indices=indices)


def test_weights_A_uniform(tiny_loaders):
    L, H, nb = tiny_loaders["L"], tiny_loaders["H"], tiny_loaders["num_bins"]
    student = RNN(L, 16, nb, 1).to(device)
    teacher = RNN(L, 16, nb, 1).to(device)
    w = _compute_arm_weights("A", student, teacher,
                             tiny_loaders["sf"], tiny_loaders["tf"],
                             tiny_loaders["indices"], L, H)
    assert set(w) == set(tiny_loaders["indices"])
    assert all(v == 1.0 for v in w.values())


def test_weights_E_bounded(tiny_loaders):
    L, H, nb = tiny_loaders["L"], tiny_loaders["H"], tiny_loaders["num_bins"]
    student = RNN(L, 16, nb, 1).to(device)
    teacher = RNN(L, 16, nb, 1).to(device)
    w = _compute_arm_weights("E", student, teacher,
                             tiny_loaders["sf"], tiny_loaders["tf"],
                             tiny_loaders["indices"], L, H)
    assert set(w) == set(tiny_loaders["indices"])
    assert all(0.0 <= v <= 4.0 for v in w.values())  # compute_weights E 分支恒落于 [0, W_MAX=4]
