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


# ==================== Task 3: _iterate_student ====================
from fgl_common import compute_shared_bin_edges, create_time_series_dataset
from fgl_common.training import _iterate_student, evaluate


@pytest.fixture(scope="module")
def tiny_setup(tiny_loaders):
    """训练一个共享 teacher + round-0 student_0 供 _iterate_student 测试。"""
    L, H, nb = tiny_loaders["L"], tiny_loaders["H"], tiny_loaders["num_bins"]
    data = _tiny_series()
    bin_edges, _, _ = compute_shared_bin_edges(data, L, nb)
    student_train, student_val, student_test, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=H, num_bins=nb,
        val_size=0.2, test_size=0.2, offset=0, batch_size=8, bin_edges=bin_edges)
    teacher_train, teacher_val, _, _, _ = create_time_series_dataset(
        data=data, lookback_window=L, forecasting_horizon=1, num_bins=nb,
        val_size=0.2, test_size=0.2, offset=H - 1, batch_size=8, bin_edges=bin_edges)
    torch.manual_seed(0)
    teacher = RNN(L, 16, nb, 1).to(device)
    ce = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    for _ in range(3):
        teacher.train()
        for _, x, y in teacher_train:
            x = x.float().to(device).view(-1, 1, L)
            opt.zero_grad(); ce(teacher(x), y.long().to(device)).backward(); opt.step()
    teacher.eval()
    # round-0 student(uniform KL,from scratch)
    torch.manual_seed(1)
    student_0 = RNN(L, 16, nb, 1).to(device)
    from fgl_common import KL
    opt = torch.optim.Adam(student_0.parameters(), lr=1e-3)
    for _ in range(3):
        student_0.train()
        for (_, xs, ys), (_, xt, _) in zip(student_train, teacher_train):
            xs = xs.float().to(device).view(-1, 1, L)
            out = student_0(xs)
            xt = xt.float().to(device).view(-1, 1, L)
            with torch.no_grad():
                tlog = teacher(xt)
            loss = 0.5 * ce(out, ys.long().to(device)) + KL(out, tlog, 4.0, 0.5)
            opt.zero_grad(); loss.backward(); opt.step()
    student_0.eval()
    return dict(teacher=teacher, student_0=student_0,
                student_train=student_train, teacher_train=teacher_train,
                student_val=student_val, student_test=student_test,
                sf=tiny_loaders["sf"], tf=tiny_loaders["tf"],
                indices=tiny_loaders["indices"], L=L, H=H, nb=nb)


def test_iterate_student_structure(tiny_setup):
    s = tiny_setup
    res = _iterate_student(
        s["student_0"], s["teacher"], "A", max_rounds=2,
        student_train=s["student_train"], teacher_train=s["teacher_train"],
        student_val=s["student_val"], student_test=s["student_test"],
        student_train_full=s["sf"], teacher_train_full=s["tf"],
        student_train_indices=s["indices"],
        L=s["L"], H=s["H"], alpha=0.5, temperature=4.0,
        round_epochs=2, patience=5, eps=0.01, N_stall=2, lr=1e-3)
    assert {"rounds_used", "total_epochs", "mse_curve_val", "mse_curve_test", "student"} <= set(res)
    assert res["rounds_used"] <= 2
    assert len(res["mse_curve_val"]) == res["rounds_used"] + 1
    assert len(res["mse_curve_test"]) == res["rounds_used"] + 1
    assert res["total_epochs"] >= res["rounds_used"]


def test_iterate_student_keeps_best_by_val(tiny_setup):
    s = tiny_setup
    res = _iterate_student(
        s["student_0"], s["teacher"], "E", max_rounds=3,
        student_train=s["student_train"], teacher_train=s["teacher_train"],
        student_val=s["student_val"], student_test=s["student_test"],
        student_train_full=s["sf"], teacher_train_full=s["tf"],
        student_train_indices=s["indices"],
        L=s["L"], H=s["H"], alpha=0.5, temperature=4.0,
        round_epochs=2, patience=5, eps=0.01, N_stall=2, lr=1e-3)
    # 返回的 student 应对应 val 最优那一轮
    best_val = min(res["mse_curve_val"])
    assert abs(evaluate(res["student"], s["student_val"], s["L"]) - best_val) < 1e-4


def test_iterate_student_maxrounds0_is_round0(tiny_setup):
    s = tiny_setup
    res = _iterate_student(
        s["student_0"], s["teacher"], "A", max_rounds=0,
        student_train=s["student_train"], teacher_train=s["teacher_train"],
        student_val=s["student_val"], student_test=s["student_test"],
        student_train_full=s["sf"], teacher_train_full=s["tf"],
        student_train_indices=s["indices"],
        L=s["L"], H=s["H"], alpha=0.5, temperature=4.0,
        round_epochs=2, patience=5, eps=0.01, N_stall=2, lr=1e-3)
    assert res["rounds_used"] == 0
    assert len(res["mse_curve_val"]) == 1
    # 与输入 student_0 的 val MSE 一致(未训练)
    assert abs(res["mse_curve_val"][0] - evaluate(s["student_0"], s["student_val"], s["L"])) < 1e-4


# ==================== Task 4: run_iterative_distillation ====================
from fgl_common import run_iterative_distillation


def _tiny_data():
    return _tiny_series(n=400)


def test_run_iterative_four_arms_structure():
    res = run_iterative_distillation(
        _tiny_data(), L=20, H=15, num_bins=50, epochs=3, round_epochs=2,
        batch_size=8, K=2, seed=0, verbose=False)
    assert set(res) == {"A_single", "E_single", "A_iter", "E_iter"}
    expected = {"teacher_mse", "baseline_mse", "student_mse", "fgl_delta", "init_delta",
                "rounds_used", "total_epochs", "mse_curve_val", "mse_curve_test"}
    for arm, r in res.items():
        assert expected <= set(r), f"{arm} missing keys"


def test_all_arms_share_round0():
    res = run_iterative_distillation(
        _tiny_data(), L=20, H=15, num_bins=50, epochs=3, round_epochs=2,
        batch_size=8, K=2, seed=0, verbose=False)
    a0 = res["A_single"]["mse_curve_test"][0]
    for arm in ("E_single", "A_iter", "E_iter"):
        assert abs(res[arm]["mse_curve_test"][0] - a0) < 1e-6, f"{arm} round-0 differs"


def test_A_single_is_round0_only():
    res = run_iterative_distillation(
        _tiny_data(), L=20, H=15, num_bins=50, epochs=3, round_epochs=2,
        batch_size=8, K=2, seed=0, verbose=False)
    assert res["A_single"]["rounds_used"] == 0
    assert len(res["A_single"]["mse_curve_val"]) == 1


def test_E_single_uses_one_round():
    res = run_iterative_distillation(
        _tiny_data(), L=20, H=15, num_bins=50, epochs=3, round_epochs=2,
        batch_size=8, K=2, seed=0, verbose=False)
    assert res["E_single"]["rounds_used"] == 1


# ==================== E-soft variant (sigmoid + soft floor) ====================
from fgl_common import compute_weights


def test_compute_weights_E_soft_bounded_and_ordered():
    """E-soft: sigmoid 软地板,权重恒落于 [w_floor=0.2, W_MAX=4.0],
    大正 gap 权重高、负 gap(学生优于老师)权重低(近地板)。"""
    indices = [0, 1, 2, 3]
    se = {0: 0.1, 1: 0.5, 2: 0.9, 3: 0.1}      # student per-sample MSE
    te = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.9}      # teacher per-sample MSE
    # signed gaps = se - te: {0:0, 1:+0.4, 2:+0.8, 3:-0.8}
    w, raw, norm = compute_weights('E-soft', se, te, indices)
    assert set(w) == set(indices)
    assert all(0.2 - 1e-6 <= v <= 4.0 + 1e-6 for v in w.values())  # [w_floor, W_MAX]
    assert w[2] > w[3]    # 大正 gap (0.8) > 负 gap (-0.8)
    assert w[3] < 1.0     # 负 gap 样本拿到接近地板的低权重
    assert w[2] > w[0]    # 大正 gap > 零 gap


def test_compute_weights_E_soft_w_floor_param():
    """w_floor 参数抬高软地板:权重下界 ≈ w_floor。"""
    indices = [0, 1, 2, 3]
    se = {0: 0.1, 1: 0.5, 2: 0.9, 3: 0.1}
    te = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.9}
    w_default, _, _ = compute_weights('E-soft', se, te, indices)
    w_hi, _, _ = compute_weights('E-soft', se, te, indices, w_floor=0.5)
    # 抬高地板 → 最小权重上升
    assert min(w_hi.values()) > min(w_default.values())
    assert all(v >= 0.5 - 1e-6 for v in w_hi.values())   # 新下界 0.5
    assert all(v <= 4.0 + 1e-6 for v in w_hi.values())    # 上界不变
