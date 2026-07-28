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
