from fgl_common.transfer_analysis import compute_transfer_ratios, summarize_transfer


def test_summary_and_ratio():
    rows = [
        {"path": "a", "condition": "zero_shot", "teacher_mse": 2.0, "seed": 0},
        {"path": "a", "condition": "zero_shot", "teacher_mse": 4.0, "seed": 1},
    ]
    summary = summarize_transfer(rows)
    assert summary[0]["n"] == 2
    assert summary[0]["mean"] == 3.0
    paired = [
        {"teacher_gain": 2.0, "student_gain": 1.0},
        {"teacher_gain": 4.0, "student_gain": 2.0},
    ]
    ratios = compute_transfer_ratios(paired)
    assert all(abs(r["transfer_ratio"] - 0.5) < 1e-12 for r in ratios)
