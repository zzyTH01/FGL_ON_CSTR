import numpy as np
import torch

from fgl_common.teacher_transfer import run_teacher_student_transfer


def test_transfer_rows_and_ratio():
    N, L, H = 260, 8, 3
    x = torch.stack([torch.linspace(0, 1, N + L), torch.sin(torch.linspace(0, 6, N + L))], dim=1)
    rows = run_teacher_student_transfer(
        x.numpy(), L=L, H=H, num_bins=10, seed=0, epochs=1, batch_size=32
    )
    conditions = {r["condition"] for r in rows}
    assert {"baseline", "more_epochs", "lower_lr", "ensemble", "regularized"} <= conditions
    for row in rows:
        if row["teacher_gain"] > 1e-8:
            assert abs(row["transfer_ratio"] - row["student_gain"] / row["teacher_gain"]) < 1e-6
