import pickle
import torch

from fgl_common.transfer import evaluate_teacher_transfer, split_lopo


def test_split_lopo_excludes_target():
    paths = ["tau10.pkl", "tau20.pkl", "tau30.pkl"]
    source, target = split_lopo(paths, "tau20.pkl")
    assert target == "tau20.pkl"
    assert source == ["tau10.pkl", "tau30.pkl"]


def test_evaluate_teacher_transfer_conditions(tmp_path):
    N, L, bins = 260, 8, 10
    for tau in (10, 20):
        data = torch.stack(
            [torch.linspace(0, 1, N + L), torch.linspace(0, 1, N + L) + tau / 100],
            dim=1,
        )
        with open(tmp_path / f"tau{tau}.pkl", "wb") as f:
            pickle.dump(data, f)
    teacher = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(L, bins))
    rows = evaluate_teacher_transfer(
        str(tmp_path / "tau10.pkl"),
        teacher,
        L=L,
        H=1,
        hidden=8,
        finetune_epochs=1,
        scratch_epochs=1,
        num_bins=bins,
        seed=0,
    )
    assert {r["condition"] for r in rows} == {"zero_shot", "finetuned", "scratch"}
    assert all(0 <= r["teacher_mse"] < bins ** 2 for r in rows)
