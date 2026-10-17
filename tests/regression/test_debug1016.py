import pickle
from pathlib import Path

import numpy as np
import torch

from fgl_common.data import create_time_series_dataset
from fgl_common.training import compute_shared_bin_edges
from fgl_common.continuous_fgl import _cont_mse

from fgl_common.transfer import evaluate_teacher_transfer

ROOT = Path(__file__).resolve().parents[2]
CSTR = ROOT / "cstr"


class OneD(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(1, 2), keepdim=True)


def _series(n=120):
    return [(float(i), float(i)) for i in range(n)]


def test_offset_is_applied_before_splitting():
    L, H = 4, 3
    off = H - 1
    student = create_time_series_dataset(_series(), L, H, 10, 0.2, 0.2, 0, MSE=True)
    teacher = create_time_series_dataset(_series(), L, 1, 10, 0.2, 0.2, off, MSE=True)
    s = next(iter(student[2]))[2]
    t = next(iter(teacher[2]))[2]
    assert len(student[2]) == len(teacher[2])
    assert torch.equal(s, t)


def test_shared_bin_edges_use_training_targets_only():
    data = _series()
    data[90] = (90.0, 10_000.0)
    edges, lo, hi = compute_shared_bin_edges(data, 4, 10)
    assert lo == 4.0 and hi == 73.0


def test_continuous_mse_is_physical_and_prevents_broadcast():
    assert _cont_mse(OneD(), torch.zeros(3, 4), torch.full((3,), 1.0), y_std=2.0) == 4.0



def test_transfer_uses_shared_edges_and_converged_control(tmp_path):
    def save_tau(tau):
        raw = torch.linspace(0, 0.9, 300) + tau / 10
        data = torch.stack([raw, raw], dim=1)
        with open(tmp_path / f"tau{tau}.pkl", "wb") as f:
            pickle.dump(data, f)

    save_tau(1); save_tau(9)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.flat = torch.nn.Flatten(); self.fc = torch.nn.Linear(8, 50)
        def forward(self, x):
            return self.fc(self.flat(x))
    teacher = Tiny()
    rows = evaluate_teacher_transfer(str(tmp_path / "tau1.pkl"), teacher, L=8, finetune_epochs=1, scratch_epochs=1)
    assert {r["condition"] for r in rows} == {"zero_shot", "finetuned", "scratch", "scratch_converged"}


def _assert_pattern(relative, pattern):
    source = (CSTR / relative).read_text()
    assert pattern in source, (relative, pattern)


def test_visualization_selects_models_by_validation_mse():
    _assert_pattern("plot_pretrain_finetune_eiter.py", "snaps.append((r, copy.deepcopy(s), float(mv), float(mt)))")
    _assert_pattern("plot_pretrain_finetune_eiter.py", "min(snaps, key=lambda t: t[2])")
    _assert_pattern("plot_method_comparison.py", "_cap.append((round_idx, _copy.deepcopy(student), float(mse_val), float(mse_test)))")
    _assert_pattern("plot_method_comparison.py", "best_round, best_model, best_val_mse, best_mse = min(captured, key=lambda t: t[2])")
    _assert_pattern("plot_method_comparison.py", "min(captured, key=lambda t: t[2])")


def test_teacher_sweep_uses_validation_for_configuration_and_training():
    _assert_pattern("run_teacher_sweep.py", "x_val, y_val = _to_tensor(TX[n_tr:n_tr+n_val]")
    _assert_pattern("run_teacher_sweep.py", "val_mse = _eval_mse(model, x_val, y_val, Ys)")
    _assert_pattern("run_teacher_sweep.py", "teacher_val_mse_mean")
    _assert_pattern("run_teacher_sweep.py", "best = min(rows, key=lambda r: r[\"teacher_val_mse_mean\"])")
    _assert_pattern("run_teacher_sweep.py", "_train_regressor(teacher, tx_tr, y_tr, tx_val, y_val")


def test_foundation_teacher_selects_by_validation_and_summarizes_sources():
    _assert_pattern("run_foundation_teacher.py", "ft_val_mse = _eval(finetuned, x_val, y_val, ys)")
    _assert_pattern("run_foundation_teacher.py", "scratch_val_mse = _eval(model, x_val, y_val, ys)")
    _assert_pattern("run_foundation_teacher.py", "t_ft_val < t_scratch_val")
    _assert_pattern("run_foundation_teacher.py", "sorted({r[\"teacher_best_source\"] for r in g})")


def test_complex_generator_connects_delayed_value_and_naming():
    source = (CSTR / "generate_complex.py").read_text()
    assert source.count("dh_smooth = filter_beta * delayed_val") >= 3
    assert "delay_bufs = [np.zeros(max(1, int(tau))) for tau in taus]" in source
    assert "delay_buf = np.zeros(max(1, int(tau_delay)))" in source
    assert "(50, 0.02, 0.5)" in source
    assert "{ve:.1e}" in source


def test_student_sweep_squeezes_only_extra_prediction_dimension():
    source = (CSTR / "run_student_sweep.py").read_text()
    assert source.count("out = _forward_1d(model, xs.to(device))") == 2
    assert "model(sx_val.to(device))" not in source
    assert "args.fresh = False" in source


def test_continuous_iterative_fgl_prevents_teacher_broadcast():
    import warnings
    from fgl_common.continuous_fgl import run_continuous_iterative_fgl

    class LinearOneD(torch.nn.Module):
        def __init__(self, L):
            super().__init__()
            self.fc = torch.nn.Linear(L, 1)

        def forward(self, x):
            return self.fc(x.squeeze(1))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_continuous_iterative_fgl(
            _series(120), L=4, H=2, model_fn=LinearOneD,
            epochs=1, round_epochs=1, K=1, batch_size=16, verbose=False,
        )
    broadcast_warnings = [w for w in caught if "broadcasting" in str(w.message)]
    assert broadcast_warnings == []


def test_param_mod_volume_parameters_change_trajectory():
    from cstr.generate_complex import simulate_param_mod
    a = simulate_param_mod(
        tau_delay=5, volume_start=8e-6, volume_end=14e-6, mod_period=20.0,
        t_end=4.0, dt=0.1, t_onset=0.0,
    )[:, 0]
    b = simulate_param_mod(
        tau_delay=5, volume_start=6e-6, volume_end=16e-6, mod_period=30.0,
        t_end=4.0, dt=0.1, t_onset=0.0,
    )[:, 0]
    assert not np.allclose(a, b)




def test_seq2seq_shared_bin_edges_are_train_only():
    from fgl_common.training import _shared_seq_bin_edges
    data = _series()
    data[90] = (90.0, 10_000.0)
    edges, lo, hi = _shared_seq_bin_edges(data, lookback_window=4, horizon=3, num_bins=10)
    assert lo == 4.0 and hi == 75.0


def test_temporal_rnn_recurs_over_time():
    from fgl_common.models import TemporalRNN
    torch.manual_seed(0)
    model = TemporalRNN(4, 4, 1, 1)
    assert model.rnn.input_size == 1
    assert model(torch.zeros(2, 1, 4)).shape == (2, 1)


def test_delayed_control_uses_delayed_sample_for_each_tau():
    src = (CSTR / "generate_complex.py").read_text()
    assert "dh_smooths[i] = filter_beta * delayed_val" in src
    assert "dh_smooth = filter_beta * delayed_val" in src
    assert "np.zeros(max(1, int(tau / dt)))" not in src
    assert "np.zeros(max(1, int(tau_delay / dt)))" not in src
    assert "filter_beta = 0.03" in src
    assert "signs=(1, 1)" in src
    assert "feedback_amplitude=0.9" in src
    assert src.count("(50, 0.02, 0.5)") >= 1
    assert "{ve:.1e}" in src


def test_tau_changes_delayed_dynamics_after_buffer_warmup():
    from cstr.generate_complex import simulate_param_mod
    args = dict(t_end=4.0, dt=0.1, t_onset=0.0, feedback_amplitude=0.05)
    a = simulate_param_mod(tau_delay=1, **args)[:, 0]
    b = simulate_param_mod(tau_delay=10, **args)[:, 0]
    assert not np.allclose(a[15:], b[15:])


def test_stage_both_loads_validation_selected_teacher():
    source = (CSTR / "run_teacher_sweep.py").read_text()
    assert 'float(r["teacher_val_mse_mean"])' in source
    assert source.count('best = min(sweep_rows, key=lambda r: float(r["teacher_mse_mean"]))') == 1


def test_foundation_family_files_exist():
    from cstr.run_foundation_teacher import FAMILIES
    missing = [name for names in FAMILIES.values() for name in names
               if not (CSTR / "data" / name).exists()]
    assert missing == []


def test_foundation_csv_is_saved_before_summary_printing():
    source = (CSTR / "run_foundation_teacher.py").read_text()
    assert source.index("os.replace(tmp, _CSV_PATH)") < source.index("print(\"SUMMARY\")")
    assert "({src}," not in source


def test_student_sweep_selects_architecture_by_validation():
    source = (CSTR / "run_student_sweep.py").read_text()
    assert '"student_val_mse": student_val_mse' in source
    assert 'float(r["student_val_mse"])' in source
    assert 'float(r["student_mse"])' not in source.split("def _best_from_csv")[1]


def test_student_sweep_preserves_validation_columns():
    source = (CSTR / "run_student_sweep.py").read_text()
    assert '"baseline_val_mse",\n              "student_mse", "student_val_mse"' in source


def test_student_sweep_csv_save_preserves_columns():
    source = (CSTR / "run_student_sweep.py").read_text()
    save = source.index("def _save_rows(")
    assert source.index('"baseline_val_mse"', save) < source.index('"student_mse"', save)
    assert source.index('"student_val_mse"', save) < source.index("os.replace(tmp, _CSV_PATH)", save)
