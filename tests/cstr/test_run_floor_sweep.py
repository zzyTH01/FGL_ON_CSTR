import sys, pathlib, math, csv as _csv
_CSTR = pathlib.Path(__file__).resolve().parents[2] / "cstr"
sys.path.insert(0, str(_CSTR))

import numpy as np
import run_floor_sweep
from run_floor_sweep import run, COLUMNS


def _series(n=400, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    s = np.sin(t * 0.2) * 50.0 + 100.0 + rng.randn(n)
    return [(float(s[i]), float(s[i])) for i in range(n)]


def test_default_grids():
    cells = run_floor_sweep.default_tau100_grid()
    # 9 surface cells + 4 extra H=15 cells = 13 unique
    assert (20, 15) in cells and (100, 30) in cells and (85, 15) in cells and (120, 15) in cells
    assert len(cells) == 13
    assert len(set(cells)) == 13  # no duplicates


def test_run_writes_csv_with_all_columns(tmp_path):
    data = _series()
    entry = {"label": "tiny", "data": data, "tau": 100, "periodicity": 0.5}
    rows = run(entries=[entry], cells_by_dataset={"tiny": [(20, 15)]},
               seeds=[0], alpha=0.5, temperature=4.0, bins=20, epochs=3,
               round_epochs=2, batch_size=32, patience=2, conv_epochs=3,
               conv_patience=2, K=2, outdir=str(tmp_path), verbose=False)
    csv_path = tmp_path / "floor_sweep.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        reader = _csv.DictReader(f)
        header = reader.fieldnames
        row = next(reader)
    for col in COLUMNS:
        assert col in header, f"missing column {col}"
    for vc in ("baseline_mse", "baseline_converged_mse", "teacher_mse",
               "fgl_student_mse", "A_iter_mse", "E_iter_mse"):
        v = float(row[vc])
        assert math.isfinite(v) and v >= 0, f"{vc}={v}"
    assert int(row["LplusH_minus_1"]) == 20 + 15 - 1
