import sys, pathlib, csv as _csv
_CSTR = pathlib.Path(__file__).resolve().parents[2] / "cstr"
sys.path.insert(0, str(_CSTR))

from analyze_floor import analyze


def _write_floor_csv(path):
    rows = []
    for L in (20, 50, 100):
        for H in (5, 15, 30):
            for seed in (0, 1, 2):
                tm = 10.0 + L * 0.1 + H * 0.5  # teacher_mse
                rows.append({"dataset": "tau100", "tau": "100", "periodicity": 0.5,
                             "L": L, "H": H, "LplusH_minus_1": L + H - 1, "seed": seed,
                             "baseline_mse": tm * 3.0,
                             "baseline_converged_mse": tm * 2.0,
                             "teacher_mse": tm,
                             "fgl_student_mse": tm * 2.5,
                             "A_iter_mse": tm * 2.2,
                             "E_iter_mse": tm * 2.0})
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _write_lyap_csv(path):
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["file", "tau", "lyap", "N"])
        w.writeheader()
        w.writerow({"file": "tau100", "tau": "100", "lyap": "0.02", "N": "6000"})


def test_analyze_finds_known_relation(tmp_path):
    csv_path = tmp_path / "floor_sweep.csv"
    lyap_path = tmp_path / "lyapunov_tau.csv"
    _write_floor_csv(csv_path)
    _write_lyap_csv(lyap_path)

    stats = analyze(csv_path=str(csv_path), lyap_path=str(lyap_path),
                    outdir=str(tmp_path), conclusion_dir=str(tmp_path),
                    deep_label="tau100")
    # floor = 2 * teacher exactly => R^2 ~ 1
    assert stats["h3_floor_vs_teacher_r2"] > 0.99
    md = (tmp_path / "floor_determinants.md").read_text()
    for h in ("H1", "H2", "H3", "H4"):
        assert h in md
    # at least the H3 figure written
    assert (tmp_path / "floor_h3_floor_vs_teacher.png").exists()
