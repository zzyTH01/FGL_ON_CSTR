import os, subprocess, sys, pathlib

_REPO = pathlib.Path(__file__).resolve().parents[2]


def test_floor_sweep_listed_and_off():
    out = subprocess.check_output(
        [sys.executable, str(_REPO / "cstr" / "run.py"), "--list"], text=True)
    assert "floor_sweep" in out
    # the line for floor_sweep must show it as off (enabled=False)
    line = [ln for ln in out.splitlines() if ln.strip().startswith("floor_sweep")][0]
    assert "off" in line


def test_experiments_dict_registered():
    # import the module directly via file path (cstr is not a package)
    import importlib.util
    spec = importlib.util.spec_from_file_location("cstr_run_mod", _REPO / "cstr" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "floor_sweep" in mod.EXPERIMENTS
    assert mod.EXPERIMENTS["floor_sweep"]["enabled"] is False
