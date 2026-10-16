import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cstr"))
import run


def test_next_phase_entries_registered_and_disabled():
    expected = {"transfer_generalization", "teacher_student_relation", "complex_multichannel"}
    assert expected <= set(run.EXPERIMENTS)
    assert all(run.EXPERIMENTS[k]["enabled"] is False for k in expected)
