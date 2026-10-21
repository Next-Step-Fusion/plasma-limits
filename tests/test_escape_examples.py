"""The ESCAPE examples run as scripts and write only their plot."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from plasma_limits.pedestal_prediction.ESCAPE_interface import ESCAPE_ROOT

EXAMPLES_DIR = Path(__file__).parents[1] / "examples"


def _submodule_status():
    out = subprocess.run(["git", "-C", str(ESCAPE_ROOT), "status", "--porcelain"],
                         capture_output=True, text=True, check=True)
    return out.stdout


@pytest.mark.slow
@pytest.mark.parametrize("name", ["ARC", "CENTAUR"])
def test_example_runs(name, tmp_path):
    status_before = _submodule_status()
    run = subprocess.run([sys.executable, str(EXAMPLES_DIR / f"ESCAPE_{name}_example.py")], cwd=tmp_path,
                         capture_output=True, text=True, env={**os.environ, "MPLBACKEND": "Agg"})
    assert run.returncode == 0, run.stderr[-3000:]
    assert "5 iterations, converged: False" in run.stdout
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == [
        Path("outputs"), Path(f"outputs/ESCAPE_{name}_profiles.png")]
    assert _submodule_status() == status_before
