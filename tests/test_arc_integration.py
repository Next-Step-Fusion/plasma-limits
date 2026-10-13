"""ARC example run of ESCAPE (1D scipy, real EPED1) against a recorded baseline.

Run with:  uv run pytest -m integration
To record a new baseline on purpose:  ESCAPE_UPDATE_BASELINE=1 uv run pytest -m integration
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent
DRIVER = TESTS_DIR / "arc_integration_driver.py"
BASELINE = TESTS_DIR / "data" / "arc_1d_baseline.npy"
ESCAPE_ROOT = TESTS_DIR.parent / "dependencies" / "ESCAPE"


def _submodule_status():
    out = subprocess.run(["git", "-C", str(ESCAPE_ROOT), "status", "--porcelain"],
                         capture_output=True, text=True, check=True).stdout
    return sorted(line for line in out.splitlines() if "__pycache__" not in line)


def _diff(a, b, path="result"):
    """Paths where a and b differ; arrays must match in dtype, shape and every value."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys {sorted(a.keys() ^ b.keys())}"]
        return [d for k in a for d in _diff(a[k], b[k], f"{path}[{k!r}]")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        same = (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype
                and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
        return [] if same else [path]
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} != {type(b).__name__}"]
    return [] if a == b or (a != a and b != b) else [path]


@pytest.mark.integration
def test_arc_run_matches_baseline(tmp_path):
    out_dir, result_file = tmp_path / "out", tmp_path / "result.npy"
    out_dir.mkdir()
    status_before = _submodule_status()
    env = dict(os.environ, MPLBACKEND="Agg")
    subprocess.run([sys.executable, str(DRIVER), str(out_dir), str(result_file)],
                   env=env, check=True)
    assert _submodule_status() == status_before, "the ARC run changed files in the ESCAPE submodule"

    result = np.load(result_file, allow_pickle=True).item()
    print(f"ESCAPE import route: {result['route']}")
    iterations = result["iterations"]
    assert len(iterations) == 5

    if os.environ.get("ESCAPE_UPDATE_BASELINE") == "1":
        np.save(BASELINE, {"iterations": iterations}, allow_pickle=True)
        pytest.fail(f"Baseline written to {BASELINE}. Unset ESCAPE_UPDATE_BASELINE and run again.")

    baseline = np.load(BASELINE, allow_pickle=True).item()["iterations"]
    differences = _diff(iterations, baseline, "iterations")
    assert not differences, "ARC run differs from the baseline at:\n" + "\n".join(differences[:20])
