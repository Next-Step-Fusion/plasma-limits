"""ARC example runs of ESCAPE (real EPED1) against recorded baselines.

- 1d: 1D scipy, in this interpreter, compared bit for bit, in both driver modes
  (file: ESCAPE_solve defaults; memory: the in-memory keywords).
- 3d: 3D Firedrake, in an interpreter that has Firedrake, compared to rtol=1e-10.
  PyOP2 builds its kernels for the local CPU, so a baseline from another
  machine can differ in the last bits. The case is skipped if no interpreter
  can import Firedrake.

Run with:  uv run pytest -m integration
To record a new baseline on purpose:
    ESCAPE_UPDATE_BASELINE=1 uv run pytest -m integration -k "1d-file or 3d-file"
Both modes of a model compare with the same baseline.

The 3D case finds Firedrake through FIREDRAKE_PYTHON, then this interpreter,
then <repo>/firedrake-venv. PETSC_DIR, PETSC_ARCH, CC and CXX come from the
environment, otherwise from the defaults of firedrake-build/env.sh. PyOP2
compiles with the bare name ``mpicc`` from PETSc's configuration, so the run
puts a folder with the right ``mpicc`` first in PATH.
"""
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
DRIVER = TESTS_DIR / "arc_integration_driver.py"
ESCAPE_ROOT = REPO_ROOT / "dependencies" / "ESCAPE"
FIREDRAKE_BUILD = REPO_ROOT / "firedrake-build"
RTOL_3D = 1e-10

FIREDRAKE_DEFAULTS = {
    "PETSC_DIR": str(FIREDRAKE_BUILD / "petsc"),
    "PETSC_ARCH": "arch-firedrake-default",
    "CC": "/usr/bin/mpicc",
    "CXX": "/usr/bin/mpicxx",
}


def _submodule_status():
    out = subprocess.run(["git", "-C", str(ESCAPE_ROOT), "status", "--porcelain"],
                         capture_output=True, text=True, check=True).stdout
    return sorted(line for line in out.splitlines() if "__pycache__" not in line)


def _close(a, b, rtol, atol):
    if rtol == 0:
        return a == b or (a != a and b != b)
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol) or (a != a and b != b)


def _diff(a, b, path="result", rtol=0.0):
    """Paths where a and b differ.

    Keys, lengths, types, dtypes and shapes must match exactly. With rtol=0 every
    value must match exactly. With rtol>0, floats may differ by rtol relative,
    plus an absolute rtol * max|b| per array so that values near zero can pass.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys {sorted(a.keys() ^ b.keys())}"]
        return [d for k in a for d in _diff(a[k], b[k], f"{path}[{k!r}]", rtol)]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]", rtol)]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.dtype == b.dtype and a.shape == b.shape):
            return [path]
        if rtol == 0 or a.dtype.kind not in "fc" or b.size == 0:
            same = np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")
        else:
            atol = rtol * float(np.nanmax(np.abs(b), initial=0.0))
            same = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        return [] if same else [path]
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, (float, np.floating)):
        return [] if _close(a, b, rtol, rtol * abs(b)) else [path]
    return [] if a == b or (a != a and b != b) else [path]


def _firedrake_python():
    """An interpreter that should have Firedrake, or None."""
    if os.environ.get("FIREDRAKE_PYTHON"):
        return Path(os.environ["FIREDRAKE_PYTHON"])
    if importlib.util.find_spec("firedrake") is not None:
        return Path(sys.executable)
    venv_python = REPO_ROOT / "firedrake-venv" / "bin" / "python"
    return venv_python if venv_python.exists() else None


def _firedrake_runner(tmp_path):
    """The interpreter and subprocess environment for a Firedrake run; skips if there is none."""
    python = _firedrake_python()
    if python is None or not python.exists():
        pytest.skip(f"no Python with Firedrake (FIREDRAKE_PYTHON={os.environ.get('FIREDRAKE_PYTHON')})")

    env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1")
    if "PETSC_DIR" not in env and Path(FIREDRAKE_DEFAULTS["PETSC_DIR"]).is_dir():
        env["PETSC_DIR"] = FIREDRAKE_DEFAULTS["PETSC_DIR"]
        env.setdefault("PETSC_ARCH", FIREDRAKE_DEFAULTS["PETSC_ARCH"])
    for name in ("CC", "CXX"):
        if name not in env and Path(FIREDRAKE_DEFAULTS[name]).exists():
            env[name] = FIREDRAKE_DEFAULTS[name]

    mpi_bin = FIREDRAKE_BUILD / "mpi-bin"
    if not mpi_bin.is_dir():
        mpi_bin = tmp_path / "mpi-bin"
        mpi_bin.mkdir()
        for link, compiler in (("mpicc", env.get("CC")), ("mpicxx", env.get("CXX"))):
            if compiler:
                (mpi_bin / link).symlink_to(compiler)
    env["PATH"] = f"{mpi_bin}{os.pathsep}{env.get('PATH', '')}"

    check = subprocess.run([str(python), "-c", "import firedrake, escape"],
                           env=env, capture_output=True, text=True)
    if check.returncode != 0:
        tail = "\n".join(check.stderr.strip().splitlines()[-5:])
        pytest.skip(f"Firedrake can't be imported with {python}:\n{tail}")
    return python, env


@pytest.mark.integration
@pytest.mark.parametrize("model, mode", [("1d", "file"), ("1d", "memory"), ("3d", "file")])
def test_arc_run_matches_baseline(model, mode, tmp_path):
    if model == "3d":
        python, env = _firedrake_runner(tmp_path)
        rtol = RTOL_3D
    else:
        python, env = Path(sys.executable), dict(os.environ, MPLBACKEND="Agg")
        rtol = 0.0
    baseline_file = TESTS_DIR / "data" / f"arc_{model}_baseline.npy"

    out_dir, result_file = tmp_path / "out", tmp_path / "result.npy"
    out_dir.mkdir()
    status_before = _submodule_status()
    subprocess.run([str(python), str(DRIVER), str(out_dir), str(result_file),
                    "--model", model, "--mode", mode], env=env, check=True)
    assert _submodule_status() == status_before, "the ARC run changed files in the ESCAPE submodule"

    result = np.load(result_file, allow_pickle=True).item()
    iterations = result["iterations"]
    assert len(iterations) == 5
    if mode == "memory":
        extras = result["extras"]
        assert len(extras) == 5
        for it, extra in zip(iterations, extras):
            assert extra["betan"].shape == (1,) and np.isfinite(extra["betan"]).all()
            assert extra["T_e_pres"].shape == it["psi_N"].shape
            assert np.isfinite(extra["T_e_pres"]).all()

    if os.environ.get("ESCAPE_UPDATE_BASELINE") == "1" and mode == "file":
        np.save(baseline_file, {"iterations": iterations}, allow_pickle=True)
        pytest.fail(f"Baseline written to {baseline_file}. Unset ESCAPE_UPDATE_BASELINE and run again.")

    baseline = np.load(baseline_file, allow_pickle=True).item()["iterations"]
    differences = _diff(iterations, baseline, "iterations", rtol)
    assert not differences, f"ARC {model} {mode} run differs from the baseline at:\n" + "\n".join(differences[:20])
