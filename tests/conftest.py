"""Shared fixtures and helpers.

The files in tests/data/ load with numpy alone:
- arc_reference.npy: the inputs and the ESCAPE state of a reference ARC run
  (3D Firedrake, EPED1), without the Julia objects.
- arc_1d_baseline.npy, arc_3d_baseline.npy: the iteration dicts of the ARC
  runs of test_arc_integration.py.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from plasma_limits.pedestal_prediction.ESCAPE_interface import PedestalPlasmaState

TESTS_DIR = Path(__file__).parent
REF_DIR = TESTS_DIR / "data"

# IterationRecord field -> key of the dict that ESCAPE_solve gives to on_iteration
RECORD_ARRAY_KEYS = {"x": "x", "ne_x": "ne", "nFC": "nFC", "nCX": "nCX", "psi_N_pres": "psi_N",
                     "T_e_pres": "T_e_pres", "psi_N_ne": "psi_ne_eval", "ne": "n_e",
                     "psi_N_Te": "psi_Te_eval", "Te": "T_e"}
RECORD_VALUE_KEYS = {"Te_ped_eV": "Te_ped_eV", "ne_ped": "ne_ped_val", "psi_ped": "psi_ped",
                     "pedestal_height": "pedestal_height", "pedestal_width": "pedestal_width",
                     "model": "model", "implementation": "implementation", "params": "SOLVE_KW"}


def _load_npy_dict(path):
    return np.load(path, allow_pickle=True).item()


def _load_tests_module(name):
    """A module of tests/ loaded by file path (tests/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, TESTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


@pytest.fixture(scope="session")
def diff():
    """_diff(a, b, path="result", rtol=0.0): the paths where a and b differ."""
    return _diff


@pytest.fixture(scope="session")
def driver():
    """tests/arc_integration_driver.py (imports ESCAPE)."""
    return _load_tests_module("arc_integration_driver")


@pytest.fixture(scope="session")
def arc_integration():
    """tests/test_arc_integration.py, for its Firedrake runner."""
    return _load_tests_module("test_arc_integration")


@pytest.fixture(scope="session")
def arc_1d_iterations():
    """The five iteration dicts of arc_1d_baseline.npy (ESCAPE's file path, 1D scipy, EPED1)."""
    return _load_npy_dict(REF_DIR / "arc_1d_baseline.npy")["iterations"]


@pytest.fixture(scope="session")
def record_array_keys():
    return dict(RECORD_ARRAY_KEYS)


@pytest.fixture(scope="session")
def record_value_keys():
    return dict(RECORD_VALUE_KEYS)


@pytest.fixture(scope="session")
def ref_arc():
    """The reference ARC run without the Julia EPED model and NeBCs."""
    return _load_npy_dict(REF_DIR / "arc_reference.npy")


@pytest.fixture(scope="session")
def ref_eq(ref_arc):
    """The read_eqdsk dict of the ARC g-file, exactly as ESCAPE consumed it."""
    return ref_arc["state"]["eq"]


@pytest.fixture(scope="session")
def ref_profiles(ref_arc):
    """The input kinetic profiles (kprof_params) of the reference run."""
    return ref_arc["profiles"]


@pytest.fixture(scope="session")
def ref_free_params(ref_arc):
    """The Saarelma-Connor free parameters of the reference run."""
    return ref_arc["free_params"]


@pytest.fixture(scope="session")
def arc_state(ref_arc, ref_eq, ref_profiles):
    """The reference ARC inputs as a PedestalPlasmaState (g-file mapping done here, on the test side)."""
    eq = ref_eq
    return PedestalPlasmaState(
        R=np.linspace(eq["rleft"], eq["rleft"] + eq["rdim"], eq["nr"]),
        Z=np.linspace(eq["zmid"] - eq["zdim"] / 2, eq["zmid"] + eq["zdim"] / 2, eq["nz"]),
        psi=eq["psirz"],
        psi_axis=eq["psimag"], psi_boundary=eq["psibry"],
        R_axis=eq["raxis"], Z_axis=eq["zaxis"], Ip=eq["ip"],
        F=eq["fpol"], pressure=eq["pres"],
        boundary=eq["rzout"],
        psi_N_Te=ref_profiles["psin_Te"], Te=ref_profiles["T_e"],
        psi_N_ne=ref_profiles["psin_ne"], ne=ref_profiles["n_e"],
        Z_eff=ref_arc["Zeff"], P_tot_e=ref_arc["P_tot_e"],
    )


@pytest.fixture
def synthetic_inputs():
    """Keyword arguments of a small valid PedestalPlasmaState (5x5 grid, 6-point profiles).

    The values are consistent in shape only; they are not a physical equilibrium.
    """
    R = np.linspace(1.0, 3.0, 5)
    Z = np.linspace(-1.0, 1.0, 5)
    RR, ZZ = np.meshgrid(R, Z)
    theta = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    psi_N = np.linspace(0.0, 1.0, 6)
    return dict(
        R=R, Z=Z, psi=(RR - 2.0) ** 2 + ZZ ** 2,
        psi_axis=0.0, psi_boundary=1.0, R_axis=2.0, Z_axis=0.0, Ip=1e6,
        F=np.full(6, 5.0),
        boundary=np.column_stack([2.0 + 0.9 * np.cos(theta), 0.9 * np.sin(theta)]),
        psi_N_Te=psi_N, Te=np.linspace(5.0, 0.1, 6),
        psi_N_ne=psi_N, ne=np.linspace(1e20, 3e19, 6),
        Z_eff=1.5, P_tot_e=1e7,
    )
