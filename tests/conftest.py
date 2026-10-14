"""Shared fixtures: reference outputs of an ESCAPE ARC run (3D, Firedrake, EPED1).

The files in tests/data/ were written by ESCAPE's ARC driver
and load with numpy alone. ne_and_Te_iter_k.npy holds iteration k of the ESCAPE
loop. arc_reference.npy holds the plain-data part of the driver's out_dict.npy
(see extract_arc_reference.py).
"""
from pathlib import Path

import numpy as np
import pytest

REF_DIR = Path(__file__).parent / "data"
N_REF_ITERS = 5


def _load_npy_dict(path):
    return np.load(path, allow_pickle=True).item()


@pytest.fixture(scope="session")
def ref_iters():
    """The five per-iteration dicts saved by ESCAPE_solve, in order."""
    return [_load_npy_dict(REF_DIR / f"ne_and_Te_iter_{k}.npy") for k in range(N_REF_ITERS)]


@pytest.fixture(scope="session")
def ref_arc():
    """The ARC driver's out_dict without the Julia EPED model and NeBCs."""
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
