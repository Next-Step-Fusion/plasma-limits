"""Shared fixtures: reference outputs of an ESCAPE ARC run (3D, Firedrake, EPED1).

The files in src/pedestal_prediction/test/ were written by ESCAPE's ARC driver.
ne_and_Te_iter_k.npy holds iteration k of the ESCAPE loop and needs only numpy.
out_dict.npy also stores the Julia EPED model and an ESCAPE boundary-condition
object, so rebuilding it needs juliacall, EPEDNN and ESCAPE on sys.path. That
setup runs only when a test requests ref_out_dict (or a fixture derived from it).
"""
import sys

import numpy as np
import pytest

from pedestal_prediction.ESCAPE_interface import ESCAPE_ROOT

REF_DIR = ESCAPE_ROOT.parent / "test"  # src/pedestal_prediction/test
N_REF_ITERS = 5
EPEDNN_UUID = "e64856f0-3bb8-4376-b4b7-c03396503991"


def _load_npy_dict(path):
    return np.load(path, allow_pickle=True).item()


def _prepare_unpickling():
    """Make the non-numpy objects in out_dict.npy rebuildable; return Julia's Main.

    The Julia setup is the same as ESCAPE's epednn_call.py:104-122: Julia can
    only rebuild the stored EPED model if the module defining its type is loaded.
    """
    if str(ESCAPE_ROOT) not in sys.path:
        sys.path.append(str(ESCAPE_ROOT))  # for src.saarelma_connor.bc_ig_helpers.NeBCs

    import juliapkg
    juliapkg.add("EPEDNN", uuid=EPEDNN_UUID,
                 path=str(ESCAPE_ROOT / "dependencies" / "EPEDNN.jl"), dev=True)
    juliapkg.resolve()
    from juliacall import Main as jl
    jl.seval("using EPEDNN")
    return jl


@pytest.fixture(scope="session")
def ref_iters():
    """The five per-iteration dicts saved by ESCAPE_solve, in order."""
    return [_load_npy_dict(REF_DIR / f"ne_and_Te_iter_{k}.npy") for k in range(N_REF_ITERS)]


@pytest.fixture(scope="session")
def ref_out_dict():
    """The ARC driver's out_dict, with the Julia EPED model rebuilt."""
    jl = _prepare_unpickling()
    try:
        return _load_npy_dict(REF_DIR / "out_dict.npy")
    except Exception as err:
        raise RuntimeError(
            f"Could not rebuild out_dict.npy with the local Julia {jl.VERSION}. "
            "Julia's serialization format can change between minor versions; "
            "regenerate out_dict.npy with this Julia version."
        ) from err


@pytest.fixture(scope="session")
def ref_eq(ref_out_dict):
    """The read_eqdsk dict of the ARC g-file, exactly as ESCAPE consumed it."""
    return ref_out_dict["state"]["eq"]


@pytest.fixture(scope="session")
def ref_profiles(ref_out_dict):
    """The input kinetic profiles (kprof_params) of the reference run."""
    return ref_out_dict["profiles"]


@pytest.fixture(scope="session")
def ref_free_params(ref_out_dict):
    """The Saarelma-Connor free parameters of the reference run."""
    return ref_out_dict["free_params"]
