"""Smoke tests for the reference-data fixtures in conftest.py."""
import json
import subprocess
import sys
import textwrap

import numpy as np

ITER_KEYS = {
    "x", "ne", "nFC", "nCX", "T_e", "psi_N", "psi_Te_eval", "psi_ne_eval", "n_e",
    "Te_ped_eV", "ne_ped_val", "psi_ped", "pedestal_height", "pedestal_width",
    "SOLVE_KW", "model", "implementation", "diagnostics",
}
PLAIN_SCALARS = (np.generic, bool, int, float, complex, str, type(None))


def _leaves(obj, path="d"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _leaves(value, f"{path}[{key!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            yield from _leaves(value, f"{path}[{i}]")
    else:
        yield path, obj


def test_ref_iters(ref_iters):
    assert len(ref_iters) == 5
    for k, it in enumerate(ref_iters):
        assert ITER_KEYS <= it.keys(), f"iteration {k}"
        assert ("loop_tol" in it) == (k > 0), f"iteration {k}"


def test_ref_eq_grid(ref_eq):
    assert ref_eq["psirz"].shape == (129, 129)
    assert ref_eq["rzout"].shape == (102, 2)


def test_arc_reference_is_plain(ref_arc):
    for path, leaf in _leaves(ref_arc):
        if isinstance(leaf, np.ndarray):
            assert leaf.dtype != object, path
        else:
            assert isinstance(leaf, PLAIN_SCALARS), f"{path}: {type(leaf)}"


def test_arc_reference_loads_with_numpy_only():
    """Loading arc_reference.npy must not import Julia or ESCAPE."""
    code = textwrap.dedent("""
        import json, sys
        import numpy as np
        from pedestal_prediction.ESCAPE_interface import ESCAPE_ROOT
        d = np.load(ESCAPE_ROOT.parent / "test" / "arc_reference.npy", allow_pickle=True).item()
        print(json.dumps({
            "loaded": [m for m in ("juliacall", "src") if m in sys.modules],
            "has_state": "state" in d,
        }))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    result = json.loads(out.stdout)
    assert result["loaded"] == []
    assert result["has_state"]
