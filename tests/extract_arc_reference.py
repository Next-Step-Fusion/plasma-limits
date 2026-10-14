"""Write arc_reference.npy: the plain-data part of the ARC driver's out_dict.npy.

out_dict.npy also stores the Julia EPED model and an ESCAPE boundary-condition
object, so loading it needs juliacall, EPEDNN and ESCAPE's ``src`` package.
This script loads it once with that setup, keeps only numpy arrays, numbers,
strings, None, and dicts and lists of these, and saves the result. The output
loads with numpy alone.

Run by hand from the repo root:  uv run python tests/extract_arc_reference.py
"""
import sys
from pathlib import Path

import numpy as np

from plasma_limits.pedestal_prediction.ESCAPE_interface import ESCAPE_ROOT

REF_DIR = Path(__file__).parent / "data"
EPEDNN_UUID = "e64856f0-3bb8-4376-b4b7-c03396503991"
PLAIN_SCALARS = (np.generic, bool, int, float, complex, str, type(None))
_DROP = object()  # marks a value that plain() removes


def _prepare_unpickling():
    """Make the Julia model and NeBCs in out_dict.npy rebuildable.

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


def _is_plain_leaf(obj):
    if isinstance(obj, np.ndarray):
        return obj.dtype != object
    return isinstance(obj, PLAIN_SCALARS)


def plain(obj, path, dropped):
    """Return obj without its non-plain parts; append the path of each dropped part.

    A list or tuple is kept only if all its items are plain, so indices never shift.
    """
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            kept = plain(value, f"{path}[{key!r}]", dropped)
            if kept is not _DROP:
                out[key] = kept
        return out
    if isinstance(obj, (list, tuple)):
        sub = []
        items = [plain(v, f"{path}[{i}]", sub) for i, v in enumerate(obj)]
        if sub:
            dropped.append(f"{path} ({type(obj).__name__} with non-plain items)")
            return _DROP
        return type(obj)(items)
    if _is_plain_leaf(obj):
        return obj
    dropped.append(f"{path} ({type(obj).__module__}.{type(obj).__name__})")
    return _DROP


def assert_same(kept, orig, path="d"):
    """Check that every kept value equals the original value at the same path."""
    if isinstance(kept, dict):
        for key, value in kept.items():
            assert_same(value, orig[key], f"{path}[{key!r}]")
    elif isinstance(kept, (list, tuple)):
        assert len(kept) == len(orig), path
        for i, (a, b) in enumerate(zip(kept, orig)):
            assert_same(a, b, f"{path}[{i}]")
    elif isinstance(kept, np.ndarray):
        assert kept.dtype == orig.dtype and np.array_equal(kept, orig, equal_nan=kept.dtype.kind in "fc"), path
    else:
        assert type(kept) is type(orig), path
        assert kept == orig or (kept != kept and orig != orig), path  # NaN == NaN here


def main():
    _prepare_unpickling()
    orig = np.load(REF_DIR / "out_dict.npy", allow_pickle=True).item()
    dropped = []
    kept = plain(orig, "d", dropped)
    out = REF_DIR / "arc_reference.npy"
    np.save(out, kept, allow_pickle=True)

    reloaded = np.load(out, allow_pickle=True).item()
    assert_same(reloaded, orig)
    print(f"Wrote {out}")
    print("Dropped:")
    for p in dropped:
        print(f"  {p}")


if __name__ == "__main__":
    main()
