"""Import behaviour of pedestal_prediction.ESCAPE_interface."""
import importlib.util
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from pedestal_prediction.ESCAPE_interface import ESCAPE_ROOT, _escape

REPO_ROOT = Path(__file__).resolve().parents[1]

# (file relative to ESCAPE_ROOT, pattern) -> number of lines with the pattern.
WRITE_PATTERNS = ("savefig", "np.save", "open(", "pickle.dump", ".write(", "VTKFile")
KNOWN_WRITES = {
    ("src/saarelma_connor/saarelma_connor_base.py", "savefig"): 1,  # _plot_profiles, overridden
    ("src/adas/adas_ionisation.py", "open("): 1,  # reads an ADAS table
    ("src/adas/adas_cx.py", "open("): 1,  # reads an ADAS table
    ("helpers/ped_width_proxy.py", "savefig"): 1,  # in the __main__ block only
}


def _run_in_subprocess(code, cwd=None):
    """Run code in a fresh interpreter and return its JSON output."""
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                         capture_output=True, text=True, check=True, cwd=cwd)
    return json.loads(out.stdout)


def test_import_is_lightweight():
    """Importing the interface must not load ESCAPE, Firedrake, Julia or pyplot,
    and must not change sys.path."""
    result = _run_in_subprocess("""
        import json, sys
        path_before = list(sys.path)
        import pedestal_prediction.ESCAPE_interface as ei
        heavy = ["src", "firedrake", "juliacall", "matplotlib.pyplot"]
        print(json.dumps({
            "loaded": [m for m in heavy if m in sys.modules],
            "path_unchanged": sys.path == path_before,
            "escape_found": (ei.ESCAPE_ROOT / "src" / "ESCAPE").is_dir(),
        }))
    """)
    assert result["loaded"] == []
    assert result["path_unchanged"]
    assert result["escape_found"]


def test_escape_loader():
    esc = _escape()
    assert esc.ESCAPE.__module__ == "src.ESCAPE.ESCAPE_api"
    assert Path(inspect.getfile(esc.ESCAPE)).resolve().is_relative_to(ESCAPE_ROOT)
    assert _escape() is esc
    assert esc.firedrake_available == (importlib.util.find_spec("firedrake") is not None)
    assert (esc.firedrake_error is None) == esc.firedrake_available


def test_escape_loader_from_repo_root():
    """From the repo root, 'src' is a namespace package with two portions
    (plasma-limits/src first). ESCAPE must still be found."""
    result = _run_in_subprocess("""
        import inspect, json, sys
        from pathlib import Path
        from pedestal_prediction.ESCAPE_interface import ESCAPE_ROOT, _escape
        esc_file = Path(inspect.getfile(_escape().ESCAPE)).resolve()
        print(json.dumps(dict(
            under_root=esc_file.is_relative_to(ESCAPE_ROOT),
            n_src_portions=len(list(sys.modules["src"].__path__)),
        )))
    """, cwd=REPO_ROOT)
    assert result["under_root"]
    assert result["n_src_portions"] == 2


def test_escape_loader_reports_shadowing_src(tmp_path):
    """A regular 'src' package earlier on sys.path gives an error that names ESCAPE_ROOT."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    result = _run_in_subprocess(f"""
        import json, sys
        sys.path.insert(0, {str(tmp_path)!r})
        from pedestal_prediction.ESCAPE_interface import ESCAPE_ROOT, _escape
        try:
            _escape()
            error = None
        except ImportError as err:
            error = str(err)
        print(json.dumps(dict(error=error, root=str(ESCAPE_ROOT))))
    """)
    assert result["error"] is not None
    assert result["root"] in result["error"]


def test_ped_width_loaded_by_path():
    """_ped_width gives the upstream value and does not import ESCAPE's 'helpers' package."""
    result = _run_in_subprocess("""
        import json, sys
        import numpy as np
        from pedestal_prediction.ESCAPE_interface import ESCAPE_ROOT, _ped_width
        psi = np.linspace(0.85, 1.0, 100)
        ne = 1e20 * (0.6 + 0.4 * np.tanh((0.96 - psi) / 0.02))
        width = _ped_width(psi, ne)
        helpers_loaded = "helpers" in sys.modules
        sys.path.append(str(ESCAPE_ROOT))
        from helpers.ped_width_proxy import ped_width
        print(json.dumps(dict(width=width, upstream=ped_width(psi, ne),
                              helpers_loaded=helpers_loaded)))
    """)
    assert not result["helpers_loaded"]
    assert result["width"] == result["upstream"]
    assert 0 < result["width"] < 0.15


def test_plot_profiles_writes_nothing(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    esc = _escape()
    assert issubclass(esc.Model, esc.ESCAPE)
    monkeypatch.chdir(tmp_path)
    figs_before = plt.get_fignums()
    x = np.linspace(-0.05, 0.0, 10)
    esc.Model._plot_profiles(None, x, x, x, x, title="profiles")
    assert list(tmp_path.iterdir()) == []
    assert plt.get_fignums() == figs_before


def test_escape_file_writes_are_known():
    """Every file write in the ESCAPE code that the interface loads is known.
    A new one must be blocked in _InMemoryESCAPE or added to KNOWN_WRITES."""
    files = _run_in_subprocess("""
        import json, sys
        from pedestal_prediction.ESCAPE_interface import _escape, _ped_width_module
        _escape()
        files = [m.__file__ for name, m in sys.modules.items()
                 if (name == "src" or name.startswith("src.")) and getattr(m, "__file__", None)]
        files.append(_ped_width_module().__file__)
        print(json.dumps(files))
    """)
    hits = {}
    for f in files:
        rel = Path(f).resolve().relative_to(ESCAPE_ROOT).as_posix()
        for line in Path(f).read_text().splitlines():
            if line.lstrip().startswith("#"):
                continue
            for pattern in WRITE_PATTERNS:
                if pattern in line:
                    hits[(rel, pattern)] = hits.get((rel, pattern), 0) + 1
    assert hits == KNOWN_WRITES, (
        "ESCAPE has a new file write: block it in _InMemoryESCAPE or update KNOWN_WRITES")

    esc = _escape()
    assert "savefig" in inspect.getsource(esc.ESCAPE._plot_profiles)
    assert "savefig" not in inspect.getsource(esc.Model._plot_profiles)
    proxy_src = (ESCAPE_ROOT / "helpers" / "ped_width_proxy.py").read_text()
    assert "savefig" in proxy_src.split("if __name__ == '__main__':")[1]
