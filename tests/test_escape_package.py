"""Layout and imports of the installed ESCAPE package (dependencies/ESCAPE/src/escape)."""
import json
import subprocess
import sys
import textwrap

from plasma_limits.pedestal_prediction.ESCAPE_interface import ESCAPE_ROOT

PACKAGE_DIR = ESCAPE_ROOT / "src" / "escape"


def _run_in_subprocess(code):
    """Run code in a fresh interpreter and return its JSON output."""
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_installed_escape_imports():
    result = _run_in_subprocess("""
        import json, os, sys
        from pathlib import Path
        path_before = list(sys.path)
        from escape.ESCAPE.ESCAPE_api import ESCAPE
        from escape.ESCAPE import ESCAPE_solve, ESCAPE_state
        from escape.saarelma_connor import saarelma_connor_base, solver_3d
        from escape.epednn import epednn_call
        from escape.adas import adas_ionisation
        from escape.radas import radas_rates
        print(json.dumps(dict(
            api_file=sys.modules[ESCAPE.__module__].__file__,
            same_state_class=saarelma_connor_base.ESCAPE_state is ESCAPE_state.ESCAPE_state,
            loaded=[m for m in ("src", "helpers") if m in sys.modules],
            path_unchanged=sys.path == path_before,
            epednn_root=str(Path(epednn_call.__file__).resolve().parents[3]),
            adas_dat=sorted(f for f in os.listdir(adas_ionisation.ADAS_DIR) if f.endswith(".dat")),
            radas_data=os.path.isdir(radas_rates.DATA_DIR),
        )))
    """)
    assert result["api_file"].startswith(str(PACKAGE_DIR))
    assert result["same_state_class"]
    assert result["loaded"] == []
    assert result["path_unchanged"]
    assert result["epednn_root"] == str(ESCAPE_ROOT)
    assert (ESCAPE_ROOT / "dependencies" / "EPEDNN.jl").is_dir()
    assert result["adas_dat"] == ["ccd96_d.dat", "scd12_h.dat"]
    assert result["radas_data"]


def test_helpers_import_without_optional_dependencies():
    """OpenFUSIONToolkit and omfit are imported inside the helper functions only."""
    result = _run_in_subprocess("""
        import json, sys
        from escape.helpers import load_equil, parse_equil, parse_prof, ped_width_proxy
        print(json.dumps(dict(
            loaded=[m for m in ("OpenFUSIONToolkit", "omfit_classes", "helpers") if m in sys.modules],
            ped_width_file=ped_width_proxy.__file__,
        )))
    """)
    assert result["loaded"] == []
    assert result["ped_width_file"].startswith(str(PACKAGE_DIR / "helpers"))
    assert not (ESCAPE_ROOT / "helpers").exists()


def test_no_old_imports_left():
    """No ESCAPE source imports the old src package or the old top-level helpers package."""
    package_lines = [(path, line) for path in PACKAGE_DIR.rglob("*.py")
                     for line in path.read_text().splitlines()]
    assert [p for p, line in package_lines if line.startswith(("from src.", "import src."))] == []

    sources = [p for p in ESCAPE_ROOT.rglob("*")
               if p.suffix in (".py", ".ipynb")
               and not {"old", "dependencies"} & set(p.relative_to(ESCAPE_ROOT).parts)]
    old_imports = ("from helpers.", "import helpers", "from ESCAPE.helpers.", '"from helpers.',
                   "from src.", "import src.", '"from src.', '"import src.', '"# from src.')
    offenders = [str(p.relative_to(ESCAPE_ROOT)) for p in sources
                 if any(line.lstrip().startswith(old_imports) for line in p.read_text().splitlines())]
    assert offenders == []


def test_src_holds_only_known_packages():
    """The editable install puts ESCAPE_ROOT/src on sys.path, so every folder there is importable."""
    folders = {d.name for d in (ESCAPE_ROOT / "src").iterdir()
               if d.is_dir() and d.name != "__pycache__" and not d.name.endswith(".egg-info")}
    assert folders == {"escape", "old"}
