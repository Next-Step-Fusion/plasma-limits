"""Layout and imports of the installed ESCAPE package (dependencies/ESCAPE/src/escape)."""
import json
import subprocess
import sys
import textwrap

from pedestal_prediction.ESCAPE_interface import ESCAPE_ROOT

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
        from escape.ESCAPE.ESCAPE_api import ESCAPE
        from escape.ESCAPE import ESCAPE_solve, ESCAPE_state
        from escape.saarelma_connor import saarelma_connor_base, solver_3d
        from escape.epednn import epednn_call
        from escape.adas import adas_ionisation
        from escape.radas import radas_rates
        print(json.dumps(dict(
            api_file=sys.modules[ESCAPE.__module__].__file__,
            same_state_class=saarelma_connor_base.ESCAPE_state is ESCAPE_state.ESCAPE_state,
            src_loaded="src" in sys.modules,
            solve_root=str(ESCAPE_solve.ROOT),
            epednn_root=str(Path(epednn_call.__file__).resolve().parents[3]),
            adas_dat=sorted(f for f in os.listdir(adas_ionisation.ADAS_DIR) if f.endswith(".dat")),
            radas_data=os.path.isdir(radas_rates.DATA_DIR),
        )))
    """)
    assert result["api_file"].startswith(str(PACKAGE_DIR))
    assert result["same_state_class"]
    assert not result["src_loaded"]
    assert result["solve_root"] == str(ESCAPE_ROOT)
    assert result["epednn_root"] == str(ESCAPE_ROOT)
    assert (ESCAPE_ROOT / "dependencies" / "EPEDNN.jl").is_dir()
    assert result["adas_dat"] == ["ccd96_d.dat", "scd12_h.dat"]
    assert result["radas_data"]


def test_no_src_imports_left_in_package():
    offenders = [f"{path.relative_to(ESCAPE_ROOT)}:{n}"
                 for path in PACKAGE_DIR.rglob("*.py")
                 for n, line in enumerate(path.read_text().splitlines(), 1)
                 if line.startswith(("from src.", "import src."))]
    assert offenders == []


def test_src_holds_only_known_packages():
    """The editable install puts ESCAPE_ROOT/src on sys.path, so every folder there is importable."""
    folders = {d.name for d in (ESCAPE_ROOT / "src").iterdir()
               if d.is_dir() and d.name != "__pycache__" and not d.name.endswith(".egg-info")}
    assert folders == {"escape", "old"}
