"""Import behaviour of plasma_limits.pedestal_prediction.ESCAPE_interface."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

SUBMODULE = Path(__file__).resolve().parents[1] / "dependencies" / "ESCAPE"


def _run_in_subprocess(code):
    """Run code in a fresh interpreter and return its JSON output."""
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_import_is_lightweight():
    """Importing the interface must not load ESCAPE, Firedrake, Julia or pyplot,
    and must not change sys.path."""
    result = _run_in_subprocess("""
        import json, sys
        path_before = list(sys.path)
        import plasma_limits.pedestal_prediction.ESCAPE_interface as ei
        heavy = ["escape", "firedrake", "juliacall", "matplotlib.pyplot"]
        print(json.dumps({
            "loaded": [m for m in heavy if m in sys.modules],
            "path_unchanged": sys.path == path_before,
            "escape_root": str(ei.ESCAPE_ROOT),
        }))
    """)
    assert result["loaded"] == []
    assert result["path_unchanged"]
    assert result["escape_root"] == str(SUBMODULE), "escape is not installed from the submodule"
