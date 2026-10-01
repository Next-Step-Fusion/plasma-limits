"""Run ESCAPE (git submodule) device-prediction scripts from plasma-limits."""
import os
import subprocess
import sys
from pathlib import Path

ESCAPE_ROOT = Path(__file__).resolve().parent / 'ESCAPE'
ARC_DIR = ESCAPE_ROOT / 'examples' / 'device_prediction' / 'ARC'


def run_ARC_scan(output_dir=None, sc_model=None, sc_implementation=None, ne_grad_bc_loc=None):
    """Run ESCAPE's ARC scan script (scan_SCfp_ESCAPE_ARC.py).

    The script resolves the ESCAPE repo root and its input files (geqdsk-ARCv3a,
    ARC_ne.csv, ARC_Te.csv) relative to the working directory, so it is run in
    a subprocess from its own directory.

    Parameters
    ----------
    output_dir : str or Path, optional
        Where the script writes its outputs (passed as OUTPUT_DIR). A relative
        path is taken relative to the caller's working directory, not the
        script's. If None, the script's default under
        ESCAPE/examples/device_prediction/ARC/outputs/ is used.
    sc_model, sc_implementation, ne_grad_bc_loc : str, optional
        Passed to the script as SC_MODEL, SC_IMPLEMENTATION, NE_GRAD_BC_LOC.
        If None, the script's defaults ('3D', 'firedrake', 'inner') are used.

    Returns
    -------
    Path
        The script's output directory.
    """
    env = os.environ.copy()
    for key, val in [('SC_MODEL', sc_model),
                     ('SC_IMPLEMENTATION', sc_implementation),
                     ('NE_GRAD_BC_LOC', ne_grad_bc_loc)]:
        if val is not None:
            env[key] = val

    if output_dir is not None:
        env['OUTPUT_DIR'] = str(Path(output_dir).resolve())  # absolute, since the script runs from ARC_DIR

    subprocess.run([sys.executable, 'scan_SCfp_ESCAPE_ARC.py'], cwd=ARC_DIR, env=env, check=True)

    if 'OUTPUT_DIR' in env:
        return Path(env['OUTPUT_DIR'])
    return ARC_DIR / 'outputs' / (f"ARC_ESCAPE_{env.get('SC_MODEL', '3D')}"
                                  f"{env.get('SC_IMPLEMENTATION', 'firedrake')}"
                                  f"{env.get('NE_GRAD_BC_LOC', 'inner')}")


if __name__ == '__main__':
    print(f'ESCAPE outputs written to {run_ARC_scan()}')
