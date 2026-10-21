"""Run ESCAPE on a CENTAUR state from NSFsim with the in-memory interface (1D scipy model, EPED1).

The input is the last time step of the NSFsim history file CENTAUR/centaur.h5.
This script reads it with h5py and fills in a PedestalPlasmaState; no g-file
or profile file is written. The script saves the plot of the profiles to
outputs/ in the working directory.

DISCLAIMER: CENTAUR has negative triangularity. ESCAPE is known to give
unphysical results for negative triangularity (because of EPED), and the results of this run
are not physical. The only purpose of this example is to show the data flow
from an NSFsim history file to run_escape.

Needs ESCAPE (editable install). Run from the repository root:
    uv run python examples/ESCAPE_CENTAUR_example.py
"""
from pathlib import Path

import h5py
import numpy as np

from ESCAPE_ARC_example import plot_history
from plasma_limits.pedestal_prediction.ESCAPE_interface import (FreeParams, PedestalPlasmaState, SCConfig,
                                                              load_eped_model, run_escape)

HISTORY_FILE = Path(__file__).parent / "CENTAUR" / "centaur.h5"


def centaur_state(path, step=-1):
    """The PedestalPlasmaState of one time step of an NSFsim history file.

    NSFsim gives psi in Wb (not divided by 2*pi), with psi[i_R, i_Z].
    The kinetic profiles are on a rho_tor grid, with psi(rho_tor) in psi_profile.
    """
    with h5py.File(path, "r") as f:
        s = {key: value[step] for key, value in f["history/state"].items()}

    two_pi = 2 * np.pi
    psi_profile = s["psi_profile"]
    psi_N = (psi_profile - psi_profile[0]) / (psi_profile[-1] - psi_profile[0])  # psi_profile[-1] = psi_separatrix

    return PedestalPlasmaState(
        R=s["x_psi"],
        Z=s["y_psi"],
        psi=s["psi"].T / two_pi,  # (R, Z) -> (Z, R), Wb -> Wb/rad
        psi_axis=s["psi_axis"] / two_pi,
        psi_boundary=s["psi_separatrix"] / two_pi,  # psi_boundary in the file is the psi_N = 0.995 surface
        R_axis=s["magnetic_center"][0],
        Z_axis=s["magnetic_center"][1],
        Ip=s["plasma_current"],
        F=s["eqdsk_data_fpol_out"],  # on a uniform psi_N grid (eqdsk_data_psix_out)
        pressure=s["eqdsk_data_pres_out"],
        boundary=s["boundary"][:-1],  # as the g-file writer of NSFsim: the last point isn't part of the LCFS
        psi_N_Te=psi_N, Te=s["electron_temperature_profile"] / 1e3,  # eV -> keV
        psi_N_ne=psi_N, ne=s["electron_density_profile"] * 1e19,  # 1e19 m^-3 -> m^-3
        species="D-T",
        Z_eff=s["zeff"],
        P_tot_e=10.0e6,  # assumed heating power to the electrons [W]
    )


if __name__ == "__main__":
    print("DISCLAIMER: ESCAPE is unphysical for negative triangularity. This example only shows the data flow.")
    state = centaur_state(HISTORY_FILE)
    sc = SCConfig(
        x_res=50,
        picard_rtol=1e-8,
        free_params=FreeParams(alpha_crit=10, C_KBM=0.1, De_chie_etg=1, nFC_x0=1e15, ncx_x0_ratio=17.8),
        first_step="skip",  # with these free parameters, the Eq. 6 first step makes the BVP singular
    )
    result = run_escape(state, load_eped_model(), sc, max_iter=5)

    print(f"{result.n_iter} iterations, converged: {result.converged}")
    print(f"pedestal height {result.final.pedestal_height:.6f} MPa, "
          f"width {result.final.pedestal_width:.6f} psi_N")
    plot_path = Path("outputs") / "ESCAPE_CENTAUR_profiles.png"
    plot_history(result, plot_path)
    print(f"plot saved to {plot_path}")
