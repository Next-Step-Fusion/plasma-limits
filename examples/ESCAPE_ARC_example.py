"""Run ESCAPE on ARC with the in-memory interface (1D scipy model, EPED1).

The inputs are the files of ESCAPE's ARC example: the g-file geqdsk-ARCv3a and
the digitized profiles ARC_ne.csv and ARC_Te.csv. This script reads them and
fills in a PedestalPlasmaState. run_escape writes no file. The script saves
the plot of the profiles to outputs/ in the working directory.

Needs ESCAPE (editable install) and freeqdsk. Run from the repository root:
    uv run python examples/ESCAPE_ARC_example.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from freeqdsk import geqdsk

from plasma_limits.pedestal_prediction.ESCAPE_interface import (ESCAPE_ROOT, PedestalPlasmaState,
                                                              load_eped_model, run_escape)


def load_profile(path):
    """psi_N and the values of a Web Plot Digitizer CSV, sorted, with psi_N in [0, 1]."""
    data = np.loadtxt(path, delimiter=",")
    psi_N = np.clip(data[:, 0], 0.0, 1.0)  # digitization noise can put a point past the separatrix
    order = np.argsort(psi_N, kind="stable")
    psi_N, values = psi_N[order], data[order, 1]
    keep = np.append(np.diff(psi_N) > 0, True)  # after the clip, keep the outermost point at psi_N = 1
    return psi_N[keep], values[keep]


def arc_state(arc_dir):
    with open(arc_dir / "geqdsk-ARCv3a") as f:
        g = geqdsk.read(f)
    psi_N_ne, ne = load_profile(arc_dir / "ARC_ne.csv")  # 1e20 m^-3
    psi_N_Te, Te = load_profile(arc_dir / "ARC_Te.csv")  # keV

    return PedestalPlasmaState(
        # The g-file gives the grid as end points and sizes
        R=np.linspace(g.rleft, g.rleft + g.rdim, g.nx),
        Z=np.linspace(g.zmid - g.zdim / 2, g.zmid + g.zdim / 2, g.ny),
        psi=g.psi.T,  # freeqdsk stores psi as (R, Z); the state needs (Z, R)
        psi_axis=g.simagx,
        psi_boundary=g.sibdry,
        R_axis=g.rmagx,
        Z_axis=g.zmagx,
        Ip=g.cpasma,
        F=g.fpol,
        pressure=g.pres,
        boundary=np.column_stack([g.rbdry, g.zbdry]),
        psi_N_Te=psi_N_Te, Te=Te,
        psi_N_ne=psi_N_ne, ne=ne * 1e20,
        Z_eff=1.52,
        P_tot_e=(21.5 + 0.8 + 227) * 0.5 * 1e6,  # table 4 of Hillesheim et al. 2026
    )


def plot_history(result, path):
    """ne, Te and the neutral densities of each iteration.

    ESCAPE predicts only the pedestal. Inside it, ESCAPE keeps the shape of the
    input profile and shifts it to the new pedestal top: these parts are dotted.
    """
    fig, (ax_ne, ax_Te, ax_n0) = plt.subplots(1, 3, figsize=(15, 4))
    for record in result.history:
        color = f"C{record.iteration}"
        # ESCAPE's ne is the shifted core, then the Saarelma-Connor solution on its x grid
        n_ped = len(record.x)
        psi_N_ped = record.psi_N_ne[-n_ped:]
        ax_ne.plot(record.psi_N_ne[:-n_ped + 1], record.ne[:-n_ped + 1], ":", color=color)
        ax_ne.plot(psi_N_ped, record.ne[-n_ped:], color=color, label=f"iteration {record.iteration}")

        core = record.psi_N_Te <= record.psi_ped
        ax_Te.plot(record.psi_N_Te[core], record.Te[core], ":", color=color)
        ax_Te.plot(record.psi_N_Te[~core], record.Te[~core], color=color)
        ax_Te.plot(record.psi_ped, record.Te_ped_eV / 1e3, "o", color=color)

        ax_n0.plot(psi_N_ped, record.nFC, color=color)
        ax_n0.plot(psi_N_ped, record.nCX, "--", color=color)

    ax_ne.set(xlabel=r"$\psi_N$", ylabel=r"$n_e$ [m$^{-3}$]")
    ax_ne.legend()
    ax_Te.set(xlabel=r"$\psi_N$", ylabel=r"$T_e$ [keV]")
    ax_Te.legend(handles=[plt.Line2D([], [], color="k", label="predicted"),
                          plt.Line2D([], [], color="k", ls=":", label="input shape, shifted"),
                          plt.Line2D([], [], color="k", ls="", marker="o", label="pedestal top")])
    ax_n0.set(xlabel=r"$\psi_N$", ylabel=r"neutral density [m$^{-3}$]", yscale="log")
    n0_max = max(max(record.nFC.max(), record.nCX.max()) for record in result.history)
    ax_n0.set_ylim(1e-6 * n0_max, 3 * n0_max)
    ax_n0.legend(handles=[plt.Line2D([], [], color="k", label="Franck-Condon"),
                          plt.Line2D([], [], color="k", ls="--", label="charge exchange")])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    state = arc_state(ESCAPE_ROOT / "examples" / "device_prediction" / "ARC")

    # Load EPED once. In a scan or a time loop, give the same model to each run.
    eped = load_eped_model()

    # To keep each iteration on disk, give run_escape a callback:
    # def save(record):
    #     np.save(f"ARC_iter_{record.iteration}.npy", record.to_dict(), allow_pickle=True)
    # result = run_escape(state, eped, max_iter=5, on_iteration=save)
    result = run_escape(state, eped, max_iter=5)

    print(f"{result.n_iter} iterations, converged: {result.converged}")
    print(f"pedestal height {result.final.pedestal_height:.6f} MPa, "
          f"width {result.final.pedestal_width:.6f} psi_N")
    plot_path = Path("outputs") / "ESCAPE_ARC_profiles.png"
    plot_history(result, plot_path)
    print(f"plot saved to {plot_path}")
