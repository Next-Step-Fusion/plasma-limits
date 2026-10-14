"""In-memory interface to the ESCAPE pedestal model (the ``escape`` package).

ESCAPE couples the Saarelma-Connor pedestal density model with EPEDNN. Its
own entry point, ``ESCAPE_solve``, reads and writes files. This module wraps
the same model without any file I/O: the caller builds all inputs in memory,
and results are returned in memory. The only files read from disk are the
EPED model weights and the atomic-rate tables bundled with ESCAPE.

Public API (added step by step):
    PedestalPlasmaState  -- every physical input (equilibrium, profiles, species)
    FreeParams, SCConfig -- Saarelma-Connor free parameters and solver settings
    EPEDConfig           -- which EPED model to use
    load_eped_model      -- load an EPED model once per process
    run_escape           -- run the ESCAPE loop, returning an ESCAPEResult
    IterationRecord, ESCAPEResult, ESCAPERunError

ESCAPE is imported lazily, on first use, so importing this module needs only
the standard library and numpy.

ESCAPE must be an editable install from an ESCAPE checkout, because it finds
its Julia EPEDNN model relative to its own source files. ``ESCAPE_ROOT`` is
that checkout; ESCAPE_io.py and the tests use it to find ESCAPE's example and
data files.
"""
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["PedestalPlasmaState"]


def _escape_root():
    """Root of the ESCAPE checkout that ``escape`` is installed from, or None if it isn't installed.

    find_spec doesn't import the package.
    """
    spec = importlib.util.find_spec("escape")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parents[2]  # <root>/src/escape/__init__.py


ESCAPE_ROOT = _escape_root()

_ARRAY_FIELDS = ("R", "Z", "psi", "F", "pressure", "boundary",
                 "psi_N_Te", "Te", "psi_N_ne", "ne",
                 "psi_N_Ti", "Ti", "psi_N_ni", "ni")


@dataclass(frozen=True, eq=False, kw_only=True)
class PedestalPlasmaState:
    """All physical inputs of one ESCAPE run. All fields are keyword-only.

    The user fills in every field directly. The user is responsible for
    correct data and units. The state keeps float64, read-only copies of all
    arrays, so a later change to the caller's arrays has no effect on it.

    Equilibrium grid:
        R (nr,), Z (nz,): uniform 1D grids [m], nr == nz.
        psi (nz, nr): poloidal flux on the (Z, R) grid. Unit: Wb/rad
            (COCOS 7, flux divided by 2*pi) if pol_norm is False,
            Wb if pol_norm is True.
    Equilibrium scalars:
        psi_axis, psi_boundary: psi at the magnetic axis and at the LCFS.
        R_axis, Z_axis: magnetic axis [m].
        Ip: plasma current [A], sign kept.
    Flux functions:
        F (n_psi,): R*B_phi [T m] on a uniform psi_N grid from 0 to 1.
            n_psi sets the number of points of ESCAPE's 1D psi grid.
        pressure (n_psi,) or None: pressure [Pa] on the same grid. Necessary
            only for EPED with pres_gfile=True.
    Boundary:
        boundary (N, 2): LCFS points (R, Z) [m].
    Kinetic profiles (each psi_N grid increases strictly and ends at 1):
        psi_N_Te, Te: electron temperature [keV].
        psi_N_ne, ne: electron density [m^-3].
        psi_N_Ti, Ti: ion temperature [keV], optional.
        psi_N_ni, ni: ion density [m^-3], optional.
    Species and parameters:
        species: 'D' or 'D-T'. Sets the effective ion mass (2.0 or 2.5).
        Z_i: ion charge number.
        Z_eff: effective charge.
        P_tot_e: total heating power to the electrons [W].
        T_rat: Ti/Te, used when Ti_follows_Te is True.
        Ti_follows_Te: if True, Ti = T_rat * Te. A Ti given here is used only
            until the first Te update, at the end of iteration 0.
        quasineutral: if True, ni = ne. An ni given here is used only until
            the first ne update, after the Saarelma-Connor solve of
            iteration 0.
        regime: ESCAPE's regime_flag ('PT H-mode' or 'NT'). It has no effect
            in ESCAPE at this time.
        ne_sep: electron density at the separatrix [m^-3], the outer boundary
            condition. None uses ne[-1].
        pol_norm: True if psi is not divided by 2*pi (see psi).
    """

    R: np.ndarray
    Z: np.ndarray
    psi: np.ndarray

    psi_axis: float
    psi_boundary: float
    R_axis: float
    Z_axis: float
    Ip: float

    F: np.ndarray
    pressure: np.ndarray | None = None

    boundary: np.ndarray

    psi_N_Te: np.ndarray
    Te: np.ndarray
    psi_N_ne: np.ndarray
    ne: np.ndarray
    psi_N_Ti: np.ndarray | None = None
    Ti: np.ndarray | None = None
    psi_N_ni: np.ndarray | None = None
    ni: np.ndarray | None = None

    species: str = "D"
    Z_i: int = 1
    Z_eff: float
    P_tot_e: float
    T_rat: float = 1.0
    Ti_follows_Te: bool = True
    quasineutral: bool = True
    regime: str = "PT H-mode"
    ne_sep: float | None = None
    pol_norm: bool = False

    def __post_init__(self):
        for name in _ARRAY_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            arr = np.array(value, dtype=np.float64)
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)
