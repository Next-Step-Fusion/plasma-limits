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
_PROFILE_PAIRS = (("psi_N_Te", "Te"), ("psi_N_ne", "ne"),
                  ("psi_N_Ti", "Ti"), ("psi_N_ni", "ni"))
_SPECIES = ("D", "D-T")
_GRID_RTOL = 1e-9
_PSI_N_END_TOL = 1e-12


def _check_uniform_grid(name, x):
    """ESCAPE rebuilds R and Z with linspace from their end points, so other grids are misread."""
    if x.ndim != 1 or len(x) < 2:
        raise ValueError(f"{name} must be a 1D array with at least 2 points, got shape {x.shape}")
    step = np.diff(x)
    if not np.all(step > 0):
        raise ValueError(f"{name} must increase strictly")
    if not np.allclose(step, (x[-1] - x[0]) / (len(x) - 1), rtol=_GRID_RTOL, atol=0.0):
        raise ValueError(f"{name} must be uniform: ESCAPE rebuilds it with linspace from its end points")


def _check_profile(grid_name, grid, name, values):
    if grid.ndim != 1 or values.ndim != 1:
        raise ValueError(f"{grid_name} and {name} must be 1D arrays")
    if len(grid) != len(values):
        raise ValueError(f"len({grid_name}) = {len(grid)} != len({name}) = {len(values)}")
    if len(grid) < 2:
        raise ValueError(f"{grid_name} must have at least 2 points")
    if not np.all(np.diff(grid) > 0):
        raise ValueError(f"{grid_name} must increase strictly")
    if abs(grid[-1] - 1.0) > _PSI_N_END_TOL:
        raise ValueError(f"{grid_name} must end at the separatrix (1.0), got {grid[-1]!r}")


@dataclass(frozen=True, eq=False, kw_only=True)
class PedestalPlasmaState:
    """All physical inputs of one ESCAPE run. All fields are keyword-only.

    The user fills in every field directly. The user is responsible for
    correct data and units. The state keeps float64, read-only copies of all
    arrays, so a later change to the caller's arrays has no effect on it.
    The constructor raises ValueError on shapes, grids and field combinations
    that make ESCAPE crash or misread the input.

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
        self._validate()

    def _validate(self):
        """Raise ValueError on inputs that make ESCAPE crash or misread them without an error.

        Units and physical values are not checked.
        """
        _check_uniform_grid("R", self.R)
        _check_uniform_grid("Z", self.Z)
        if len(self.R) != len(self.Z):
            raise ValueError(
                f"len(R) = {len(self.R)} != len(Z) = {len(self.Z)}: ESCAPE needs nr == nz, "
                "because ESCAPE_state.calc_B(rgrid, zgrid) evaluates B point by point")
        if self.psi.shape != (len(self.Z), len(self.R)):
            raise ValueError(f"psi must have shape (len(Z), len(R)) = {(len(self.Z), len(self.R))}, "
                             f"got {self.psi.shape}")
        if self.F.ndim != 1:
            raise ValueError(f"F must be a 1D array, got shape {self.F.shape}")
        if self.pressure is not None and self.pressure.shape != self.F.shape:
            raise ValueError(f"pressure must have the shape of F {self.F.shape}, got {self.pressure.shape}")
        if self.boundary.ndim != 2 or self.boundary.shape[1] != 2 or len(self.boundary) < 3:
            raise ValueError(f"boundary must have shape (N, 2) with N >= 3, got {self.boundary.shape}")

        for grid_name, name in _PROFILE_PAIRS:
            grid, values = getattr(self, grid_name), getattr(self, name)
            if grid is None and values is None:
                continue
            if grid is None or values is None:
                raise ValueError(f"give both {grid_name} and {name}, or neither")
            _check_profile(grid_name, grid, name, values)
        if not self.Ti_follows_Te and self.Ti is None:
            raise ValueError("Ti_follows_Te=False needs Ti and psi_N_Ti")
        if not self.quasineutral and self.ni is None:
            raise ValueError("quasineutral=False needs ni and psi_N_ni")

        if self.species not in _SPECIES:
            raise ValueError(f"species must be one of {_SPECIES}, got {self.species!r}")
