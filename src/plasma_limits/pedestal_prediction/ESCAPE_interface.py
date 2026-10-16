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
import copy
import dataclasses
import importlib.util
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["PedestalPlasmaState", "FreeParams", "SCConfig"]


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

    def _to_escape_dict(self):
        """The equilibrium as ESCAPE's equil_params: the read_eqdsk keys that ESCAPE reads.

        ESCAPE keeps references to the arrays, so each call returns fresh, writable copies.
        Without a pressure, 'pres' is zeros: ESCAPE uses its length for the 1D psi grid
        and reads its values only for EPED with pres_gfile=True.
        """
        R, Z = self.R, self.Z
        return {
            "nr": len(R),
            "nz": len(Z),
            "rleft": float(R[0]),
            "rdim": float(R[-1] - R[0]),
            "zmid": float((Z[0] + Z[-1]) / 2),
            "zdim": float(Z[-1] - Z[0]),
            "psirz": self.psi.copy(),
            "raxis": float(self.R_axis),
            "zaxis": float(self.Z_axis),
            "psimag": float(self.psi_axis),
            "psibry": float(self.psi_boundary),
            "fpol": self.F.copy(),
            "pres": np.zeros(len(self.F)) if self.pressure is None else self.pressure.copy(),
            "ip": float(self.Ip),
            "rzout": self.boundary.copy(),
        }

    def _to_kprof_params(self):
        """The kinetic profiles as ESCAPE's kprof_params, with fresh, writable copies.

        Ti and ni are included only when given; otherwise ESCAPE sets them from
        Te (Ti_follows_Te) and ne (quasineutral).
        """
        pairs = {"T_e": self.Te, "psin_Te": self.psi_N_Te, "n_e": self.ne, "psin_ne": self.psi_N_ne}
        if self.Ti is not None:
            pairs.update(T_i=self.Ti, psin_Ti=self.psi_N_Ti)
        if self.ni is not None:
            pairs.update(n_i=self.ni, psin_ni=self.psi_N_ni)
        return {key: arr.copy() for key, arr in pairs.items()}


@dataclass(frozen=True, kw_only=True)
class FreeParams:
    """Free parameters of the Saarelma-Connor model. The defaults are the ARC values.

    alpha_crit: critical normalized pressure gradient for the KBM transport.
    C_KBM: KBM diffusion coefficient.
    De_chie_etg: ratio of the ETG particle diffusivity to the electron heat diffusivity.
    nFC_x0: Franck-Condon neutral density at the separatrix [m^-3].
    ncx_x0_ratio: ratio of the charge-exchange to the Franck-Condon neutral
        density at the separatrix (3D model only).
    """

    alpha_crit: float = 0.01
    C_KBM: float = 0.3
    De_chie_etg: float = 0.1
    nFC_x0: float = 1e15
    ncx_x0_ratio: float = 15


_SOLVE_KEYWORD_FIELDS = ("tanh_center_psiN", "psiN_pwg")


@dataclass(frozen=True, kw_only=True)
class SCConfig:
    """Saarelma-Connor solver settings of one ESCAPE run. All fields are keyword-only.

    The defaults are the sc_params of ESCAPE's ARC driver, with the 1D model
    and the scipy implementation. Each key has the name and meaning that
    ESCAPE gives it (init_saarelmaconnor and the solve_* methods of the
    Saarelma-Connor solvers). The solver ignores the keys that the selected
    model and implementation do not use.

    Inner boundary: psi_N_inner_boundary, or None to find it from
        nFC_threshold and nCX_threshold.
    Model: model ('1D' or '3D'), implementation ('scipy' or 'firedrake').
        3D needs Firedrake, also with 'scipy'.
    Shared solver settings: x_method, verbose, x_res, ne_grad_bc_loc,
        picard_max_it, picard_rtol, picard_relax, reuse_setup.
    scipy settings: bvp_tol, bvp_max_nodes.
    Firedrake settings: fe_degree, grad_bc_tol, grad_bc_max_it, grad_bc_seed,
        linear_solver, ksp_rtol, ksp_max_it.
    3D settings: nCX_ic, nFC_ic, neutrals_treatment, n_neutral_sub.
    1D settings: eq6_form, first_step.
    free_params: FreeParams.
    Initial guess of ne in iteration 0:
        initial_guess: 'tanh' or 'tanh_linear'.
        tanh_center_psiN: center of the tanh [psi_N]. None uses the solver default.
        psiN_pwg: pedestal top (1 - pedestal width) [psi_N], for 'tanh_linear'.
    solver_kwargs: more solver keywords (for example dne_method, ne_inner_bc),
        added to the sc_params as they are. The config keeps a read-only copy.
    """

    psi_N_inner_boundary: float | None = 0.85
    nFC_threshold: float | None = None
    nCX_threshold: float | None = None
    x_method: str = "radas"
    verbose: bool = False
    model: str = "1D"
    implementation: str = "scipy"

    x_res: int = 40
    ne_grad_bc_loc: str = "inner"
    picard_max_it: int = 50
    picard_rtol: float = 1e-6
    picard_relax: float = 1.0
    reuse_setup: bool = False

    bvp_tol: float = 1e-6
    bvp_max_nodes: int = 5000

    fe_degree: int = 2
    grad_bc_tol: float = 1e-8
    grad_bc_max_it: int = 25
    grad_bc_seed: float | None = None
    linear_solver: str = "lu"
    ksp_rtol: float = 1e-8
    ksp_max_it: int = 200

    nCX_ic: str = "scale nFC"
    nFC_ic: str = "solve"
    neutrals_treatment: str = "fem"
    n_neutral_sub: int = 4001

    eq6_form: str = "complete"
    first_step: str = "auto"

    free_params: FreeParams = FreeParams()

    initial_guess: str = "tanh"
    tanh_center_psiN: float | None = None
    psiN_pwg: float | None = None

    solver_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "solver_kwargs",
                           types.MappingProxyType(copy.deepcopy(dict(self.solver_kwargs))))

    def to_params(self):
        """The sc_params dict for ESCAPE_solve, with free_params as a plain dict.

        ESCAPE_solve changes the dict that it gets (initial_guess, tanh_center,
        x_pwg), and a changed dict changes the next run that uses it. Thus each
        call returns a new dict, with deep copies of the solver_kwargs values.
        """
        params = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)
                  if f.name not in _SOLVE_KEYWORD_FIELDS and f.name != "solver_kwargs"}
        params["free_params"] = dataclasses.asdict(self.free_params)
        params.update(copy.deepcopy(dict(self.solver_kwargs)))
        return params

    def _solve_keywords(self):
        """The ESCAPE_solve keywords that set the initial guess in psi_N."""
        return {name: getattr(self, name) for name in _SOLVE_KEYWORD_FIELDS}
