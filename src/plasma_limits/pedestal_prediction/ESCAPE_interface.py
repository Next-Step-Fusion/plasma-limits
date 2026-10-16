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
    EPEDModel            -- a loaded EPED model with its EPEDConfig
    load_eped_model      -- load an EPED model, once before the runs
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

__all__ = ["PedestalPlasmaState", "FreeParams", "SCConfig", "EPEDConfig", "EPEDModel", "load_eped_model",
           "IterationRecord", "ESCAPEResult", "ESCAPERunError", "run_escape"]


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

    def _solve_keywords(self):
        """The ESCAPE_solve keywords of the state. They are also keywords of ESCAPE(...)."""
        return {
            "equil_params": self._to_escape_dict(),
            "kprof_params": self._to_kprof_params(),
            "Z_i": self.Z_i,
            "Z_eff": self.Z_eff,
            "P_tot_e": self.P_tot_e,
            "ne_x0": self.ne_sep,
            "quasineutral_flag": self.quasineutral,
            "T_rat_flag": self.Ti_follows_Te,
            "T_rat": self.T_rat,
            "pol_norm": self.pol_norm,
            "species": self.species,
            "regime_flag": self.regime,
        }


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
_MODELS = ("1D", "3D")
_IMPLEMENTATIONS = ("scipy", "firedrake")
_INITIAL_GUESSES = ("tanh", "tanh_linear")


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
        A key must not be a field name, 'tanh_center' or 'x_pwg': set these
        through the fields.

    The constructor raises ValueError on settings that make ESCAPE crash
    after the setup of the run. Numbers are not checked.
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
        self._validate()

    def _validate(self):
        if self.model not in _MODELS:
            raise ValueError(f"model must be one of {_MODELS}, got {self.model!r}")
        if self.implementation not in _IMPLEMENTATIONS:
            raise ValueError(f"implementation must be one of {_IMPLEMENTATIONS}, got {self.implementation!r}")

        if self.psi_N_inner_boundary is None:
            if self.model == "3D":
                raise ValueError("model='3D' needs psi_N_inner_boundary (solve_coupled_nondim)")
            if self.nFC_threshold is None or self.nCX_threshold is None:
                raise ValueError("psi_N_inner_boundary=None needs nFC_threshold and nCX_threshold "
                                 "(init_saarelmaconnor)")
        if self.model == "3D":
            for name in ("nFC_ic", "nCX_ic"):
                if getattr(self, name) == "state":
                    raise ValueError(
                        f"{name}='state' needs neutral profiles from an earlier solve, which iteration 0 "
                        "does not have (bc_ig_helpers._state_on_desc)")

        if self.initial_guess not in _INITIAL_GUESSES:
            raise ValueError(f"initial_guess must be one of {_INITIAL_GUESSES}, got {self.initial_guess!r}")
        if self.initial_guess == "tanh_linear" and self.psiN_pwg is None:
            raise ValueError("initial_guess='tanh_linear' needs psiN_pwg (build_ne_initial_guess)")

        reserved = {f.name for f in dataclasses.fields(self)} | {"tanh_center", "x_pwg"}
        clashes = sorted(reserved & set(self.solver_kwargs))
        if clashes:
            raise ValueError(f"solver_kwargs must not contain {clashes}: set them through the "
                             "SCConfig fields (tanh_center and x_pwg through tanh_center_psiN and psiN_pwg)")

    def _check_firedrake(self, available, error):
        """Raise ImportError, caused by error, if the run needs Firedrake and it is not available.

        The 3D model needs Firedrake also with the scipy implementation.
        """
        if not available and (self.model == "3D" or self.implementation == "firedrake"):
            raise ImportError(f"model={self.model!r} with implementation={self.implementation!r} "
                              "needs Firedrake, which can't be imported") from error

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


_EPED_MODELS = ("EPED1", "EPED_SPARC")


@dataclass(frozen=True, kw_only=True)
class EPEDConfig:
    """EPED settings of one ESCAPE run. All fields are keyword-only.

    name: 'EPED1' (EPEDNN.jl through juliacall) or 'EPED_SPARC' (epednn_mit,
        TensorFlow). EPED_SPARC needs the epednn_mit sources in the ESCAPE
        checkout (dependencies/epednn_mit/src).
    pres_gfile: if True, EPED uses PedestalPlasmaState.pressure. If False,
        it uses the pressure from the kinetic profiles.

    The constructor raises NotImplementedError for 'EPED_SCOPING' and
    ValueError for other names.
    """

    name: str = "EPED1"
    pres_gfile: bool = False

    def __post_init__(self):
        if self.name == "EPED_SCOPING":
            raise NotImplementedError(
                "EPED_SCOPING can't run: epednn_class.feed_epednn does not calculate its inputs qstar and nsfrac")
        if self.name not in _EPED_MODELS:
            raise ValueError(f"EPED model name must be one of {_EPED_MODELS}, got {self.name!r}")


@dataclass(frozen=True, eq=False, kw_only=True)
class EPEDModel:
    """A loaded EPED model and the EPEDConfig it was loaded with.

    ESCAPE selects how it calls the model from the model name, so the two
    are kept together. Make it with load_eped_model.
    """

    config: EPEDConfig
    model: object


def load_eped_model(config=EPEDConfig()):
    """Load the EPED model of config and return it as an EPEDModel.

    Load the model once, before a scan or a time loop, and give the same
    EPEDModel to each run: each call of this function loads the model again.
    For EPED1, the model is called once at the center of its training range,
    so that Julia compiles the call here and not in the first run.
    Julia is not safe across fork, so parallel runs must start their
    processes with 'spawn'.
    """
    if not isinstance(config, EPEDConfig):
        raise TypeError(f"config must be an EPEDConfig, got {type(config).__name__}")
    from escape.epednn.epednn_call import load_epednn_model
    model = load_epednn_model(config.name)
    if config.name == "EPED1":
        _warm_up_eped1(model)
    return EPEDModel(config=config, model=model)


def _warm_up_eped1(model):
    """Call EPED1 with the same signature as epednn_class.feed_epednn.

    The inputs are the centers of the training bounds, so EPEDNN prints no
    extrapolation warning.
    """
    center = np.array(model.xbounds, dtype=np.float64).mean(axis=1)
    model(*(float(x) for x in center), only_powerlaw=False, warn_nn_train_bounds=True)


_PLAIN_SCALARS = (np.generic, bool, int, float, complex, str, type(None))


def _is_plain(obj):
    if isinstance(obj, (dict, list, tuple)):
        return True
    if isinstance(obj, np.ndarray):
        return obj.dtype != object
    return isinstance(obj, _PLAIN_SCALARS)


def _plain(obj):
    """A deep copy of obj that keeps only values that load with numpy alone.

    Dicts, lists and tuples are kept, and their other values are dropped.
    """
    if isinstance(obj, dict):
        return {key: _plain(value) for key, value in obj.items() if _is_plain(value)}
    if isinstance(obj, list):
        return [_plain(value) for value in obj if _is_plain(value)]
    if isinstance(obj, tuple):
        return tuple(_plain(value) for value in obj if _is_plain(value))
    if isinstance(obj, np.ndarray):
        return obj.copy()
    return obj


def _read_only_copy(value):
    arr = np.array(value)
    arr.flags.writeable = False
    return arr


# IterationRecord field -> key of the dict that ESCAPE_solve gives to on_iteration
_RECORD_ARRAYS = (("x", "x"), ("ne_x", "ne"), ("nFC", "nFC"), ("nCX", "nCX"),
                  ("psi_N_pres", "psi_N"), ("T_e_pres", "T_e_pres"),
                  ("psi_N_ne", "psi_ne_eval"), ("ne", "n_e"),
                  ("psi_N_Te", "psi_Te_eval"), ("Te", "T_e"))
_RECORD_VALUES = (("Te_ped_eV", "Te_ped_eV"), ("ne_ped", "ne_ped_val"), ("psi_ped", "psi_ped"),
                  ("pedestal_height", "pedestal_height"), ("pedestal_width", "pedestal_width"),
                  ("model", "model"), ("implementation", "implementation"))


@dataclass(frozen=True, eq=False, kw_only=True)
class IterationRecord:
    """The result of one ESCAPE iteration. All arrays are read-only copies.

    iteration: index of the ESCAPE iteration, from 0.
    Saarelma-Connor solution, on the x grid [m, 0 at the separatrix, negative inside]:
        x, ne_x: electron density [m^-3].
        nFC, nCX: Franck-Condon and charge-exchange neutral densities [m^-3].
    psi_N_pres, T_e_pres: the Te [eV] that the Saarelma-Connor solve used.
    Full profiles after the iteration, with the names and units of PedestalPlasmaState:
        psi_N_ne, ne: electron density [m^-3].
        psi_N_Te, Te: electron temperature [keV], with the new EPED pedestal.
    Pedestal:
        Te_ped_eV: Te at the pedestal top [eV].
        ne_ped: ne at the pedestal top [m^-3].
        psi_ped: pedestal top [psi_N].
        pedestal_height: EPED pedestal pressure [MPa].
        pedestal_width: EPED pedestal width [psi_N].
        betan: normalized beta that EPED got.
    loop_tol: relative change of the pedestal height plus that of the width,
        from the previous iteration. None in iteration 0.
    params: the sc_params of the solve.
    diagnostics: the solver diagnostics, with only values that load with numpy alone.
    model, implementation: the Saarelma-Connor model and implementation used.
    """

    iteration: int

    x: np.ndarray
    ne_x: np.ndarray
    nFC: np.ndarray
    nCX: np.ndarray
    psi_N_pres: np.ndarray
    T_e_pres: np.ndarray

    psi_N_ne: np.ndarray
    ne: np.ndarray
    psi_N_Te: np.ndarray
    Te: np.ndarray

    Te_ped_eV: float
    ne_ped: float
    psi_ped: float
    pedestal_height: float
    pedestal_width: float
    betan: float

    loop_tol: float | None
    params: dict
    diagnostics: dict
    model: str
    implementation: str

    @classmethod
    def _from_sol(cls, iteration, sol):
        """The record of the dict that ESCAPE_solve gives to on_iteration.

        sol shares its arrays and SOLVE_KW with the running solve, so the
        record keeps copies. Iteration 0 has no loop_tol key.
        """
        return cls(
            iteration=iteration,
            **{name: _read_only_copy(sol[key]) for name, key in _RECORD_ARRAYS},
            **{name: sol[key] for name, key in _RECORD_VALUES},
            betan=float(np.asarray(sol["betan"]).item()),
            loop_tol=sol.get("loop_tol"),
            params=copy.deepcopy(sol["SOLVE_KW"]),
            diagnostics=_plain(sol["diagnostics"]),
        )

    def to_dict(self):
        """The fields as a dict of deep copies."""
        return dataclasses.asdict(self)


@dataclass(frozen=True, eq=False, kw_only=True)
class ESCAPEResult:
    """The result of one ESCAPE run.

    converged: True if the last iteration changed the pedestal by less than tol.
    history: an IterationRecord for each iteration, in order.
    state: ESCAPE's state object at the end of the run, if requested. It holds
        Julia objects, so to_dict() does not include it.
    """

    converged: bool
    history: tuple
    state: object = None

    def __post_init__(self):
        object.__setattr__(self, "history", tuple(self.history))

    @property
    def n_iter(self):
        return len(self.history)

    @property
    def final(self):
        return self.history[-1]

    def to_dict(self):
        """The result without state, as values that load with numpy alone."""
        return {
            "converged": self.converged,
            "n_iter": self.n_iter,
            "history": [record.to_dict() for record in self.history],
        }


class ESCAPERunError(RuntimeError):
    """An ESCAPE run failed. history holds the IterationRecords made before the failure."""

    def __init__(self, message, history=()):
        super().__init__(message)
        self.history = tuple(history)

    def __reduce__(self):
        return type(self), (self.args[0], self.history)


def run_escape(plasma, eped, sc=SCConfig(), *, max_iter=500, tol=1e-5, on_iteration=None,
               keep_state=False, verbose=False):
    """Run the ESCAPE loop (ESCAPE_solve) for one plasma, in memory.

    plasma: PedestalPlasmaState.
    eped: EPEDModel from load_eped_model. Load it once and give it to each run.
    sc: SCConfig. Its verbose field controls the solver output. With
        model='3D' and implementation='firedrake', sc.verbose=True also makes
        ESCAPE save one PNG per iteration into the working directory.
    max_iter: maximum number of ESCAPE iterations.
    tol: the loop stops when loop_tol (see IterationRecord) is less than tol.
    on_iteration: None, or a callable that gets the IterationRecord of each
        iteration, when the iteration is complete.
    keep_state: if True, the result keeps ESCAPE's state object.
    verbose: ESCAPE's own verbose flag (the setup of the state).

    No iteration file is written. ESCAPE_solve prints its progress to stdout.

    Returns an ESCAPEResult. converged is True if the run stopped on tol
    before max_iter.

    Raises TypeError, ValueError or ImportError on arguments that can't run,
    before ESCAPE starts. Raises ESCAPERunError, with the records made so far,
    if an exception occurs during the run, also in on_iteration.
    """
    if not isinstance(eped, EPEDModel):
        raise TypeError(f"eped must be an EPEDModel (see load_eped_model), got {type(eped).__name__}")
    if eped.config.pres_gfile and plasma.pressure is None:
        raise ValueError("EPEDConfig.pres_gfile=True needs PedestalPlasmaState.pressure")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")

    import matplotlib.pyplot as plt
    from escape.ESCAPE.ESCAPE_solve import ESCAPE_solve
    from escape.saarelma_connor import solver_3d

    sc._check_firedrake(solver_3d._FIREDRAKE_AVAILABLE, solver_3d._FIREDRAKE_IMPORT_ERR)

    history = []

    def collect(iteration, sol):
        record = IterationRecord._from_sol(iteration, sol)
        history.append(record)
        if on_iteration is not None:
            on_iteration(record)

    try:
        # TODO: when ESCAPE has a plot keyword in solve_coupled_nondim (U17), set it in
        # sc_params for 3D Firedrake only, so that sc.verbose=True saves no PNG.
        # Then remove the PNG note from the docstring (TASK.md, Step 10.1).
        with plt.ioff():
            state = ESCAPE_solve(
                sc_params=sc.to_params(),
                epednn_params={"epednn_model": eped.config.name, "pres_gfile": eped.config.pres_gfile},
                **sc._solve_keywords(),
                **plasma._solve_keywords(),
                ESCAPE_iter_max=max_iter,
                ESCAPE_tol_max=tol,
                out_dir=None,
                verbose=verbose,
                epednn_model=eped.model,
                on_iteration=collect,
            )
    except Exception as err:
        raise ESCAPERunError(f"ESCAPE run failed after {len(history)} iteration records: {err!r}",
                             history) from err

    return ESCAPEResult(
        converged=len(history) > 1 and history[-1].loop_tol < tol,
        history=history,
        state=state if keep_state else None,
    )
