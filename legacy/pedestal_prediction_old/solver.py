import os
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.integrate import simpson, solve_bvp, cumulative_trapezoid
from scipy import constants
import matplotlib.pyplot as plt
from matplotlib.path import Path as _MplPath
from src.adas.adas_ionisation import scd_adas
from src.adas.adas_cx import scx_adas
from src.radas.radas_rates import scd_radas, scx_radas
try:
    from firedrake import (
        IntervalMesh, FunctionSpace, MixedFunctionSpace, Function,
        TestFunctions, TrialFunction, Constant, DirichletBC,
        dx, ds, split, solve, lhs, rhs,
        LinearVariationalProblem, LinearVariationalSolver,
        SpatialCoordinate,
    )
    _FIREDRAKE_AVAILABLE = True
except Exception as _firedrake_import_err:  # raise an error if Firedrake is not available
    _FIREDRAKE_AVAILABLE = False
    _FIREDRAKE_IMPORT_ERR = _firedrake_import_err


class saarelma_connor:
    """

    Description
    ----------
    Creates instance of a tokamak pedestal configuration that the Saarelma-Connor (S. Saarelma et al 2024 Nucl. Fusion 64 076025) model can be applied to
    Dependencies:
    - juliacall (install with pip install juliacall) for EPEDNN interfacing
    - EPEDNN (install by git clone)
    - OpenFUSIONToolkit -> TokaMaker

    Uses COCOS 7 coordinate convention (same as TokaMaker) as defined by https://crppwww.epfl.ch/~sauter/cocos/Sauter_COCOS_Tokamak_Coordinate_Conventions.pdf

    Parameters
    ----------
    E_FC : float
        Energy of Franck-Condon neutrals as defined in Mahdavi M.A., Maingi R., Groebner R.J., Leonard A.W., Osborne T.H. and Porter G. 2003 Phys. Plasmas 10 3984 J
    Z_i : int
        Z of ions
    M_i : float
        Proton mass, kg
    M_e : float
        Electron mass, kg
    P_tot_e : float
        Total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox, W
    alpha_crit : float
        FREE PARAMETER, Critical alpha value for onset of infinite-n ballooning instability, dimensionless
    C_KBM : float
        FREE PARAMETER, KBM diffusion coefficient, m^2/s
    De_chie_etg : float
        FREE PARAMETER, ETG diffusion coefficient, m^2/s
    nFC_x0 : float
        m^-3, FREE PARAMETER, Franck-Condon neutral density at the separatrix (boundary condition)
    ne_x0 : float
        m^-3, electron density at the separatrix (boundary condition)
    psi_N_inner_boundary : float
        psi_N to choose the inner boundary boundary condition. If None, inner boundary is chosen based on nFC_threshold and nCX_threshold.
    nCX_x0 : float
        m^-3, CX neutral density at the separatrix (boundary condition). If None, defaults to ncx_x0_ratio * nFC_x0
    ncx_x0_ratio : float
        ratio of nCX at the separatrix to nFC at the separatrix. Used if nCX_x0 is None
    nFC_threshold : float or None
        Fraction of the separatrix FC neutral density (nFC_x0) below which the
        inner boundary is placed.  Default 0.01 (1 %).  Set to None to disable. Not used if psi_N_inner_boundary!=None
    nCX_threshold : float or None
        Fraction of the peak estimated CX neutral density below which the inner
        boundary is placed.  Default 0.01 (1 %).  Set to None to disable. Not used if psi_N_inner_boundary!=None
    mhd_loc : string
        Location of MHD equilibrium parameters, currently supporting: Tokamaker eqdsk
    kprof_loc : string
        Location of kinetic equilibrium parameters, currently supporting: p-file
    mhd_fp : string
        Filepath to mhd_loc-type file
    kprof_fp : string
        Filepath to kprof_loc-type file
    T_rat_flag : bool
        True if the temperature ratio is given, False if the temperature ratio is to be calculated
    T_rat : float
        Temperature ratio between ions and electrons, dimensionless
        Ignored if T_rat_flag is False
        Default is 1
    pol_norm : bool
        True if the poloidal flux is normalized by 2pi, False if the poloidal flux is not normalized by 2pi
    species : string
        Species of ions, currently supporting: D, D-T
    error_check : bool
        True if you want to use the exact published equations from Samuli's 2023 paper, False if you want to use the corrected equations
    equations_to_solve : string
        Equations to solve, currently supporting: 'coupled' (full 3-fluid system of ODEs), 'SC' (simplified 3-fluid system to one ODE)
    initial_guess : string
        Initial guess for the electron density profile, currently supporting: pfile
    nCX_ic : string
        Initial condition for the CX density profile, currently supporting: 'solve' (solve for the CX density profile), 'scale nFC' (scale the FC density profile by the specified initial condition ratio)
    verbose : bool
        True if verbose output is desired, False if verbose output is not desired
    """
    def __init__(
        self,
        E_FC = 3 * 1.60218e-19, # J,
        Z_i = 1, # Z of ions
        M_i = 1.673e-27, # kg, mass of hydrogen nuclei
        M_e = 9.109e-31, # kg, mass of electron
        P_tot_e = None, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox
        alpha_crit = None, # FREE PARAMETER
        C_KBM = None, # FREE PARAMETER
        De_chie_etg = None, # FREE PARAMETER
        nFC_x0 = None, # m^-3, FREE PARAMETER, Franck-Condon neutral density at the separatrix
        ne_x0 = None, # m^-3, electron density at the separatrix (boundary condition, default is to use from pfile)
        psi_N_inner_boundary = 0.85, # normalized poloidal flux at the inner boundary (boundary condition); overridden by find_inner_boundary if nFC_threshold or nCX_threshold is set
        nCX_x0=None, # m^-3, CX neutral density at the separatrix (Dirichlet BC at x = 0). If None, defaults to 0.1 * nFC_x0 (used in coupled solver)
        ncx_x0_ratio = None, # ratio of nCX at the separatrix to nFC at the separatrix (used in coupled solver)
        nFC_threshold = 0.01, # fraction of nFC at the separatrix below which the inner boundary is placed (None to disable)
        nCX_threshold = 0.01, # fraction of nCX at the separatrix below which the inner boundary is placed (None to disable)
        mhd_loc = 'eqdsk', # location of MHD equilibrium parameters, currently supporting: Tokamaker eqdsk
        kprof_loc = 'pfile', # location of kinetic parameters, currently supporting: p-file
        mhd_fp = None, # filepath to MHD paramter file
        kprof_fp = None, # filepath to kinetic paramter file
        manual_profs = None, # manual profiles for the electron temperature and density, currently supporting: 'pfile', 'epednn'
        T_rat_flag = True, # True if using a temperature ratio between ions and electrons, False if doing something else
        T_rat = 1,
        pol_norm = False, # True for when the poloidal flux is not normalized by 2pi. COCOS 7 convention is pol_norm=False, so poloidal flux is normalized by 2pi
        species = 'D', # species of ions, currently supporting: D, D-T
        error_check = False, # check if Samuli's 2023 paper Eq. 15 is correct or not
        equations_to_solve = 'coupled', # equations to solve, currently supporting: 'coupled', 'SC'
        initial_guess = 'pfile', # initial guess for the electron density profile, currently supporting: pfile, linear
        nCX_ic="solve",
        regime_flag = 'PT H-mode', # regime of the plasma, currently supporting: 'PT H-mode', 'NT'
        x_method = 'radas', # method to use for the cross-section rates, currently supporting: 'adas', 'radas'
        verbose = False,
    ):

        # User-specified flags
        self.regime_flag = regime_flag
        self.equations_to_solve = equations_to_solve
        self.error_check = error_check
        self.T_rat_flag = T_rat_flag
        self.T_rat = T_rat
        self.verbose = verbose
        if self.verbose:
            self.bvp_verbose = 2
        else:
            self.bvp_verbose = 0
        self.pol_norm = pol_norm
        self.psi_N_inner_boundary = psi_N_inner_boundary
        self.initial_guess = initial_guess

        # User-specified thresholds for the inner boundary
        self.nFC_threshold = nFC_threshold
        self.nCX_threshold = nCX_threshold

        self.ncx_x0_ratio = ncx_x0_ratio

        self.mu0 = 4 * np.pi * 10**-7 # N/A**2, vacuum magnetic permeability constant
        self.P_tot_e = P_tot_e

        self.M_i = M_i
        if species == 'D':
            self.M_eff = 2.0
        elif species == 'D-T':
            self.M_eff = 2.5
        else:
            assert False, 'species must be D or D-T'

        # Load in quantities
        self.mhd_load(mhd_loc,mhd_fp) # load in MHD quantities
        self.kprof_load(kprof_loc,kprof_fp,manual_profs=manual_profs) # load in kinetic quantities
        
        # Calculate the magnetic field at each RZ grid point, sets self.B
        self.calc_B(self.rgrid,self.zgrid)

        # calculate the flux surface-averaged |grad(r)| and |grad(r)|^2 and some other quantities like r_psi (outboard midplane minor radius for each flux surface)
        self.calc_gradr() # only a function of geometry

        # Calculate velocities
        self.Z_i = Z_i
        self.e_i = Z_i * constants.e # C
        k_B = 1.38064852e-23 # J/K, Boltzmann constant
        self.V_th_i = np.sqrt(2*k_B*self.T_i_K/(M_i*self.M_eff)) # m/s, per psi_N_eval for Ti
        self.V_th_e = np.sqrt(2*k_B*self.T_e_K/M_e) # m/s, per psi_N_eval for Te
        self.V_FC = np.sqrt(8*E_FC/((np.pi**2) * M_i*self.M_eff)) # m/s
        self.V_cx = np.sqrt(2*k_B*self.T_i_K/(np.pi * M_i*self.M_eff)) # m/s, per psi_N_eval for Ti

        # Load in cross-section rates, which are only a function of temperature if we use an average density (predictive models could provide a guess density)
        self.cross_section_rates(species=species,x_method=x_method) # load in cross-sections
        # self.S_i = self.sigma_i # m^3/s, ionization <sigma v> profile on psi_Te_eval (scd_adas already returns the rate coefficient)
        # self.S_cx = self.sigma_cx * self.V_th_i # m^3/s, CX rate coefficient profile on psi_Te_eval


        # Setup for diffusion coefficient that does not include free parameters and n_e
        self.c_s = (self.e_i * self.T_e * 1e3 / (M_i * self.M_eff)) ** 0.5 # m/s, cs = (e*T_e/mD)^1/2, T_e in keV -> eV via 1e3, as defined in W. Guttenfelder et al 2021 Nucl. Fusion 61 056005
        V_th_i_rz = self.psi_rz_expand(self.V_th_i, psi_N_A='T_e')
        self.rho_s = V_th_i_rz*M_i*self.M_eff / (self.e_i * self.B) # m, known on each RZ grid point
        self.rho_s = self.fsa(self.rho_s,flux_surfaces='T_e') # m, known on each flux surface, outputs nan for psi_N < 0.01 or psi_N > 0.99
        valid = ~np.isnan(self.rho_s)
        self.rho_s = interp1d(self.psi_Te_eval[valid], self.rho_s[valid], kind='linear',bounds_error=False, fill_value='extrapolate')(self.psi_Te_eval) # removes nan values from rho_s
        self.mu0 = 4 * np.pi * 10**-7 # N/A**2, vacuum magnetic permeability constant

        # Interpolate T_e, c_s, rho_s from psi_Te_eval onto the pressure psi_N grid
        T_e_pres = interp1d(self.psi_Te_eval, self.T_e, kind='linear',
                            bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        self.T_e_pres = T_e_pres * (1e3) # eV, on psi_N_pres grid
        if getattr(self, 'T_i_from_profile', False):
            self.T_i_pres = interp1d(
                self.psi_Ti_eval_pfile, self.T_i_pfile, kind='linear',
                bounds_error=False, fill_value='extrapolate'
            )(self.psi_N_pres) * (1e3)  # eV
        elif self.T_rat_flag:
            self.T_i_pres = T_e_pres * self.T_rat * (1e3) # eV, on psi_N_pres grid
        else:
            raise NotImplementedError("T_rat_flag must be True for now")
        self.n_e_pres = interp1d(self.psi_ne_eval, self.n_e_pfile, kind='linear',
                            bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        self.n_i_pres = interp1d(self.psi_ni_eval, self.n_i_pfile, kind='linear',
                            bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        self.c_s = interp1d(self.psi_Te_eval, self.c_s, kind='linear',
                       bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        self.rho_s = interp1d(self.psi_Te_eval, self.rho_s, kind='linear',
                         bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)

        # Also store pfile information for T_e and T_i
        self.T_e_pres_pfile = interp1d(self.psi_Te_eval_pfile, self.T_e_pfile, kind='linear',
                            bounds_error=False, fill_value='extrapolate')(self.psi_N_pres) * (1e3) # eV, on psi_N_pres grid
        if getattr(self, 'T_i_from_profile', False):
            self.T_i_pres_pfile = interp1d(
                self.psi_Ti_eval_pfile, self.T_i_pfile, kind='linear',
                bounds_error=False, fill_value='extrapolate'
            )(self.psi_N_pres) * (1e3)  # eV
        elif self.T_rat_flag:
            self.T_i_pres_pfile = self.T_e_pres_pfile * self.T_rat # eV, on psi_N_pres grid
        else:
            raise NotImplementedError("T_rat_flag must be True for now")

        grad_Te = np.gradient(self.T_e_pres * (1.60218e-19), self.r_psi) # gradient in J/m, T_e_pres is in eV
        self.D_ETG_x = P_tot_e / (self.S_plasma * abs(grad_Te)) # evaluated at each psi_N_pres, not including free parameter De_chie_etg and n_e
        self.D_NEO = 0.05 * (self.c_s * self.rho_s**2) / self.a

        # Outer boundary condition for electrons and FC neutrals
        self.ne_x0 = self.n_e_pfile[-1]
        if nFC_x0 is None:
            nFC_x0 = self.n_e_pfile[-1] * 1e-4
            if self.verbose:
                print('nFC_x0 = ', nFC_x0)

        # Set free parameters
        self.update_free_params(
            alpha_crit, C_KBM, De_chie_etg, nFC_x0,
            nFC_threshold=nFC_threshold,
            nCX_threshold=nCX_threshold,
            psi_N_inner_boundary=psi_N_inner_boundary,
            ncx_x0_ratio=ncx_x0_ratio,
        )

        if equations_to_solve == 'coupled':
            if nCX_x0 is not None:
                self.nCX_x0 = nCX_x0
            elif ncx_x0_ratio is not None:
                self.ncx_x0_ratio * self.nFC_x0 # set nCX boundary condition
            else:
                raise ValueError("nCX_x0 or ncx_x0_ratio must be specified")
            self._fd_cache = {}

    def calc_pressure_quantities_sc(self,n_e,x):
        """Calculate the pressure, alpha, and D_KBM on the psi_N_pres grid."""
        n_e = interp1d(x, n_e, kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        _pres = n_e * self.T_e_pres * 1.60218e-19 # Pa
        _alpha = (
            -(2 * np.gradient(self.V_plasma, self.psi_pres) / ((2 * np.pi) ** 2))
            * self.mu0
            * np.gradient(_pres, self.psi_pres)
            * np.sqrt(self.V_plasma / (2 * self.Rmajor * np.pi ** 2))
        )
        self._D_KBM = np.where(
            _alpha > self.alpha_crit,
            self.C_KBM*(_alpha-self.alpha_crit)*(self.c_s*self.rho_s**2)/self.a,
            0)

    def update_free_params(self, alpha_crit, C_KBM, De_chie_etg, nFC_x0,
                           nFC_threshold=None, nCX_threshold=None,
                           psi_N_inner_boundary=None,
                           ncx_x0_ratio=None,
                           ne_inner=None, dne_dx_inner=None, ne_x0=None,
                           clear_solution=False):
        """Update free parameters.

        Call this instead of constructing a new instance when the MHD equilibrium
        is fixed.

        Parameters
        ----------
        alpha_crit : float
        C_KBM : float
        De_chie_etg : float
        nFC_x0 : float
        nFC_threshold : float or None, optional
            Override the FC-neutral threshold used by find_inner_boundary.
            If omitted, keeps the value from construction.
        nCX_threshold : float or None, optional
            Override the CX-neutral threshold used by find_inner_boundary.
            If omitted, keeps the value from construction.
        psi_N_inner_boundary : float, optional
            If given, the inner boundary is placed at this psi_N value directly
            and the adaptive threshold-based logic is disabled (both thresholds
            forced to None for this run).
        ncx_x0_ratio : float, optional
            Override the ratio of nCX at the separatrix to nFC at the separatrix (used in coupled solver).
            If omitted, keeps the value from construction.
        ne_inner : float, optional
            Override the electron density at the inner boundary.
            If omitted, keeps the value from construction.
        dne_dx_inner : float, optional
            Override the derivative of the electron density at the inner boundary.
            If omitted, keeps the value from construction.
        ne_x0 : float, optional
            Override the electron density at the outer boundary.
            If omitted, keeps the value from construction.
        clear_solution : bool, default True
            If True, drop cached BVP solution attributes from a previous
            :meth:`solve` call.  Set False inside the Picard loop.
        """
        
        # Set/update free parameters alpha_crit, C_KBM, De_chie_etg, nFC_x0, psi_N_inner_boundary
        self.alpha_crit = alpha_crit
        self.C_KBM = C_KBM
        self.De_chie_etg = De_chie_etg
        self.nFC_x0 = nFC_x0
        if psi_N_inner_boundary is not None:
            self.psi_N_inner_boundary = float(psi_N_inner_boundary)
            self.nFC_threshold = None
            self.nCX_threshold = None
        else:
            if nFC_threshold is not None:
                self.nFC_threshold = nFC_threshold
            if nCX_threshold is not None:
                self.nCX_threshold = nCX_threshold
        if ncx_x0_ratio is not None:
            self.ncx_x0_ratio = ncx_x0_ratio
            self.nCX_x0 = self.ncx_x0_ratio * self.nFC_x0

        if ne_inner is not None:
            self.ne_inner = ne_inner
        if dne_dx_inner is not None:
            self.dne_dx_inner = dne_dx_inner
        if ne_x0 is not None:
            self.ne_x0 = ne_x0

        if clear_solution:
            for _attr in ('sol', 'sol_first', 'x_sol', 'ne_sol', 'dne_dx_sol',
                          'exp_term_arr', 'nFC_sol', 'integral_from_0'):
                if hasattr(self, _attr):
                    delattr(self, _attr)

        if self.equations_to_solve == 'coupled':
            self.invalidate_firedrake_cache()


    def inner_boundary_limits(self, outer_threshold=None, safety_margin=0.01,
                              x_res=100, psi_N_default=0.85):
        """Return (psi_N_inner_limit, psi_N_outer_limit) — the valid range for
        psi_N_inner_boundary given current parameters (D_ped, nFC_x0).

        The OUTER limit (closest to separatrix, largest psi_N) is determined by
        the nFC/nCX threshold logic in ``find_inner_boundary``.
        The INNER limit (deepest in core, smallest psi_N) is found by scanning
        from the separatrix inward until the p-file ``dn_e/dx`` is no longer
        steeper than ``safety_margin * min(dn_e/dx)``.

        Parameters
        ----------
        outer_threshold : float, optional
            Threshold applied to BOTH nFC_threshold and nCX_threshold when
            computing the outer limit.  Default: keep current thresholds.
        safety_margin : float, default ``0.01``
            Fraction of the most negative p-file ``dn_e/dx`` used as the inner
            slope cutoff (see ``psi_inner_safety_margin`` in scan notebooks).
        x_res : int
            Resolution passed to ``setup_solver_grids`` if it has not yet been
            called on this instance.
        psi_N_default : float, default ``0.85``
            Default psi_N value used when no valid inner boundary is found.
        """
        # Ensure form-factor / solver-grid setup has been done.
        if not hasattr(self, 'fFC'):
            self.form_factor(type='FC')
        if not hasattr(self, 'fCX'):
            self.form_factor(type='cx')
        if not hasattr(self, 'x_init') or not hasattr(self, 'S_i_pres'):
            self.setup_solver_grids(res=x_res)

        # ---- OUTER limit ----------------------------------------------------
        saved_thr_fc = self.nFC_threshold
        saved_thr_cx = self.nCX_threshold
        saved_psi    = self.psi_N_inner_boundary
        saved_x_in   = self.x_inner

        if outer_threshold is not None:
            self.nFC_threshold = float(outer_threshold)
            self.nCX_threshold = float(outer_threshold)

        # Start from the default so find_inner_boundary doesn't compound onto
        # a previously narrowed value.
        self.find_inner_boundary()
        psi_N_outer = float(self.psi_N_inner_boundary)

        # Restore
        self.nFC_threshold = saved_thr_fc
        self.nCX_threshold = saved_thr_cx
        self.psi_N_inner_boundary = saved_psi
        self.x_inner = saved_x_in

        # ---- INNER limit ----------------------------------------------------
        dne_dx = np.gradient(self.n_e_pres, self.x_init)
        slope_cut = float(safety_margin) * float(np.min(dne_dx))
        psi_N_inner = None
        for i in range(len(dne_dx)):
            j = len(dne_dx) - i - 1  # separatrix -> core
            if dne_dx[j] >= slope_cut:
                psi_N_inner = float(self.psi_N_pres[j])
                break
        if psi_N_inner is None:
            if self.verbose:
                print(
                    "No valid inner boundary found, defaulting to psi_N=0.85 "
                    "for inner boundary"
                )
            psi_N_inner = psi_N_default

        # Guarantee monotonic ordering (inner <= outer) even with edge cases.
        if psi_N_inner > psi_N_outer:
            psi_N_inner, psi_N_outer = psi_N_outer, psi_N_inner

        return psi_N_inner, psi_N_outer

    def find_boundary_points(self,eq):
        """Find the top/bottom/inboard/outboard extrema of the separatrix.

        Parameters
        ----------
        eq : dict
            Equilibrium dictionary returned by ``read_eqdsk``.  Must contain
            keys: ``nr``, ``nz``, ``rleft``, ``rdim``, ``zmid``, ``zdim``,
            ``psirz``, ``raxis``, ``zaxis``, ``psimag``, ``psibry``.
            Optionally ``rzout`` (boundary R,Z points).

        Returns
        -------
        result : dict
            ``'top'``      – ``(R, Z)`` of the upper boundary point
            ``'bottom'``   – ``(R, Z)`` of the lower boundary point
            ``'outboard'``  – ``(R, Z)`` of the outboard boundary point
            ``'inboard'``   – ``(R, Z)`` of the inboard boundary point
        """

        psi = eq['psirz']
        psibry = eq['psibry']

        if 'rzout' in eq and eq['rzout'] is not None and len(eq['rzout']) > 0:
            bdy = eq['rzout']
        else:
            nr = eq['nr']
            nz = eq['nz']
            r = np.linspace(eq['rleft'], eq['rleft'] + eq['rdim'], nr)
            z = np.linspace(eq['zmid'] - eq['zdim']/2, eq['zmid'] + eq['zdim']/2, nz)
            import matplotlib
            matplotlib.use('Agg')
            fig, ax = plt.subplots()
            cs = ax.contour(r, z, psi, levels=[psibry])
            bdy = np.vstack(cs.allsegs[0])
            plt.close(fig)
        itop = np.argmax(bdy[:, 1])
        ibot = np.argmin(bdy[:, 1])
        iout = np.argmax(bdy[:, 0])
        iin  = np.argmin(bdy[:, 0])
        top      = (bdy[itop, 0], bdy[itop, 1])
        bottom   = (bdy[ibot, 0], bdy[ibot, 1])
        outboard = (bdy[iout, 0], bdy[iout, 1])
        inboard  = (bdy[iin,  0], bdy[iin,  1])

        return {
            'top': top,
            'bottom': bottom,
            'outboard': outboard,
            'inboard': inboard,
        }

    def plasma_surface_area_and_volume(self):
        """Compute the plasma surface area and enclosed volume at each flux surface.

        For each psi in np.linspace(psimag, psibry, len(self.pres)), extracts
        the flux surface contour from the 2D psi grid, then computes the
        toroidal surface area (Pappus' theorem) and volume (exact
        piecewise-linear revolution integral) of the surface of revolution.

        Parameters
        ----------
        self : object
            instance of saarelma_connor class

        Sets
        ----
        self.S_plasma : ndarray, shape (n_psi,)
            Toroidal surface area (m^2) at each flux surface.
        self.V_plasma : ndarray, shape (n_psi,)
            Enclosed toroidal volume (m^3) at each flux surface.
        """

        n_psi = len(self.psi_pres)

        self.S_plasma = np.zeros(n_psi)
        self.V_plasma = np.zeros(n_psi)

        # Magnetic axis, used to pick the closed core contour (not an open
        # SOL / divertor-leg segment) at each psi level.  On double-null
        # (e.g. SPARC / ARC) equilibria a given psi level also produces open
        # contours that can be *longer* than the closed core surface, so the
        # previous "longest segment" heuristic silently returned garbage
        # volumes / areas (non-monotonic V, sign-flipping dV/dpsi).  Reuse the
        # same selector as fsa() / calc_gradr().
        R_axis = self.eq['raxis']
        Z_axis = self.eq['zaxis']

        # Extract the flux surface contour from the 2D psi grid
        fig, ax = plt.subplots()
        for i in range(n_psi):
            ax.cla()
            cs = ax.contour(self.rgrid, self.zgrid, self.psi_RZ,
                            levels=[self.psi_pres[i]])

            segs = cs.allsegs[0]
            seg = self._select_core_contour(segs, R_axis, Z_axis)
            if seg is None:
                # No closed contour around the axis at this psi level
                # (e.g. exactly at / beyond the separatrix); filled from
                # valid neighbours below.
                self.S_plasma[i] = np.nan
                self.V_plasma[i] = np.nan
                continue

            R = seg[:, 0]
            Z = seg[:, 1]

            # Close the contour so the integral spans a full 2*pi
            if not (np.isclose(R[0], R[-1]) and np.isclose(Z[0], Z[-1])):
                R = np.append(R, R[0])
                Z = np.append(Z, Z[0])

            dZ = np.diff(Z)
            dR = np.diff(R)
            R_i  = R[:-1]
            R_ip = R[1:]

            # Toroidal volume:  V = (pi/3) |sum (Z_{i+1}-Z_i)(R_i^2 + R_i*R_{i+1} + R_{i+1}^2)|
            # Exact integral of pi*R^2 dZ for piecewise-linear boundary segments
            self.V_plasma[i] = (np.pi / 3.0) * abs(np.sum(dZ * (R_i**2 + R_i * R_ip + R_ip**2))) # m^3, volume enclosed by the plasma per poloidal flux

            # Poloidal cross-section area: Shoelace formula - general to any polygon (Pappus' theorem)
            dl = np.sqrt(dR**2 + dZ**2)
            self.S_plasma[i] = 2.0 * np.pi * np.sum(0.5 * (R_i + R_ip) * dl) # m^2, total surface area of plasma

        plt.close(fig)

        # Fill NaN entries (surfaces where no closed core contour was found,
        # typically right at / beyond the separatrix on diverted equilibria)
        # by extrapolating from the nearest valid neighbours, mirroring
        # calc_gradr().
        for arr in (self.S_plasma, self.V_plasma):
            valid = np.isfinite(arr)
            if valid.any() and not valid.all():
                arr[:] = interp1d(self.psi_N_pres[valid], arr[valid],
                                  kind='linear', bounds_error=False,
                                  fill_value='extrapolate')(self.psi_N_pres)

    def calc_B(self,R_eval,Z_eval):
        """Calculate magnetic field at some point in the plasma
            Always use (rho,theta,var_zeta) coordinate convention as defined by https://crppwww.epfl.ch/~sauter/cocos/Sauter_COCOS_Tokamak_Coordinate_Conventions.pdf

           Note: sigma_Bb is not important for this model, we will always use sigma_Bp=1.
        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        R_eval : float or array
            radial location at which to evaluate the magnetic field
        Z_eval : float or array
            vertical location at which to evaluate the magnetic field
        """

        F = self.eq['fpol']
        psi_F = np.linspace(self.eq['psimag'], self.eq['psibry'], len(F))

        e_Bp = 1 if self.pol_norm else 0

        r = self.rgrid
        z = self.zgrid
        spl = RectBivariateSpline(z, r, self.eq['psirz'])

        R_eval_arr = np.atleast_1d(R_eval)
        Z_eval_arr = np.atleast_1d(Z_eval)

        psi = spl(Z_eval_arr, R_eval_arr, grid=False)
        dpsi_dR = spl(Z_eval_arr, R_eval_arr, dx=0, dy=1, grid=False) # specifying dx, dy specifies the derivative order in the respective direction
        dpsi_dZ = spl(Z_eval_arr, R_eval_arr, dx=1, dy=0, grid=False)
        F_interp = interp1d(psi_F, F, kind='linear', bounds_error=False, fill_value="extrapolate")
        B_R = (1 / ((2*np.pi)**e_Bp)) * dpsi_dZ / R_eval_arr # R component of the magnetic field
        B_Z = (1 / ((2*np.pi)**e_Bp)) * -dpsi_dR / R_eval_arr # Z component of the magnetic field
        B_phi = (F_interp(psi) / R_eval_arr) # T, toroidal magnetic field

        self.B = np.sqrt(B_R**2 + B_Z**2 + B_phi**2) # T, total magnetic field at each R_eval, Z_eval
        return self.B, [B_R, B_Z, B_phi]

    def mhd_load(self,mhd_loc,fp):
        """Load and calculate various MHD equilibrium parameters using method specified by mhd_eq_loc flag. 
        
        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        mhd_loc : string
            which method to use to load MHD parameters.
        fp : string
            filepath to file with MHD parameters.
             
        """

        if mhd_loc == 'eqdsk':
            from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk
            self.eq = read_eqdsk(fp)

            bdry = self.find_boundary_points(eq=self.eq)

            rmax_top = bdry['top'][0]
            rmax_bottom = bdry['bottom'][0]
            zmax_top = bdry['top'][1]
            zmax_bottom = bdry['bottom'][1]
            rmax_outboard = bdry['outboard'][0]
            rmax_inboard = bdry['inboard'][0]
            z_outboard = bdry['outboard'][1]
            # zmax_inboard = bdry['inboard'][1]

            # Geometric parameters
            self.Raxis = self.eq['raxis'] # m, location of magnetic axis relative to device rotational line of toroidal symmetry
            self.Rmajor = (rmax_outboard + rmax_inboard) / 2 # m
            self.a = (rmax_outboard - rmax_inboard) / 2 # minor radius # m
            delta_u = (self.Rmajor - rmax_top) / self.a
            delta_l = (self.Rmajor - rmax_bottom) / self.a
            self.delta = (delta_u + delta_l) / 2 # dimensionless, total triangularity
            self.kappa = (zmax_top - zmax_bottom) / (2*self.a) # dimensionless, elongation

            # Plasma parameters (skip the magnetic axis to avoid degenerate zero-area/volume flux surface)
            self.Ip = self.eq['ip'] / 1e6 # MA, Plasma current
            self.psi_pres = np.linspace(self.eq['psimag'], self.eq['psibry'], len(self.eq['pres']))[1:]
            self.psi_N_pres = (self.psi_pres - self.eq['psimag']) / (self.eq['psibry'] - self.eq['psimag'])

            # self.pres_gfile = self.eq['pres'][1:] # pressure is NOT an input to this model but using this for plotting - want to use pfile pressure instead

            # Grids
            self.rgrid = np.linspace(self.eq['rleft'],self.eq['rleft']+self.eq['rdim'],self.eq['nr']) # m, 1D R grid
            self.zgrid = np.linspace(self.eq['zmid']-self.eq['zdim']/2,self.eq['zmid']+self.eq['zdim']/2,self.eq['nz']) # m, 1D Z grid
            self.psi_RZ = self.eq['psirz'] # 2D poloidal flux array at each RZ grid point
            self.psi_RZ_N = (self.psi_RZ - self.eq['psimag']) / (self.eq['psibry'] - self.eq['psimag']) # normalized poloidal flux at each RZ grid point
            # self.rsep_mid = (((rmax_outboard - self.Raxis)**2) + ((z_outboard - self.eq['zaxis'])**2))**5 # separatrix radius at midplane

            self.plasma_surface_area_and_volume()

    def OMFITnc_load(self, filename):
        """
        Reads an OMFITnc file, averages T_e and n_e across the first dimension,
        and interpolates them onto a common, unified psi_N grid.

        Parameters:
            filename (str): Path to the NetCDF file.

        Returns:
            tuple: (psi_N_unified, Te_1d, ne_1d) as 1D numpy arrays.
            Te_1d in keV, ne_1d in m^-3.

        # Example usage:
        # psi_grid, Te_profile, ne_profile = self.OMFITnc_load('my_plasma_data.cdf')
        """
        from omfit_classes.omfit_nc import OMFITnc
        nc = OMFITnc(filename)

        def _nc_array(var_name):
            """OMFITnc variables are SortedDicts; numeric data lives under 'data'."""
            var = nc[var_name]
            if hasattr(var, 'keys') and 'data' in var:
                return np.asarray(var['data'], dtype=float)
            return np.asarray(var, dtype=float)

        def _nc_has(var_name):
            return var_name in nc

        def _reduce_time(arr):
            """Average over a leading time axis if present."""
            arr = np.asarray(arr, dtype=float)
            if arr.ndim > 1:
                return np.mean(arr, axis=0)
            return arr

        # 1. Extract and average the physics variables
        Te_raw = _reduce_time(_nc_array('T_e'))
        ne_raw = _reduce_time(_nc_array('n_e'))

        # Convert Te to keV if the file stores eV (IDA OMFITnc files use eV)
        te_unit = ''
        if hasattr(nc['T_e'], 'keys') and 'unit' in nc['T_e']:
            te_unit = str(nc['T_e']['unit']).lower()
        if te_unit in ('ev', 'electron volt', 'electron-volt'):
            Te_raw = Te_raw / 1e3
        elif te_unit in ('kev',):
            pass
        elif np.nanmax(np.abs(Te_raw)) > 100:
            # Heuristic: values >> 100 are almost certainly eV, not keV
            Te_raw = Te_raw / 1e3

        # 2. Determine psi_N grids (files may use psi_n / psi_N / separate Te,ne grids)
        if _nc_has('psi_N_Te') and _nc_has('psi_N_ne'):
            psi_Te = _reduce_time(_nc_array('psi_N_Te'))
            psi_ne = _reduce_time(_nc_array('psi_N_ne'))
        else:
            for psi_key in ('psi_n', 'psi_N', 'psiN', 'psin'):
                if _nc_has(psi_key):
                    break
            else:
                raise KeyError(
                    f"No psi_N grid found in {filename}. "
                    f"Available keys: {[k for k in nc.keys() if not str(k).startswith('__')]}"
                )
            psi_shared = _reduce_time(_nc_array(psi_key))
            psi_Te = psi_shared
            psi_ne = psi_shared

        # Keep only the closed-flux domain for interpolation onto [0, 1]
        def _clip_profile(psi, prof):
            psi = np.asarray(psi, dtype=float).ravel()
            prof = np.asarray(prof, dtype=float).ravel()
            order = np.argsort(psi)
            psi, prof = psi[order], prof[order]
            mask = (psi >= 0.0) & (psi <= 1.0)
            if np.count_nonzero(mask) < 2:
                mask = np.ones_like(psi, dtype=bool)
            # Drop duplicate psi points that break interp1d
            _, uniq = np.unique(psi[mask], return_index=True)
            uniq = np.sort(uniq)
            return psi[mask][uniq], prof[mask][uniq]

        psi_Te, Te_raw = _clip_profile(psi_Te, Te_raw)
        psi_ne, ne_raw = _clip_profile(psi_ne, ne_raw)
        num_points = max(len(psi_Te), len(psi_ne))

        # 3. Unified evaluation grid on [0, 1]
        psi_N_unified = np.linspace(0.0, 1.0, num_points)

        # 4. Interpolate both profiles onto the unified grid
        Te_interp_func = interp1d(
            psi_Te, Te_raw, kind='cubic', bounds_error=False, fill_value='extrapolate'
        )
        ne_interp_func = interp1d(
            psi_ne, ne_raw, kind='cubic', bounds_error=False, fill_value='extrapolate'
        )

        Te_unified = Te_interp_func(psi_N_unified)
        ne_unified = ne_interp_func(psi_N_unified)

        return psi_N_unified, Te_unified, ne_unified

    def kprof_load(self,kprof_loc='pfile',kprof_fp=None,T_prof=None,T_prof_psi_N=None,manual_profs=None):
        """Load kinetic equilibrium parameters using method specified by kprof_loc flag. 
        Parameters that will be loaded include: T_e, n_e
        Calculates: dn_e/dx|x=-inf, T_i
        
        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        kprof_loc : string
            which method to use to load kinetic parameters.
        kprof_fp : string
            filepath to kprof_loc-type file with kinetic parameters.
        T_prof : string
            temperature profile if using the EPEDNN model
        T_prof_psi_N : array
            psi_N values at which T_e is evaluated if using the EPEDNN model
        """

        # currently self.T_e = self.T_e_pfile, fix when cleaning up code
        if kprof_loc == 'pfile' and manual_profs is None: # use true pfile for n_e and T_e

            def read_pfile(path):
                data = {}
                key = ''
                with open(path) as f:
                    for line in f:
                        if '3 N Z A' in line:
                            break
                        if line.startswith('201'):
                            key = line.split()[2]
                            data[key] = np.array([])
                            data[f'{key}_psi'] = np.array([])
                        else:
                            psi, dat, _ = line.split()
                            psi = float(psi)
                            dat = float(dat)
                            data[key] = np.append(data[key], dat)
                            data[f'{key}_psi'] = np.append(data[f'{key}_psi'], psi)
                return data

            # Extract profiles
            pf = read_pfile(kprof_fp)

            # Store pfile information
            self.T_e_pfile = pf['te(KeV)'] # T_e values (keV) evaluated at psi_Te_eval
            self.T_e = self.T_e_pfile # keV
            self.psi_Te_eval_pfile = pf['te(KeV)_psi'] # psi_N values at which T_e is evaluated
            self.psi_Te_eval = self.psi_Te_eval_pfile # psi_N values at which T_e is evaluated

            self.n_e_pfile = pf['ne(10^20/m^3)'] * 1e20 # n_e values (10^20/m^3 -> m^-3) evaluated at psi_ne_eval
            self.psi_ne_eval = pf['ne(10^20/m^3)_psi'] # psi_N values at which n_e is evaluated

        elif kprof_loc == 'pfile' and manual_profs is not None: # use manual profiles for T_e but not n_e
            def read_pfile(path):
                data = {}
                key = ''
                with open(path) as f:
                    for line in f:
                        if '3 N Z A' in line:
                            break
                        if line.startswith('201'):
                            key = line.split()[2]
                            data[key] = np.array([])
                            data[f'{key}_psi'] = np.array([])
                        else:
                            psi, dat, _ = line.split()
                            psi = float(psi)
                            dat = float(dat)
                            data[key] = np.append(data[key], dat)
                            data[f'{key}_psi'] = np.append(data[f'{key}_psi'], psi)
                return data

            # Extract profiles
            pf = read_pfile(kprof_fp)

            # Store pfile information
            self.n_e_pfile = pf['ne(10^20/m^3)'] * 1e20 # n_e values (10^20/m^3 -> m^-3) evaluated at psi_ne_eval
            self.psi_ne_eval = pf['ne(10^20/m^3)_psi'] # psi_N values at which n_e is evaluated

            # Store manual profiles information
            self.T_e_pfile = manual_profs['Te'] # keV
            self.T_e = self.T_e_pfile # keV
            self.psi_Te_eval_pfile = manual_profs['psi_N_Te'] # psi_N values at which T_e is evaluated
            self.psi_Te_eval = self.psi_Te_eval_pfile # psi_N values at which T_e is evaluated

        elif kprof_loc == 'OMFITnc':
            psi_N_unified, self.T_e_pfile, self.n_e_pfile = self.OMFITnc_load(kprof_fp)
            self.T_e = self.T_e_pfile # keV
            self.psi_Te_eval_pfile = psi_N_unified
            self.psi_Te_eval = self.psi_Te_eval_pfile # psi_N values at which T_e is evaluated
            self.psi_ne_eval = psi_N_unified

        elif kprof_loc == 'manual rho grid':
            # Specify full T_e and corresponding psi_N profile
            # rho is treated as sqrt(psi_N): pass sqrt(psi_N) if your file's rho is toroidal.
            self.T_e_pfile = manual_profs['Te'] # T_e values (keV) evaluated at psi_Te_eval
            self.T_e = self.T_e_pfile # keV
            self.psi_Te_eval_pfile = self.psi_N_pres[-1] * (manual_profs['rho_Te']**2) # psi_N values at which T_e is evaluated
            self.psi_Te_eval = self.psi_Te_eval_pfile # psi_N values at which T_e is evaluated

            # Specify n_e'(psi_N=0.85) and n_e(psi_N=1.0) boundary conditions
            # n_e_pfile is only used for the cross_sections and the boundary conditions
            self.n_e_pfile = manual_profs['ne'] * 1e20 # n_e values (10^20/m^3 -> m^-3) evaluated at psi_ne_eval
            self.psi_ne_eval = self.psi_N_pres[-1] * (manual_profs['rho_ne']**2) # psi_N values at which n_e is evaluated

            # Optional ion profiles. If omitted, Ti = T_rat * Te and ni = ne.
            if 'Ti' in manual_profs and manual_profs['Ti'] is not None:
                rho_Ti = manual_profs.get('rho_Ti', manual_profs['rho_Te'])
                psi_Ti = self.psi_N_pres[-1] * (np.asarray(rho_Ti, dtype=float)**2)
                Ti_keV = np.asarray(manual_profs['Ti'], dtype=float)
                self.T_i_pfile = Ti_keV
                self.psi_Ti_eval_pfile = psi_Ti
                self.psi_Ti_eval = psi_Ti
                # Working Ti on the Te psi_N grid (velocities / FSA use Te grid).
                self.T_i = interp1d(
                    psi_Ti, Ti_keV, kind='linear',
                    bounds_error=False, fill_value='extrapolate'
                )(self.psi_Te_eval)
                self.T_i_from_profile = True
            if 'ni' in manual_profs and manual_profs['ni'] is not None:
                rho_ni = manual_profs.get('rho_ni', manual_profs['rho_ne'])
                self.n_i_pfile = np.asarray(manual_profs['ni'], dtype=float) * 1e20  # 10^20/m^3 -> m^-3
                self.psi_ni_eval = self.psi_N_pres[-1] * (np.asarray(rho_ni, dtype=float)**2)

        elif kprof_loc == 'manual EPEDNN loop':
            assert manual_profs is not None, 'T and n_e, n_CX, n_FC profiles must be provided if T_e_source is epednn'
            self.T_e_pfile = manual_profs['Te'] # keV
            self.T_e = self.T_e_pfile # keV
            self.psi_Te_eval_pfile = manual_profs['psi_N_Te'] # psi_N values at which T_e is evaluated
            self.psi_Te_eval = self.psi_Te_eval_pfile # psi_N values at which T_e is evaluated
            self.n_e_pfile = manual_profs['ne'] # m^-3; n_e values evaluated at psi_ne_eval
            self.psi_ne_eval = manual_profs['psi_N_n'] # psi_N values at which n_e is evaluated
            self.nCX_manual = manual_profs['nCX'] # m^-3; evaluated at manual_profs['psi_N_n']
            self.nFC_manual = manual_profs['nFC'] # m^-3; evaluated at manual_profs['psi_N_n']
            self.psi_N_n_manual = manual_profs['psi_N_n'] # psi_N values at which n_e, n_CX, n_FC are evaluated

        elif kprof_loc == 'manual profs':
            assert manual_profs is not None, 'T and n_e, profiles must be provided'
            self.T_e_pfile = manual_profs['Te'] # keV
            self.T_e = self.T_e_pfile # keV
            self.psi_Te_eval_pfile = manual_profs['psi_N_Te'] # psi_N values at which T_e is evaluated
            self.psi_Te_eval = self.psi_Te_eval_pfile # psi_N values at which T_e is evaluated
            self.n_e_pfile = manual_profs['ne'] # m^-3; n_e values evaluated at psi_ne_eval
            self.psi_ne_eval = manual_profs['psi_N_ne'] # psi_N values at which n_e is evaluated

        else:
            assert False, 'kprof_loc method not supported'

        self.T_e_K = self.T_e * 1e3 * 11604.52 # T_e values (K) evaluated at psi_Te_eval

        # Ion temperature: profile if provided above, else T_rat * Te.
        if not getattr(self, 'T_i_from_profile', False):
            if self.T_rat_flag:
                self.T_i = self.T_e * self.T_rat # keV
                self.T_i_pfile = self.T_e_pfile * self.T_rat
                self.psi_Ti_eval = self.psi_Te_eval
                self.psi_Ti_eval_pfile = self.psi_Te_eval_pfile
            else:
                raise NotImplementedError("T_rat_flag must be True when Ti is not provided")
        self.T_i_K = self.T_i * 1e3 * 11604.52 # K

        # Ion density: profile if provided above, else ni = ne (quasineutrality).
        if not hasattr(self, 'n_i_pfile'):
            self.n_i_pfile = np.asarray(self.n_e_pfile, dtype=float).copy()
            self.psi_ni_eval = self.psi_ne_eval


    def cross_section_rates(self,species='D',x_method='radas'):
        """Calculate ionization and charge-exchange rate coefficients.

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        species : string
            species of ions: 'D' or 'D-T' (D-T averages D and T RADAS rates)
        x_method : string
            'radas' (ADF11 from local RADAS dumps) or 'adas' (bundled ADAS files)
        """

        if x_method == 'adas':
            if species == 'D':
                
                # charge-exchange cross-section
                '''sigma_cx_perE = np.array([3.81*10**(-18), 3.85*10**(-18), 3.44*10**(-18), 2.71*10**(-18), 1.74*10**(-18), 8.10*10**(-20), 9.56*10**(-22), 1.46*10**(-23)]) # m^2
                E = np.array([3.23*10**(-16), 9.68*10**(-16), 3.23*10**(-15), 6.45*10**(-15), 9.68*10**(-15), 3.23*10**(-14), 9.68*10**(-14), 2.26*10**(-13)]) # J
                sigma_cx_interp = interp1d(E, sigma_cx_perE, kind='linear',fill_value='extrapolate',bounds_error=False)
                self.sigma_cx = sigma_cx_interp(0.5 * (self.M_i*self.M_eff) * self.V_cx**2) # m^2, charge-exchange cross-section'''

                # ionization rate coefficient profile: scd_adas(n_e, T_e[eV]) at each psi_Te_eval point.
                # also including CX rate coefficients from ADAS ADF11 https://open.adas.ac.uk/detail/adf11/ccd96/ccd96_d.dat
                # n_e is given on psi_ne_eval, so interpolate it onto psi_Te_eval first.
                n_e_at_Te = interp1d(self.psi_ne_eval, self.n_e_pfile, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')(self.psi_Te_eval)
                n_e_input = np.mean(n_e_at_Te)
                T_e_eV = self.T_e * 1e3 # keV -> eV
                self.S_i = np.array([
                    # scd_adas(n_e_at_Te[i], T_e_eV[i]) for i in range(len(self.psi_Te_eval))
                    scd_adas(n_e_input, T_e_eV[i]) for i in range(len(self.psi_Te_eval))
                ]) # m^3/s, on psi_Te_eval
                self.S_cx = scx_adas(np.ones_like(T_e_eV) * n_e_input, T_e_eV)

            else:
                assert False, 'species not supported for x_method=adas (use D)'
        elif x_method == 'radas':
            # RADAS ADF11 SCD (effective ionisation) and CCD (CX cross-coupling).
            # Same evaluation pattern as adas: mean ne, rate vs Te profile.
            # Tables are 2D (ne, Te); ne is fixed at the profile mean.
            n_e_at_Te = interp1d(self.psi_ne_eval, self.n_e_pfile, kind='linear',
                                bounds_error=False, fill_value='extrapolate')(self.psi_Te_eval)
            ne_arr = n_e_at_Te
            # n_e_input = float(np.mean(n_e_at_Te))
            # ne_arr = np.ones_like(T_e_eV) * n_e_input
            T_e_eV = self.T_e * 1e3  # keV -> eV

            if species == 'D':
                self.S_i = np.asarray(scd_radas(ne_arr, T_e_eV, isotope='D'), dtype=float)
                self.S_cx = np.asarray(scx_radas(ne_arr, T_e_eV, isotope='D'), dtype=float)
            elif species == 'D-T':
                S_i_D = np.asarray(scd_radas(ne_arr, T_e_eV, isotope='D'), dtype=float)
                S_i_T = np.asarray(scd_radas(ne_arr, T_e_eV, isotope='T'), dtype=float)
                S_cx_D = np.asarray(scx_radas(ne_arr, T_e_eV, isotope='D'), dtype=float)
                S_cx_T = np.asarray(scx_radas(ne_arr, T_e_eV, isotope='T'), dtype=float)
                self.S_i = 0.5 * (S_i_D + S_i_T)
                self.S_cx = 0.5 * (S_cx_D + S_cx_T)
            else:
                assert False, "species must be 'D' or 'D-T' for x_method=radas"
                            
        else:
            assert False, f"x_method must be 'radas' or 'adas', got {x_method!r}"


    def form_factor(self,x,type = 'ex'):
        """Calculate the form factor for FC or charge-exchange cases
        Currently just sets to 1, but can be updated to use a more sophisticated to account for poloidal asymmetries in the FC and CX neutral profiles.

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        x : array
            radial grid to evaluate the form factor on
        type : string
            type of form factor to calculate, supporting: FC, cx
        """
        assert type != 'FC' or type != 'cx', 'form factor must be for FC or cx'

        # grad(r) and nFC or nCX are needed

        if type == 'FC':
            self.fFC = np.ones_like(x)
        elif type == 'cx':
            self.fCX = np.ones_like(x)


    def setup_solver_grids(self,res = 100):
        """Setup the grids for the solver and calculates the flux surface-averaged |grad(r)| and |grad(r)|^2
        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        res : int, optional
            number of points to use in the radial grid if using the first method of defining the radial grid

        Returns
        -------
        self.x_prev : ndarray, shape (n_psi,)
            Radial grid (shifted so zero is at the separatrix) only at midplane, defined on the psi_N_pres grid.
        self.gradr_fsa : ndarray, shape (n_psi,)
            Flux-surface-averaged |grad(r)| at each psi_N_pres surface.
        self.gradr2_fsa : ndarray, shape (n_psi,)
            Flux-surface-averaged |grad(r)|^2 at each psi_N_pres surface.
        self.S_i_pres : ndarray, shape (n_psi,)
            Ionization cross-section at each psi_N_pres surface.
        self.S_cx_pres : ndarray, shape (n_psi,)
            Charge-exchange cross-section at each psi_N_pres surface.
        self.V_cx_pres : ndarray, shape (n_psi,)
            Volume of the charge-exchange neutral at each psi_N_pres surface.
        """

        # one method of defining the radial grid, requires uncommenting the rsep_mid definition in the mhd_load function
        # self.rmid = np.linspace(0, self.rsep_mid, res) # m, radial grid (shifted so zero is at the magnetic axis)
        # self.xmid = self.rmid - self.rsep_mid # m, radial grid (shifted so zero is at the separatrix)

        # another method of defining the radial grid
        # self.r_psi is the outboard midplane minor radius for each flux surface for the psi_N_pres grid
        self.x_init = self.r_psi - self.r_psi[-1] # m, radial grid (shifted so zero is at the separatrix) only at midplane, defined on the psi_N_pres grid
        self.x_prev = self.x_init.copy() # m, radial grid (shifted so zero is at the separatrix) only at midplane, defined on the psi_N_pres grid

        # interpolate quantities on Te grid to psi_N_pres grid
        self.S_i_pres = interp1d(self.psi_Te_eval, self.S_i, kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        self.S_cx_pres = interp1d(self.psi_Te_eval, self.S_cx, kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        self.V_cx_pres = interp1d(self.psi_Te_eval, np.abs(self.V_cx), kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)

        # note all 1D quantities are now defined on the psi_N_pres grid, which is the same as the x_prev grid

        self.x_inner = interp1d(self.psi_N_pres, self.x_prev, kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi_N_inner_boundary)

        # Equilibrium-only coefficient for the Connor-Hastie alpha when
        # written in the dp/dx form (the chain rule absorbs the dx/dpsi
        # Jacobian).  Precomputed on the parent (sorted) self.x_init grid
        # so that downstream Firedrake-DOF grids can use np.interp without
        # ever needing np.gradient on a possibly unsorted DOF array.
        # See calc_pressure_quantities and App. A.7 in the writeup.
        _dxdpsi_xinit = np.gradient(self.x_init, self.psi_pres)
        _dVdpsi_xinit = np.gradient(self.V_plasma, self.psi_pres)
        self._alpha_nodp_xinit = (
            -(2 * _dVdpsi_xinit / ((2 * np.pi) ** 2))
            * self.mu0
            * np.sqrt(self.V_plasma / (2 * self.Rmajor * np.pi ** 2))
            * _dxdpsi_xinit
        ) # alpha = self._alpha_nodp_xinit * dp/dx (with dp/dx evaluated on the same grid)

    def find_inner_boundary(self):
        """Adaptively locate the inner boundary by finding where the neutral
        densities (FC and/or CX) fall below user-supplied thresholds.

        Uses the p-file n_e profile as a proxy for the pre-solve electron
        density to estimate nFC(x) via its exponential attenuation integral
        (Eq. 11) and nCX(x) via the algebraic closure (Eq. 12).  The inner
        boundary is placed at the outermost x (closest to the separatrix)
        where *both* active thresholds are satisfied simultaneously.

        If neither threshold is set (both are None), the method returns
        immediately without changing ``self.psi_N_inner_boundary`` or
        ``self.x_inner``.

        Parameters used from self
        -------------------------
        self.nFC_threshold : float or None
            nFC/nFC(x=0) must drop below this fraction.  None disables.
        self.nCX_threshold : float or None
            nCX/nCX_peak must drop below this fraction.  None disables.
        self.n_e_pres, self.x_init : array
            p-file density and radial grid.
        self.S_i_pres, self.S_cx_pres : array
            Ionization and CX rate coefficients on psi_N_pres grid.
        self.D_ped, self.gradr2_fsa, self.V_cx_pres : array
            Diffusion and geometry arrays on psi_N_pres grid.
        self.V_FC, self.fFC, self.fCX : float
            FC neutral speed and form factors.
        self.nFC_x0 : float
            FC neutral density at the separatrix (boundary condition).

        Updates
        -------
        self.psi_N_inner_boundary : float
            Updated to the adaptively found inner boundary psi_N.
        self.x_inner : float
            Updated to the corresponding physical x coordinate (m).
        """
        if self.nFC_threshold is None and self.nCX_threshold is None:
            self.x_inner = interp1d(self.psi_N_pres, self.x_init, kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi_N_inner_boundary)
            return  # use fixed psi_N_inner_boundary and corresponding x_inner

        ne   = self.n_e_pres
        Si   = self.S_i_pres
        Scx  = self.S_cx_pres
        Dped = self.D_NEO + self._D_KBM + (self.D_ETG_x / ne)
        gr2  = self.gradr2_fsa
        Vcx  = self.V_cx_pres      # array, local thermal CX speed
        Vfc  = abs(self.V_FC)      # scalar FC speed
        fFC  = self.fFC            # form factor (= 1 currently)
        fCX  = self.fCX            # form factor (= 1 currently)
        x    = self.x_init         # physical x, separatrix = 0, inward < 0

        # ------------------------------------------------------------------ #
        # nFC estimate: integrate from separatrix inward (descending x)       #
        # nFC(x) / nFC_x0 = exp( ∫_0^x  ne*(Si+Scx)/(fFC*Vfc)  dx' )        #
        # ------------------------------------------------------------------ #
        order_desc = np.argsort(x)[::-1]   # index order: separatrix → core
        x_desc     = x[order_desc]
        integrand  = (ne * (Si + Scx)) / (fFC * Vfc)
        cumint     = cumulative_trapezoid(integrand[order_desc], x_desc, initial=0.0)
        nFC_ratio_desc = np.exp(cumint)    # decays < 1 going inward

        # map back to original (psi_N ascending) order
        nFC_ratio = np.empty_like(nFC_ratio_desc)
        nFC_ratio[order_desc] = nFC_ratio_desc

        # ------------------------------------------------------------------ #
        # nCX estimate: algebraic closure (Eq. 12)                            #
        # nCX = -(gr2*Dped/(Vcx*fCX))*(dne/dx - dne_dx_inner)               #
        #        - (Vfc*fFC/(Vcx*fCX))*((Si+Scx/2)/(Si+Scx))*nFC            #
        # ------------------------------------------------------------------ #
        dne_dx = np.gradient(ne, x)
        # Use the innermost p-file gradient as the dne_dx_neginf proxy
        dne_dx_inner_est = dne_dx[np.argmin(x)]

        f_arr     = gr2 * Dped
        flux_term = -(f_arr / (Vcx * fCX)) * (dne_dx - dne_dx_inner_est)
        fc_term   = -(Vfc * fFC / (Vcx * fCX)) * ((Si + Scx / 2) / (Si + Scx)) * (self.nFC_x0 * nFC_ratio)
        nCX_est   = flux_term + fc_term
        nCX_est   = np.maximum(nCX_est, 0.0)   # physical lower bound

        nCX_peak  = np.max(nCX_est)
        nCX_ratio = nCX_est / nCX_peak if nCX_peak > 0 else np.zeros_like(nCX_est)

        # ------------------------------------------------------------------ #
        # Find the outermost x (closest to separatrix) where both active      #
        # thresholds are satisfied.                                            #
        # Work in ascending-x order (core first, separatrix last).            #
        # ------------------------------------------------------------------ #
        asc      = np.argsort(x)
        x_asc    = x[asc]
        psi_asc  = self.psi_N_pres[asc]

        # build combined mask: only consider thresholds the user activated
        mask = np.ones(len(x), dtype=bool)
        if self.nFC_threshold is not None:
            mask &= nFC_ratio[asc] < self.nFC_threshold
        if self.nCX_threshold is not None:
            mask &= nCX_ratio[asc] < self.nCX_threshold

        crossing = np.where(mask)[0]

        if len(crossing) == 0:
            import warnings
            warnings.warn(
                f"Neutral densities never fall below the requested thresholds "
                f"(nFC_threshold={self.nFC_threshold}, nCX_threshold={self.nCX_threshold}) "
                f"across the available p-file domain.  Keeping the fixed inner boundary "
                f"at psi_N = {self.psi_N_inner_boundary:.3f}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        # The outermost (highest x, most separatrix-side) crossing point that
        # still satisfies both thresholds. We scan inward from the separatrix
        # (highest x in ascending order = last element) and take the first hit.
        idx       = crossing[-1]
        x_new     = float(x_asc[idx])
        psi_new   = float(psi_asc[idx])

        self.psi_N_inner_boundary = psi_new
        print(f"psi_N_inner_boundary: {self.psi_N_inner_boundary:.4f}")
        self.x_inner = x_new
        print(f"x_inner: {self.x_inner:.4f} m")

    @staticmethod
    def _select_core_contour(segs, R_axis, Z_axis, closure_tol=0.05):
        """Pick the closed flux-surface contour that encloses the magnetic
        axis from a list of matplotlib contour segments.

        Previously the *longest* segment was used, but on double-null
        (e.g. SPARC) or diverted equilibria a given psi level also produces
        open SOL / divertor-leg / private-flux contours which can be longer
        than the closed core surface.  Integrating over one of those open,
        theta-folded curves silently corrupts the flux-surface average
        (it even produced *negative* <|grad r|^2>).

        Parameters
        ----------
        segs : list of (N, 2) ndarray
            Contour segments (R, Z) for one psi level.
        R_axis, Z_axis : float
            Magnetic axis position (m).
        closure_tol : float
            Segment counts as closed if the gap between its endpoints is
            below ``closure_tol`` times its perimeter.

        Returns
        -------
        seg : ndarray or None
            The longest closed segment enclosing the axis, or None if no
            segment qualifies (caller should treat the surface as invalid,
            e.g. NaN + neighbour fill).
        """
        candidates = []
        for s in segs:
            if len(s) < 4:
                continue
            perim = np.hypot(np.diff(s[:, 0]), np.diff(s[:, 1])).sum()
            if perim <= 0.0:
                continue
            gap = np.hypot(s[0, 0] - s[-1, 0], s[0, 1] - s[-1, 1])
            if gap > closure_tol * perim:
                continue  # open contour (SOL / divertor leg)
            if not _MplPath(s).contains_point((R_axis, Z_axis)):
                continue  # closed but not around the axis (island etc.)
            candidates.append(s)
        if not candidates:
            return None
        return max(candidates, key=lambda s: len(s))

    @staticmethod
    def _sort_dedup_close_theta(theta_c, R_c, Z_c, min_dtheta=1e-12):
        """Sort contour points by poloidal angle, drop near-duplicate
        angles, and close the contour over a full 2*pi.

        Near-duplicate angles (e.g. the coincident first/last vertices of
        a closed matplotlib contour, or point clusters near an X-point)
        create ~1e-16-wide intervals; Simpson's nonuniform weights blow up
        on the huge interval-length ratios and amplify round-off into
        O(1e-4) errors in the flux-surface average.

        Parameters
        ----------
        theta_c, R_c, Z_c : ndarray
            Poloidal angle and contour coordinates (unsorted).
        min_dtheta : float
            Minimum allowed angular spacing between consecutive points.

        Returns
        -------
        theta_c, R_c, Z_c : ndarray
            Sorted, deduplicated arrays with the closing point
            (theta[0] + 2*pi, R[0], Z[0]) appended.
        """
        idx = np.argsort(theta_c)
        theta_c, R_c, Z_c = theta_c[idx], R_c[idx], Z_c[idx]

        keep = np.concatenate(([True], np.diff(theta_c) > min_dtheta))
        theta_c, R_c, Z_c = theta_c[keep], R_c[keep], Z_c[keep]

        # Close the contour so the integral spans a full 2*pi; drop the
        # last point first if it would duplicate the closing point.
        if theta_c[0] + 2 * np.pi - theta_c[-1] <= min_dtheta:
            theta_c, R_c, Z_c = theta_c[:-1], R_c[:-1], Z_c[:-1]
        theta_c = np.append(theta_c, theta_c[0] + 2 * np.pi)
        R_c = np.append(R_c, R_c[0])
        Z_c = np.append(Z_c, Z_c[0])
        return theta_c, R_c, Z_c

    def calc_gradr(self):
        """Compute <|grad(r)|> at each flux surface.

        r is defined as the outboard-midplane minor radius for each flux
        surface, making it a proper flux-surface label (one value per surface).
        By the chain rule:
            |grad(r)| = |dr/dpsi| * |grad(psi)|
        This varies poloidally because |grad(psi)| is larger where flux
        surfaces are compressed (inboard side) and smaller where they are
        spread apart (outboard side).

        The flux surface average is:
            <|grad(r)|> = ∮ R² |grad(r)| dθ / ∮ R² dθ

        Sets
        ----
        self.r_psi : ndarray, shape (n_psi,)
            Outboard midplane minor radius for each flux surface (m).
        self.gradr_c : ndarray, shape (n_psi,)
            |grad(r)| at each contour point on each flux surface.
        self.gradr_fsa : ndarray, shape (n_psi,)
            Flux-surface-averaged |grad(r)| at each psi_N_pres surface.
        self.gradr2_fsa : ndarray, shape (n_psi,)
            Flux-surface-averaged |grad(r)|^2 at each psi_N_pres surface.
        """
        R_axis = self.eq['raxis']
        Z_axis = self.eq['zaxis']

        psi_spl = RectBivariateSpline(self.zgrid, self.rgrid, self.psi_RZ)
        n_psi = len(self.psi_N_pres)

        # r(psi): find outboard midplane crossing for each flux surface
        R_out = np.linspace(R_axis, self.rgrid[-1], 500)
        Z_mid = np.full_like(R_out, Z_axis)
        psi_mid = psi_spl(Z_mid, R_out, grid=False)
        sort_idx = np.argsort(psi_mid)
        psi_to_R = interp1d(psi_mid[sort_idx], R_out[sort_idx], kind='linear',
                            bounds_error=False, fill_value=np.nan)

        self.r_psi = np.zeros(n_psi)
        for i, psi_val in enumerate(self.psi_pres):
            self.r_psi[i] = psi_to_R(psi_val) - R_axis

        dr_dpsi = np.gradient(self.r_psi, self.psi_N_pres) # (m) / (dimensionless), change in r_midplane(psi_N) over psi_N

        self.gradr_fsa = np.zeros(n_psi)
        self.gradr2_fsa = np.zeros(n_psi)
        fig, ax = plt.subplots()
        for i, psi_val in enumerate(self.psi_pres):
            ax.cla()
            cs = ax.contour(self.rgrid, self.zgrid, self.psi_RZ,
                            levels=[psi_val])
            segs = cs.allsegs[0]
            seg = self._select_core_contour(segs, R_axis, Z_axis)
            if seg is None:
                # No closed contour around the axis at this psi level
                # (e.g. exactly at / beyond the separatrix); filled from
                # valid neighbours below.
                self.gradr_fsa[i] = np.nan
                self.gradr2_fsa[i] = np.nan
                continue
            R_c, Z_c = seg[:, 0], seg[:, 1]

            theta_c = np.arctan2(Z_c - Z_axis, R_c - R_axis) # theta at all points on contour
            theta_c, R_c, Z_c = self._sort_dedup_close_theta(theta_c, R_c, Z_c)

            # |grad(psi)| at each contour point from the equilibrium spline
            dpsi_dR = psi_spl(Z_c, R_c, dx=0, dy=1, grid=False) # value at each point on the contour
            dpsi_dZ = psi_spl(Z_c, R_c, dx=1, dy=0, grid=False) # value at each point on the contour
            grad_psi_mag = np.sqrt(dpsi_dR**2 + dpsi_dZ**2) # value at each point on the contour

            # |grad(r)| = |dr/dpsi| * |grad(psi)| at each contour point on each flux surface i
            gradr_c = np.abs(dr_dpsi[i]) * grad_psi_mag

            den = simpson(R_c**2, theta_c)
            self.gradr_fsa[i] = simpson(R_c**2 * gradr_c, theta_c) / den
            self.gradr2_fsa[i] = simpson(R_c**2 * gradr_c**2, theta_c) / den
        plt.close(fig)

        # Fill NaN/zero entries (near-axis or separatrix edge cases) by extrapolating from the nearest valid neighbours.
        for arr in (self.r_psi, self.gradr_fsa, self.gradr2_fsa):
            valid = np.isfinite(arr) & (arr != 0)
            if valid.any() and not valid.all():
                arr[:] = interp1d(self.psi_N_pres[valid], arr[valid],
                                  kind='linear', bounds_error=False,
                                  fill_value='extrapolate')(self.psi_N_pres)

    def non_dimensionalize(self, x, y, L=None, n0=None):
        """Non-dimensionalize the BVP variables.

        Introduces xi = x / L and N = ne / n0 so that both the independent
        and dependent variables are O(1).

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        x : ndarray
            Physical radial grid (m).
        y : ndarray, shape (2, n_points)
            [ne, dne/dx] guess in physical units.
        L : float, optional
            Length scale (m).  Default ``|x_inner|``.
        n0 : float, optional
            Density scale (m^-3).  Default ``ne_x0``.

        Sets
        ----
        self._L, self._n0 : float
            Stored scales for later de-normalization.
        self.xi : ndarray
            Normalized grid (dimensionless, -1 to 0).
        self.N_guess : ndarray, shape (2, n_points)
            [N, dN/dxi] initial guess in normalized units.
        self.dNdxi_neginf : float
            Normalized Neumann BC value.
        """
        if L is None:
            L = abs(self.x_inner)
        if n0 is None:
            n0 = self.ne_x0
        self._L = L
        self._n0 = n0

        self.xi = x / L
        N = y[0] / n0
        dNdxi = y[1] * (L / n0)   # dne/dx = (n0/L)*dN/dxi  =>  dN/dxi = (L/n0)*dne/dx
        self.N_guess = np.vstack([N, dNdxi])

        self.dNdxi_neginf = self.dne_dx_neginf * (L / n0)


    def first_step(self,resolution=200):
        """Solve Equation (16) in S. Saarelma et al 2023 Nucl. Fusion 63 052002
        This is the simplified BVP (no charge-exchange neutrals) used as the
        initial guess for the full iterative solve of Equation (15).

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        resolution : int, optional
            number of points to use in the radial grid solve_bvp() call

        Boundary conditions:
            dn_e/dx = dne_dx_neginf  at x_inner (psi_N adaptively set by
                                      find_inner_boundary, defaulting to
                                      psi_N_inner_boundary = 0.85)
            n_e     = ne_x0          at x = 0   (separatrix)

        Uses coefficient interpolators set up by solve() and stored as
        self._D_ped_x, self._gradr2_x, self._Si_x, self._Scx_x,
        self._P_x, self._dPdx_x, self._x_inner.

        Sets
        ----
        self.ne_first_sol : BVP solution object from solve_bvp.
        """

        # FIRST set boundary conditions
        # Adaptively locate the inner boundary where both neutral species
        # have attenuated below the requested thresholds (no-op when both
        # thresholds are None, in which case psi_N_inner_boundary is unchanged).
        if self.psi_N_inner_boundary is None:
            self.find_inner_boundary()
        else:
            self.x_inner = self.psi_N_inner_boundary

        # Calculate boundary condition from profiles at psi_N = 0.85
        # self.n_e_pres = interp1d(self.psi_ne_eval, self.n_e, kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        self.dne_dx = np.gradient(self.n_e_pres, self.x_init) # (particles/m^3) / m, electron density gradient
        dne_dx_interp = interp1d(self.psi_N_pres, self.dne_dx, kind='linear', bounds_error=False, fill_value='extrapolate')
        self.dne_dx_neginf = dne_dx_interp(self.psi_N_inner_boundary)

        # Physical sanity check: the inner-boundary gradient must be negative.
        # A zero or positive value means the chosen boundary is inside the flat
        # core or on a density inversion, which violates the model's assumption
        # that the pedestal gradient is steeper everywhere inside the domain.
        if self.dne_dx_neginf >= 0:
            raise ValueError(
                f"Inner boundary condition dne/dx = {self.dne_dx_neginf:.3e} m^-4 "
                f"at psi_N = {self.psi_N_inner_boundary:.4f} is zero or positive. "
                f"The Neumann BC must be strictly negative (density decreasing "
                f"outward). The inner boundary may be sitting inside the flat "
                f"core or on a density inversion. Consider lowering "
                f"nFC_threshold / nCX_threshold, or increasing "
                f"psi_N_inner_boundary to move the boundary further outward."
            )

        # Set up ODE
        f0_arr = self.gradr2_fsa * (self.D_NEO + self._D_KBM)
        f1_arr = self.gradr2_fsa * self.D_ETG_x

        df0_arr = np.gradient(f0_arr, self.x_prev)
        df1_arr = np.gradient(f1_arr, self.x_prev)

        f0_x = interp1d(self.x_prev, f0_arr, kind='linear', bounds_error=False, fill_value='extrapolate')
        df0_dx = interp1d(self.x_prev, df0_arr, kind='linear', bounds_error=False, fill_value='extrapolate')
        f1_x = interp1d(self.x_prev, f1_arr, kind='linear', bounds_error=False, fill_value='extrapolate')
        df1_dx = interp1d(self.x_prev, df1_arr, kind='linear', bounds_error=False, fill_value='extrapolate')

        # x-based interpolators for the ionization and CX rate coefficient profiles
        S_i_x = interp1d(self.x_prev, self.S_i_pres, kind='linear', bounds_error=False, fill_value='extrapolate')
        S_cx_x = interp1d(self.x_prev, self.S_cx_pres, kind='linear', bounds_error=False, fill_value='extrapolate')

        # Build physical guess, then non-dimensionalize to help solver converge
        x_grid = np.linspace(self.x_inner, 0, resolution)
        ne_guess = interp1d(self.x_init, self.n_e_pres, kind='linear',
                            bounds_error=False, fill_value='extrapolate')(x_grid)
        dne_guess = np.gradient(ne_guess, x_grid)
        dne_guess[0] = self.dne_dx_neginf
        Y_guess = np.vstack([ne_guess, dne_guess])

        n0_inner = interp1d(self.x_prev, self.n_e_pres, kind='linear', bounds_error=False, fill_value='extrapolate')(x_inner) 
        self.non_dimensionalize(x=x_grid, y=Y_guess, n0=n0_inner)
        L = self._L
        n0 = self._n0

        def ode(xi, Y):
            N, dNdxi = Y
            x = L * xi # map back to physical coordinate for interpolators
            
            f0 = f0_x(x)
            df0dx = df0_dx(x)
            f1 = f1_x(x)
            df1dx = df1_dx(x)
            
            S_i = S_i_x(x)
            S_cx = S_cx_x(x)

            # Prevent division by zero mathematically if N drops near 0 during BVP iterations
            N_safe = np.maximum(N, 1e-6)
            if 1e-6 in N_safe:
                import warnings
                warnings.warn(
                    "N_safe is very close to zero and got clipped to 1e-6",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # Total flux multiplier
            F = f0 + (f1 / (n0 * N_safe))

            # Non-dimensional coefficients
            C_A = (n0 * L * (S_i + S_cx)) / (abs(self.V_FC) * self.fFC * F)
            C_B = C_A * self.dNdxi_neginf
            C_K = (L / F) * (df0dx + (df1dx / (n0 * N_safe)))
            C_N = f1 / (n0 * (N_safe**2) * F)

            if self.verbose:
                print('iteration ODE eval')
                print(f"C_A : {np.max(C_A):.3e}, min: {np.min(C_A):.3e}")
                print(f"C_B : {np.max(C_B):.3e}, min: {np.min(C_B):.3e}")
                print(f"C_K : {np.max(C_K):.3e}, min: {np.min(C_K):.3e}")
                print(f"C_N : {np.max(C_N):.3e}, min: {np.min(C_N):.3e}")
                
            d2Ndxi2 = C_A * N * dNdxi - C_B * N - C_K * dNdxi + C_N * (dNdxi**2)
            return np.vstack([dNdxi, d2Ndxi2])

        def bc(Ya, Yb):
            return np.array([
                Ya[1] - self.dNdxi_neginf,  # Neumann BC at xi = -1
                Yb[0] - self.ne_x0/n0,      # Dirichlet BC: N = ne/n0 at xi = x = 0 (separatrix)
            ])

        sol = solve_bvp(ode, bc, self.xi, self.N_guess, max_nodes=5000, verbose=self.bvp_verbose)
        if not sol.success:
            raise RuntimeError(f"first_step BVP failed: {sol.message}")

        # De-normalize back to physical units for downstream use
        sol.x = L * sol.x
        sol.y[0] = n0 * sol.y[0]
        sol.y[1] = (n0 / L) * sol.y[1]
        self.sol = sol

    def compute_post_solve_SC_neutrals(self):
        """Franck--Condon and charge-exchange densities after ``solve_simplified()``.

        Uses the same closures as ``find_inner_boundary`` (Eqs.~(11)--(12) in
        Saarelma et al., 2023): exponential attenuation of FC neutrals
        from the separatrix and the algebraic CX relation.

        Returns
        -------
        nFC, nCX : ndarray
            Neutral densities (m^-3) on ``x``.
        """
        x = self.x_sol
        dne_dx = self.dne_dx_sol

        ne_x = interp1d(self.x_sol, self.ne_sol, kind='linear', bounds_error=False, fill_value='extrapolate')

        D_ped = self.D_NEO + self._D_KBM + (self.D_ETG_x / ne_x(self.x_prev))

        Si = interp1d(self.x_init, self.S_i_pres, kind='linear',
                      bounds_error=False, fill_value='extrapolate')(x)
        Scx = interp1d(self.x_init, self.S_cx_pres, kind='linear',
                       bounds_error=False, fill_value='extrapolate')(x)
        gr2 = interp1d(self.x_init, self.gradr2_fsa, kind='linear',
                       bounds_error=False, fill_value='extrapolate')(x)
        Dped = interp1d(self.x_prev, D_ped, kind='linear',
                        bounds_error=False, fill_value='extrapolate')(x)
        Vcx = interp1d(self.x_init, self.V_cx_pres, kind='linear',
                       bounds_error=False, fill_value='extrapolate')(x)

        Vfc = abs(self.V_FC)
        fFC = self.fFC
        fCX = self.fCX

        integrand = self.ne_sol * (Si + Scx) / (fFC * Vfc)
        order_desc = np.argsort(x)[::-1]
        x_desc = x[order_desc]
        cumint = cumulative_trapezoid(integrand[order_desc], x_desc, initial=0.0)
        integral_from_0 = np.empty_like(cumint)
        integral_from_0[order_desc] = cumint
        nFC = self.nFC_x0 * np.exp(integral_from_0)

        flux_term = -(gr2 * Dped / (Vcx * fCX)) * (dne_dx - self.dne_dx_neginf)
        fc_term = -(Vfc * fFC / (Vcx * fCX)) * ((Si + Scx / 2) / (Si + Scx)) * nFC
        nCX = np.maximum(flux_term + fc_term, 0.0)
        return nFC, nCX

    def solve_SC(self,soln_method='sc_2order',tol=1e-3,max_iter=50,x_res=100,free_params=None):
        """Iteratively solve Equation (15) in S. Saarelma et al 2023 Nucl. Fusion 63 052002

        1. Sets up coefficient interpolators mapping psi_N quantities to x.
        2. Calls first_step() to solve the simplified Eq 16 (no CX neutrals).
        3. Iterates the full Eq 15 (with CX neutrals and the integral term)
           until the n_e profile converges.

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        soln_method : string
            method to use to determine the solution, supporting: sc_2order
        tol : float
            Relative convergence tolerance on n_e for solve_bvp() call.
        max_iter : int
            Maximum number of Picard iterations for solve_bvp() call.
        x_res : int
            Number of points to use in the radial grid for solve_bvp() call.
        free_params : dict, optional
            Dictionary of free parameters to update. If None, uses the free parameters from construction.

        Sets
        ----
        self.ne_sol : ndarray
            Converged electron density profile on self.x_sol.
        self.x_sol : ndarray
            Radial grid (x = 0 at separatrix, x < 0 inside).
        self.nFC_sol, self.nCX_sol : ndarray
            FC and CX neutral densities on ``self.x_sol`` (Eqs.~(11)--(12)).
        """

        # form factors are currently set to 1, assuming poloidal symmetry as done in S. Saarelma et al 2024 Nucl. Fusion 64 076025 Section 3
        self.form_factor(type='FC')
        self.form_factor(type='cx')
        self.setup_solver_grids(res=x_res)

        # Set free parameters
        self.update_free_params(
            free_params['alpha_crit'], free_params['C_KBM'], free_params['De_chie_etg'], free_params['nFC_x0'],
            nFC_threshold=free_params['nFC_threshold'],
            nCX_threshold=free_params['nCX_threshold'],
            psi_N_inner_boundary=free_params['psi_N_inner_boundary'],
        )

        if soln_method == 'sc_2order':
            for i in range(max_iter):

                # Current iteration of n_e
                if i == 0:
                    if self.initial_guess == 'pfile':
                        ne_sol_prev = self.n_e_pres
                    else:
                        raise ValueError(f"Invalid initial guess: {self.initial_guess}")
                else:
                    # Previous solution in physical units (after de-normalization)
                    self.x_prev = self.sol.x
                    ne_sol_prev = self.sol.y[0]

                # Calculate pressure, alpha, and D_KBM
                self.calc_pressure_quantities_sc(n_e=ne_sol_prev,x=self.x_prev)

                # First step (Eq 16 in Saarelma et al., 2023, no CX neutrals)
                if i == 0:
                    self.first_step(resolution=x_res)
                    continue

                # Individual transport component interpolators
                D_NC_int = interp1d(self.x_init, self.D_NEO, kind='linear', bounds_error=False, fill_value='extrapolate')
                D_KBM_int = interp1d(self.x_init, self._D_KBM, kind='linear', bounds_error=False, fill_value='extrapolate')
                D_ETG_x_int = interp1d(self.x_init, self.D_ETG_x, kind='linear', bounds_error=False, fill_value='extrapolate')

                gradr2_fsa_int = interp1d(
                    self.x_init, self.gradr2_fsa, kind='linear',
                    bounds_error=False, fill_value='extrapolate',
                )
                V_CX_int = interp1d(
                    self.x_init, self.V_cx_pres, kind='linear',
                    bounds_error=False, fill_value='extrapolate',
                )
                S_i_int = interp1d(
                    self.x_init, self.S_i_pres, kind='linear',
                    bounds_error=False, fill_value='extrapolate',
                )
                S_cx_int = interp1d(
                    self.x_init, self.S_cx_pres, kind='linear',
                    bounds_error=False, fill_value='extrapolate',
                )

                # Iterated exponential term (computed in physical coordinates).
                S_i_on_xprev = S_i_int(self.x_prev)
                S_cx_on_xprev = S_cx_int(self.x_prev)
                integrand = (ne_sol_prev * (S_i_on_xprev + S_cx_on_xprev)) / (self.fFC * abs(self.V_FC))
                
                # To make the sign unambiguous with x<0 inward, integrate on a
                # descending-x ordering (separatrix -> core), then map back.
                order_desc = np.argsort(self.x_prev)[::-1]
                x_desc = self.x_prev[order_desc]
                integrand_desc = integrand[order_desc]
                integral_desc = cumulative_trapezoid(integrand_desc, x_desc, initial=0.0)
                integral_from_0 = np.empty_like(integral_desc)
                integral_from_0[order_desc] = integral_desc
                if i == 0:
                    self.integral_from_0 = integral_from_0 # debugging
                exp_term_arr = np.exp(integral_from_0)

                # Split non-linear components based on the LaTeX derivation
                f0_arr = gradr2_fsa_int(self.x_prev) * (D_NC_int(self.x_prev) + D_KBM_int(self.x_prev))
                f1_arr = gradr2_fsa_int(self.x_prev) * D_ETG_x_int(self.x_prev)
                
                df0_arr = np.gradient(f0_arr, self.x_prev)
                df1_arr = np.gradient(f1_arr, self.x_prev)

                # Callable interpolators (all in physical x)
                f0_x = interp1d(self.x_prev, f0_arr, kind='linear', bounds_error=False, fill_value='extrapolate')
                df0_dx = interp1d(self.x_prev, df0_arr, kind='linear', bounds_error=False, fill_value='extrapolate')
                f1_x = interp1d(self.x_prev, f1_arr, kind='linear', bounds_error=False, fill_value='extrapolate')
                df1_dx = interp1d(self.x_prev, df1_arr, kind='linear', bounds_error=False, fill_value='extrapolate')
                exp_term_prev = interp1d(self.x_prev, exp_term_arr, kind='linear', bounds_error=False, fill_value='extrapolate')

                # Build physical guess on uniform grid, then non-dimensionalize
                x_grid = np.linspace(self.x_inner, 0, x_res)
                ne_guess = interp1d(self.x_prev, ne_sol_prev, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')(x_grid)
                dne_guess = np.gradient(ne_guess, x_grid)
                Y_guess = np.vstack([ne_guess, dne_guess])

                n0_inner = interp1d(self.x_prev, ne_sol_prev, kind='linear', bounds_error=False, fill_value='extrapolate')(self.x_inner) 
                self.non_dimensionalize(x=x_grid, y=Y_guess, n0=n0_inner)
                L = self._L
                n0 = self._n0

                def ode_solv(xi, Y):
                    N, dNdxi = Y
                    x = L * xi
                    Vcx = V_CX_int(x)
                    
                    f0 = f0_x(x)
                    df0dx = df0_dx(x)
                    f1 = f1_x(x)
                    df1dx = df1_dx(x)
                    
                    exp_term = exp_term_prev(x)
                    S_i = S_i_int(x)
                    S_cx = S_cx_int(x)

                    # Protection against non-physical solver guesses (N near 0)
                    N_safe = np.maximum(N, 1e-6)
                    if 1e-6 in N_safe:
                        import warnings
                        warnings.warn(
                            "N_safe is very close to zero and got clipped to 1e-6",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    F = f0 + (f1 / (n0 * N_safe))

                    C_cx = 1 - (abs(self.V_FC) * self.fFC / (abs(Vcx) * self.fCX)) * ((S_i + S_cx / 2) / (S_i + S_cx))

                    # Mathematically consistent dimensionless coefficients
                    C_A = (n0 * L * S_i) / (abs(Vcx) * self.fCX)
                    C_B = C_A * self.dNdxi_neginf
                    C_E = (L**2 * S_i * C_cx * self.nFC_x0 * exp_term) / F
                    C_K = (L / F) * (df0dx + (df1dx / (n0 * N_safe)))
                    C_N = f1 / (n0 * (N_safe**2) * F)

                    if self.error_check:
                        # Fallback utilizing an effective D_ped equivalent if the error check overrides coefficients
                        D_ped_eff = D_NC_int(x) + D_KBM_int(x) + (D_ETG_x_int(x) / (n0 * N_safe))
                        C_A = C_A * abs(Vcx) * self.fCX / D_ped_eff
                        C_B = C_B * abs(Vcx) * self.fCX / D_ped_eff

                    if i==0 and self.verbose:
                        print('iteration: ', i+1)
                        print(f"C_E : {np.max(C_E):.3e}, min: {np.min(C_E):.3e}")
                        print("exp_term max/min", np.max(exp_term), np.min(exp_term))
                        print("nFC_x0", self.nFC_x0)
                        print("F max/min", np.max(F), np.min(F))
                        print("C_cx max/min", np.max(C_cx), np.min(C_cx))
                        print("Si max/min", np.max(S_i), np.min(S_i))

                    d2Ndxi2 = C_A * N * dNdxi - C_B * N - C_E * N - C_K * dNdxi + C_N * (dNdxi**2)
                    return np.vstack([dNdxi, d2Ndxi2])

                def bc_solv(Ya, Yb):
                    return np.array([
                        Ya[1] - self.dNdxi_neginf, # Neumann boundary condition at x_inner
                        Yb[0] - self.ne_x0/n0,     # Dirichlet BC: N = ne/n0 at xi = x = 0 (separatrix)
                    ])

                sol = solve_bvp(ode_solv, bc_solv, self.xi, self.N_guess, max_nodes=5000, verbose=self.bvp_verbose)
                if not sol.success:
                    raise RuntimeError(f"step {i} BVP failed: {sol.message}")

                # De-normalize back to physical units
                sol.x = L * sol.x
                sol.y[0] = n0 * sol.y[0]
                sol.y[1] = (n0 / L) * sol.y[1]

                # Check convergence in physical units
                ne_sol_prev_interp = interp1d(self.x_prev, ne_sol_prev, kind='linear',
                                              bounds_error=False, fill_value='extrapolate')(sol.x)
                residual = np.max(np.abs(sol.y[0] - ne_sol_prev_interp)) / np.max(np.abs(sol.y[0]))
                if self.verbose:
                    print(f"  Eq 15 iteration {i}: residual = {residual:.2e}")

                self.sol = sol
                if i==0:
                    self.sol_first = sol # debugging

                if residual < tol:
                    break

            # Final solution in physical units
            self.x_sol = self.sol.x
            self.ne_sol = self.sol.y[0]
            self.dne_dx_sol = self.sol.y[1]

            # FC/CX on the converged BVP grid (Eqs. 11--12)
            self.exp_term_arr = exp_term_arr  # last Picard iterate, on x_prev
            self.nFC_sol, self.nCX_sol = self.compute_post_solve_SC_neutrals()
            if not (len(self.x_sol) == len(self.ne_sol) == len(self.nFC_sol)):
                raise RuntimeError(
                    "Parent solve profile length mismatch on x_sol: "
                    f"x={len(self.x_sol)}, ne={len(self.ne_sol)}, "
                    f"nFC={len(self.nFC_sol)}"
                )

            return self.x_sol, self.ne_sol, self.dne_dx_sol


    def invalidate_firedrake_cache(self):
        """Drop cached Firedrake meshes, coefficients, and linear solvers.

        Called automatically by :meth:`update_free_params`.  Call manually
        after changing equilibrium inputs or other quantities that affect
        ``setup_solver_grids`` / ``calc_gradr``.
        """
        self._fd_cache = {}

    def _plot_profiles(self, x_dofs, ne, nFC, nCX, title=""):
        """Plot n_e, <n_FC>, <n_CX> vs x with a secondary psi_N axis on top.

        Used as a diagnostic from ``solve_firedrake`` (gated on ``verbose``)
        to visualise either the initial guess or any intermediate / final
        profile triple.

        Parameters
        ----------
        x_dofs : ndarray
            Per-DOF x coordinates (m), unsorted (as stored in dat.data).
        ne, nFC, nCX : ndarray
            Profiles in the same DOF order as ``x_dofs`` (m^-3).
        title : str
            Figure suptitle.
        """
        # Sort by ascending x for clean line plots (DOF order is not guaranteed
        # to be spatial).
        sort_idx = np.argsort(x_dofs) # if setup correctly, this should not do anything
        x_plot = x_dofs[sort_idx]
        profiles = [ne[sort_idx], nFC[sort_idx], nCX[sort_idx]]
        labels   = [r"$n_e$",
                    r"$\langle n_{FC} \rangle$",
                    r"$\langle n_{CX} \rangle$"]
        colours  = ["tab:blue", "tab:orange", "tab:green"]

        # x <-> psi_N mapping built off the parent grid (self.x_init is
        # monotonically increasing toward the separatrix; self.psi_N_pres
        # is the corresponding normalised poloidal flux).  Used by
        # secondary_xaxis to render psi_N on top of each panel.
        x_to_psiN = interp1d(self.x_init, self.psi_N_pres,
                            kind='linear', bounds_error=False,
                            fill_value='extrapolate')
        psiN_to_x = interp1d(self.psi_N_pres, self.x_init,
                            kind='linear', bounds_error=False,
                            fill_value='extrapolate')

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        if title:
            fig.suptitle(title, fontsize=12)
        for ax, prof, label, col in zip(axes, profiles, labels, colours):
            ax.plot(x_plot, prof, lw=2, color=col)
            ax.set_xlabel(r"$x$ (m)")
            ax.set_ylabel(f"{label} (m$^{{-3}}$)")
            ax.grid(True, alpha=0.3)
            secax = ax.secondary_xaxis('top',
                                    functions=(x_to_psiN, psiN_to_x))
            secax.set_xlabel(r"$\psi_N$")
        # plt.show()
        plt.savefig(f'{title}.png')

    @staticmethod
    def _build_petsc_solver_parameters(linear_solver="lu",
                                    ksp_rtol=1e-8,
                                    ksp_max_it=200):
        """PETSc options for SNES ``solve(F == 0, ...)`` (full Newton path)."""
        linear_solver = str(linear_solver).lower()
        if linear_solver not in ("lu", "gamg"):
            raise ValueError(
                f"linear_solver must be 'lu' or 'gamg', got {linear_solver!r}."
            )

        params = {
            "snes_type": "newtonls",
            "snes_max_it": 50,
            "snes_atol": 1.0e-8,
            "snes_rtol": 1.0e-8,
            "snes_linesearch_type": "bt",
            "mat_type": "aij",
        }

        if linear_solver == "lu":
            params.update({
                "ksp_type": "preonly",
                "pc_type": "lu",
            })
        else:
            params.update({
                "ksp_type": "gmres",
                "ksp_rtol": float(ksp_rtol),
                "ksp_max_it": int(ksp_max_it),
                "ksp_gmres_restart": 30,
                "pc_type": "gamg",
                "mg_levels_ksp_type": "richardson",
                "mg_levels_pc_type": "sor",
                "mg_levels_ksp_max_it": 5,
            })

        return params

    def _ensure_firedrake_coefficient_grids(self, x_res, force=False):
        """Run ``form_factor`` + ``setup_solver_grids`` once per ``x_res``."""
        key = int(x_res)
        if force or self._fd_cache.get("x_res") != key:
            self.setup_solver_grids(res=x_res)
            self.form_factor(type='FC',x=self.x_init)
            self.form_factor(type='cx',x=self.x_init)
            self._fd_cache["x_res"] = key

    def calc_pressure_quantities(self, n_e, average_alpha_pedestal=True):
        """Pressure, ``alpha``, and ``D_KBM`` on the ``x_dofs`` (from separatrix to x_inner) grid.

        When ``average_alpha_pedestal`` is True (default), use the mean of
        local ``alpha`` over the pedestal ``x in [x_inner, 0]`` in Eq.~(25)
        instead of a flux-surface-local ``alpha``.

        Implementation notes
        --------------------
        ``self._fd_cache["x_dofs"]`` is the per-DOF x coordinate as stored by
        Firedrake; for ``CG_p`` with ``p>=2`` the DOFs are grouped by entity
        (vertices first, then cell interiors) and so the array is **not**
        monotonic in space.  ``np.gradient`` interprets consecutive samples
        as adjacent in x, so we must sort to spatial order before any
        finite-difference call and unsort the outputs that are subsequently
        written into ``Function.dat.data`` (which expects DOF order).

        The equilibrium-only piece of ``alpha_nodp`` is precomputed on the
        parent ``self.x_init`` grid in :meth:`setup_solver_grids`; here we
        only need ``np.interp`` (which does not require its first argument
        to be sorted).  The single ``np.gradient`` left at runtime is
        ``dpdx``, whose dependence on ``n_e`` forces the sort.
        """
        x_dofs_arr = self._fd_cache["x_dofs"]
        sort_idx = np.argsort(x_dofs_arr)
        unsort_idx = np.argsort(sort_idx) # inverse permutation: a[sort][unsort] == a
        x = x_dofs_arr[sort_idx]
        n_e_sorted = np.asarray(n_e)[sort_idx]

        # All quantities below are evaluated in sorted-x (spatial) order.
        T_e = np.interp(x, self.x_init, self.T_e_pres)
        T_i = np.interp(x, self.x_init, self.T_i_pres)
        self.T_e_xdofs = T_e
        G_KBM_grid = self.C_KBM * (self.c_s * self.rho_s**2) / self.a # on psi_Te_eval/x_init grid
        G_KBM = np.interp(x, self.x_init, G_KBM_grid)
        alpha_nodp = np.interp(x, self.x_init, self._alpha_nodp_xinit)

        _pres = n_e_sorted * (T_e + T_i) * 1.60218e-19 # Pa = J/m^3, neglecting neutral pressures and assuming quasi-neutral plasma
        dpdx = np.gradient(_pres, x) # x is now monotonic
        _alpha = alpha_nodp * dpdx # standard Connor-Hastie alpha, sorted-x order

        if average_alpha_pedestal:
            alpha_bar = float(np.mean(_alpha))
            self.alpha_bar_ped = alpha_bar
            gate = alpha_bar > self.alpha_crit
            D_KBM_sorted = np.where(gate, (alpha_bar - self.alpha_crit) * G_KBM, 0.0)
            A_KBM_sorted = np.where(gate, -G_KBM * self.alpha_crit, 0.0)
            B_KBM_sorted = np.where(gate, G_KBM * alpha_nodp, 0.0)
        else:
            gate = _alpha > self.alpha_crit
            D_KBM_sorted = np.where(gate, (_alpha - self.alpha_crit) * G_KBM, 0.0)
            A_KBM_sorted = np.where(gate, -G_KBM * self.alpha_crit, 0.0)
            B_KBM_sorted = np.where(gate, G_KBM * alpha_nodp, 0.0)

        # Outputs assigned to Firedrake Function.dat.data in solve_coupled
        # must be in DOF order: undo the sort.
        self._D_KBM = D_KBM_sorted[unsort_idx]
        self._A_KBM = A_KBM_sorted[unsort_idx]
        self._B_KBM = B_KBM_sorted[unsort_idx]

        # Diagnostic attributes; keep DOF order so they line up with x_dofs.
        self.G_KBM = G_KBM[unsort_idx]
        self.alpha_nodp = alpha_nodp[unsort_idx]
        self.pres = _pres[unsort_idx]

    def _firedrake_mesh_key(self, x_left, mesh_n, fe_degree):
        return (round(float(x_left), 12), int(mesh_n), int(fe_degree))

    def construct_C_ETG(self): # on x_init grid
        DeltaT_e = np.gradient(self.T_e_pres * (1.60218e-19), self.x_init) # gradient in J/m, T_e_pres is in eV
        self.C_ETG = self.De_chie_etg * self.P_tot_e / (self.S_plasma * abs(DeltaT_e)) # evaluated at x_init

    def _ne_on_parent_grid(self, ne_mesh_data):
        """Interpolate FE ``n_e`` DOF values onto ``self.x_init``."""
        x_dofs = self._fd_cache["x_dofs"]
        order = np.argsort(x_dofs)
        return np.interp(
            self.x_init, x_dofs[order], ne_mesh_data[order],
            left=np.nan, right=np.nan,
        )

    def _ensure_firedrake_discretization(self, x_left, mesh_n, fe_degree, force=False):
        """Build or reuse mesh, spaces, and coefficient Functions on the mesh."""
        mesh_key = self._firedrake_mesh_key(x_left, mesh_n, fe_degree)
        if (not force
                and self._fd_cache.get("mesh_key") == mesh_key
                and "mesh" in self._fd_cache):
            return (
                self._fd_cache["mesh"],
                self._fd_cache["V"],
                self._fd_cache["W"],
                self._fd_cache["x_dofs"],
                self._fd_cache["g_fd"],
                self._fd_cache["Si_fd"],
                self._fd_cache["Scx_fd"],
                self._fd_cache["Vcx_fd"],
            )

        mesh = IntervalMesh(mesh_n, x_left, 0.0)
        V = FunctionSpace(mesh, "CG", fe_degree)
        W = MixedFunctionSpace([V, V, V])

        x_coord_func = Function(V).interpolate(SpatialCoordinate(mesh)[0])
        x_dofs = x_coord_func.dat.data.copy() # value of x at each finite element node

        def _make_func(arr, name=""): # interpolate from self.x_init to x_dofs
            f = Function(V, name=name)
            f.dat.data[:] = np.interp(x_dofs, self.x_init, arr)
            return f

        g_fd = _make_func(self.gradr2_fsa, "gradr2_fsa")
        Si_fd = _make_func(self.S_i_pres, "S_i")
        Scx_fd = _make_func(self.S_cx_pres, "S_cx")
        Vcx_fd = _make_func(abs(self.V_cx_pres), "V_cx")

        self._fd_cache.pop("u", None)
        self._fd_cache.pop("u_prev", None)
        self._fd_cache.update({
            "mesh_key": mesh_key,
            "mesh": mesh,
            "V": V,
            "W": W,
            "x_dofs": x_dofs,
            "g_fd": g_fd,
            "Si_fd": Si_fd,
            "Scx_fd": Scx_fd,
            "Vcx_fd": Vcx_fd,
        })
        return mesh, V, W, x_dofs, g_fd, Si_fd, Scx_fd, Vcx_fd

    def _get_or_create_mixed_solution(self, W, force=False):
        """Return cached ``(u, u_prev)`` on ``W``, or allocate new Functions."""
        if (not force
                and self._fd_cache.get("W") is W
                and "u" in self._fd_cache
                and "u_prev" in self._fd_cache):
            return self._fd_cache["u"], self._fd_cache["u_prev"]

        u = Function(W, name="u")
        u_prev = Function(W, name="u_prev")
        self._fd_cache["W"] = W
        self._fd_cache["u"] = u
        self._fd_cache["u_prev"] = u_prev
        return u, u_prev

    def _build_f1_weak_form(
            self,
            ne, ne_for_etg,
            g_fd, C_ETG_fd, D_NEO_fd, D_KBM_fd,
            Si_fd, nFC, nCX, v_e, ne_inner_bc, dne_dx_inner_c):
        """Electron-density weak form for ``v_e``."""
        ne_dx = ne.dx(0)

        T_tot_fd = self._fd_cache.get("T_tot_fd")
        dT_tot_dx_fd = self._fd_cache.get("dT_tot_dx_fd")
        A_KBM_fd = self._fd_cache.get("A_KBM_fd")
        B_KBM_fd = self._fd_cache.get("B_KBM_fd")
        if (T_tot_fd is None or dT_tot_dx_fd is None
                or A_KBM_fd is None or B_KBM_fd is None):
            raise RuntimeError(
                "Eq. A33 weak form requires T_tot_fd, dT_tot_dx_fd, "
                "A_KBM_fd, and B_KBM_fd in _fd_cache (built in solve_firedrake)."
            )
        # Expanded flux inside <|grad r|^2> ... v_e' (Labbate appendix A33).
        flux_a33 = (
            (C_ETG_fd / ne) * ne_dx
            + A_KBM_fd * ne_dx
            + B_KBM_fd * T_tot_fd * ne_dx * ne_dx
            + B_KBM_fd * ne * dT_tot_dx_fd * ne_dx
            + D_NEO_fd * ne_dx
        )
        
        F1 = (g_fd * flux_a33 * v_e.dx(0) - ne * Si_fd * (nFC + nCX) * v_e) * dx

        if ne_inner_bc == "neumann":
            T_tot_fd = self._fd_cache["T_tot_fd"]
            dT_tot_dx_fd = self._fd_cache["dT_tot_dx_fd"]
            A_KBM_fd = self._fd_cache["A_KBM_fd"]
            B_KBM_fd = self._fd_cache["B_KBM_fd"]
            D_bc_a33 = (
                C_ETG_fd / ne
                + A_KBM_fd
                + B_KBM_fd * T_tot_fd * dne_dx_inner_c
                + B_KBM_fd * ne * dT_tot_dx_fd
                + D_NEO_fd
            )
            F1 = F1 + g_fd * D_bc_a33 * dne_dx_inner_c * v_e * ds(1)
        return F1

    def newton_solve(self, linear_solver="lu", ksp_rtol=1e-6, ksp_max_it=1000, params_dict=None, v=False):
        """Full Newton via SNES on the nonlinear residual F(u)=0.
        Parameters
        ----------
        linear_solver : str, default "lu"
            Linear solver to use for the Newton solve
        ksp_rtol : float, default 1e-6
            Relative tolerance for the linear solver
        ksp_max_it : int, default 1000
            Maximum number of iterations for the linear solver
        params_dict : dict
            Dictionary containing the parameters for the Newton solve
        v : bool, default False
            Whether to print verbose output
        """
        W = params_dict['W']
        g_fd = params_dict['g_fd']
        C_ETG_fd = params_dict['C_ETG_fd']
        D_NEO_fd = params_dict['D_NEO_fd']
        D_KBM_fd = params_dict['D_KBM_fd']
        Si_fd = params_dict['Si_fd']
        Scx_fd = params_dict['Scx_fd']
        Vcx_fd = params_dict['Vcx_fd']
        VFC_const = params_dict['VFC_const']
        fFC_const = params_dict['fFC_const']
        fCX_const = params_dict['fCX_const']
        half = params_dict['half']
        ne_inner_bc = params_dict['ne_inner_bc']
        dne_dx_inner_c = params_dict['dne_dx_inner_c']
        bcs = params_dict['bcs']
        u = params_dict['u']
        v_e, v_F, v_C = TestFunctions(W)
        ne_curr, nFC_curr, nCX_curr = split(u)

        F1 = self._build_f1_weak_form(
            ne_curr, ne_curr,
            g_fd, C_ETG_fd, D_NEO_fd, D_KBM_fd,
            Si_fd, nFC_curr, nCX_curr, v_e,
            ne_inner_bc, dne_dx_inner_c
        )
        F2 = ((VFC_const * fFC_const * g_fd * nFC_curr).dx(0) * v_F
                - ne_curr * (Si_fd + Scx_fd) * nFC_curr * v_F) * dx
        F3 = ((Vcx_fd * fCX_const * g_fd * nCX_curr).dx(0) * v_C
                - ne_curr * (
                    Si_fd * nCX_curr - half * Scx_fd * nFC_curr
                ) * v_C) * dx

        F = F1 + F2 + F3
        newton_params = self._build_petsc_solver_parameters(
            linear_solver=linear_solver,
            ksp_rtol=ksp_rtol,
            ksp_max_it=ksp_max_it,
        )
        solve(F == 0, u, bcs=bcs, solver_parameters=newton_params)

    def solve_coupled(self,
                      x_res=200,
                      fe_degree=2,
                      ne_inner_bc="neumann",
                      bc_origin="p-file",
                      ne_inner=None,
                      dne_dx_inner=None,
                      initial_guess="linear",
                      tanh_width=None,
                      tanh_center=None,
                      linear_solver="lu",
                      ksp_rtol=1e-8,
                      ksp_max_it=200,
                      reuse_setup=True,
                      verbose=None):
        """Firedrake-based solver for the *coupled three-equation* Saarelma-Connor
        neutral-transport pedestal model in its un-reduced form.

        System solved
        =============
        Origin: Saarelma & Connor 2023 Nucl. Fusion 63 052002 Eq. (4)-(6)
        Unknowns:
            n_e(x)         (electron density; treated as a flux function)
            <n_FC>(x)      (FSA Franck-Condon neutral density)
            <n_CX>(x)      (FSA charge-exchange neutral density)
        with flux-surface average (FSA) defined by  <A>(x) = oint R^2 A dtheta / oint R^2 dtheta
        on the radial coordinate x = r - r_sep (separatrix at x = 0, core at x<0).

        With form factors
            f_FC(x) = <|grad r|^2 n_FC> / ( <n_FC> <|grad r|^2> )
            f_CX(x) = <|grad r|^2 n_CX> / ( <n_CX> <|grad r|^2> )
        the coupled system to solve is:

            d/dx [ <|grad r|^2> D dn_e/dx ]  with
                D = D_NEO + D_KBM + C_ETG/n_e,
                D_KBM from Eqs. (24)--(25) (pedestal-averaged alpha),
                = - n_e S_i ( <n_FC> + <n_CX> )

            |V_FC| d/dx [ f_FC <n_FC> <|grad r|^2> ]
                = + n_e (S_i + S_CX) <n_FC>

            d/dx [ |V_CX| f_CX <n_CX> <|grad r|^2> ]
                =  n_e ( <n_CX> S_i - (S_CX/2) <n_FC> )

        Boundary conditions
        ===================
            n_e (x = 0)         = ne_x0         (Dirichlet at separatrix)
            n_e (x = x_inner)   = ne_inner      (Dirichlet, inner pedestal boundary)
            <n_FC>(x = 0)       = nFC_x0        (Dirichlet at separatrix)
            <n_CX>(x = 0)       = nCX_x0        (Dirichlet at separatrix)
        The first-order ODEs for <n_FC> and <n_CX> have characteristic flowing
        from the separatrix inward (V_{*,r} < 0), so a single Dirichlet BC at
        x = 0 is well-posed.  The diffusion equation is second-order in n_e and
        takes Dirichlet at both ends.

        Parameters
            ----------
            x_res : int
                Resolution of solver grid
            fe_degree : int
                Polynomial degree of the CG finite-element basis.
            ne_inner_bc : {"neumann", "dirichlet"}, default "neumann"
                Whether to use a Neumann or Dirichlet BC at the inner electron density boundary
            bc_origin : {"p-file", "user"}, default "p-file"
                Origin of the inner electron density boundary condition:
                - ``"p-file"`` (default): read the inner-boundary values
                    from the p-file.
                - ``"user"``: use the values provided by the user.
            ne_inner : float or None
                Value of n_e at the inner boundary x = x_inner used for the Dirichlet BC when ne_inner_bc == "dirichlet" and you want a user-specified value (if bc_origin == "user")
            dne_dx_inner : float or None
                Value of dn_e/dx at the inner boundary x = x_inner used for the Neumann BC when ne_inner_bc == "neumann" and you want a user-specified value (if bc_origin == "user")
            initial_guess : {"linear", "pfile", "tanh"}
                * ``"linear"`` (default) -- build the initial guess from the
                four prescribed boundary conditions only:
                    n_e  : linear from ne_inner at x_inner to ne_x0 at 0
                    n_FC : linear from 0           at x_inner to nFC_x0 at 0
                    n_CX : linear from 0           at x_inner to nCX_x0 at 0
                No use of the p-file profile or the analytical exponential
                decay estimate.
                * ``"pfile"`` -- legacy guess: p-file electron density profile
                and an analytical FC exponential decay (same construction as
                the parent solver's first-step routine).
                * ``"tanh"`` -- H-mode-pedestal-like tanh initial guess.
                n_e transitions from ne_inner_val (pedestal top) to ne_x0
                (separatrix) through a tanh of half-width ``tanh_width/2``
                centred at ``tanh_center``.  Neutrals get the mirrored
                shape (peaked at the separatrix, decaying inward).  Default
                width / centre give a typical narrow pedestal just inside
                the separatrix.
            tanh_width : float or None
                Pedestal width (m) for the ``"tanh"`` initial guess.  The
                tanh argument is ``(x - tanh_center) / (tanh_width/2)``.
                If ``None`` (default), set to ``0.1 * |x_inner|``
            tanh_center : float or None
                Pedestal-foot x-position (m) for the ``"tanh"`` initial
                guess; should be <= 0.  If ``None`` (default), set to
                ``-tanh_width`` so the pedestal sits just inside the
                separatrix.
            linear_solver : {"lu", "gamg"}, default ``"lu"``
                Picard: direct/iterative solve of each linear sub-step
                (:meth:`_build_picard_linear_solver_parameters`).  Newton:
                linear solve of each SNES Jacobian step
                (:meth:`_build_petsc_solver_parameters`).

                * ``"lu"`` -- sparse direct LU (default; fast for 1D meshes).
                * ``"gamg"`` -- GMRES + PETSc algebraic multigrid (GAMG)
                preconditioner.  Prefer this when refining ``mesh_n`` /
                ``fe_degree`` makes direct solves expensive.
            ksp_rtol : float, default ``1e-8``
                GMRES relative tolerance when ``linear_solver="gamg"``.
            ksp_max_it : int, default ``200``
                Maximum GMRES iterations per Newton step when using ``"gamg"``.
            reuse_setup : bool, default ``True``
                If True, reuse cached ``setup_solver_grids`` / mesh / coefficient
                projections when ``x_res``, ``mesh_n``, ``fe_degree``, and
                ``x_inner`` are unchanged since the last solve.  Call
                :meth:`invalidate_firedrake_cache` after changing free parameters
                or equilibrium inputs (also done automatically by
                :meth:`update_free_params`).
            verbose : bool or None
                Override ``self.verbose`` for the duration of this solve.

            Sets
            ----
            self.x_sol     : ndarray
            self.ne_sol    : ndarray
            self.nFC_sol   : ndarray
            self.nCX_sol   : ndarray
                Converged profiles on the (sorted) DOF grid.
            self.u_fd      : Firedrake mixed Function (raw FE solution).
        """

        if not _FIREDRAKE_AVAILABLE:
            raise ImportError(
                "Firedrake is not available in this environment.  "
                "Install Firedrake (https://www.firedrakeproject.org/) "
                f"to use this solver.  Original import error:\n"
                f"  {_FIREDRAKE_IMPORT_ERR}"
            )

        v = self.verbose if verbose is None else bool(verbose)
        force_setup = not reuse_setup # True is reuse_setup is False

        self._ensure_firedrake_coefficient_grids(x_res, force=force_setup)
        self.construct_C_ETG()

        # set the inner boundary location and read off the values used for
        # either the Dirichlet or Neumann BC at x = x_inner.
        if self.psi_N_inner_boundary is None:
            self.find_inner_boundary() # set self.psi_N_inner_boundary and self.x_inner, where nFC and nCX fall below their thresholds
        else:
            self.x_inner = np.interp(self.psi_N_inner_boundary, self.psi_N_pres, self.x_init)

        # ne(x_inner) -- used for the Dirichlet BC if requested, and for the
        # initial guess at the inner boundary in all cases.
        if bc_origin == "p-file":
            ne_inner_val = float(np.interp(self.x_inner, self.x_init, self.n_e_pres))
            self.ne_inner = ne_inner_val

            # dne/dx(x_inner) -- used for the Neumann BC if requested. Computed the same way as solver.first_step (np.gradient on the p-file n_e).
            dne_dx_pres        = np.gradient(self.n_e_pres, self.x_init)
            dne_dx_inner_val   = float(np.interp(self.x_inner, self.x_init, dne_dx_pres))
        elif bc_origin == "user":
            ne_inner_val = float(ne_inner)
            dne_dx_inner_val = float(dne_dx_inner)
        elif bc_origin == "p-file user combo":
            if ne_inner_bc == "neumann":
                dne_dx_pres        = np.gradient(self.n_e_pres, self.x_init)
                dne_dx_inner_val   = float(np.interp(self.x_inner, self.x_init, dne_dx_pres))

                ne_inner_val = float(ne_inner)
                self.ne_inner = ne_inner_val
            elif ne_inner_bc == "dirichlet":
                ne_inner_val = float(ne_inner)
                self.ne_inner = ne_inner_val

                dne_dx_inner_val = float(dne_dx_inner)
        else:
            raise ValueError(
                f"bc_origin must be 'p-file' or 'user' or 'p-file user combo', got {bc_origin!r}."
            )

        ne_inner_bc = str(ne_inner_bc).lower()
        if ne_inner_bc not in ("dirichlet", "neumann"):
            raise ValueError(
                f"ne_inner_bc must be 'dirichlet' or 'neumann', "
                f"got {ne_inner_bc!r}."
            )

        if float(self.x_inner) >= 0.0:
            raise ValueError(
                f"x_inner = {self.x_inner} must be strictly less than 0 (separatrix)."
            )

        if v:
            print(f"[coupled solver] x_inner        = {self.x_inner:.4e} m")
            print(f"[coupled solver] ne_inner_bc    = {ne_inner_bc!r}")
            print(f"[coupled solver] ne(x_inner)    = {ne_inner_val:.3e} m^-3 ({bc_origin})")
            print(f"[coupled solver] dne/dx(x_in)   = {dne_dx_inner_val:.3e} m^-4 ({bc_origin})")
            print(f"[coupled solver] ne_x0          = {self.ne_x0:.3e} m^-3")
            print(f"[coupled solver] nFC_x0         = {self.nFC_x0:.3e} m^-3")
            print(f"[coupled solver] nCX_x0         = {self.nCX_x0:.3e} m^-3")

        mesh, V, W, x_dofs, g_fd, Si_fd, Scx_fd, Vcx_fd = (
            self._ensure_firedrake_discretization(
                float(self.x_inner), x_res, fe_degree, force=force_setup,
            )
        )

        # define constants for the Firedrake system
        VFC_const = Constant(abs(self.V_FC))
        fFC_const = Constant(self.fFC)
        fCX_const = Constant(self.fCX)
        half      = Constant(0.5)

        u, u_prev = self._get_or_create_mixed_solution(W, force=force_setup)

        # set up initial guess
        if initial_guess == "linear":
            # BC-only initial guess: linear ramps between the four
            # prescribed Dirichlet values.  n_FC and n_CX ramp linearly
            # from 0 at x_inner (physically reasonable -- the neutrals
            # have been ionised away deep in the plasma) to their
            # separatrix values at x = 0.
            xi = (x_dofs - self.x_inner) / (0 - self.x_inner) # x range is from x_inner to 0
            if ne_inner_bc == "dirichlet":  # ne_inner_val is known
                ne_init_data = ne_inner_val + (self.ne_x0 - ne_inner_val) * xi
            elif ne_inner_bc == "neumann":  # derive ne_inner_val from the slope
                if dne_dx_inner_val >= 0:
                    raise ValueError(
                        f"dne/dx(x_inner) = {dne_dx_inner_val:.3e} m^-4 must be "
                        "negative for Neumann boundary condition (n_e decreases "
                        "outward in the pedestal)."
                    )
                ne_inner_val  = dne_dx_inner_val * self.x_inner + self.ne_x0
                self.ne_inner = ne_inner_val
                ne_init_data  = ne_inner_val + (self.ne_x0 - ne_inner_val) * xi
            nFC_init_data = self.nFC_x0 * xi
            nCX_init_data = self.nCX_x0 * xi

            u.subfunctions[0].dat.data[:] = ne_init_data
            u.subfunctions[1].dat.data[:] = nFC_init_data
            u.subfunctions[2].dat.data[:] = nCX_init_data
            u_prev.assign(u)

        elif initial_guess == "pfile":
            # p-file n_e; n_FC from d<n_FC>/dx = ne (Si+Scx)/(fFC |V_FC|) <n_FC>
            # integrated separatrix -> inner, evaluated only at mesh DOFs (x_dofs).
            ne_init_data = np.interp(x_dofs, self.x_init, self.n_e_pres)

            order_desc = np.argsort(x_dofs)[::-1]
            x_desc = x_dofs[order_desc]
            ne_desc = np.interp(x_desc, self.x_init, self.n_e_pres)
            Si_desc = np.interp(x_desc, self.x_init, self.S_i_pres)
            Scx_desc = np.interp(x_desc, self.x_init, self.S_cx_pres)
            integrand_init = (
                ne_desc * (Si_desc + Scx_desc) / (self.fFC * abs(self.V_FC))
            )
            cumint_desc = cumulative_trapezoid(integrand_init, x_desc, initial=0.0)
            nFC_on_desc = self.nFC_x0 * np.exp(cumint_desc)

            nFC_init_data = np.empty_like(x_dofs)
            nFC_init_data[order_desc] = nFC_on_desc

            ratio_CX = self.nCX_x0 / self.nFC_x0 if self.nFC_x0 > 0 else 0.0
            nCX_init_data = ratio_CX * nFC_init_data

            u.subfunctions[0].dat.data[:] = ne_init_data
            u.subfunctions[1].dat.data[:] = nFC_init_data
            u.subfunctions[2].dat.data[:] = nCX_init_data
            u_prev.assign(u)

        elif initial_guess == "tanh":
            # H-mode-pedestal-like tanh initial guess.
            #   s_ne(x)   = 0.5 * (1 - tanh((x - center) / (width/2)))
            #               -> 1 deep inside the pedestal, -> 0 at the separatrix
            #   n_e(x)    = ne_x0 + (ne_inner_val - ne_x0) * s_ne(x)
            # Neutrals get the mirrored shape (peaked at separatrix):
            #   s_neut(x) = 1 - s_ne(x)
            #   n_FC(x)   = nFC_x0 * s_neut(x)
            #   n_CX(x)   = nCX_x0 * s_neut(x)
            width = float(tanh_width) if tanh_width is not None else 0.1 * abs(self.x_inner)
            if width <= 0:
                raise ValueError(f"tanh_width must be positive, got {width}.")

            if ne_inner_bc == "dirichlet":  # ne_inner_val is known
                center = float(tanh_center) if tanh_center is not None else -width
                s_ne   = 0.5 * (1.0 - np.tanh((x_dofs - center) / (0.5 * width)))
                s_neut = 1.0 - s_ne
                ne_init_data  = self.ne_x0 + (ne_inner_val - self.ne_x0) * s_ne
                nFC_init_data = self.nFC_x0 * s_neut
                nCX_init_data = self.nCX_x0 * s_neut
            elif ne_inner_bc == "neumann":  # derive ne_inner_val from the slope
                if dne_dx_inner_val >= 0:
                    raise ValueError(
                        f"dne/dx(x_inner) = {dne_dx_inner_val:.3e} m^-4 must be "
                        "negative for Neumann boundary condition (n_e decreases "
                        "outward in the pedestal)."
                    )
                # default: x_inner sits one width inside the foot of the tanh
                # so sech^2 at x_inner is well-conditioned (~0.07)
                center = (float(tanh_center) if tanh_center is not None
                          else self.x_inner + width)
                arg = (self.x_inner - center) / (0.5 * width)
                sech2 = 1.0 / np.cosh(arg) ** 2
                if sech2 < 1e-6:
                    raise ValueError(
                        "tanh transition is too far from x_inner to match the "
                        "requested slope; reduce |center - x_inner| or increase width."
                    )
                ne_inner_val  = self.ne_x0 - dne_dx_inner_val * width / sech2
                self.ne_inner = ne_inner_val

                s_ne   = 0.5 * (1.0 - np.tanh((x_dofs - center) / (0.5 * width)))
                s_neut = 1.0 - s_ne
                ne_init_data  = self.ne_x0 + (ne_inner_val - self.ne_x0) * s_ne
                nFC_init_data = self.nFC_x0 * s_neut
                nCX_init_data = self.nCX_x0 * s_neut

            u.subfunctions[0].dat.data[:] = ne_init_data
            u.subfunctions[1].dat.data[:] = nFC_init_data
            u.subfunctions[2].dat.data[:] = nCX_init_data
            u_prev.assign(u)

        else:
            raise ValueError(
                f"Unknown initial_guess={initial_guess!r}; expected "
                f"'linear', 'pfile', or 'tanh'."
            )

        # D_KBM (Eqs. 24--25, pedestal-averaged alpha) from the initial n_e.
        # Newton: frozen after this step.
        self.calc_pressure_quantities(
            u.subfunctions[0].dat.data, average_alpha_pedestal=True, # this is n_e on x_dofs, which goes from separatrix to x_inner
        )
        C_ETG = np.interp(x_dofs, self.x_init, self.C_ETG)
        if v:
            print(f'D_ETG = {C_ETG/ne_init_data}\n\n')
            print(f'D_NEO = {self.D_NEO}\n\n')
            print(f'D_KBM = {self._D_KBM}\n\n')
        
        f = Function(V, name="C_ETG")
        f.dat.data[:] = C_ETG
        self._fd_cache["C_ETG_fd"] = f

        f = Function(V, name="D_NEO")
        f.dat.data[:] = np.interp(x_dofs, self.x_init, self.D_NEO) # self.D_NEO is on self.x_init grid
        self._fd_cache["D_NEO_fd"] = f

        f = Function(V, name="D_KBM")
        f.dat.data[:] = self._D_KBM
        self._fd_cache["D_KBM_fd"] = f

        f = Function(V, name="A_KBM")
        f.dat.data[:] = self._A_KBM
        self._fd_cache["A_KBM_fd"] = f

        f = Function(V, name="B_KBM")
        f.dat.data[:] = self._B_KBM
        self._fd_cache["B_KBM_fd"] = f

        # Frozen background T_e + T_i and d(T_e+T_i)/dx on x for Eq. A33
        T_e_pres = self.T_e_pres * 1.60218e-19 # J
        T_i_pres = self.T_i_pres * 1.60218e-19 # J
        T_tot_on_x = np.interp(x_dofs, self.x_init, T_e_pres + T_i_pres)
        dT_tot_on_x = np.interp(
            x_dofs, self.x_init, np.gradient(T_e_pres + T_i_pres, self.x_init),
        )

        f = Function(V, name="T_tot")
        f.dat.data[:] = T_tot_on_x
        self._fd_cache["T_tot_fd"] = f
        f = Function(V, name="dT_tot_dx")
        f.dat.data[:] = dT_tot_on_x
        self._fd_cache["dT_tot_dx_fd"] = f

        if v:
            self._plot_profiles(
                x_dofs=x_dofs,
                ne=u.subfunctions[0].dat.data,
                nFC=u.subfunctions[1].dat.data,
                nCX=u.subfunctions[2].dat.data,
                title=f"Initial guess: '{initial_guess}'",
            )

        # ------------------------------------------------------------------
        # Boundary conditions for the coupled (n_e, <n_FC>, <n_CX>) system.
        # IntervalMesh boundary IDs:  1 = left (x = x_inner),  2 = right (x = 0).
        #
        # The system is order 4 overall (n_e is 2nd order, <n_FC> and <n_CX>
        # are 1st order each) so it needs exactly 4 BCs.  Three of them are
        # always Dirichlet at the separatrix (where the physical values are
        # known):
        #
        #   n_e (x = 0)        = ne_x0
        #   <n_FC>(x = 0)      = nFC_x0
        #   <n_CX>(x = 0)      = nCX_x0
        #
        # x = 0 is also the *inflow* boundary for the first-order ODEs (because
        # V_{FC,r}, V_{CX,r} < 0 -- neutrals stream inward), so a single
        # Dirichlet at x = 0 is the well-posed choice for each neutral
        # species.  Putting them at x_inner instead would be ill-posed
        # (outflow boundary).
        #
        # The fourth BC -- on n_e at x_inner -- is the only one where there's
        # a real modelling choice: Neumann (match the core-side particle flux)
        # or Dirichlet (anchor to a known core-side density).
        #
        # NEUMANN vs DIRICHLET AT x_inner -- PROS AND CONS
        # -------------------------------------------------
        # Neumann  (dn_e/dx |_{x_inner} = dne_dx_inner):
        #   + Saarelma-consistent (assumption A7 in docs/derivation_eq{15,16}.tex).
        #     The pedestal-top density emerges from the model rather than being
        #     prescribed, so the solve is genuinely *predictive* for n_e(x_inner).
        #   + Physically matches the pedestal to the core-side particle flux
        #     Gamma_e ~ -D_ped <|grad r|^2> dn_e/dx, which is what you usually
        #     know (or believe) from the core transport / source modelling.
        #   - Sensitive to noise in the p-file gradient at x_inner.
        #   - Two Neumann BCs (one at each end) would be ill-posed; this
        #     setup uses Neumann at x_inner + Dirichlet at x=0, which is fine.
        #
        # Dirichlet (n_e(x_inner) = ne_inner):
        #   + Forces the solve to pass through a known pedestal-top density,
        #     useful when reproducing a specific experimental profile.
        #   + Robust to noise in the inner-boundary *gradient*.
        #   - No longer predictive at the pedestal top; you are inputting one
        #     of the things the model could otherwise tell you.
        #   - Inconsistent with how the parent solver's first_step / Eq.(15),(16)
        #     derivations close the inner boundary.
        #
        # If x_inner is deep enough that the p-file n_e value and the p-file
        # n_e gradient at that point are mutually consistent (and consistent
        # with the model), Neumann and Dirichlet converge to similar profiles.
        # ------------------------------------------------------------------
        ne_x0_c    = Constant(self.ne_x0)
        ne_inner_c = Constant(ne_inner_val)        # used only for Dirichlet
        nFC_x0_c   = Constant(self.nFC_x0)
        nCX_x0_c   = Constant(self.nCX_x0)
        dne_dx_inner_c = Constant(dne_dx_inner_val)  # used only for Neumann

        bcs = [
            DirichletBC(W.sub(0), ne_x0_c,  2),  # n_e (0)        -- separatrix
            DirichletBC(W.sub(1), nFC_x0_c, 2),  # <n_FC>(0)      -- separatrix
            DirichletBC(W.sub(2), nCX_x0_c, 2),  # <n_CX>(0)      -- separatrix
        ]
        if ne_inner_bc == "dirichlet":
            bcs.append(DirichletBC(W.sub(0), ne_inner_c, 1))  # n_e (x_inner)

        # Variational forms for each of the three equations
        #
        # Second-order diffusion for n_e: D = D_NEO + D_KBM + C_ETG/n_e (Plan A).
        # D_KBM from Eqs. (24)--(25) with pedestal-averaged alpha; updated each
        # Picard step, frozen for Newton from the initial guess.
        #
        # First-order ODE for <n_FC>.  We use the strong
        # residual form (no IBP) so the Galerkin solution converges to
        # the same profile that a forward marching solve from the inflow
        # boundary x = 0 would produce:
        #     int [ |V_FC| d/dx(f_FC g n_FC) - n_e (S_i + S_CX) n_FC ] v_F dx = 0.
        #
        # First-order ODE for <n_CX>, written symmetrically:
        #     int [ |V_CX| d/dx(f_CX g n_CX)
        #           - n_e ( S_i n_CX - (S_CX/2) n_FC ) ] v_C dx = 0,
        # ------------------------------------------------------------------
        C_ETG_fd = self._fd_cache["C_ETG_fd"]
        D_NEO_fd = self._fd_cache["D_NEO_fd"]
        D_KBM_fd = self._fd_cache["D_KBM_fd"]
        params_dict = {
            'W': W,
            'g_fd': g_fd,
            'C_ETG_fd': C_ETG_fd,
            'D_NEO_fd': D_NEO_fd,
            'D_KBM_fd': D_KBM_fd,
            'Si_fd': Si_fd,
            'Scx_fd': Scx_fd,
            'Vcx_fd': Vcx_fd,
            'VFC_const': VFC_const,
            'fFC_const': fFC_const,
            'fCX_const': fCX_const,
            'half': half,
            'u_prev': u_prev,
            'ne_inner_bc': ne_inner_bc,
            'dne_dx_inner_c': dne_dx_inner_c,
            'bcs': bcs,
            'u': u,
        }
        self.newton_solve(linear_solver=linear_solver, ksp_rtol=ksp_rtol, ksp_max_it=ksp_max_it, params_dict=params_dict, v=v)

        # extract the converged profiles from the Firedrake solution and save/return them
        ne_fd, nFC_fd, nCX_fd = u.subfunctions

        sort_x_idx  = np.argsort(x_dofs)
        self.x_sol   = x_dofs[sort_x_idx]
        self.ne_sol  = ne_fd .dat.data[sort_x_idx]
        self.nFC_sol = nFC_fd.dat.data[sort_x_idx]
        self.nCX_sol = nCX_fd.dat.data[sort_x_idx]

        self.u_fd = u
        self.W_fd = W
        self.V_fd = V

        return self.x_sol, self.ne_sol, self.nFC_sol, self.nCX_sol



    def fsa(self,A,flux_surfaces='T_e'):
        """Flux surface average a quantity as defined by ⟨A⟩= int(R^2Adθ)/ int(R^2dθ) in S. Saarelma et al 2023 Nucl. Fusion 63 052002

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        A : array
            2D array of A values at each R_grid, Z_grid.
        flux_surfaces : string
            which flux surface to average over, supporting: T_e, psi_N_pres
             
        """

        R_axis = self.eq['raxis']
        Z_axis = self.eq['zaxis']

        if flux_surfaces == 'T_e':
            psi_N_vals = self.psi_Te_eval
        elif flux_surfaces == 'psi_N_pres':
            psi_N_vals = self.psi_N_pres
        else:
            assert False, 'valid flux_surfaces method must be provided'

        A_clean = np.where(np.isfinite(A), A, 0.0) # replace nan values with 0.0
        A_spl = RectBivariateSpline(self.zgrid, self.rgrid, A_clean)
        fsa_A = np.full(len(psi_N_vals), np.nan)

        fig, ax = plt.subplots()
        for i, psi_val in enumerate(psi_N_vals):
            if psi_val <= 0.01 or psi_val >= 0.99:
                continue

            ax.cla()
            cs = ax.contour(self.rgrid, self.zgrid, self.psi_RZ_N,
                            levels=[psi_val])

            segs = cs.allsegs[0]
            # Closed contour around the axis = the real flux surface
            # (not open SOL/divertor legs or islands).
            seg = self._select_core_contour(segs, R_axis, Z_axis)
            if seg is None:
                continue  # stays NaN; callers interpolate over gaps
            R_c, Z_c = seg[:, 0], seg[:, 1] # R, Z coordinates of the contour

            # poloidal angle measured from the magnetic axis
            # R_c_ax = (((R_c - R_axis)**2) + ((Z_c - Z_axis)**2))**0.5
            # theta_c = np.arcsin( (Z_c - Z_axis) / R_c_ax )
            theta_c = np.arctan2(Z_c - Z_axis, R_c - R_axis) # theta at all points on contour
            theta_c, R_c, Z_c = self._sort_dedup_close_theta(theta_c, R_c, Z_c)

            A_c = A_spl(Z_c, R_c, grid=False)

            den = simpson(R_c**2, theta_c)
            if abs(den) < 1e-30:
                continue
            fsa_A[i] = simpson(R_c**2 * A_c, theta_c) / den

        plt.close(fig)

        return fsa_A

    def psi_rz_expand(self,A,psi_N_A='T_e'):
        """For A defined for each psi_N, expand to all R_grid, Z_grid.

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        A : array
            1D array of A values at each psi_N.
        psi_N_A : array
            1D array of psi_N values at which A is defined.

        Returns
        -------
        A_expanded : array
            2D array of A values at each R_grid, Z_grid.
             
        """

        # psi_N values at which A is defined
        if psi_N_A == 'T_e':
            psi_N_A = self.psi_Te_eval # 1D array of psi_N values at which T_e is evaluated
        else:
            assert False, 'valid psi_N_A method must be provided'

        # Interpolate: psi_N -> A, then evaluate on the 2D psi_N map
        A_interp = interp1d(psi_N_A, A, kind='linear',
                            bounds_error=False, fill_value=np.nan)
        return A_interp(self.psi_RZ_N)



    def calc_volavgP(self,x_ne,ne_pedestal,psiN_Te,Te_prev,EPEDNN_core='pfile',pres_gfile=False):
        """Calculate the volume-averaged pressure

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        x_ne : array
            x values at which the ne model is evaluated (ne is only in pedestal)
        ne_pedestal : array
            pedestal density profile in m^-3
        psiN_Te : array
            psi_N values at which the Te model is evaluated (Te is for the full plasma)
        Te_prev : array
            previous temperature profile in eV
        EPEDNN_core : string
            'pfile' 
            'pfile T, stiched ne'
            'previous T, stiched ne'
        pres_gfile : boolean
            if True, use the pressure from the g-file

        Sets
        ----
        self.volavgP : float
            Volume-averaged pressure (same units as self.pres).
        """

        if pres_gfile: 
            pressure = self.eq['pres']
            psi_N_plasma = self.psi_N_pres
        else:
            if EPEDNN_core == 'pfile': # always fixed to p-file n_e and T_e
                psi_N_plasma = self.psi_N_pres
                n_e_plasma = interp1d(self.psi_N_pres, self.n_e_pres, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_N_plasma)
                T_tot_plasma = interp1d(self.psi_N_pres, self.T_e_pres_pfile + self.T_i_pres_pfile, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_N_plasma)

            elif EPEDNN_core == 'pfile T, stiched ne': # pfile T_e and stiched n_e
                # Calculate core n_e
                psi_N_core = np.linspace(0, self.psi_N_inner_boundary, 75)
                n_e_core = interp1d(self.psi_ne_eval, self.n_e_pfile, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_N_core)
                
                # Calculate total n_e and T_e
                # psi_N_ped = interp1d(self.x_init, self.psi_N_pres, kind='linear', bounds_error=False, fill_value='extrapolate')(self.x_dofs_si)
                psi_N_ped = interp1d(self.x_init, self.psi_N_pres, kind='linear', bounds_error=False, fill_value='extrapolate')(x_ne)
                psi_N_plasma = np.concatenate([psi_N_core, psi_N_ped])
                n_e_plasma = np.concatenate([n_e_core, ne_pedestal])

                # x_dofs_si is in Firedrake DOF order (not spatial); sort to psi_N
                # before np.gradient (same issue as dpdx in update_alpha).
                sort_idx = np.argsort(psi_N_plasma)
                psi_N_plasma = psi_N_plasma[sort_idx]
                n_e_plasma = n_e_plasma[sort_idx]
                _, uniq_idx = np.unique(psi_N_plasma, return_index=True)
                psi_N_plasma = psi_N_plasma[uniq_idx]
                n_e_plasma = n_e_plasma[uniq_idx]

                T_tot_plasma = interp1d(self.psi_N_pres, self.T_e_pres + self.T_i_pres, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_N_plasma)

            elif EPEDNN_core == 'previous T, stiched ne': # previous T_e and stiched n_e
                # Calculate core n_e
                psi_N_core = np.linspace(0, self.psi_N_inner_boundary, 75)
                n_e_core = interp1d(self.psi_ne_eval, self.n_e_pfile, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_N_core)
                
                # Calculate total n_e and T_e
                # psi_N_ped = interp1d(self.x_init, self.psi_N_pres, kind='linear', bounds_error=False, fill_value='extrapolate')(self.x_dofs_si)
                psi_N_ped = interp1d(self.x_init, self.psi_N_pres, kind='linear', bounds_error=False, fill_value='extrapolate')(x_ne)
                psi_N_plasma = np.concatenate([psi_N_core, psi_N_ped])
                n_e_plasma = np.concatenate([n_e_core, ne_pedestal])

                # x_dofs_si is in Firedrake DOF order (not spatial); sort to psi_N
                # before np.gradient (same issue as dpdx in update_alpha).
                sort_idx = np.argsort(psi_N_plasma)
                psi_N_plasma = psi_N_plasma[sort_idx]
                n_e_plasma = n_e_plasma[sort_idx]
                _, uniq_idx = np.unique(psi_N_plasma, return_index=True)
                psi_N_plasma = psi_N_plasma[uniq_idx]
                n_e_plasma = n_e_plasma[uniq_idx]

                if self.T_rat_flag:
                    Ti_prev = Te_prev * self.T_rat
                else:
                    raise ValueError('T_rat_flag must be True if T_rat is provided')
                T_tot_plasma = interp1d(psiN_Te, Te_prev + Ti_prev, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_N_plasma)

            elif EPEDNN_core == 'previous T, varying ne': # stiched T_e and varyped outputted n_e
                raise NotImplementedError('previous T, varying ne is not yet implemented')

            else:
                assert False, 'EPEDNN_betan method not supported'

                # In the future, the code below will be stitched with a core simulation
                """
                T_tot_xdofs = self.T_e_xdofs
                if self.T_rat_flag: # use both electron and ion temperatures, neglect neutrals
                    T_i_xdofs = np.interp(self.x_sol, self.x_init, self.T_i_pres)
                    T_tot_xdofs = T_tot_xdofs + T_i_xdofs
                ne = self.ne_sol + core_n_e
                pressure = self.ne_sol * T_tot_xdofs * 1.60218e-19  # Pa, assuming quasi-neutrality

                psi_N_xdofs = np.interp(self.x_sol, self.x_init, self.psi_N_pres) # convert x_dofs to psi_N
                pressure = np.interp(self.psi_N_pres, psi_N_xdofs, pressure) # Pa on psi_N_pres grid

                self.volavgP = (simpson(pressure * dV_dpsi, self.psi_N_pres)
                                / simpson(dV_dpsi, self.psi_N_pres))
                """

            pressure = (n_e_plasma * T_tot_plasma) * constants.e # Pa

        # Calculate pressure and volavgP
        V_full_plasma = interp1d(self.psi_N_pres, self.V_plasma, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_N_plasma)
        dV_dpsi = np.gradient(V_full_plasma, psi_N_plasma)
        self.volavgP = (simpson(pressure * dV_dpsi, psi_N_plasma)
                        / simpson(dV_dpsi, psi_N_plasma))

    def calc_betan(self,x_ne,ne_pedestal,psiN_Te,Te_prev,EPEDNN_core='pfile',pres_gfile=False):
        """Calculate the normalized beta

        Parameters
        ----------
        self : object
            instance of saarelma_connor class

        Sets
        -------
        self.Betan : float
            Normalized beta, dimensionless
        """

        self.calc_volavgP(x_ne,ne_pedestal,psiN_Te,Te_prev,EPEDNN_core,pres_gfile)

        _, [B_R, B_Z, _] = self.calc_B(self.eq['rzout'][:, 0], self.eq['rzout'][:, 1])
        bp_lcfs = np.sqrt(B_R**2 + B_Z**2)
        # bp_avg = np.mean(bp_lcfs)

        betat = self.volavgP / (self.bt**2 / (2 * self.mu0))
        self.betat = betat
        # betap = self.volavgP / (bp_avg**2 / (2 * self.mu0))
        # beta = ((1/betat) + (1/betap))**(-1)

        # EPED / Troyon: β_N = β_t[%] * a * abs(B_t) [T] / I_p[MA] -> this is what OpenFUSIONToolkit uses for β_N
        if self.verbose:
            print(f'betat: {betat}, a: {self.a}, bt: {self.bt}, Ip: {self.Ip}')
        self.betan = 100 * betat * (self.a * abs(self.bt) / abs(self.Ip))

    def setup_epednn(self, model='EPED1'):
        """Setup the EPEDNN model with quantities from the Saarelma-Connor setup

        Parameters
        ----------
        self : object
            instance of saarelma_connor class

        Returns
        -------
        pedestal_pressure : float
            Pedestal pressure (MPa)
        pedestal_width : float
            Pedestal width (normalized poloidal flux)
        """

        print("Setting up EPEDNN...")

        if model == 'EPED1':
            # Requires dependency "juliacall" to translate Python inputs to FUSE EPED.jl
            # Requires dependency EPEDNN

            import juliapkg

            # 1. Tell juliapkg to add your local EPEDNN package in development mode.
            # This registers it with the isolated Julia environment PythonCall uses.
            epednn_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "dependencies",
                "EPEDNN.jl",
            )
            if not os.path.isdir(epednn_path):
                raise FileNotFoundError(
                    f"EPEDNN.jl not found at {epednn_path}. "
                    "Initialize it with: git submodule update --init dependencies/EPEDNN.jl"
                )
            juliapkg.add(
                "EPEDNN",
                uuid="e64856f0-3bb8-4376-b4b7-c03396503991",
                path=epednn_path,
                dev=True,
            )

            # 2. Resolve and instantiate. THIS is what fixes your missing dependency error!
            juliapkg.resolve()

            # 3. Now that the environment is set up, load juliacall and your package
            from juliacall import Main as jl
            jl.seval('using EPEDNN')

            # from juliacall import Main as jl

            # 1. Load the Julia EPEDNN module (Assuming EPEDNN is already installed in your Julia environment)
            '''
                # To install EPEDNN, run the following command in your terminal:
                conda activate sc_ped
                cd /Users/nelsonlab/codes/saarelma-conner-ped
                julia

                # if you need to install julia, run the following command in your terminal:
                curl -fsSL https://install.julialang.org | sh
                # then restart terminal
                julia --version
                # if this doesn't work, you could try the following although I did not verify this works:
                echo 'export PATH="$HOME/.juliaup/bin:$PATH"' >> ~/.zshrc
                source ~/.zshrc
                julia --version

                # Then in Julia:
                using Pkg
                Pkg.activate(".")  # optional but recommended: use this repo as the active Julia project
                Pkg.develop(path="dependencies/EPEDNN.jl")
                Pkg.instantiate()

                #Then in Julia: 
                using EPEDNN


                Make sure that EPEDNN submodule is installed, the General Julia package registry is installed, and juliacall and juliapkg are installed.
                The following commands may help:

                pip install juliapkg
                git submodule update --init dependencies/EPEDNN.jl
                # Only needed if juliapkg fails on registry download:
                git clone --depth 1 https://github.com/JuliaRegistries/General.git \
                ~/.julia/registries/General
                pip install juliacall
            '''
            # you can run Julia commands in Python using jl.seval('command')
            # jl.seval('using Pkg')
            # jl.seval('Pkg.activate(".")  # optional but recommended: use this repo as the active Julia project')
            # jl.seval('Pkg.develop(path="/Users/nelsonlab/codes/saarelma-conner-ped/dependencies/EPEDNN.jl")')
            # jl.seval('Pkg.instantiate()')
            # jl.seval('using EPEDNN')

            # 2. Load the pre-trained EPED neural network model
            # This mimics the EPEDNN.loadmodelonce("EPED1NNmodel.bson") step
            model_filename = "EPED1NNmodel.bson" 
            self.epednn_model = jl.EPEDNN.loadmodelonce(model_filename)
        elif model == 'EPED_SPARC':
            # Vendored package lives at dependencies/epednn_mit/src/epednn_mit/;
            # it is not installed into the env by default, so put src/ on sys.path.
            import sys
            from pathlib import Path
            epednn_root = Path(__file__).resolve().parent.parent / "dependencies" / "epednn_mit"
            epednn_src = epednn_root / "src"
            if not epednn_src.is_dir():
                raise FileNotFoundError(
                    f"epednn_mit not found at {epednn_src}. "
                    "Expected dependencies/epednn_mit/src in the repo."
                )
            if str(epednn_src) not in sys.path:
                sys.path.insert(0, str(epednn_src))
            from epednn_mit.models.sparc.tensorflow_model import generate_epednn_mit_sparc_tensorflow
            from epednn_mit.utils.load import load_weights
            weights_dir = epednn_src / "epednn_mit" / "models" / "sparc"
            weights = load_weights(sorted(weights_dir.glob("*sparc*.pkl")))
            if not weights:
                raise FileNotFoundError(f"No *sparc*.pkl weights found in {weights_dir}")
            self.epednn_model = generate_epednn_mit_sparc_tensorflow(weights)

        self.bt = np.array(self.calc_B(self.eq['raxis'],self.eq['zaxis'])[1][2])
        # print(f'bt: {self.bt}')


    def feed_epednn(self, model='EPED1', ne_ped=None, x_ne=None, psiN_Te=None, Te_prev=None, EPEDNN_core='pfile', pres_gfile=False):
        """Feed the Saarelma-Connor solution to the EPEDNN model"""
        
        # Define inputs in Python
        # These map exactly to the InputEPED struct we saw in the Julia code
        if ne_ped is None:
            self._neped = interp1d(self.x_sol, self.ne_sol, kind='linear', bounds_error=False, fill_value='extrapolate')(self.x_inner) / (1e19) # m^-3 -> 10^19 m^-3
            self._x_ne = self.x_sol
        else:
            self._neped = ne_ped
            if x_ne is None:
                print('Warning: Most functionalities require x_ne to be provided if ne_ped is provided')
                # assert False, 'x_ne must be provided if ne_ped is provided'
                self._x_ne = None
            else:
                self._x_ne = x_ne
        self.calc_betan(self._x_ne,self._neped,psiN_Te,Te_prev,EPEDNN_core,pres_gfile)
        print(f'betan: {self.betan}')

        if self._x_ne is None:
            ne_ped_h = self._neped
        else:
            ne_ped_h = interp1d(self._x_ne, self._neped, kind='linear', bounds_error=False, fill_value='extrapolate')(self.x_inner) / (1e19) # m^-3 -> 10^19 m^-3

        inputs = {
            "a": float(self.a),           # Minor radius (m)
            "betan": float(self.betan[0]),       # Normalized beta
            "bt": float(abs(self.bt[0])),                # Toroidal magnetic field at the magnetic axis (T)
            "delta": float(self.delta),       # Effective triangularity
            "ip": float(abs(self.Ip)),          # Plasma current (MA)
            "kappa": float(self.kappa),       # Elongation
            "m": float(self.M_eff),           # Effective mass (must be 2.0 for D or 2.5 for D-T)
            "neped": float(ne_ped_h),       # Pedestal density (in 10^19 m^-3)
            "r": float(self.Rmajor),           # Major radius (m)
            "zeffped": float(self.Z_i)      # Effective charge
        }

        if model == 'EPED1':
            # Call the Julia model using the Python inputs
            # We pass the inputs into the Julia function, along with the keyword arguments
            solution = self.epednn_model(
                inputs["a"], 
                inputs["betan"], 
                inputs["bt"], 
                inputs["delta"],
                inputs["ip"], 
                inputs["kappa"], 
                inputs["m"], 
                inputs["neped"],
                inputs["r"], 
                inputs["zeffped"],
                only_powerlaw=False,        # Set to True if you only want the scaling law
                warn_nn_train_bounds=True   # Warns if inputs are outside the training data. Good for debugging
            )

            # Extract the results back into Python
            # The solution structure has pressure and width for different modes (GH, G, H)
            self.pedestal_pressure = solution.pressure.GH.H  # in MPa
            self.pedestal_width = solution.width.GH.H        # in normalized poloidal flux


        elif model == 'EPED_SPARC':
            ''' Training dataset was on (in order of input position from the EPEDNN_MIT README): 
            Ip:     [  1.6  , 14.3   ]
            Bt:     [  7.2  , 12.2   ]
            R:      [  1.85 ,  1.85  ]
            a:      [  0.57 ,  0.57  ]
            kappa:  [  1.53 ,  2.29  ]
            delta:  [  0.39 ,  0.59  ]
            neped:  [  2.84 , 90.235 ]
            betan:  [  0.8  ,  1.6   ]
            zeff:   [  1.3  ,  2.5   ]
            '''
            if (inputs["bt"] - 12.2 < 0.5) and (inputs["bt"] > 12.2):
                print('Warning: bt is close to 12.2 but is greater than 12.2, setting bt to 12.2')
                inputs["bt"] = 12.2
            if (inputs["a"] - 0.57 < 1e-3) and (inputs["a"] != 0.57):
                print('Warning: a is close to 0.57, setting a to 0.57 or else EPEDNN behaves badly')
                inputs["a"] = 0.57
            if (inputs["r"] - 1.85 < 1e-3) and (inputs["r"] != 1.85):
                print('Warning: r is close to 1.85, setting r to 1.85 or else EPEDNN behaves badly')
                inputs["r"] = 1.85
            
            x = np.atleast_2d([
                inputs["ip"], 
                inputs["bt"], 
                inputs["r"], 
                inputs["a"], 
                inputs["kappa"], 
                inputs["delta"], 
                inputs["neped"], 
                inputs["betan"], 
                inputs["zeffped"]
            ])
            solution = self.epednn_model.predict(x)[0]  # [[ped_height, ped_width]]
            print(solution)
            self.pedestal_pressure = solution[0] / 1000     # in MPa -> kPa
            self.pedestal_width = solution[1]              # in normalized poloidal flux

        # Apply ELM-free regime scaling
        if self.regime_flag == 'PT H-mode':
            pass
        elif self.regime_flag == 'NT':
            self.pedestal_pressure = self.pedestal_pressure * self.NT_scaling
            raise NotImplementedError('NT ELM-free regime scaling is not yet implemented')
        else:
            assert False, 'specified regime_flag not supported'

        return self.pedestal_pressure, self.pedestal_width, self.betan