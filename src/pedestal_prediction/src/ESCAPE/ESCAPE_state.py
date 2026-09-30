import numpy as np
from scipy import constants
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from matplotlib.path import Path as _MplPath


class ESCAPE_state:
    """

    Description
    ----------
    Initialize and hold state information in ESCAPE model. State includes:
    - All equilibrium information.
    - Density profiles and corresponding psi_n grids.
    - Temperature profiles and corresponding psi_n grids.
    - Global ESCAPE constants.
    - Global ESCAPE flags.


    Parameters
    ----------
    Z_i : int
        Z of ions
    P_tot_e : float
        Total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox, W
    ne_x0 : float
        m^-3, electron density at the separatrix (boundary condition)
    equil_params : dict
        Dictionary of required equilibrium parameters
    kprof_params : dict
        Dictionary of required kinetic profile parameters
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
    verbose : bool
        True if verbose output is desired, False if verbose output is not desired
    """
    def __init__(
        self,
        Z_i = 1, # Z of ions
        Z_eff = 1, # effective Z in plasma
        P_tot_e = None, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox
        ne_x0 = None, # m^-3, electron density at the separatrix (boundary condition, default is to use from profiles)        
        equil_params = None, # dictionary of required equilibrium parameters
        kprof_params = None, # dictionary of required kinetic profile (density, temperature) parameters
        quasineutral_flag = True,
        T_rat_flag = True, # True if using a temperature ratio between ions and electrons, False if doing something else
        T_rat = 1,
        pol_norm = False, # True for when the poloidal flux is not normalized by 2pi. COCOS 7 convention is pol_norm=False, so poloidal flux is normalized by 2pi
        species = 'D', # species of ions, currently supporting: D, D-T
        regime_flag = 'PT H-mode', # regime of the plasma, currently supporting: 'PT H-mode', 'NT'
        verbose = False,
    ):

        # User-specified flags
        self.regime_flag = regime_flag
        self.T_rat_flag = T_rat_flag
        self.T_rat = T_rat
        self.verbose = verbose
        self.pol_norm = pol_norm
        self.species = species
        self.quasineutral_flag = quasineutral_flag

        # Other constants
        self.E_FC = 3 * 1.60218e-19 # J, Energy of Franck-Condon neutrals as defined in Mahdavi M.A., Maingi R., Groebner R.J., Leonard A.W., Osborne T.H. and Porter G. 2003 Phys. Plasmas 10 3984 J
        self.mu0 = 4 * np.pi * 10**-7 # N/A**2, vacuum magnetic permeability constant
        self.P_tot_e = P_tot_e
        self.M_e = 9.109e-31 # kg, mass of electron
        self.M_i = 1.673e-27 # kg, mass of hydrogen nuclei
        if species == 'D':
            self.M_eff = 2.0
        elif species == 'D-T':
            self.M_eff = 2.5
        else:
            assert False, 'species must be D or D-T'

        # Load in equilibrium and kinetic profile quantities
        self.set_equil_params(eq=equil_params)
        self.set_kprof_params(kprof_params=kprof_params)
        
        # Calculate the magnetic field at each RZ grid point, sets self.B
        self.calc_B(self.rgrid,self.zgrid)

        # calculate the flux surface-averaged |grad(r)| and |grad(r)|^2 and some other quantities like r_psi (outboard midplane minor radius for each flux surface)
        self.calc_gradr() # only a function of geometry

        self.Z_i = Z_i
        self.Z_eff = Z_eff
        self.e_i = Z_i * constants.e # C
        self.k_B = 1.38064852e-23 # J/K, Boltzmann constant

        # Outer Dirichlet boundary condition for electrons
        if ne_x0 is None:
            self.ne_x0 = self.n_e[-1]
            self.ne_x0_manual = False
        else:
            self.ne_x0 = ne_x0
            self.ne_x0_manual = True


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

    def set_equil_params(self,eq):
        self.eq = eq

        bdry = self.find_boundary_points(eq=eq)

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

    def set_kprof_params(self,kprof_params):
        """
        
        Description
        -----------
        Sets the provided profile in kprof_params (could be density, temperature, or both).
        Called once from __init__ and again whenever the profiles are updated (e.g. each
        ESCAPE iteration). Profiles not in kprof_params keep their current values, except
        that T_i follows T_e (T_rat_flag) and n_i follows n_e (quasineutral_flag) when
        they are not given explicitly.

        Optional neutral profiles (3D SC solver): 'nFC'/'psin_nFC' and 'nCX'/'psin_nCX'
        (m^-3 on psi_N). If absent they are left unchanged, or initialized empty on the
        first call.
        """

        #-------- electron information (required) --------#
        if 'T_e' in kprof_params and 'psin_Te' in kprof_params:
            self.T_e = kprof_params['T_e'] # keV
            self.psi_Te_eval = kprof_params['psin_Te'] # psi_N values at which T_e is evaluated
            self.T_e_K = self.T_e * 1e3 * 11604.52 # T_e values (K) evaluated at psi_Te_eval
        if 'n_e' in kprof_params and 'psin_ne' in kprof_params:
            self.n_e = kprof_params['n_e'] # m^(-3)
            self.psi_ne_eval = kprof_params['psin_ne'] # psi_N values at which n_e is evaluated


        #-------- ion information (not required) --------#

        # ion temperatures
        if 'T_i' in kprof_params and 'psin_Ti' in kprof_params:
            self.T_i = kprof_params['T_i'] # keV
            self.psi_Ti_eval = kprof_params['psin_Ti']
            self.T_i_K = self.T_i * 1e3 * 11604.52 # K
        elif self.T_rat_flag:
            self.T_i = self.T_e * self.T_rat # keV
            self.psi_Ti_eval = self.psi_Te_eval
            self.T_i_K = self.T_i * 1e3 * 11604.52 # K
        elif not hasattr(self, 'T_i'):
            raise NotImplementedError("if T_i is not provided, must specify T_rat_flag")

        # ion densities
        if 'n_i' in kprof_params and 'psin_ni' in kprof_params:
            self.n_i = kprof_params['n_i'] # m^(-3)
            self.psi_ni_eval = kprof_params['psin_ni']
        elif self.quasineutral_flag:
            self.n_i = self.n_e # m^(-3)
            self.psi_ni_eval = self.psi_ne_eval
        elif not hasattr(self, 'n_i'):
            raise NotImplementedError("if n_i is not provided, must use quasi-neutrality")


        #-------- neutral information (only required for 3D SC solver) --------#
        if 'nFC' in kprof_params and 'psin_nFC' in kprof_params:
            self.nFC = kprof_params['nFC'] # m^(-3)
            self.psi_nFC_eval = kprof_params['psin_nFC']
        elif not hasattr(self, 'nFC'): # initialize
            self.nFC = np.array([])
            self.psi_nFC_eval = np.array([])
        if 'nCX' in kprof_params and 'psin_nCX' in kprof_params:
            self.nCX = kprof_params['nCX'] # m^(-3)
            self.psi_nCX_eval = kprof_params['psin_nCX']
        elif not hasattr(self, 'nCX'): # initialize
            self.nCX = np.array([])
            self.psi_nCX_eval = np.array([])
