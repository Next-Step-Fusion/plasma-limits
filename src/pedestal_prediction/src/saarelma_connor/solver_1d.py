"""Non-dimensional Firedrake and scipy solvers for the original (implicit)
Saarelma-Connor pedestal model ("1D").

Full documentation: docs/solver_1d_documentation.tex.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.integrate import cumulative_trapezoid, solve_bvp

try:
    from firedrake import (
        IntervalMesh, FunctionSpace, Function, TestFunction,
        Constant, DirichletBC, dx, ds, solve, assemble, SpatialCoordinate,
    )
    _FIREDRAKE_AVAILABLE = True
    _FIREDRAKE_IMPORT_ERR = None
except Exception as _firedrake_import_err:
    _FIREDRAKE_AVAILABLE = False
    _FIREDRAKE_IMPORT_ERR = _firedrake_import_err

from src.saarelma_connor import bc_ig_helpers as bcig


# Conversion constant (eV -> J)
_EV2J = 1.60218e-19


class OneDSolverMixin:
    """Mixin adding the original implicit Saarelma-Connor n_e-only solver
    (report Eqs. 6-7) to saarelma_connor. The attributes set after a solve are
    listed in docs/solver_1d_documentation.tex.
    """

    # ------------------------------------------------------------------
    # Implementation-shared setup
    # ------------------------------------------------------------------

    def _check_free_params_sc(self):
        """Backwards-compatible alias for :meth:`solver.check_free_params`."""
        self.check_free_params("1D")

    def _ensure_sc_setup(self, x_res, free_params=None, force=False):
        """Apply free parameters, build the solver grids, form factors and
        C_ETG, locate the inner boundary, and set the non-dim reference scales.
        """
        if not hasattr(self, "_fd_cache"):
            self._fd_cache = {}

        self.apply_free_params(free_params, model="1D")

        # form_factor + setup_solver_grids (cached per x_res), exactly as
        # the parent's coupled solver does, then the ETG coefficient
        # (which depends on the free parameter De_chie_etg).
        self._ensure_firedrake_coefficient_grids(x_res, force=force)
        self.construct_C_ETG()

        # Inner boundary location (same logic as solver_nondim).
        if self.psi_N_inner_boundary is None:
            if not hasattr(self, "_D_KBM"):
                # find_inner_boundary needs a D_KBM estimate; a zeroed
                # array (KBM off) is the conservative pre-solve choice.
                self._D_KBM = np.zeros_like(self.x_init)
            self.find_inner_boundary()
        else: # should mostly be using this branch
            self.x_inner = float(np.interp(
                self.psi_N_inner_boundary, self.psi_N_pres, self.x_init
            ))

        if float(self.x_inner) >= 0.0:
            raise ValueError(
                f"x_inner = {self.x_inner} must be strictly less than 0 "
                "(separatrix)."
            )

        # Reference scales.
        self._L_sc = float(abs(self.x_inner))
        self._n0_sc = float(self.ne_x0)
        S0 = float(self.S_i_pres[-1])
        if not np.isfinite(S0) or S0 <= 0.0:
            raise RuntimeError(
                f"S_i(0) = {S0!r} m^3/s is not a usable separatrix "
                "ionisation rate; cannot non-dimensionalise."
            )
        self._S0_sc = S0
        self._V0_sc = self._L_sc * self._n0_sc * S0          # m/s
        self._D0_sc = (self._L_sc ** 2) * self._n0_sc * S0   # m^2/s

    def _shoot_outer_grad_sc(
        self, F, N, bcs, snes_params,
        slope_c, target, tol, max_it, verbose, tag,
    ):
        """Secant-iterate the free ds(1) slope ``slope_c`` until N'(0) ==
        ``target``, restarting each attempt from the same reference state;
        returns a diagnostics dict.
        """
        grad_sep_form = N.dx(0) * ds(2)   # ds(2) has unit measure in 1D
        scale = max(abs(target), 1e-30)   # tol is relative to the target
        history = []
        n_solves = 0
        n_backtracks = 0

        # Fixed restart state, and SNES settings that stop it exiting on
        # the step-size test before it has done any work.
        ref = N.copy(deepcopy=True)
        shoot_params = dict(snes_params)
        shoot_params["snes_stol"] = 0.0

        def _attempt(s):
            """Solve at slope ``s`` from the reference; return (r, ok)."""
            nonlocal n_solves
            N.assign(ref)
            slope_c.assign(s)
            n_solves += 1
            try:
                solve(F == 0, N, bcs=bcs, solver_parameters=shoot_params)
            except Exception:
                N.assign(ref)         # discard the failed line search
                return None, False
            return float(assemble(grad_sep_form)) - target, True

        s = float(slope_c)
        r, ok = _attempt(s)
        if not ok:
            raise RuntimeError(
                f"[{tag} grad-bc] the nonlinear solve failed at the seed "
                f"slope N'(-1) = {s:.3e}; the separatrix-gradient mode has "
                "nothing to iterate from.  Try a different initial_guess "
                "or grad_bc_seed (which seeds the search)."
            )

        s_prev = r_prev = None
        step_prev = None
        converged = False
        for it in range(1, int(max_it) + 1):
            history.append({"iteration": it, "slope": s, "residual": r})
            if verbose:
                print(f"[{tag} grad-bc] it {it:3d}: N'(-1) = {s:.6e}, "
                      f"N'(0) - target = {r:.3e}")
            if abs(r) <= tol * scale:
                converged = True
                break

            if s_prev is None:
                # Bootstrap the secant with a finite-difference step.
                step = 0.05 * abs(s) if s != 0.0 else 1e-3
            else:
                denom = r - r_prev
                if denom == 0.0:
                    raise RuntimeError(
                        f"[{tag} grad-bc] secant iteration stalled: the "
                        "separatrix gradient did not respond to a change "
                        "in the inner flux."
                    )
                step = -r * (s - s_prev) / denom
                if step_prev is not None and abs(step) > 4.0 * abs(step_prev):
                    step = np.sign(step) * 4.0 * abs(step_prev)

            # Backtrack while SNES fails.
            ok = False
            for _ in range(12):
                r_new, ok = _attempt(s + step)
                if ok:
                    break
                n_backtracks += 1
                step *= 0.5
            if not ok:
                raise RuntimeError(
                    f"[{tag} grad-bc] the nonlinear solve kept failing while "
                    f"searching for the separatrix gradient (last good slope "
                    f"N'(-1) = {s:.3e}, residual {r:.3e}).  Eq. (7) may have "
                    "no solution near the slope this condition demands; try "
                    "picard_relax < 1 or a coarser grad_bc_tol."
                )

            s_prev, r_prev = s, r
            s, r = s + step, r_new
            step_prev = step

        if not converged:
            raise RuntimeError(
                f"[{tag} grad-bc] the separatrix-gradient condition did not "
                f"converge in {max_it} iterations (last relative residual "
                f"{abs(history[-1]['residual']) / scale:.3e}, tolerance "
                f"{tol:.3e}); consider raising grad_bc_max_it or relaxing "
                f"grad_bc_tol."
            )
        slope_c.assign(s)
        return {
            "converged":   converged,
            "iterations":  len(history),
            "solves":      n_solves,
            "backtracks":  n_backtracks,
            "slope":       s,
            "rel_residual": abs(r) / scale,
            "history":     history,
        }

    def calc_D_KBM_average_sc(self, n_e_ped, x_ped):
        """KBM diffusivity on x_init from the pedestal-averaged Connor-Hastie
        alpha of ``n_e_ped`` on ``x_ped``, gated as a whole (Saarelma Eqs.
        24-25). Also sets alpha_bar_ped, alpha_frac_above, kbm_gate_on,
        alpha_local_ped and _D_KBM.
        """
        x_ped = np.asarray(x_ped, dtype=float)
        n_e_ped = np.asarray(n_e_ped, dtype=float)

        T_e = np.interp(x_ped, self.x_init, self.T_e_pres)   # eV
        T_i = np.interp(x_ped, self.x_init, self.T_i_pres)   # eV
        alpha_nodp = np.interp(x_ped, self.x_init, self._alpha_nodp_xinit)

        # Pa; neutral pressures neglected, quasi-neutral plasma assumed.
        _pres = n_e_ped * (T_e + T_i) * _EV2J
        dpdx = np.gradient(_pres, x_ped)                     # Pa/m
        _alpha = alpha_nodp * dpdx                           # dimensionless

        alpha_bar = float(np.mean(_alpha))
        gate = alpha_bar > self.alpha_crit
        self.alpha_bar_ped = alpha_bar
        self.alpha_frac_above = float(np.mean(_alpha > self.alpha_crit))
        self.kbm_gate_on = bool(gate)
        self.alpha_local_ped = _alpha
        self.pres_ped = _pres

        G_KBM_xinit = self.C_KBM * (self.c_s * self.rho_s ** 2) / self.a
        D_KBM_xinit = np.where(
            gate, (alpha_bar - self.alpha_crit) * G_KBM_xinit, 0.0
        )
        self._D_KBM = D_KBM_xinit
        return D_KBM_xinit

    # ------------------------------------------------------------------
    # FC attenuation kernel  E(x)  (exponent of Saarelma Eq. 11)
    # ------------------------------------------------------------------

    def _exp_kernel_sc(self, n_e, x):
        """FC attenuation kernel E(x) = exp[int_0^x n_e (S_i + S_CX) / (f_FC
        |V_FC|) dx'] on the ascending grid ``x``.
        """
        x = np.asarray(x, dtype=float)
        n_e = np.asarray(n_e, dtype=float)
        Si = np.interp(x, self.x_init, self.S_i_pres)
        Scx = np.interp(x, self.x_init, self.S_cx_pres)
        fFC = np.interp(x, self.x_init, self.fFC)
        integrand = n_e * (Si + Scx) / (fFC * abs(self.V_FC))

        # int_0^x with x < 0: integrate on a descending ordering
        # (separatrix -> core) so the cumulative integral starts at 0.
        order_desc = np.argsort(x)[::-1]
        cum_desc = cumulative_trapezoid(
            integrand[order_desc], x[order_desc], initial=0.0
        ) # results in a negative value for the sum, since integrand should be positive and integrating from 0 to x < 0
        integral_from_0 = np.empty_like(cum_desc)
        integral_from_0[order_desc] = cum_desc
        return np.exp(integral_from_0)

    def _C_cx_sc(self, x):
        """CX-flux coefficient C_CX(x) of report Eq. (7) on ``x`` (m)."""
        x = np.asarray(x, dtype=float)
        Si = np.interp(x, self.x_init, self.S_i_pres)
        Scx = np.interp(x, self.x_init, self.S_cx_pres)
        Vcx = np.interp(x, self.x_init, np.abs(self.V_cx_pres))
        fFC = np.interp(x, self.x_init, self.fFC)
        fCX = np.interp(x, self.x_init, self.fCX)
        return 1.0 - (abs(self.V_FC) * fFC / (Vcx * fCX)) * (
            (Si + 0.5 * Scx) / (Si + Scx)
        )

    def _post_solve_neutrals_sc(self):
        """Reconstruct <n_FC> and <n_CX> (m^-3) on ``self.x_sol`` from the
        converged n_e via Saarelma Eqs. (11)-(12).
        """
        x = np.asarray(self.x_sol, dtype=float)
        n_e = np.asarray(self.ne_sol, dtype=float)
        dne_dx = np.asarray(self.dne_dx_sol, dtype=float)

        nFC = self.nFC_x0 * self._exp_kernel_sc(n_e, x)

        g = np.interp(x, self.x_init, self.gradr2_fsa)
        D_ped = (np.interp(x, self.x_init, self.D_NEO + self._D_KBM)
                 + np.interp(x, self.x_init, self.C_ETG) / n_e)
        Vcx = np.interp(x, self.x_init, np.abs(self.V_cx_pres))
        fCX = np.interp(x, self.x_init, self.fCX)

        flux_term = -(g * D_ped / (Vcx * fCX)) * (dne_dx - self.dne_dx_neginf)
        fc_term = (self._C_cx_sc(x) - 1.0) * nFC
        nCX = np.maximum(flux_term + fc_term, 0.0)
        return nFC, nCX

    # ------------------------------------------------------------------
    # Picard bookkeeping shared by both implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _check_eq6_form_sc(eq6_form):
        """Validate and normalise the ``eq6_form`` switch."""
        eq6_form = str(eq6_form).lower()
        if eq6_form not in ("complete", "paper"):
            raise ValueError(
                "eq6_form must be 'complete' (docs/derivation_eq16.tex "
                "Eq. 16-full) or 'paper' (report Eq. 6 / Saarelma Eq. 16 "
                f"as printed), got {eq6_form!r}."
            )
        if eq6_form == "paper":
            import warnings
            warnings.warn(
                "eq6_form='paper' is the report Eq. 6 / Saarelma Eq. 16 "
                "as printed, which drops the shape factor "
                "1 + S_CX/(2 S_i) and cancels <|grad r|^2> against D_ped.  "
                "On the DIII-D 158091 case this makes it ~35x stiffer than "
                "the complete form (docs/derivation_eq16.tex Eq. 16-full) "
                "and no implementation can solve it in double precision.  "
                "Use eq6_form='complete' instead.",
                RuntimeWarning, stacklevel=3
            )
        return eq6_form

    @staticmethod
    def _check_first_step_sc(first_step):
        """Validate and normalise the ``first_step`` switch."""
        first_step = str(first_step).lower()
        if first_step not in ("auto", "eq6", "skip"):
            raise ValueError(
                "first_step must be 'auto', 'eq6', or 'skip', got "
                f"{first_step!r}."
            )
        return first_step

    def _first_step_failed_sc(self, tag, first_step, err):
        """Handle a failed Eq. (6) first step: re-raise if
        ``first_step='eq6'``, otherwise warn and fall back to 'skip'.
        """
        msg = (
            f"[{tag}] the first step (Eq. 6, eq6_form="
            f"{getattr(self, 'eq6_form', '?')!r}) has no usable solution "
            f"for this case: {err}.  Eq. (6) is a positive-feedback "
            "amplifier and admits no profile reaching n_e(0) = ne_x0 when "
            "the inner-boundary gradient, the pedestal width, or the "
            "D_ped shape make the amplification exp(int C_A6 N dxi) too "
            "large.  Remedies: first_step='skip' (start the Eq. 7 Picard "
            "loop from the initial guess, which is all Saarelma et al. "
            "use Eq. 16 for), a shallower dne_dx_bc, or an inner "
            "boundary closer to the separatrix."
        )
        if first_step == "eq6":
            raise RuntimeError(msg) from err
        import warnings
        warnings.warn(msg + "  Falling back to first_step='skip'.",
                      RuntimeWarning, stacklevel=3)
        self.first_step_used = "skip (Eq. 6 failed)"
        return False

    def _record_picard_sc(self, tag, converged, n_it, history,
                          picard_max_it, picard_relax, picard_rtol):
        """Store ``picard_info`` / ``kbm_info`` and raise if diverged."""
        self.picard_info = {
            "converged": converged,
            "iterations": n_it,
            "gate_mode": "average",
            "history": history,
        }
        self.kbm_info = {
            "treatment": "picard",
            "picard_gate_mode": "average",
            "eq6_form": getattr(self, "eq6_form", None),
            "first_step": getattr(self, "first_step_used", None),
            "alpha_crit": float(self.alpha_crit),
            "picard_max_it": int(picard_max_it),
            "picard_rtol": float(picard_rtol),
            "picard_relax": float(picard_relax),
            "picard_converged": converged,
            "picard_iterations": n_it,
        }
        if not converged:
            last = history[-1]["dne_rel"] if history else float("nan")
            msg = (
                f"[{tag}] Picard loop did not converge in {picard_max_it} "
                f"iterations (last |dne|_rel = {last:.3e})"
            )
            # Diagnose a KBM gate limit cycle (alpha_bar straddles alpha_crit).
            gates = [h["kbm_gate_on"] for h in history[-6:]]
            if len(set(gates)) > 1:
                a_lo = min(h["alpha_bar"] for h in history[-6:])
                a_hi = max(h["alpha_bar"] for h in history[-6:])
                msg += (
                    f".  The KBM gate is flip-flopping: alpha_bar cycles "
                    f"over [{a_lo:.4f}, {a_hi:.4f}] across "
                    f"alpha_crit = {float(self.alpha_crit):.4f}, so the "
                    "frozen D_KBM alternates between its on and off "
                    "branches.  Damp it with picard_relax < 1 (e.g. 0.5), "
                    "or move alpha_crit away from alpha_bar"
                )
            else:
                msg += (
                    "; consider increasing picard_max_it or setting "
                    "picard_relax < 1"
                )
            raise RuntimeError(msg + ".")

    # ------------------------------------------------------------------
    # scipy implementation
    # ------------------------------------------------------------------

    def solve_sc_scipy(self,
                       x_res=200,
                       free_params=None,
                       ne_grad_bc_loc="inner",
                       dne_dx_bc=None,
                       dne_dx_neginf=None,
                       initial_guess="pfile",
                       tanh_width=None,
                       tanh_center=None,
                       eq6_form="complete",
                       first_step="auto",
                       picard_max_it=50,
                       picard_rtol=1e-6,
                       picard_relax=1.0,
                       bvp_tol=1e-6,
                       bvp_max_nodes=5000,
                       reuse_setup=True,
                       verbose=None):
        """Implicit scipy ``solve_bvp`` solver: an Eq. (6) first step, then
        Picard iteration of Eq. (7) with E(x) and D_KBM frozen from the
        previous iterate. Parameters are documented in
        docs/solver_1d_documentation.tex; returns the common result dict.
        """
        v = self.verbose if verbose is None else bool(verbose)
        picard_relax = float(picard_relax)
        if not (0.0 < picard_relax <= 1.0):
            raise ValueError(
                f"picard_relax must be in (0, 1], got {picard_relax}."
            )
        eq6_form = self._check_eq6_form_sc(eq6_form)
        self.eq6_form = eq6_form
        first_step = self._check_first_step_sc(first_step)
        self.first_step_used = first_step
        ne_grad_bc_loc = bcig.check_ne_bc_loc(ne_grad_bc_loc)
        self.ne_grad_bc_loc = ne_grad_bc_loc

        self._ensure_sc_setup(x_res, free_params=free_params,
                              force=not reuse_setup)
        L = self._L_sc
        n0 = self._n0_sc
        # Only the two conditions belonging to ne_grad_bc_loc are looked up.
        nebcs = bcig.resolve_ne_bcs(
            self, ne_grad_bc_loc,
            dne_dx_bc=dne_dx_bc,
            require_negative_slope=True,
        )
        self.ne_bcs = nebcs
        dN_bc = (L / n0) * nebcs.dne_dx        # non-dim Neumann value
        # Saarelma's constant of integration C in the source term
        # (N' - N'_in).  NOT a boundary condition -- see bc_ig_helpers --
        # so it keeps the pedestal-top slope in both pathways.
        dne_dx_C = bcig.resolve_integration_constant(
            self, dne_dx_neginf=dne_dx_neginf,
        )
        dN_in = (L / n0) * dne_dx_C

        # Frozen equilibrium interpolators (SI, functions of x).
        def _mk(arr):
            return interp1d(self.x_init, arr, kind='linear',
                            bounds_error=False, fill_value='extrapolate')

        g_x = _mk(self.gradr2_fsa)
        Si_x = _mk(self.S_i_pres)
        Scx_x = _mk(self.S_cx_pres)
        Vcx_x = _mk(np.abs(self.V_cx_pres))
        fFC_x = _mk(self.fFC)
        fCX_x = _mk(self.fCX)
        Vfc = abs(self.V_FC)

        # Non-dimensional solver grid and initial guess.
        xi_grid = np.linspace(-1.0, 0.0, int(x_res))
        x_grid = L * xi_grid
        ne_init = bcig.build_ne_initial_guess(
            self, x_grid, initial_guess, nebcs,
            tanh_width=tanh_width, tanh_center=tanh_center,
        )
        self.ne_init = ne_init

        if v:
            print(nebcs.describe(prefix="[sc scipy] "))
            print(f"[sc scipy] N'(bc)          = {dN_bc:.3e}")
            print(f"[sc scipy] N'_in (source C)= {dN_in:.3e}  "
                  f"(model constant, not a BC)")
            print(f"[sc scipy] scales: L = {L:.4e} m, n0 = {n0:.4e} m^-3, "
                  f"[D]_0 = {self._D0_sc:.4e} m^2/s")

        def _build_f_interp(D_KBM_xinit):
            """PCHIP interpolators for f0 = g (D_NEO + D_KBM), f1 = g C_ETG and
            their analytic x-derivatives.
            """
            f0_arr = self.gradr2_fsa * (self.D_NEO + D_KBM_xinit)
            f1_arr = self.gradr2_fsa * self.C_ETG
            f0_spl = PchipInterpolator(self.x_init, f0_arr, extrapolate=True)
            f1_spl = PchipInterpolator(self.x_init, f1_arr, extrapolate=True)
            return (f0_spl, f1_spl, f0_spl.derivative(),
                    f1_spl.derivative())

        def _conductance(xi, N, f0_x, f1_x, df0_x, df1_x):
            """Return x, f, C_K and C_N at (xi, N), flooring N where it appears
            in a denominator.
            """
            x = L * xi
            N_safe = np.maximum(N, 1e-8)
            f1 = f1_x(x)
            F = f0_x(x) + f1 / (n0 * N_safe)  # = <|grad r|^2> D_ped, m^2/s
            C_K = (L / F) * (df0_x(x) + df1_x(x) / (n0 * N_safe))
            C_N = f1 / (n0 * (N_safe ** 2) * F)
            return x, F, C_K, C_N

        if ne_grad_bc_loc == "inner":
            def bc(Ya, Yb):
                return np.array([
                    Ya[1] - dN_bc,   # Neumann at xi = -1 (inner)
                    Yb[0] - 1.0,     # Dirichlet at xi = 0: N = ne_x0/n0 = 1
                ])
        else:
            # Both conditions at the separatrix (Saarelma Sec. 2.3); the
            # inner boundary is left free.  solve_bvp only needs the right
            # *number* of residuals, so this needs no shooting.
            def bc(Ya, Yb):
                return np.array([
                    Yb[0] - 1.0,     # Dirichlet at xi = 0: N = ne_x0/n0 = 1
                    Yb[1] - dN_bc,   # Neumann at xi = 0 (separatrix)
                ])

        # --------------------------------------------------------------
        # Step 1: no-CX first step (report Eq. 6 / Saarelma Eq. 16),
        # with D_KBM frozen from the initial guess.
        # --------------------------------------------------------------
        self.calc_D_KBM_average_sc(ne_init, x_grid)
        D_KBM_frozen = self._D_KBM.copy()
        f0_x, f1_x, df0_x, df1_x = _build_f_interp(D_KBM_frozen)

        def ode_first(xi, Y):
            N, dN = Y
            x, _F, C_K, C_N = _conductance(
                xi, N, f0_x, f1_x, df0_x, df1_x,
            )
            Si = Si_x(x)
            Scx = Scx_x(x)
            if eq6_form == "paper":
                # Eq. 6 as printed: the RHS D_ped cancels the <|grad r|^2>
                # D_ped of the transport operator, leaving 1/<|grad r|^2>.
                C_A6 = n0 * L * (Si + Scx) / (g_x(x) * Vfc * fFC_x(x))
            else:
                # Eq. (16-full): <|grad r|^2> D_ped kept on both sides
                # (cancels exactly), shape factor 1 + S_CX/(2 S_i) kept.
                C_A6 = (n0 * L * (Si + Scx)
                        / ((1.0 + 0.5 * Scx / Si) * Vfc * fFC_x(x)))
            d2N = C_A6 * N * (dN - dN_in) - C_K * dN + C_N * dN ** 2
            return np.vstack([dN, d2N])

        dne_guess = np.gradient(ne_init, x_grid)
        Y_guess = np.vstack([ne_init / n0, dne_guess * (L / n0)])
        sol = None
        if first_step != "skip":
            sol = solve_bvp(ode_first, bc, xi_grid, Y_guess,
                            tol=bvp_tol, max_nodes=int(bvp_max_nodes),
                            verbose=2 if v else 0)
            err = None
            if not sol.success:
                err = RuntimeError(sol.message)
            elif sol.y[0].min() <= 0.0:
                # A "converged" collapsed profile is not a usable start:
                # Eq. 6 can be satisfied by a density that crosses zero
                # inside the pedestal, which no Eq. 7 iterate can recover
                # from (and which is unphysical anyway).
                err = RuntimeError(
                    f"the solution is not positive (min n_e = "
                    f"{n0 * sol.y[0].min():.3e} m^-3)"
                )
            if err is not None:
                self._first_step_failed_sc("sc scipy", first_step, err)
                sol = None
            else:
                self.sol_first = sol
                if v:
                    print(f"[sc scipy] first step (Eq. 6) converged, "
                          f"alpha_bar(guess) = {self.alpha_bar_ped:.4f}, "
                          f"KBM {'ON' if self.kbm_gate_on else 'OFF'}")

        # State carried between Picard iterations: the previous iterate on
        # its own (possibly adapted) grid.  Without step 1 that is simply
        # the initial guess on the uniform grid.
        if sol is None:
            x_prev = x_grid
            N_prev, dN_prev = Y_guess
            if v:
                print("[sc scipy] first step skipped; the Eq. 7 Picard loop "
                      f"starts from initial_guess={initial_guess!r}")
        else:
            x_prev = L * sol.x
            N_prev, dN_prev = sol.y[0], sol.y[1]

        # --------------------------------------------------------------
        # Step 2: Picard iterations of the full Eq. (7).
        # --------------------------------------------------------------
        picard_history = []
        picard_converged = False
        n_picard = 0
        E_frozen = None
        prev_gate = bool(self.kbm_gate_on)

        for it in range(1, int(picard_max_it) + 1):
            n_picard = it

            # Previous iterate in SI on its (possibly adapted) grid.
            ne_prev = n0 * N_prev

            # Refreeze E(x) and D_KBM (averaged-alpha gate) from the
            # previous iterate, with optional under-relaxation -- mirrors
            # solver_nondim's picard_gate_mode="average" branch.
            E_new = np.interp(
                x_grid, x_prev, self._exp_kernel_sc(ne_prev, x_prev)
            )
            if E_frozen is None:
                E_frozen = E_new
            else:
                E_frozen = (picard_relax * E_new
                            + (1.0 - picard_relax) * E_frozen)
            E_x = interp1d(x_grid, E_frozen, kind='linear',
                           bounds_error=False, fill_value='extrapolate')

            ne_prev_on_grid = np.interp(x_grid, x_prev, ne_prev)
            D_KBM_new = self.calc_D_KBM_average_sc(ne_prev_on_grid, x_grid)
            gate_now = bool(self.kbm_gate_on)
            D_KBM_frozen = (picard_relax * D_KBM_new
                            + (1.0 - picard_relax) * D_KBM_frozen)
            self._D_KBM = D_KBM_frozen
            f0_x, f1_x, df0_x, df1_x = _build_f_interp(D_KBM_frozen)

            def ode_full(xi, Y):
                N, dN = Y
                x, F, C_K, C_N = _conductance(
                    xi, N, f0_x, f1_x, df0_x, df1_x,
                )
                Si = Si_x(x)
                Scx = Scx_x(x)
                Vcx = Vcx_x(x)
                fCX = fCX_x(x)
                C_cx = 1.0 - (Vfc * fFC_x(x) / (Vcx * fCX)) * (
                    (Si + 0.5 * Scx) / (Si + Scx)
                )
                C_A = n0 * L * Si / (Vcx * fCX)
                C_E = (L ** 2) * Si * C_cx * self.nFC_x0 * E_x(x) / F
                d2N = (C_A * N * (dN - dN_in) - C_E * N
                       - C_K * dN + C_N * dN ** 2)
                return np.vstack([dN, d2N])

            # Warm start from the previous iterate on the uniform grid.
            N_guess = np.interp(x_grid, x_prev, N_prev)
            dN_guess = np.interp(x_grid, x_prev, dN_prev)
            sol = solve_bvp(ode_full, bc, xi_grid,
                            np.vstack([N_guess, dN_guess]),
                            tol=bvp_tol, max_nodes=int(bvp_max_nodes),
                            verbose=2 if v else 0)
            if not sol.success:
                raise RuntimeError(
                    f"[sc scipy] Picard iteration {it} BVP failed: "
                    f"{sol.message}"
                )
            x_prev = L * sol.x
            N_prev, dN_prev = sol.y[0], sol.y[1]

            ne_new_on_grid = n0 * np.interp(x_grid, x_prev, N_prev)
            dn_rel = (np.max(np.abs(ne_new_on_grid - ne_prev_on_grid))
                      / max(np.max(np.abs(ne_new_on_grid)), 1e-300))

            picard_history.append({
                "iteration": it,
                "alpha_bar": float(self.alpha_bar_ped),
                "alpha_frac_above": float(self.alpha_frac_above),
                "kbm_gate_on": gate_now,
                "dne_rel": float(dn_rel),
            })
            if v:
                print(
                    f"[sc scipy] picard it {it:3d}: |dne|_rel = {dn_rel:.3e}, "
                    f"alpha_bar = {self.alpha_bar_ped:.4f}, "
                    f"KBM {'ON' if gate_now else 'OFF'}"
                )
            if dn_rel < float(picard_rtol) and gate_now == prev_gate:
                picard_converged = True
                break
            prev_gate = gate_now

        self._record_picard_sc("sc scipy", picard_converged, n_picard,
                               picard_history, picard_max_it, picard_relax,
                               picard_rtol)

        # De-normalise and store.
        self.hat_x_sol = sol.x.copy()
        self.hat_ne_sol = sol.y[0].copy()
        self.x_sol = L * sol.x
        self.ne_sol = n0 * sol.y[0]
        self.dne_dx_sol = (n0 / L) * sol.y[1]
        self.E_sol = np.interp(self.x_sol, x_grid, E_frozen)
        self.sol = sol

        if v:
            print(f"[sc scipy] solved.  n_e in "
                  f"[{self.ne_sol.min():.3e}, {self.ne_sol.max():.3e}] m^-3")
        return self._build_result_dict('1D', 'scipy')

    # ------------------------------------------------------------------
    # Firedrake implementation
    # ------------------------------------------------------------------

    def _ensure_firedrake_mesh_sc(self, mesh_n, fe_degree, force=False):
        """Build (or reuse) the non-dim CG mesh on xi in [-1, 0]; returns
        (mesh, V, xi_dofs).
        """
        if not _FIREDRAKE_AVAILABLE:
            raise ImportError(
                "Firedrake is not available in this environment.  "
                "Install Firedrake (https://www.firedrakeproject.org/) "
                "to use this solver.  Original import error:\n"
                f"  {_FIREDRAKE_IMPORT_ERR}"
            )

        mesh_key = ("sc", int(mesh_n), int(fe_degree))
        if (not force
                and self._fd_cache.get("mesh_key_sc") == mesh_key
                and "mesh_sc" in self._fd_cache):
            return (
                self._fd_cache["mesh_sc"],
                self._fd_cache["V_sc"],
                self._fd_cache["xi_dofs_sc"],
            )

        mesh = IntervalMesh(int(mesh_n), -1.0, 0.0)
        V = FunctionSpace(mesh, "CG", int(fe_degree))
        xi_dofs = Function(V).interpolate(
            SpatialCoordinate(mesh)[0]
        ).dat.data.copy()

        self._fd_cache.pop("N_sc", None)
        self._fd_cache.update({
            "mesh_key_sc": mesh_key,
            "mesh_sc": mesh,
            "V_sc": V,
            "xi_dofs_sc": xi_dofs,
        })
        return mesh, V, xi_dofs

    def _build_sc_coefficients(self, V, x_dofs_si):
        """Frozen non-dim equilibrium coefficient Functions on ``V`` (g,
        hat_Si, hat_Scx, hat_Vcx, fFC, fCX, hat_D_NEO, hat_C_ETG, C_cx).
        """
        def _mk(arr_on_xinit, name, scale=1.0):
            f = Function(V, name=name)
            f.dat.data[:] = np.interp(
                x_dofs_si, self.x_init, arr_on_xinit
            ) * scale
            return f

        # make non-dim coefficients
        coeffs = {
            "g":         _mk(self.gradr2_fsa, "gradr2_fsa"),
            "hat_Si":    _mk(self.S_i_pres, "hat_S_i", 1.0 / self._S0_sc),
            "hat_Scx":   _mk(self.S_cx_pres, "hat_S_cx", 1.0 / self._S0_sc),
            "hat_Vcx":   _mk(np.abs(self.V_cx_pres), "hat_V_cx",
                             1.0 / self._V0_sc),
            "fFC":       _mk(self.fFC, "fFC"),
            "fCX":       _mk(self.fCX, "fCX"),
            "hat_D_NEO": _mk(self.D_NEO, "hat_D_NEO", 1.0 / self._D0_sc),
            "hat_C_ETG": _mk(self.C_ETG, "hat_C_ETG",
                             1.0 / (self._n0_sc * self._D0_sc)),
        }
        C_cx_fd = Function(V, name="C_cx")
        C_cx_fd.dat.data[:] = self._C_cx_sc(x_dofs_si)
        coeffs["C_cx"] = C_cx_fd # already is non-dimensional

        self._fd_cache["coeffs_sc"] = coeffs
        return coeffs

    def solve_sc_firedrake(self,
                           x_res=200,
                           fe_degree=2,
                           free_params=None,
                           ne_grad_bc_loc="inner",
                           dne_dx_bc=None,
                           dne_dx_neginf=None,
                           grad_bc_tol=1e-8,
                           grad_bc_max_it=25,
                           grad_bc_seed=None,
                           initial_guess="pfile",
                           tanh_width=None,
                           tanh_center=None,
                           eq6_form="complete",
                           first_step="auto",
                           picard_max_it=50,
                           picard_rtol=1e-8,
                           picard_relax=1.0,
                           linear_solver="lu",
                           ksp_rtol=1e-8,
                           ksp_max_it=200,
                           reuse_setup=True,
                           verbose=None):
        """Implicit Firedrake/SNES solver for the same two-step problem as
        :meth:`solve_sc_scipy`, in conservative weak form. Parameters and weak
        forms are documented in docs/solver_1d_documentation.tex; returns the
        common result dict.
        """
        if not _FIREDRAKE_AVAILABLE:
            raise ImportError(
                "Firedrake is not available in this environment.  "
                "Install Firedrake to use this solver.  Original import "
                f"error:\n  {_FIREDRAKE_IMPORT_ERR}"
            )
        
        self._fd_cache = {}

        v = self.verbose if verbose is None else bool(verbose)
        force_setup = not reuse_setup
        picard_relax = float(picard_relax)
        if not (0.0 < picard_relax <= 1.0):
            raise ValueError(
                f"picard_relax must be in (0, 1], got {picard_relax}."
            )
        eq6_form = self._check_eq6_form_sc(eq6_form)
        self.eq6_form = eq6_form
        first_step = self._check_first_step_sc(first_step)
        self.first_step_used = first_step
        ne_grad_bc_loc = bcig.check_ne_bc_loc(ne_grad_bc_loc)
        self.ne_grad_bc_loc = ne_grad_bc_loc

        self._ensure_sc_setup(x_res, free_params=free_params,
                              force=force_setup)
        L = self._L_sc
        n0 = self._n0_sc
        # Only the two conditions belonging to ne_grad_bc_loc are looked up.
        nebcs = bcig.resolve_ne_bcs(
            self, ne_grad_bc_loc,
            dne_dx_bc=dne_dx_bc,
            require_negative_slope=True,
        )
        self.ne_bcs = nebcs
        dN_bc_val = (L / n0) * nebcs.dne_dx     # non-dim Neumann value
        # Saarelma's constant of integration C -- NOT a boundary condition,
        # so it keeps the pedestal-top slope in both pathways.
        dne_dx_C = bcig.resolve_integration_constant(
            self, dne_dx_neginf=dne_dx_neginf,
        )
        dN_in_val = (L / n0) * dne_dx_C
        hat_nFC0 = self.nFC_x0 / n0 # non-dim

        _mesh, V, xi_dofs = self._ensure_firedrake_mesh_sc(
            x_res, fe_degree, force=force_setup,
        )
        x_dofs_si = L * xi_dofs # dimensionalized
        sort_idx = np.argsort(xi_dofs)
        unsort_idx = np.argsort(sort_idx)
        x_sorted = x_dofs_si[sort_idx] # dimensionalized 
        coeffs = self._build_sc_coefficients(V, x_dofs_si)

        # Initial guess (SI on the sorted grid, then back to DOF order).
        ne_init_sorted = bcig.build_ne_initial_guess(
            self, x_sorted, initial_guess, nebcs,
            tanh_width=tanh_width, tanh_center=tanh_center,
        )
        ne_init = ne_init_sorted[unsort_idx]
        self.ne_init = ne_init

        N = Function(V, name="hat_n_e")
        N.dat.data[:] = ne_init / n0
        self._fd_cache["N_sc"] = N
        w = TestFunction(V)

        if v:
            print(nebcs.describe(prefix="[sc firedrake] "))
            print(f"[sc firedrake] N'(bc)          = {dN_bc_val:.3e}")
            print(f"[sc firedrake] N'_in (source C)= {dN_in_val:.3e}  "
                  f"(model constant, not a BC)")
            print(f"[sc firedrake] hat_nFC(0)      = {hat_nFC0:.3e}")
            print(f"[sc firedrake] scales: L = {L:.4e} m, n0 = {n0:.4e} m^-3,"
                  f" [D]_0 = {self._D0_sc:.4e} m^2/s")

        # KBM diffusivity frozen from the initial guess (average gate).
        D_KBM_xinit = self.calc_D_KBM_average_sc(ne_init_sorted, x_sorted)
        hat_D_KBM_fd = Function(V, name="hat_D_KBM_picard")
        hat_D_KBM_fd.dat.data[:] = np.interp(
            x_dofs_si, self.x_init, D_KBM_xinit
        ) / self._D0_sc
        self._fd_cache["hat_D_KBM_picard_sc"] = hat_D_KBM_fd

        # Frozen FC attenuation kernel E(xi) (unused by step 1, seeded to
        # the initial guess so the form is well defined from the start).
        E_fd = Function(V, name="E_kernel")
        E_fd.dat.data[:] = self._exp_kernel_sc(
            ne_init_sorted, x_sorted
        )[unsort_idx]
        self._fd_cache["E_kernel_sc"] = E_fd

        # Non-dim constants: dN_in_c is the source constant C (inner slope);
        # dN_bc_c is the ds(1) flux (BC value "inner", secant unknown "outer").
        dN_in_c = Constant(dN_in_val)      # source constant C
        # "outer"-mode secant seed: zero inner flux (the Galerkin natural BC).
        dN_bc_seed = (dN_bc_val if ne_grad_bc_loc == "inner"
                      else (0.0 if grad_bc_seed is None
                            else (L / n0) * float(grad_bc_seed)))
        dN_bc_c = Constant(dN_bc_seed)     # ds(1) flux: the BC value in
                                           # "inner" mode, the secant
                                           # unknown in "outer" mode
        hat_nFC0_c = Constant(hat_nFC0)
        hat_Vfc_c = Constant(abs(self.V_FC) / self._V0_sc)

        g_fd = coeffs["g"]
        hat_Si_fd = coeffs["hat_Si"]
        hat_Scx_fd = coeffs["hat_Scx"]
        hat_Vcx_fd = coeffs["hat_Vcx"]
        fFC_fd = coeffs["fFC"]
        fCX_fd = coeffs["fCX"]
        hat_D_NEO_fd = coeffs["hat_D_NEO"]
        hat_C_ETG_fd = coeffs["hat_C_ETG"]
        C_cx_fd = coeffs["C_cx"]

        # hat_f = g (D_NEO + D_KBM)/[D]_0 + g C_ETG/(n0 [D]_0 N):
        # the inline 1/N keeps the ETG nonlinearity visible to Newton.
        hat_f = g_fd * (hat_D_NEO_fd + hat_D_KBM_fd) + g_fd * hat_C_ETG_fd / N
        N_dx = N.dx(0)

        # Weak residuals.  Boundary id 1 = xi = -1 (inner, Neumann flux
        # imposed with dN_bc_c), id 2 = xi = 0 (Dirichlet).
        bnd_term = hat_f * dN_bc_c * w * ds(1)

        # Step 1 (Eq. 6): RHS6 = N D6 (hat_Si + hat_Scx)(N' - N'_in)
        #                        / (hat_V_FC f_FC), where the "D6" slot is
        # hat_f/g (the D_ped of Eq. 6 as printed) or
        # hat_f / (1 + hat_Scx/(2 hat_Si)) (Eq. 16-full).
        if eq6_form == "paper":
            D6 = hat_f / g_fd
        else:
            D6 = hat_f / (Constant(1.0)
                          + Constant(0.5) * hat_Scx_fd / hat_Si_fd)
        rhs6 = (
            N * D6 * (hat_Si_fd + hat_Scx_fd)
            * (N_dx - dN_in_c) / (hat_Vfc_c * fFC_fd)
        )
        F6 = hat_f * N_dx * w.dx(0) * dx + rhs6 * w * dx + bnd_term

        # Step 2 (Eq. 7): RHS7 = N hat_Si [ hat_f (N' - N'_in)
        #                        / (hat_V_CX f_CX) - C_CX hat_nFC0 E ]
        rhs7 = N * hat_Si_fd * (
            hat_f * (N_dx - dN_in_c) / (hat_Vcx_fd * fCX_fd)
            - C_cx_fd * hat_nFC0_c * E_fd
        )
        F7 = hat_f * N_dx * w.dx(0) * dx + rhs7 * w * dx + bnd_term

        bcs = [DirichletBC(V, Constant(1.0), 2)]   # N(0) = 1
        snes_params = self._build_petsc_solver_parameters(
            linear_solver=linear_solver,
            ksp_rtol=ksp_rtol,
            ksp_max_it=ksp_max_it,
        )

        # One nonlinear solve of the given residual: the bare SNES solve
        # in "inner" mode, the separatrix-gradient secant iteration in
        # "outer" mode.
        self.grad_bc_info = None

        def _solve_once(F_form, tag):
            if ne_grad_bc_loc == "inner":
                solve(F_form == 0, N, bcs=bcs, solver_parameters=snes_params)
            else:
                self.grad_bc_info = self._shoot_outer_grad_sc(
                    F_form, N, bcs, snes_params,
                    dN_bc_c, dN_bc_val,
                    float(grad_bc_tol), int(grad_bc_max_it), v, tag,
                )

        # --------------------------------------------------------------
        # Step 1: solve the no-CX equation.
        # --------------------------------------------------------------
        if first_step != "skip":
            err = None
            try:
                _solve_once(F6, "sc firedrake eq6")
            except Exception as exc:
                err = exc
            else:
                if N.dat.data.min() <= 0.0:
                    # See the note in solve_sc_scipy: a converged but
                    # collapsed profile is not a usable starting point.
                    err = RuntimeError(
                        f"the solution is not positive (min n_e = "
                        f"{n0 * N.dat.data.min():.3e} m^-3)"
                    )
            if err is not None:
                # Restore the initial guess: a diverged SNES leaves N in
                # whatever state the failed line search reached (and, in
                # "outer" mode, the secant may have walked dN_bc_c
                # somewhere unhelpful for the Eq. 7 restart).
                N.dat.data[:] = ne_init / n0
                dN_bc_c.assign(dN_bc_seed)
                self.grad_bc_info = None
                self._first_step_failed_sc("sc firedrake", first_step, err)
            else:
                self.hat_ne_first = N.dat.data[sort_idx].copy()
                if v:
                    print(f"[sc firedrake] first step (Eq. 6) solved, "
                          f"alpha_bar(guess) = {self.alpha_bar_ped:.4f}, "
                          f"KBM {'ON' if self.kbm_gate_on else 'OFF'}")
        elif v:
            print("[sc firedrake] first step skipped; the Eq. 7 Picard loop "
                  f"starts from initial_guess={initial_guess!r}")

        # --------------------------------------------------------------
        # Step 2: Picard loop on Eq. 7, refreezing hat_D_KBM and E(xi).
        # --------------------------------------------------------------
        picard_history = []
        picard_converged = False
        n_picard = 0
        prev_gate = bool(self.kbm_gate_on)
        prev_hat_ne = N.dat.data.copy()
        E_initialised = False

        for it in range(1, int(picard_max_it) + 1):
            n_picard = it

            # Refreeze E and D_KBM from the current iterate.
            ne_curr_sorted = n0 * N.dat.data[sort_idx]
            E_new = self._exp_kernel_sc(ne_curr_sorted, x_sorted)[unsort_idx]
            if not E_initialised: # first step in loop
                E_fd.dat.data[:] = E_new
                E_initialised = True
            else: # all other steps in the loop
                E_fd.dat.data[:] = (
                    picard_relax * E_new
                    + (1.0 - picard_relax) * E_fd.dat.data
                )

            D_KBM_xinit = self.calc_D_KBM_average_sc(ne_curr_sorted, x_sorted)
            gate_now = bool(self.kbm_gate_on)
            hat_D_KBM_new = np.interp(
                x_dofs_si, self.x_init, D_KBM_xinit
            ) / self._D0_sc
            hat_D_KBM_fd.dat.data[:] = (
                picard_relax * hat_D_KBM_new
                + (1.0 - picard_relax) * hat_D_KBM_fd.dat.data
            )

            # Solve the full equation with frozen E / D_KBM (warm start
            # from the previous iterate stored in N).
            _solve_once(F7, "sc firedrake eq7")

            hat_ne_new = N.dat.data.copy()
            dn_rel = (
                np.linalg.norm(hat_ne_new - prev_hat_ne)
                / max(np.linalg.norm(prev_hat_ne), 1e-300)
            )

            picard_history.append({
                "iteration": it,
                "alpha_bar": float(self.alpha_bar_ped),
                "alpha_frac_above": float(self.alpha_frac_above),
                "kbm_gate_on": gate_now,
                "dne_rel": float(dn_rel),
            })
            if v:
                print(
                    f"[sc firedrake] picard it {it:3d}: "
                    f"|dne|_rel = {dn_rel:.3e}, "
                    f"alpha_bar = {self.alpha_bar_ped:.4f}, "
                    f"KBM {'ON' if gate_now else 'OFF'}"
                )
            if abs(dn_rel) < float(picard_rtol) and gate_now == prev_gate:
                picard_converged = True
                break
            prev_hat_ne = hat_ne_new
            prev_gate = gate_now

        self._record_picard_sc("sc firedrake", picard_converged, n_picard,
                               picard_history, picard_max_it, picard_relax,
                               picard_rtol)

        # --------------------------------------------------------------
        # Extract converged profiles (non-dim and SI).
        # --------------------------------------------------------------
        self.hat_x_sol = xi_dofs[sort_idx]
        self.hat_ne_sol = N.dat.data[sort_idx]
        self.x_sol = L * self.hat_x_sol
        self.ne_sol = n0 * self.hat_ne_sol
        self.dne_dx_sol = np.gradient(self.ne_sol, self.x_sol)
        self.E_sol = E_fd.dat.data[sort_idx].copy()
        self.N_fd = N
        self.V_fd = V

        if ne_grad_bc_loc == "outer":
            # The inner flux was the unknown in this mode; report the
            # value the separatrix-gradient condition selected for it.
            # (Distinct from self.dne_dx_inner, which is the source
            # constant C and was an input.)
            self.dne_dx_inner_solved = float(dN_bc_c) * n0 / L
            if v:
                print(f"[sc firedrake] grad-bc converged in "
                      f"{self.grad_bc_info['iterations']} iterations; free "
                      f"dne/dx(x_inner) = "
                      f"{self.dne_dx_inner_solved:.3e} m^-4")

        if v:
            print(f"[sc firedrake] solved.  n_e in "
                  f"[{self.ne_sol.min():.3e}, {self.ne_sol.max():.3e}] m^-3")
        return self._build_result_dict('1D', 'firedrake')

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def solve_sc(self, implementation="firedrake", **kwargs):
        """Dispatch to :meth:`solve_sc_firedrake` or :meth:`solve_sc_scipy`
        according to ``implementation``.
        """
        implementation = str(implementation).lower()
        if implementation == "firedrake":
            return self.solve_sc_firedrake(**kwargs)
        if implementation == "scipy":
            return self.solve_sc_scipy(**kwargs)
        raise ValueError(
            f"implementation must be 'firedrake' or 'scipy', got "
            f"{implementation!r}."
        )