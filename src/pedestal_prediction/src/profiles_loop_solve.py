# Loop between EPEDNN and coupled version of Saarelma-Connor model to find self-consistent pedestal

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d

ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(ROOT))

from src.solver_nondim import saarelma_connor_nondim

def profiles_loop_solve(
    MHD_FP = None,
    KPROF_FP = None,
    kprof_loc = 'p',
    manual_profs = None,
    out_dir = None,
    ne_inner_bc = "neumann",
    x_res = 40,
    P_tot_e = 5e6, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox
    species = 'D',
    Z_i = 1,
    psi_N_inner = 0.85,
    free_params = None,
    eped_tol_max = 1e-5,
    eped_iter_max = 50,
    kbm_gate_eps = 0.01,
    kbm_treatment = "inline",
    picard_gate_mode = None,
    picard_max_it = 50,
    picard_rtol = 1e-8,
    picard_relax = 1.0,
    EPEDNN_core = 'pfile',
    ig = None,
    epednn_model = 'EPED1',
    verbose = False,
    verbose_sc = False,
):
    """Solve the self-consistent pedestal problem using the EPEDNN model and the Saarelma-Connor model.

    Parameters
    ----------
    MHD_FP : str
        Path to MHD equilibrium file.
    KPROF_FP : str
        Path to KPROF profile file.
    kprof_loc : str
        Location of the kinetic parameters, currently supporting: 'p', 'manual'.
    manual_profs : dict
        Dictionary of manual profiles for the electron temperature and density, currently supporting: 'pfile', 'epednn'.
    out_dir : str
        Path to directory to save successful solutions from the Saarelma-Connor model.
    ne_inner_bc : str
        Boundary condition for the electron density at the inner boundary.
    x_res : int
        Number of grid points in the radial direction.
    P_tot_e : float
        Total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox.
    species : str
        Species of ions, currently supporting: 'D', 'D-T'.
    Z_i : int
        Charge number of ions.
    psi_N_inner : float
        Inner boundary of the pedestal in normalized poloidal flux.
        Per-DOF x coordinates (m), unsorted (as stored in dat.data).
    free_params : dict
        Dictionary of free parameters for the EPEDNN model.
    eped_tol_max : float
        Maximum tolerance for the pedestal height and width.
    eped_iter_max : int
        Maximum number of iterations for the EPEDNN model.
    kbm_gate_eps : float
        Minimum gate error for the KBM model.
    kbm_treatment : str
        Treatment for the KBM model.
    picard_gate_mode : str
        Mode for the Picard gate.
    picard_max_it : int
        Maximum number of iterations for the Picard gate.
    picard_rtol : float
        Relative tolerance for the Picard gate.
    picard_relax : float
        Relaxation factor for the Picard gate.
    EPEDNN_core : str
        Core to use for the EPEDNN model.
    verbose : bool
        Whether to print verbose output.

    Returns
    -------
    psi_N_inner_boundary_new : float
        Inner boundary of the pedestal in normalized poloidal flux.
    T_prof_keV : float
        Temperature profile in keV.
    best_x : ndarray
        Best x coordinates.
    best_ne : ndarray
        Best electron density profile.
    pedestal_height : float
        Final pedestal pressure height in MPa.
    """

    te_plot_profiles = []
    ne_plot_profiles = []

    equil_tag = Path(MHD_FP).name[1:]

    # Load in free parameters
    if free_params is None:
        raise ValueError("Must specify free_params")
    alpha_crit = free_params['alpha_crit']
    C_KBM = free_params['C_KBM']
    De_chie_etg = free_params['De_chie_etg']
    nFC_x0 = free_params['nFC_x0']
    ncx_x0_ratio = free_params['ncx_x0_ratio']

    # Setup solver parameters
    SOLVE_KW = dict(
        x_res=x_res,
        fe_degree=2,
        ne_inner_bc=ne_inner_bc,   # Saarelma A7 default; see dirichlet comparison below
        linear_solver="lu",      # or "gamg" for GMRES + algebraic multigrid on J
        kbm_treatment=kbm_treatment,
        kbm_gate_eps=kbm_gate_eps, # 1e-3 minimum
        picard_gate_mode=picard_gate_mode,
        picard_max_it=picard_max_it,
        picard_rtol=picard_rtol,
        picard_relax=picard_relax,
        verbose=verbose_sc,
    )

    base_model = saarelma_connor_nondim(
            P_tot_e      = P_tot_e,
            species      = species,
            Z_i          = Z_i,
            alpha_crit   = alpha_crit,
            C_KBM        = C_KBM,
            De_chie_etg  = De_chie_etg,
            nFC_x0       = nFC_x0,
            ncx_x0_ratio = ncx_x0_ratio,
            psi_N_inner_boundary = psi_N_inner,
            mhd_fp       = MHD_FP,
            kprof_fp     = KPROF_FP,
            kprof_loc    = kprof_loc,
            verbose      = verbose_sc,
            manual_profs = manual_profs, # only used if kprof_loc is set so they are
            # psi_N_inner_boundary = 0.85, # set to None to use adaptive inner boundary method
    )
    psi_N_inner_boundary_new = psi_N_inner
    base_model.setup_epednn(model=epednn_model)
    print("Base model built.")

    # Parameters to be used in the loop
    tanh_width_new = None
    psi_N_Te_new = None
    T_prof_keV = None
    pedestal_height = None
    pedestal_width = None

    # Clear outputs from any previous scan (including appended failure logs) and setup logging files
    equil_dir = Path(out_dir) / equil_tag
    equil_dir.mkdir(parents=True, exist_ok=True)

    for path in equil_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    
    betan = -1 # initialize to -1 to indicate that betan is not yet calculated

    for eped_iter in range(eped_iter_max):
        print("------------------------------------------------")
        print(f"ESCAPE Loop Iter {eped_iter}")

        # Run solver and save outputs
        if eped_iter == 0:
            SOLVE_KW['bc_origin'] = "p-file"
            SOLVE_KW['initial_guess'] = "tanh"
            SOLVE_KW['nCX_ic'] = "scale nFC"
            SOLVE_KW['nFC_ic'] = "solve"
        elif eped_iter > 0 and ig=='solve': # bc
            SOLVE_KW['bc_origin'] = "manual EPEDNN loop"
            SOLVE_KW['initial_guess'] = "tanh" 
            SOLVE_KW['nCX_ic'] = "solve"
            SOLVE_KW['nFC_ic'] = "solve"
        elif eped_iter > 0 and ig=='manual': # bc+ig
            SOLVE_KW['bc_origin'] = "manual EPEDNN loop"
            SOLVE_KW['initial_guess'] = "manual EPEDNN loop" # use previous loop's profiles as initial guess for ne, nFC, nCX
            SOLVE_KW['nCX_ic'] = "manual EPEDNN loop"
            SOLVE_KW['nFC_ic'] = "manual EPEDNN loop"
        elif eped_iter > 0 and ig=='fix': # fix; only T changes, the densities are fixed from the pfile
            SOLVE_KW['bc_origin'] = "p-file"
            SOLVE_KW['initial_guess'] = "tanh"
            SOLVE_KW['nCX_ic'] = "solve"
            SOLVE_KW['nFC_ic'] = "solve"
        else:
            raise ValueError(f"Invalid initial guess mode: {ig}")
        x_sol, ne_sol, nFC_sol, nCX_sol, T_e_pres, psi_N_pres = base_model.solve_coupled_nondim(tanh_width=tanh_width_new, **SOLVE_KW) # tanh_width only used if initial_guess='tanh'
        # x -> psi_N on the density solution grid (for output only; same map as below)
        psi_N_ne = interp1d(np.asarray(base_model.r_psi, dtype=float) - float(base_model.r_psi[-1]),
                            np.asarray(base_model.psi_N_pres, dtype=float),
                            kind='linear', bounds_error=False, fill_value='extrapolate')(x_sol)
        sol = {'x': x_sol, 'y': ne_sol, 'nFC': nFC_sol, 'nCX': nCX_sol,
               'T_e': T_e_pres, 'psi_N': psi_N_pres, 'psi_N_ne': psi_N_ne,
               'alpha_crit': alpha_crit, 'C_KBM': C_KBM, 'De_chie_etg': De_chie_etg, 'nFC_x0': nFC_x0, 'ncx_x0_ratio': ncx_x0_ratio,
               'betan': betan, 'ig': ig}
        best_x = np.asarray(sol['x'], dtype=float)
        best_ne = np.asarray(sol['y'], dtype=float)

        # psi_N <-> x map (reuse base_model)
        psi_N_pres = np.asarray(base_model.psi_N_pres, dtype=float)
        x_grid_full = np.asarray(base_model.r_psi, dtype=float) - float(base_model.r_psi[-1])
        psi_to_x = interp1d(psi_N_pres, x_grid_full, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        psi_ped_grid = np.linspace(psi_N_inner, 1.0, x_res)
        # x_ped_grid = psi_to_x(psi_ped_grid)

        # --- Feed best profile into EPEDNN --------------------------------
        if eped_iter == 0:
            pedestal_height_prev = 0.0
            pedestal_width_prev = 0.0
            pedestal_height, pedestal_width, betan = base_model.feed_epednn(model=epednn_model, ne_ped=best_ne, x_ne=best_x, EPEDNN_core='pfile')    
        else:
            pedestal_height_prev, pedestal_width_prev = pedestal_height, pedestal_width
            pedestal_height, pedestal_width, betan = base_model.feed_epednn(model=epednn_model, ne_ped=best_ne, x_ne=best_x, psiN_Te=psi_N_Te_new, Te_prev=T_prof_keV * 1e3, EPEDNN_core=EPEDNN_core)
        tanh_width_new = psi_to_x(1-pedestal_width) * -1 # will error if result if negative which should not happen
        print(f"Pedestal height: {pedestal_height} MPa, Pedestal width: {pedestal_width} (psi_N)")

        if eped_iter > 0:
            eped_tol = ((pedestal_height - pedestal_height_prev) / pedestal_height_prev
                        + (pedestal_width - pedestal_width_prev) / pedestal_width_prev)
            print(f"Normalized pedestal pressure height and width tolerance: {eped_tol}")
            sol['loop_tol'] = eped_tol
            np.save(f'{equil_dir}/ne_and_Te_iter_{eped_iter}.npy', sol, allow_pickle=True)
            if abs(eped_tol) < eped_tol_max:
                break
        else:
            np.save(f'{equil_dir}/ne_and_Te_iter_{eped_iter}.npy', sol, allow_pickle=True)

        # --- New T_e profile (EPED1 tanh form, Eq. 1b without core H term) ---
        #   T(psi) = T_sep + aT0 * { tanh[2(1 - psi_mid)/Delta]
        #                          - tanh[2(psi - psi_mid)/Delta] }
        # On [psi_ped, 1] the tanh shape peaks at psi_ped; aT0 is fixed so
        # T(psi_ped) = Te_ped derived from EPED pedestal pressure (p_ped =
        # 2 * ne_ped * Te_ped, Ti = Te, Zeff ~ 1).
        Delta = float(pedestal_width)
        psi_mid = 1.0 - 0.5 * Delta
        psi_ped = 1.0 - Delta
        ne_ped_val = float(interp1d(best_x, best_ne, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')(
                                        psi_to_x(psi_ped)))
        T_sep_eV = float(np.asarray(base_model.T_e)[-1] * 1e3)  # keV -> eV
        eV_to_J = 1.602176634e-19
        Te_ped_eV = (float(pedestal_height) * 1.0e6 / (2.0 * ne_ped_val)) / eV_to_J
        tanh_peak = 2.0 * np.tanh(1.0)  # shape-function maximum on [psi_ped, 1]
        aT0 = (Te_ped_eV - T_sep_eV) / tanh_peak

        def _eped1_tanh_Te(psi_N):
            return T_sep_eV + aT0 * (
                np.tanh(2.0 * (1.0 - psi_mid) / Delta)
                - np.tanh(2.0 * (psi_N - psi_mid) / Delta)
            )

        # EPED tanh only on the pedestal strip [psi_ped, 1]; splice onto p-file core.
        psi_tanh = np.linspace(psi_ped, 1.0, x_res)
        Te_tanh_eV = _eped1_tanh_Te(psi_tanh)

        psi_prev = np.asarray(base_model.psi_Te_eval, dtype=float)
        Te_prev_keV = np.asarray(base_model.T_e, dtype=float)

        # Add offset to T_e core profile
        Te_prev_keV_ped = interp1d(psi_prev, Te_prev_keV, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_ped)
        Te_tanh_eV_ped = interp1d(psi_tanh, Te_tanh_eV / 1e3, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_ped)
        T_e_offset = Te_tanh_eV_ped - Te_prev_keV_ped # ped - core at psi_ped

        keep = psi_prev < psi_ped
        psi_N_Te_new = np.concatenate([psi_prev[keep], psi_tanh])
        T_prof_keV = np.concatenate([Te_prev_keV[keep] + T_e_offset, Te_tanh_eV / 1e3])

        # psi_N_inner_boundary_new = psi_ped

        if verbose:  # collect profile data for post-loop plotting
            Te_spliced_eV = T_prof_keV * 1e3
            # print(f"  Te_ped = {Te_ped_eV:.1f} eV, T_sep = {T_sep_eV:.1f} eV "
            #       f"(ne_ped = {ne_ped_val:.3e} m^-3, psi_ped = {psi_ped:.4f}, "
            #       f"Delta = {Delta:.4f})")
            # print(f"  psi_N_inner_boundary_new = {psi_N_inner_boundary_new:.4f}")
            # og_Te_peak = float(interp1d(psi_prev, Te_prev_keV * 1e3, kind='linear',
            #                             bounds_error=False, fill_value='extrapolate')(psi_ped))
            # print(f"Percent change from previous T_e at psi_ped = "
            #       f"{(100*(Te_ped_eV - og_Te_peak) / og_Te_peak):.4f}")

            if eped_iter == 0:
                te_plot_profiles.append({
                    'psi_N': np.asarray(psi_prev, dtype=float),
                    'y': np.asarray(Te_prev_keV * 1e3, dtype=float),
                    'label': 'Reference $T_e$',
                    'ls': '--',
                })
                ne_plot_profiles.append({
                    'psi_N': np.asarray(base_model.psi_N_pres, dtype=float),
                    'y': np.asarray(base_model.n_e_pres, dtype=float) / 1e19,
                    'label': 'Reference $n_e$',
                    'ls': '--',
                })

            te_plot_profiles.append({
                'psi_N': np.asarray(psi_N_Te_new, dtype=float),
                'y': np.asarray(Te_spliced_eV, dtype=float),
                'label': f'ESCAPE iteration {eped_iter}',
                'ls': '-',
            })

            x_to_psiN = interp1d(x_grid_full, psi_N_pres, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
            psi_N_ne = x_to_psiN(best_x)
            sort_idx = np.argsort(psi_N_ne)
            ne_plot_profiles.append({
                'psi_N': psi_N_ne[sort_idx],
                'y': (best_ne[sort_idx] / 1e19),
                'label': f'ESCAPE iteration {eped_iter}',
                'ls': '-',
            })

        # MAKE THIS PART FASTER
        x_to_psiN = interp1d(x_grid_full, psi_N_pres, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
        psi_N_ne = x_to_psiN(best_x)
        sort_idx = np.argsort(psi_N_ne)
        manual_profs = {
            'Te': T_prof_keV,
            'psi_N_Te': psi_N_Te_new,
            'ne': best_ne[sort_idx],
            'nCX': nCX_sol[sort_idx],
            'nFC': nFC_sol[sort_idx],
            'psi_N_n': psi_N_ne[sort_idx],
        }
        if ig == 'manual' or ig == 'solve':
            base_model = saarelma_connor_nondim( # reset base_model to the new T_e profile and define new "p-file" profiles from the manual profiles
                P_tot_e      = P_tot_e,
                species      = species,
                Z_i          = Z_i,
                alpha_crit   = alpha_crit,
                C_KBM        = C_KBM,
                De_chie_etg  = De_chie_etg,
                nFC_x0       = nFC_x0,
                ncx_x0_ratio = ncx_x0_ratio,
                mhd_fp       = MHD_FP,
                kprof_loc    = 'manual EPEDNN loop',
                kprof_fp = KPROF_FP,
                manual_profs = manual_profs,
                verbose      = False,
                psi_N_inner_boundary = psi_N_inner,
            )
        elif ig == 'fix':
            base_model = saarelma_connor_nondim( # reset base_model to the new T_e profile and use the p-file n_e
                P_tot_e      = P_tot_e,
                species      = species,
                Z_i          = Z_i,
                alpha_crit   = alpha_crit,
                C_KBM        = C_KBM,
                De_chie_etg  = De_chie_etg,
                nFC_x0       = nFC_x0,
                ncx_x0_ratio = ncx_x0_ratio,
                mhd_fp       = MHD_FP,
                kprof_loc    = 'pfile',
                kprof_fp     = KPROF_FP,
                manual_profs = manual_profs,
                verbose      = False,
                psi_N_inner_boundary = psi_N_inner,
            )
        base_model.setup_epednn(model=epednn_model)

    # gfile_pres_grid = base_model.psi_N_pres
    # gfile_pres = base_model.pres_gfile

    if te_plot_profiles:
        red_blue = LinearSegmentedColormap.from_list('red_blue', ['red', 'blue'])
        te_colors = red_blue(np.linspace(0, 1, len(te_plot_profiles)))
        ne_colors = red_blue(np.linspace(0, 1, len(ne_plot_profiles)))

        fig, ax = plt.subplots(figsize=(6, 4))
        for prof, color in zip(te_plot_profiles, te_colors):
            ax.plot(prof['psi_N'], prof['y'] / 1e3, lw=2, ls=prof['ls'],
                    color=color, label=prof['label'])
        ax.set_xlabel(r'$\psi_N$')
        ax.set_ylabel(r'$T_e$ [keV]')
        ax.set_title('Solved $T_e$ profiles')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(psi_N_inner, 1.0)
        fig.tight_layout()

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        for prof, color in zip(ne_plot_profiles, ne_colors):
            ax1.plot(prof['psi_N'], prof['y'], lw=2, ls=prof['ls'],
                     color=color, label=prof['label'])
        ax1.set_xlabel(r'$\psi_N$')
        ax1.set_ylabel(r'$n_e$ ($10^{19}$ m$^{-3}$)')
        ax1.set_title('Solved $n_e$ profiles')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xlim(psi_N_inner, 1.0)
        fig1.tight_layout()
        plt.show()

    return pedestal_width, pedestal_height, sol