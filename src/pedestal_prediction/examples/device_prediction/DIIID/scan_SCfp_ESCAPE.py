import os
import numpy as np
import sys
from pathlib import Path
import pickle # to check which state attributes np.save can store
import netCDF4 as nc # OMFIT profile files are netCDF
ROOT = Path.cwd().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.ESCAPE.ESCAPE_solve import ESCAPE_solve
from helpers.load_equil import initialize_inputs
from helpers.parse_equil import mhd_load
from helpers.parse_prof import kprof_load
from examples.device_prediction.helper_functions import calc_pressure_profile
import time


# ---------------------- COMMON USER INPUTS ----------------------
equil_num = None # None for all
# equil_list = ['125729.03589','128572.03809','128578.03658','128413.04088']
equil_list = None

alpha_crits = np.array([0.01])
nFC_x0s = np.array([1e15])
C_KBMs = np.array([0.0])
De_chie_etgs = np.array([0.5])
ncx_x0_ratios = np.array([15])

# Overridable from the environment (see submit_scan.sh)
sc_model = os.environ.get('SC_MODEL', '3D')
sc_implementation = os.environ.get('SC_IMPLEMENTATION', 'firedrake')
ne_grad_bc_loc = os.environ.get('NE_GRAD_BC_LOC', 'inner')

output_dir = f'DIIIDSnyder_ESCAPE_{sc_model}{sc_implementation}{ne_grad_bc_loc}'
# ----------------------------------------------------------------


Path(output_dir).mkdir(parents=True, exist_ok=True)

# Equilibria parameters
geqdsk_dir = Path('/mnt/homes_global/jal2351/software/sc_inputs/gHighPerfHMode')
pfile_dir = Path('/mnt/homes_global/jal2351/software/sc_inputs/OMFITnc_HighPerfHMode')
mhd_loc = 'eqdsk'
kprof_loc = 'OMFITnc'
equil_num_total = len(list(pfile_dir.glob('*.cdf')))
if equil_num is None:
    equil_num = equil_num_total
print(f'Number of equilibria found: {equil_num_total}')
print(f'Number of equilibria to process: {equil_num}')
equilibria = initialize_inputs(equil_num, geqdsk_dir, pfile_dir, p_filetype=kprof_loc, select_equil=equil_list) # (g-file, IDA .cdf) path pairs


# ── Heating power ────────────────────────────────────────────────────────────
# The DIII-D inputs carry NO heating power: the EFIT g-files have no slot for it
# and the OMFIT IDA netCDFs hold only kinetic/geometric profiles. Until per-shot
# P_NBI/P_ECH are pulled from MDSplus, use one representative value for the set.
#
# P_HEAT_TOT below is a ballpark for this HighPerfHMode database, obtained by
# integrating W_th = 3/2 * int (p_e + p_ion) dV from these very files and
# inverting IPB98(y,2) at H98 = 1: median ~7.3 MW (quartiles ~3.6-15 MW) over
# the 300 cases. Replace with measured PINJ + P_ECH per shot when available.
P_HEAT_TOT = 7.3e6      # W, total (NBI + ECH + Ohmic) heating, set-representative
ELECTRON_FRAC = 0.5     # Saarelma et al. take ~half the heating as the electron channel

# Fallback Zeff for the two files that carry no Zeff variable
# (IDA_125732_3.45_3.55_.cdf, IDA_128578_3.6_3.7_.cdf). 1.8 is the median of
# the profile-mean Zeff over the 300 files that do have it.
ZEFF_DEFAULT = 1.8
ZEFF_PSI_N = 0.95       # evaluate Zeff at the pedestal top, not on axis


def read_omfitnc_zeff(kprof_fp, psi_eval=ZEFF_PSI_N, default=ZEFF_DEFAULT):
    """Zeff at psi_N = psi_eval from an OMFIT IDA netCDF, averaged over its
    time slices. Returns `default` if the file has no usable Zeff."""
    with nc.Dataset(kprof_fp) as f:
        if 'Zeff' not in f.variables:
            print(f'  Zeff missing in {Path(kprof_fp).name}, using {default}')
            return float(default)
        z = np.ma.filled(f.variables['Zeff'][:], np.nan)   # (time, psi_n)
        psi = np.asarray(f.variables['psi_n'][:])
    z_t = np.nanmean(z, axis=0)
    good = np.isfinite(z_t)
    if not good.any():
        print(f'  Zeff all-NaN in {Path(kprof_fp).name}, using {default}')
        return float(default)
    return float(np.interp(psi_eval, psi[good], z_t[good]))


def load_equilibrium(mhd_fp, kprof_fp):
    """Build the ESCAPE_state inputs for one (g-file, IDA netCDF) pair.

    Returns
    -------
    equil_params : dict
        read_eqdsk dictionary of the g-file (psirz, fpol, pres, ip, rzout, ...),
        which is what ESCAPE_state.set_equil_params consumes.
    kprof_params : dict
        T_e [keV] / n_e [m^-3] on their psi_N grids from kprof_load. The IDA
        files carry no main-ion channels (only carbon), so n_i = n_e is set
        explicitly here (quasi-neutrality) and T_i is left out so that
        ESCAPE_state takes T_i = T_rat * T_e.
    """
    equil_params = mhd_load(None, mhd_loc, mhd_fp)
    kprof_params = kprof_load(kprof_loc=kprof_loc, kprof_fp=kprof_fp)
    kprof_params['n_i'] = kprof_params['n_e'].copy()          # m^-3
    kprof_params['psin_ni'] = kprof_params['psin_ne'].copy()
    return equil_params, kprof_params


def kprof_to_profiles(kprof_params):
    """kprof_params -> the layout calc_pressure_profile expects, so the
    experimental pedestal pressure comes from the same profiles ESCAPE is fed.
    With no Ti/ni keys, calc_pressure_profile uses p = 2*ne*Te."""
    return {
        'psi_N_ne': kprof_params['psin_ne'],
        'ne': kprof_params['n_e'],            # m^-3
        'psi_N_Te': kprof_params['psin_Te'],
        'Te': kprof_params['T_e'],            # keV
        'units': {'ne': 'm^-3', 'Te': 'keV'},
    }


def state_to_dict(state):
    """Picklable snapshot of an ESCAPE state's attributes, for np.save.

    The state object itself can't be pickled: it holds the juliacall EPEDNN
    model and (after a Firedrake solve) Firedrake meshes/Functions. Those
    attributes are left out and their names recorded under '_not_saved'.
    """
    saved, not_saved = {}, []
    for name, val in vars(state).items():
        try:
            pickle.dumps(val)
        except Exception:
            not_saved.append(name)
        else:
            saved[name] = val
    saved['_not_saved'] = not_saved
    return saved


# ESCAPE parameters
ESCAPE_tol_max = 1e-5
ESCAPE_iter_max = 5
out_dir = output_dir
ne_x0 = None, # m^-3, electron density at the separatrix (boundary condition, default is to use from profiles)        
quasineutral_flag = True
T_rat_flag = True # True if using a temperature ratio between ions and electrons, False if doing something else
T_rat = 1
pol_norm = False # True for when the poloidal flux is not normalized by 2pi. COCOS 7 convention is pol_norm=False, so poloidal flux is normalized by 2pi
species = 'D' # species of ions, currently supporting: D, D-T
regime_flag = 'PT H-mode' # regime of the plasma, currently supporting: 'PT H-mode', 'NT'
verbose_ESCAPE = False

# Saarelma-Connor initialization parameters
sc_params = {
    'psi_N_inner_boundary': 0.85, # normalized poloidal flux at the inner boundary (boundary condition); overridden by find_inner_boundary if nFC_threshold or nCX_threshold is set
    'nFC_threshold': None, # fraction of nFC at the separatrix below which the inner boundary is placed (None to disable)
    'nCX_threshold': None, # fraction of nCX at the separatrix below which the inner boundary is placed (None to disable)
    'x_method': 'radas', # method to use for the cross-section rates, currently supporting: 'adas', 'radas'
    'verbose': False,
    'model': sc_model,
    'implementation': sc_implementation,
}

# Saarelma-Connor (SC) shared solver parameters
sc_params.update({
    'x_res':40,
    'ne_grad_bc_loc':ne_grad_bc_loc,
    'picard_max_it':50,
    'picard_rtol':1e-6,
    'picard_relax':1.0,
    'reuse_setup':False,
})

# SC scipy specific
sc_params.update({
    'bvp_tol':1e-6,
    'bvp_max_nodes':5000,
})

# SC firedrake specific
sc_params.update({
    'fe_degree':2, # firedrake specific
    'grad_bc_tol':1e-8,
    'grad_bc_max_it':25,
    'grad_bc_seed':None,
    'linear_solver':"lu",
    'ksp_rtol':1e-8,
    'ksp_max_it':200,
})

# SC 3D specific
sc_params.update({
    'nCX_ic':"scale nFC", # 'scale nFC' or 'solve' or 'state'
    'nFC_ic':"solve",
    'neutrals_treatment':"fem", # only for firedrake (FEM)
    'n_neutral_sub':4001, # only for firedrake (FEM)
})

# SC 1D specific
sc_params.update({
    'eq6_form':"complete",
    'first_step':"auto",
})

# EPEDNN parameters
epednn_params = {
    'epednn_model': 'EPED1', # 'EPED1' or 'EPED_SPARC'
    'pres_gfile': False, # use pressure from kprofs
}


import itertools

# ne_x0s = [None, 1e20, 2e20] # m^-3, manually specify outer bc for electron density
ne_x0s = [None] # m^-3, manually specify outer bc for electron density

scan_total = len(alpha_crits) * len(C_KBMs) * len(De_chie_etgs) * len(nFC_x0s) * len(ncx_x0_ratios) * len(ne_x0s)
print(f'Total number of scans: {scan_total}*{equil_num}')

for mhd_fp, kprof_fp in equilibria:
    equil_tag = Path(mhd_fp).name[1:]
    out_path = Path(output_dir) / equil_tag
    out_path.mkdir(parents=True, exist_ok=True)

    # Zeff is a profile in the OMFIT netCDF, so take it per equilibrium.
    Z_eff = read_omfitnc_zeff(kprof_fp)  # dimensionless, at psi_N = ZEFF_PSI_N
    Z_i = 1

    # ESCAPE_state inputs, plus the same profiles in calc_pressure_profile form
    # so out_dict['profiles'] and the experimental pedestal pressure match them.
    equil_params, kprof_params = load_equilibrium(mhd_fp, kprof_fp)
    profiles = kprof_to_profiles(kprof_params)
    psi_N_p, p_Pa, p_mode = calc_pressure_profile(profiles)

    # P_tot_e: heating power to electrons [W]. No per-shot power in the inputs,
    # so apply the set-representative estimate (see P_HEAT_TOT above).
    P_tot_e = P_HEAT_TOT * ELECTRON_FRAC

    print(f'{equil_tag}: Z_eff = {Z_eff:.3f}, P_tot_e = {P_tot_e/1e6:.2f} MW, pressure: {p_mode}')

    out_dir_ref = Path(out_path) # out_dir / equil_num

    j=0
    for ne_x0 in ne_x0s:
        i=0
        for combo in itertools.product(alpha_crits, C_KBMs, De_chie_etgs, nFC_x0s, ncx_x0_ratios):
            free_params = {
                'alpha_crit': combo[0],
                'C_KBM': combo[1],
                'De_chie_etg': combo[2],
                'nFC_x0': combo[3],
                'ncx_x0_ratio': combo[4]
            }
            sc_params['free_params'] = free_params

            out_dir = out_dir_ref / Path(f'neouter{j}_fp{i}') # out_dir / equil_num / neouterj_fpi
            out_dir.mkdir(parents=True, exist_ok=True)


            try:
                t0 = time.perf_counter()
                state = ESCAPE_solve(
                    sc_params,
                    epednn_params,
                    Z_i = Z_i, # Z of ions
                    Z_eff = Z_eff, 
                    P_tot_e = P_tot_e, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox
                    ne_x0 = ne_x0, # m^-3, electron density at the separatrix (boundary condition, default is to use from profiles)        
                    equil_params = equil_params, # dictionary of required equilibrium parameters
                    kprof_params = kprof_params, # dictionary of required kinetic profile (density, temperature) parameters
                    quasineutral_flag = quasineutral_flag,
                    T_rat_flag = T_rat_flag, # True if using a temperature ratio between ions and electrons, False if doing something else
                    T_rat = T_rat,
                    pol_norm = pol_norm, # True for when the poloidal flux is not normalized by 2pi. COCOS 7 convention is pol_norm=False, so poloidal flux is normalized by 2pi
                    species = species, # species of ions, currently supporting: D, D-T
                    regime_flag = regime_flag, # regime of the plasma, currently supporting: 'PT H-mode', 'NT'
                    ESCAPE_iter_max = ESCAPE_iter_max, 
                    ESCAPE_tol_max = ESCAPE_tol_max,
                    out_dir = out_dir, # directory to output files
                    verbose = verbose_ESCAPE,
                )
                t1 = time.perf_counter()


                # ESCAPE pedestal from the final EPEDNN call on the returned state
                ped_wid = state.pedestal_width      # normalized poloidal flux
                # EPED1 returns MPa (as ESCAPE_solve assumes); convert to Pa to
                # match the experimental p_Pa below
                ped_h_out = state.pedestal_pressure * 1e6 # Pa

                # Experimental pedestal pressure at psi_N = 1 - ped_wid, from the
                # input profiles (psi_N_p / p_Pa built once per equilibrium above).
                psi_ped_top = 1.0 - float(ped_wid)
                p_at_ped_top = float(np.interp(psi_ped_top, psi_N_p, p_Pa))

                # Save model output
                out_dict = {
                    'mhd_fp': mhd_fp,
                    'kprof_fp': kprof_fp,
                    'profiles': profiles,
                    'ESCAPE_ped_h': ped_h_out, # Pa
                    'ESCAPE_ped_wid': ped_wid,
                    'experimental_ped_h': p_at_ped_top, # Pa
                    'free_params': free_params,
                    'p_mode': p_mode,
                    'Zeff': Z_eff,
                    'P_tot_e': P_tot_e,
                    'i': i,
                    'state': state_to_dict(state), # picklable ESCAPE state attributes
                    'ESCAPE_runtime': t1-t0,
                }
                np.save(out_dir / 'out_dict.npy', out_dict)
            except Exception as e:
                out_dict = {'failed': True, 'free_params': free_params, 'error': str(e)}
                np.save(out_dir / 'failed.npy', out_dict)

            if i%50 == 0:
                print(f'Scan {i} of {scan_total} completed')
            i+=1
        j+=1