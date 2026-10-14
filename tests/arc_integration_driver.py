"""Run ESCAPE once on the ARC example inputs and save the per-iteration results.

The inputs copy ESCAPE's examples/device_prediction/ARC/scan_SCfp_ESCAPE_ARC.py
(cited as ``ARC:<line>``), with the 1D scipy model or the 3D Firedrake model.
EPED1 runs through Julia.

Usage:  python tests/arc_integration_driver.py <out_dir> <result_file>
            [--model {1d,3d}] [--mode {file,memory}]

ESCAPE_solve writes its iteration files into <out_dir> and deletes everything
else in it. The driver then saves the plain values of all iterations as one
dict in <result_file> (np.save, loads with numpy alone).

Modes:
- file: ESCAPE_solve with its default arguments, as the author's drivers call it.
- memory: the in-memory keywords: the EPEDNN model is loaded with
  load_epednn_model and given to ESCAPE_solve as epednn_model.
"""
import argparse
from pathlib import Path

import numpy as np

from escape.ESCAPE.ESCAPE_solve import ESCAPE_solve
from escape.epednn.epednn_call import load_epednn_model
from escape.helpers.parse_equil import mhd_load

ESCAPE_ROOT = Path(__file__).resolve().parents[1] / "dependencies" / "ESCAPE"
ARC_DIR = ESCAPE_ROOT / "examples" / "device_prediction" / "ARC"
PLAIN_SCALARS = (np.generic, bool, int, float, complex, str, type(None))
N_ITER = 5  # ARC:130


def load_digitized_profile(filepath):
    """Copy of ARC:51-76: psi_N clipped to [0, 1], sorted, duplicates at 1 removed."""
    data = np.loadtxt(filepath, delimiter=',')
    psi_N, profile = data[:, 0], data[:, 1]
    psi_N = np.clip(psi_N, 0.0, 1.0)
    order = np.argsort(psi_N, kind='stable')
    psi_N, profile = psi_N[order], profile[order]
    unique = np.append(np.diff(psi_N) > 0, True)
    return psi_N[unique], profile[unique]


def arc_inputs(model='1d'):
    """The ESCAPE_solve arguments of the ARC example's first scan point.

    model='1d' is the 1D scipy model; model='3d' is the ARC default, 3D Firedrake.
    """
    psi_N_ne, ne_ref = load_digitized_profile(ARC_DIR / "ARC_ne.csv")  # ARC:83
    psi_N_Te, Te_ref = load_digitized_profile(ARC_DIR / "ARC_Te.csv")  # ARC:84
    kprof_params = {  # ARC:91-96
        'n_e': ne_ref * 1e20,
        'T_e': Te_ref,
        'psin_ne': psi_N_ne,
        'psin_Te': psi_N_Te,
    }
    equil_params = mhd_load(None, 'eqdsk', str(ARC_DIR / "geqdsk-ARCv3a"))  # ARC:43, ARC:109

    sc_params = {  # ARC:142-150
        'psi_N_inner_boundary': 0.85,
        'nFC_threshold': None,
        'nCX_threshold': None,
        'x_method': 'radas',
        'verbose': False,
    }
    if model == '3d':
        sc_params.update(model='3D', implementation='firedrake')  # ARC:30-31
    else:
        sc_params.update(model='1D', implementation='scipy')
    sc_params.update({  # ARC:153-160
        'x_res': 40,
        'ne_grad_bc_loc': 'inner',  # ARC:32
        'picard_max_it': 50,
        'picard_rtol': 1e-6,
        'picard_relax': 1.0,
        'reuse_setup': False,
    })
    sc_params.update({  # ARC:163-166
        'bvp_tol': 1e-6,
        'bvp_max_nodes': 5000,
    })
    sc_params.update({  # ARC:169-177
        'fe_degree': 2,
        'grad_bc_tol': 1e-8,
        'grad_bc_max_it': 25,
        'grad_bc_seed': None,
        'linear_solver': "lu",
        'ksp_rtol': 1e-8,
        'ksp_max_it': 200,
    })
    sc_params.update({  # ARC:180-185
        'nCX_ic': "scale nFC",
        'nFC_ic': "solve",
        'neutrals_treatment': "fem",
        'n_neutral_sub': 4001,
    })
    sc_params.update({  # ARC:188-191
        'eq6_form': "complete",
        'first_step': "auto",
    })
    sc_params['free_params'] = {  # ARC:18-22, first scan point
        'alpha_crit': 0.01,
        'C_KBM': 0.3,
        'De_chie_etg': 0.1,
        'nFC_x0': 1e15,
        'ncx_x0_ratio': 15,
    }
    epednn_params = {'epednn_model': 'EPED1', 'pres_gfile': False}  # ARC:194-197

    return dict(
        sc_params=sc_params,
        epednn_params=epednn_params,
        Z_i=1,  # ARC:209
        Z_eff=1.52,  # ARC:208
        P_tot_e=(21.5 + 0.8 + 227) * 0.5 * 1e6,  # ARC:48
        ne_x0=None,  # ARC:203
        equil_params=equil_params,
        kprof_params=kprof_params,
        quasineutral_flag=True,  # ARC:133
        T_rat_flag=True,  # ARC:134
        T_rat=1,  # ARC:135
        pol_norm=False,  # ARC:136
        species='D',  # ARC:137
        regime_flag='PT H-mode',  # ARC:138
        ESCAPE_iter_max=N_ITER,
        ESCAPE_tol_max=1e-5,  # ARC:129
        verbose=False,  # ARC:139
    )


def plain(obj):
    """obj without values that need more than numpy to load."""
    if isinstance(obj, dict):
        return {k: plain(v) for k, v in obj.items() if _keep(v)}
    if isinstance(obj, (list, tuple)):
        return type(obj)(plain(v) for v in obj if _keep(v))
    return obj


def _keep(obj):
    if isinstance(obj, (dict, list, tuple)):
        return True
    if isinstance(obj, np.ndarray):
        return obj.dtype != object
    return isinstance(obj, PLAIN_SCALARS)


def main(out_dir, result_file, model='1d', mode='file'):
    inputs = arc_inputs(model)
    if mode == 'memory':
        inputs['epednn_model'] = load_epednn_model(inputs['epednn_params']['epednn_model'])
    ESCAPE_solve(**inputs, out_dir=str(out_dir))
    iters = []
    for k in range(N_ITER):
        path = Path(out_dir) / f"ne_and_Te_iter_{k}.npy"
        if not path.exists():
            break
        iters.append(plain(np.load(path, allow_pickle=True).item()))
    np.save(result_file, {"iterations": iters}, allow_pickle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("result_file", type=Path)
    parser.add_argument("--model", choices=["1d", "3d"], default="1d")
    parser.add_argument("--mode", choices=["file", "memory"], default="file")
    args = parser.parse_args()
    main(args.out_dir, args.result_file, args.model, args.mode)
