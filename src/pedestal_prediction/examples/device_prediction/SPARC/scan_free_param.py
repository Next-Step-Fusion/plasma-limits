#!/usr/bin/env python3
"""
Free-parameter scan of the Saarelma-Connor pedestal model for the SPARC
PRD equilibrium.

Initializes MHD + kinetic profiles the same way as SPARC.ipynb (gEQDSK from
SPARCPublic / local file, TRANSP or CGYRO profiles on a psi_N-consistent rho
grid, POPCON Ohmic/RF → P_tot_e and Zeff), then explores the free-parameter
space with update_free_params + solve_coupled_nondim (no profiles_loop_solve /
EPEDNN).
"""
from __future__ import annotations

import csv
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent  # saarelma-conner-ped/
sys.path.insert(0, str(ROOT))

tokamaker_python_path = os.getenv('OFT_ROOTPATH')
if tokamaker_python_path is not None:
    sys.path.append(os.path.join(tokamaker_python_path, 'python'))

from src.solver_nondim import saarelma_connor_nondim

# ── Scan grid ────────────────────────────────────────────────────────────────
N = 3
alpha_crits = np.logspace(-1, 1, N)
C_KBMs = np.logspace(-1, 1, N)
De_chie_etgs = np.logspace(-1, 1, N)
nFC_x0s = np.logspace(15, 17, 3)
ncx_x0_ratios = np.logspace(0.1, 1.25, N)
psi_val = 0.85
x_res = 50
verbose = False

# ── SPARC input paths (same directory as SPARC.ipynb) ─────────────────────────
_SPARC_PRD = (
    'https://raw.githubusercontent.com/cfs-energy/SPARCPublic/'
    'main/PrimaryReferenceDischarge'
)
_SPARC_GEQDSK = '2 - SPARC_DN_PRD_freegs_20221013'
MHD_FP = HERE / 'gSPARC_DN_PRD.geqdsk'
TRANSP_PATH = HERE / '5 - transp_20221013.txt'
CGYRO_PATH = HERE / '6 - cgyro_20221013.txt'
POPCON_PATH = HERE / '1 - PRD_POPCON_20221013.csv'
PROFILE_SOURCE = 'cgyro' # Kinetic profile source for the solve: 'cgyro' or 'transp'


scan_success_dir = HERE / (f'scan_results_SPARC_{PROFILE_SOURCE}_N{N}_psi{psi_val}')
success_fp = Path('success_SPARC.txt')
failure_fp = Path('failure_SPARC.txt')
error_messages_fp = Path('error_messages_SPARC.txt')
SOLVE_KW = dict(
    x_res=x_res,
    fe_degree=2,
    initial_guess='tanh',
    ne_inner_bc='neumann',
    linear_solver='lu',
    nCX_ic='scale nFC',
    kbm_treatment='picard',
    kbm_gate_eps=0.1,
    picard_gate_mode='average',
    picard_max_it=50,
    picard_rtol=1e-8,
    picard_relax=1.0,
    bc_origin='p-file',
    neutrals_treatment="fem",
    verbose=verbose,
)


# ── SPARC loaders (mirrors SPARC.ipynb) ───────────────────────────────────────

def fetch_text(filename: str) -> str:
    url = f'{_SPARC_PRD}/{urllib.parse.quote(filename)}'
    print(f'Fetching {filename} from SPARCPublic...', end=' ', flush=True)
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode()
    print('done.')
    return text


def ensure_sparc_geqdsk(mhd_fp: Path = MHD_FP) -> Path:
    """Use existing gEQDSK or download from SPARCPublic."""
    if mhd_fp.exists():
        print(f'Using existing gEQDSK: {mhd_fp}')
        return mhd_fp
    mhd_fp.write_text(fetch_text(_SPARC_GEQDSK))
    print(f'Wrote {mhd_fp}')
    return mhd_fp


def _read_sparcpublic_profiles(path: Path) -> dict:
    """Parse SPARCPublic-style profile text (TRANSP / CGYRO layout).

    Expected sections: rho, polflux, q, te [keV], ti [keV], ne [10^19/m^3],
    and optionally ni. Returns Te/Ti in keV and ne/ni in m^-3.
    """
    sections = {}
    current = None
    current_unit = ''
    values = []

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('#'):
                if current is not None:
                    sections[current] = {
                        'data': np.asarray(values, dtype=float),
                        'unit': current_unit,
                    }
                body = line[1:].strip()
                name, _, unit = body.partition('|')
                current = name.strip().lower()
                current_unit = unit.strip()
                values = []
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                values.append(float(parts[-1]))

    if current is not None:
        sections[current] = {
            'data': np.asarray(values, dtype=float),
            'unit': current_unit,
        }

    if 'rho' not in sections:
        raise KeyError(f'No rho grid in {path}. Sections: {list(sections)}')

    rho = sections['rho']['data']
    out = {'rho': rho, 'units': {'rho': sections['rho']['unit']}}
    for key in ('polflux', 'q'):
        if key in sections:
            out[key] = sections[key]['data']
            out['units'][key] = sections[key]['unit']

    for src, dst in (('te', 'Te'), ('ti', 'Ti')):
        if src in sections:
            if len(sections[src]['data']) != len(rho):
                raise ValueError(f'{src} length != rho length')
            out[dst] = sections[src]['data'].copy()
            out['units'][dst] = 'keV'

    def _density_to_si(name, unit, data):
        u = unit.lower().replace(' ', '')
        if '10^20' in u or '10**20' in u or '1e20' in u:
            return data * 1e20
        if '10^19' in u or '10**19' in u or '1e19' in u:
            return data * 1e19
        if u in ('m^-3', 'm-3', '1/m^3', '/m^3'):
            return data.copy()
        raise ValueError(f'Unrecognized {name} unit {unit!r}')

    for src, dst in (('ne', 'ne'), ('ni', 'ni')):
        if src in sections:
            if len(sections[src]['data']) != len(rho):
                raise ValueError(f'{src} length != rho length')
            out[dst] = _density_to_si(src, sections[src]['unit'], sections[src]['data'])
            out['units'][dst] = 'm^-3'

    return out


def read_transp_profiles(path: Path) -> dict:
    """Parse SPARCPublic-style TRANSP profile text file."""
    return _read_sparcpublic_profiles(path)


def read_cgyro_profiles(path: Path) -> dict:
    """Parse SPARCPublic-style CGYRO profile text file.

    Same section layout as TRANSP (``6 - cgyro_20221013.txt``): rho, polflux,
    q, te/ti [keV], ne [10^19/m^3]. Returns Te/Ti in keV and ne in m^-3.
    """
    return _read_sparcpublic_profiles(path)


def build_manual_profs(profiles: dict) -> dict:
    """manual_profs for kprof_loc='manual rho grid' with psi_N-consistent rho."""
    if 'polflux' not in profiles:
        raise KeyError('Profile file needs polflux to map rho -> psi_N correctly')
    pf = profiles['polflux']
    psi_N = (pf - pf[0]) / (pf[-1] - pf[0])
    rho_psi = np.sqrt(np.clip(psi_N, 0.0, None))

    manual_profs = {
        'Te': profiles['Te'],
        'rho_Te': rho_psi,
        'ne': profiles['ne'] / 1e20,
        'rho_ne': rho_psi,
    }
    if 'Ti' in profiles:
        manual_profs['Ti'] = profiles['Ti']
        manual_profs['rho_Ti'] = rho_psi
    if 'ni' in profiles:
        manual_profs['ni'] = profiles['ni'] / 1e20
        manual_profs['rho_ni'] = rho_psi
    return manual_profs

def read_popcon(path: Path, keys):
    wanted = {k.lower(): k for k in keys}
    found = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = (row.get('Variable') or row.get('variable') or '').strip()
            if name.lower() not in wanted:
                continue
            val = float(row.get('Value') or row.get('value'))
            unit = (row.get('Unit') or row.get('unit') or '').strip()
            found[wanted[name.lower()]] = {'value': val, 'unit': unit}
    missing = [k for k in keys if k not in found]
    if missing:
        raise KeyError(f'Missing {missing} in {path}')
    return found


def _to_watts(value, unit):
    unit_to_W = {'': 1.0, 'w': 1.0, 'kw': 1e3, 'mw': 1e6, 'gw': 1e9}
    key = (unit or 'W').strip().lower()
    if key not in unit_to_W:
        raise ValueError(f'Unrecognized power unit {unit!r}')
    return value * unit_to_W[key]


def load_sparc_inputs(profile_source: str = PROFILE_SOURCE):
    """Return (mhd_fp, manual_profs, P_tot_e, Zeff) like SPARC.ipynb.

    ``profile_source`` is ``'cgyro'`` (default) or ``'transp'``.
    """
    mhd_fp = ensure_sparc_geqdsk()

    source = profile_source.strip().lower()
    if source == 'cgyro':
        prof_path = CGYRO_PATH
        profiles = read_cgyro_profiles(prof_path)
    elif source == 'transp':
        prof_path = TRANSP_PATH
        profiles = read_transp_profiles(prof_path)
    else:
        raise ValueError(
            f"profile_source must be 'cgyro' or 'transp', got {profile_source!r}"
        )
    manual_profs = build_manual_profs(profiles)
    print(f'Loaded {prof_path.name} ({source}): '
          f'{len(profiles["rho"])} profile points')

    popcon = read_popcon(POPCON_PATH, keys=('Ohmic power', 'RF power', 'Zeff'))
    P_ohmic = _to_watts(popcon['Ohmic power']['value'], popcon['Ohmic power']['unit'])
    P_RF = _to_watts(popcon['RF power']['value'], popcon['RF power']['unit'])
    Zeff = float(popcon['Zeff']['value'])
    P_tot_e = (P_ohmic + P_RF) / 2.0
    print(f'POPCON: P_ohmic={P_ohmic/1e6:.2f} MW, P_RF={P_RF/1e6:.2f} MW, '
          f'P_tot_e={P_tot_e/1e6:.2f} MW, Zeff={Zeff:.3g}')

    return str(mhd_fp), manual_profs, P_tot_e, Zeff

# ── Main scan ────────────────────────────────────────────────────────────────

def main():
    mhd_fp, manual_profs, P_tot_e, Zeff = load_sparc_inputs()

    eq_tag = Path(mhd_fp).name[1:]  # e.g. SPARC_DN_PRD.geqdsk
    eq_dir = scan_success_dir / eq_tag
    eq_dir.mkdir(parents=True, exist_ok=True)

    eq_success_fp = eq_dir / success_fp
    eq_failure_fp = eq_dir / failure_fp
    eq_error_fp = eq_dir / error_messages_fp

    for old in eq_dir.glob('ne_*.npy'):
        old.unlink()
    for fp in (eq_success_fp, eq_failure_fp, eq_error_fp):
        fp.unlink(missing_ok=True)
    with open(eq_success_fp, 'w') as f:
        f.write('eq_tag, alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, '
                'psi_N_inner, L2_error\n')
    with open(eq_failure_fp, 'w') as f:
        f.write('eq_tag, alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, '
                'psi_N_inner\n')
    with open(eq_error_fp, 'w') as f:
        f.write('eq_tag, alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, '
                'psi_N_inner, message\n')

    # Build the base model ONCE — free params are updated in the scan loop.
    print('Building Saarelma-Connor base model for SPARC...')
    base_model = saarelma_connor_nondim(
        P_tot_e=P_tot_e,
        alpha_crit=round(float(alpha_crits[0]), 3),
        C_KBM=round(float(C_KBMs[0]), 3),
        De_chie_etg=round(float(De_chie_etgs[0]), 3),
        nFC_x0=round(float(nFC_x0s[0]), 3),
        ncx_x0_ratio=round(float(ncx_x0_ratios[0]), 3),
        psi_N_inner_boundary=psi_val,
        mhd_fp=mhd_fp,
        kprof_loc='manual rho grid',
        manual_profs=manual_profs,
        species='D-T',
        Z_i=Zeff,
        x_method='radas',
        verbose=verbose,
    )
    print('Base model ready — starting free-parameter scan.')

    n_ok = n_fail = 0
    n_total = (len(alpha_crits) * len(C_KBMs) * len(De_chie_etgs)
               * len(nFC_x0s) * len(ncx_x0_ratios))
    i_combo = 0

    for alpha_crit in alpha_crits:
        for C_KBM in C_KBMs:
            for De_chie_etg in De_chie_etgs:
                for nFC_x0 in nFC_x0s:
                    for ncx_x0_ratio in ncx_x0_ratios:
                        i_combo += 1
                        ac = round(float(alpha_crit), 3)
                        ck = round(float(C_KBM), 3)
                        de = round(float(De_chie_etg), 3)
                        nf = round(float(nFC_x0), 3)
                        nc = round(float(ncx_x0_ratio), 3)
                        try:
                            base_model.update_free_params(
                                alpha_crit=ac,
                                C_KBM=ck,
                                De_chie_etg=de,
                                nFC_x0=nf,
                                ncx_x0_ratio=nc,
                                psi_N_inner_boundary=psi_val,
                            )
                            x_sol, ne_sol, nFC_sol, nCX_sol, T_e_pres, psi_N_pres = (
                                base_model.solve_coupled_nondim(**SOLVE_KW)
                            )
                            sol = {
                                'x': x_sol, 'y': ne_sol,
                                'nFC': nFC_sol, 'nCX': nCX_sol,
                                'T_e_pres': T_e_pres, 'psi_N_pres': psi_N_pres,
                            }
                        except Exception as e:
                            n_fail += 1
                            with open(eq_failure_fp, 'a') as f:
                                f.write(
                                    f'{eq_tag}, {ac}, {ck}, {de}, {nf}, {nc}, '
                                    f'{psi_val:.4f}\n'
                                )
                            with open(eq_error_fp, 'a') as f:
                                f.write(
                                    f'{eq_tag}, {ac}, {ck}, {de}, {nf}, {nc}, '
                                    f'{psi_val:.4f}, {e}\n'
                                )
                        else:
                            n_ok += 1
                            # L2 vs input kinetic profiles on the solution grid
                            ne_ref = interp1d(
                                base_model.x_init, base_model.n_e_pres,
                                kind='linear', bounds_error=False,
                                fill_value='extrapolate',
                            )(x_sol)
                            L2_error = (
                                np.linalg.norm(ne_sol - ne_ref)
                                / np.linalg.norm(ne_ref)
                            )
                            with open(eq_success_fp, 'a') as f:
                                f.write(
                                    f'{eq_tag}, {ac}, {ck}, {de}, {nf}, {nc}, '
                                    f'{psi_val:.4f}, {L2_error}\n'
                                )
                            np.save(
                                eq_dir / (
                                    f'ne_a{ac}_C{ck}_D{de}_n{nf}_nc{nc}'
                                    f'_b{psi_val:.4f}'
                                ),
                                sol,
                                allow_pickle=True,
                            )

                        if i_combo % 20 == 0 or i_combo == n_total:
                            print(f'  {i_combo}/{n_total}  '
                                  f'(ok={n_ok}, fail={n_fail})')

    print(f'Done. Saved under {eq_dir}  (ok={n_ok}, fail={n_fail})')
    del base_model


if __name__ == '__main__':
    main()
