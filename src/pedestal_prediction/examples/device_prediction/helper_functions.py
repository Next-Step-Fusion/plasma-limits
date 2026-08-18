import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import tempfile
import urllib.request # needed for geqdsk import
ROOT = Path.cwd().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.profiles_loop_solve import profiles_loop_solve

tokamaker_python_path = os.getenv('OFT_ROOTPATH')
if tokamaker_python_path is not None:
    sys.path.append(os.path.join(tokamaker_python_path,'python'))
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import create_isoflux, read_eqdsk


def read_sparcpublic_profiles(path):
    """
    Parse a SPARCPublic-style profile text file (TRANSP or CGYRO layout).

    Format: blocks headed by ``# name | unit``, each followed by
    ``index  value`` rows.

    Returns
    -------
    dict
        Keys for available quantities, each an ndarray. Always includes
        ``rho``. Electron/ion profiles added when present:
          - ne  [m^-3]
          - Te  [keV]
          - Ti  [keV]   (if present)
          - ni  [m^-3]  (if present)
        Also keeps raw auxiliaries ``polflux``, ``q`` when present, plus
        ``units`` metadata.
    """
    path = Path(path)
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
                # "# te | keV"  or  "# rho | -"
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
        raise KeyError(f'No rho grid found in {path}. Sections: {list(sections)}')

    rho = sections['rho']['data']
    out = {
        'rho': rho,
        'units': {'rho': sections['rho']['unit']},
    }
    for key in ('polflux', 'q'):
        if key in sections:
            out[key] = sections[key]['data']
            out['units'][key] = sections[key]['unit']

    # Temperatures: file stores keV
    for src, dst in (('te', 'Te'), ('ti', 'Ti')):
        if src in sections:
            if len(sections[src]['data']) != len(rho):
                raise ValueError(
                    f'{src} length {len(sections[src]["data"])} != rho length {len(rho)}'
                )
            out[dst] = sections[src]['data'].copy()
            out['units'][dst] = 'keV'

    # Densities: convert known file units -> m^-3
    def _density_to_si(name, unit, data):
        u = unit.lower().replace(' ', '')
        if '10^20' in u or '10**20' in u or '1e20' in u:
            return data * 1e20
        if '10^19' in u or '10**19' in u or '1e19' in u:
            return data * 1e19
        if u in ('m^-3', 'm-3', '1/m^3', '/m^3'):
            return data.copy()
        raise ValueError(f'Unrecognized {name} unit {unit!r} in {path}')

    for src, dst in (('ne', 'ne'), ('ni', 'ni')):
        if src in sections:
            if len(sections[src]['data']) != len(rho):
                raise ValueError(
                    f'{src} length {len(sections[src]["data"])} != rho length {len(rho)}'
                )
            out[dst] = _density_to_si(src, sections[src]['unit'], sections[src]['data'])
            out['units'][dst] = 'm^-3'

    return out


def psi_n_and_rho_psi(profiles):
    """Return (psi_N, rho_psi=sqrt(psi_N)) from polflux."""
    if 'polflux' not in profiles:
        raise KeyError('Profile file needs polflux to map rho -> psi_N correctly')
    pf = profiles['polflux']
    psi_N = (pf - pf[0]) / (pf[-1] - pf[0])
    rho_psi = np.sqrt(np.clip(psi_N, 0.0, None))
    return psi_N, rho_psi


def build_manual_profs(profiles):
    """manual_profs for kprof_loc='manual rho grid' with psi_N-consistent rho."""
    _, rho_psi = psi_n_and_rho_psi(profiles)
    manual_profs = {
        'Te': profiles['Te'],
        'rho_Te': rho_psi,
        'ne': profiles['ne'] / 1e20,   # m^-3 -> 10^20 m^-3 (solver multiplies by 1e20)
        'rho_ne': rho_psi,
    }
    if 'Ti' in profiles:
        manual_profs['Ti'] = profiles['Ti']
        manual_profs['rho_Ti'] = rho_psi
    if 'ni' in profiles:
        manual_profs['ni'] = profiles['ni'] / 1e20
        manual_profs['rho_ni'] = rho_psi
    return manual_profs


def _psi_n_from_rho(rho):
    """Map a normalized radial coordinate to psi_N (rho_psi = sqrt(psi_N))."""
    rho = np.asarray(rho, dtype=float)
    return np.clip(rho ** 2, 0.0, None)


def _interp_to_psi(psi_src, y_src, psi_dst):
    """Linear interpolate y(psi_src) onto psi_dst (both 1-D)."""
    psi_src = np.asarray(psi_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    order = np.argsort(psi_src)
    return np.interp(psi_dst, psi_src[order], y_src[order])


def _density_to_m3(ne, profiles):
    """Return density in m^-3 from SPARC profiles (SI) or manual_profs (10^20 m^-3)."""
    ne = np.asarray(ne, dtype=float)
    units = profiles.get('units') or {}
    unit = str(units.get('ne', '')).lower().replace(' ', '')
    if 'm^-3' in unit or unit in ('m-3', '1/m^3', '/m^3'):
        return ne
    if '10^20' in unit or '10**20' in unit or '1e20' in unit:
        return ne * 1e20
    if '10^19' in unit or '10**19' in unit or '1e19' in unit:
        return ne * 1e19
    # SPARCPublic kinetic dict: SI. Solver / ARC manual_profs: 10^20 m^-3.
    if 'polflux' in profiles:
        return ne
    if 'rho_ne' in profiles or 'rho_Te' in profiles:
        return ne * 1e20
    if np.nanmax(np.abs(ne)) > 1e18:
        return ne
    return ne * 1e20


def calc_pressure_profile(profiles):
    """
    Thermal pressure from kinetic profiles on a psi_N grid.

    Defaults to ``p = 2 * ne * Te`` (Te = Ti, ne = ni). If ``Ti`` is present,
    uses ``p = ne * Te + ni * Ti``, with ``ni = ne`` when ``ni`` is missing.

    Accepts either:
      - SPARCPublic ``profiles`` (``ne`` [m^-3], ``Te``/``Ti`` [keV], ``polflux``)
      - solver ``manual_profs`` (``ne`` [10^20 m^-3], ``Te``/``Ti`` [keV],
        ``rho_ne`` / ``rho_Te``; ion grids ``rho_ni`` / ``rho_Ti`` optional)

    Parameters
    ----------
    profiles : dict
        Kinetic profiles in either of the layouts above.

    Returns
    -------
    psi_N : ndarray
    p : ndarray
        Pressure in Pa, on ``psi_N``.
    mode : str
        Formula used.
    """
    eV_to_J = 1.602176634e-19

    if 'polflux' in profiles:
        psi_N, _ = psi_n_and_rho_psi(profiles)
        ne = _density_to_m3(profiles['ne'], profiles)
        Te_keV = np.asarray(profiles['Te'], dtype=float)
        Ti_keV = np.asarray(profiles['Ti'], dtype=float) if 'Ti' in profiles else None
        ni = _density_to_m3(profiles['ni'], profiles) if 'ni' in profiles else None
    elif 'psi_N_ne' in profiles and 'psi_N_Te' in profiles:
        psi_N = profiles['psi_N_ne']
        ne = profiles['ne']
        Te_keV = _interp_to_psi(profiles['psi_N_Te'], profiles['Te'], psi_N)
        Ti_keV = _interp_to_psi(profiles['psi_N_ne'], profiles['Ti'], psi_N) if 'Ti' in profiles else None
        ni = profiles['ni'] if 'ni' in profiles else None
    else:
        rho_ne = profiles.get('rho_ne', profiles.get('rho'))
        rho_Te = profiles.get('rho_Te', rho_ne)
        if rho_ne is None or 'ne' not in profiles or 'Te' not in profiles:
            raise KeyError(
                "Need either polflux+ne+Te, or rho_ne/rho_Te+ne+Te "
                f"(keys: {list(profiles)})"
            )
        psi_N = _psi_n_from_rho(rho_ne)
        ne = _density_to_m3(profiles['ne'], profiles)
        Te_keV = _interp_to_psi(_psi_n_from_rho(rho_Te), profiles['Te'], psi_N)
        if 'Ti' in profiles:
            rho_Ti = profiles.get('rho_Ti', rho_Te)
            Ti_keV = _interp_to_psi(_psi_n_from_rho(rho_Ti), profiles['Ti'], psi_N)
        else:
            Ti_keV = None
        if 'ni' in profiles:
            rho_ni = profiles.get('rho_ni', rho_ne)
            ni = _density_to_m3(
                _interp_to_psi(_psi_n_from_rho(rho_ni), profiles['ni'], psi_N),
                profiles,
            )
        else:
            ni = None

    Te_J = Te_keV * 1e3 * eV_to_J  # keV -> J

    if Ti_keV is not None:
        Ti_J = Ti_keV * 1e3 * eV_to_J
        if ni is None:
            ni = ne
            mode = 'ne*Te + ne*Ti (ni=ne)'
        else:
            mode = 'ne*Te + ni*Ti'
        p = ne * Te_J + ni * Ti_J
    else:
        p = 2.0 * ne * Te_J
        mode = '2*ne*Te'

    return psi_N, p, mode


def _print_profile_summary(label, path, profiles):
    psi_N, rho_psi = psi_n_and_rho_psi(profiles)
    print(f'Loaded {path.name} ({label}): {len(profiles["rho"])} points')
    print(f'  quantities: {[k for k in profiles if k != "units"]}')
    print(f'  psi_N mapping: using polflux -> rho_psi=sqrt(psi_N) '
          f'(max |rho_file^2 - psi_N| = {np.max(np.abs(profiles["rho"]**2 - psi_N)):.3f})')
    print(f'  Te axis  = {profiles["Te"][0]:.2f} keV,  edge = {profiles["Te"][-1]:.2f} keV')
    print(f'  ne axis  = {profiles["ne"][0]/1e20:.2f}e20 m^-3,  '
          f'edge = {profiles["ne"][-1]/1e20:.2f}e20 m^-3')
    if 'Ti' in profiles:
        print(f'  Ti axis  = {profiles["Ti"][0]:.2f} keV,  edge = {profiles["Ti"][-1]:.2f} keV')
    else:
        print('  Ti: not in file')
    if 'ni' in profiles:
        print(f'  ni axis  = {profiles["ni"][0]/1e20:.2f}e20 m^-3')
    else:
        print('  ni: not in file')


def read_popcon(path, keys):
    """
    Read named entries from the SPARCPublic POPCON CSV.

    Returns
    -------
    dict
        key -> {'value': float, 'unit': str}
    """
    import csv

    wanted = {k.lower(): k for k in keys}
    found = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
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