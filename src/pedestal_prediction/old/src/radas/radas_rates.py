"""
Ionisation (SCD) and charge-exchange (CCD) rate coefficients from local
RADAS ADF11 dumps under ``radas_dir/data_files/``.

API matches ``src.adas.adas_ionisation.scd_adas`` / ``src.adas.adas_cx.scx_adas``:
inputs ``ne`` [m^-3], ``Te`` [eV]; outputs rate coefficients [m^3/s].
"""
from __future__ import annotations

import os

import numpy as np

from src.adas.adas_ionisation import make_adf11_interpolator

RADAS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RADAS_DIR, 'radas_dir', 'data_files')

_FILE_MAP = {
    'D': {
        'scd': 'deuterium_effective_ionisation.dat',
        'ccd': 'deuterium_charge_exchange_cross_coupling.dat',
    },
    'T': {
        'scd': 'tritium_effective_ionisation.dat',
        'ccd': 'tritium_charge_exchange_cross_coupling.dat',
    },
}

_interps = {}


def _get_interp(isotope: str, kind: str):
    """Lazy-load ADF11 interpolator for isotope ('D'|'T') and kind ('scd'|'ccd')."""
    key = (isotope, kind)
    if key not in _interps:
        iso = isotope.upper()
        if iso not in _FILE_MAP:
            raise ValueError(f"isotope must be 'D' or 'T', got {isotope!r}")
        path = os.path.join(DATA_DIR, _FILE_MAP[iso][kind])
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Missing RADAS ADF11 file: {path}')
        _interps[key] = make_adf11_interpolator(path, block=0)
    return _interps[key]


def _eval_rate(isotope: str, kind: str, ne_m3, Te_eV):
    """Evaluate ADF11 rate [m^3/s] at ne [m^-3], Te [eV]."""
    interp = _get_interp(isotope, kind)
    Te_arr = np.atleast_1d(np.asarray(Te_eV, dtype=float))
    ne_arr = np.atleast_1d(np.asarray(ne_m3, dtype=float))
    if Te_arr.size != ne_arr.size:
        if ne_arr.size == 1:
            ne_arr = np.full(Te_arr.shape, float(ne_arr[0]))
        elif Te_arr.size == 1:
            Te_arr = np.full(ne_arr.shape, float(Te_arr[0]))
        else:
            raise ValueError(
                f'ne and Te shapes incompatible: ne={ne_arr.shape}, Te={Te_arr.shape}'
            )
    rate_cm3_s = interp(Te_arr, ne_arr * 1e-6)
    rate_m3_s = rate_cm3_s * 1e-6
    # Preserve scalar-in / scalar-out like scd_adas / scx_adas
    Te_in = np.asarray(Te_eV)
    ne_in = np.asarray(ne_m3)
    if Te_in.ndim == 0 and ne_in.ndim == 0:
        return float(np.asarray(rate_m3_s).ravel()[0])
    return np.asarray(rate_m3_s, dtype=float)


def scd_radas(ne_m3, Te_eV, isotope='D'):
    """
    Effective ionisation rate coefficient from RADAS SCD data.

    Parameters
    ----------
    ne_m3 : array-like
        Electron density [m^-3].
    Te_eV : array-like
        Electron temperature [eV].
    isotope : {'D', 'T'}
        Hydrogen isotope file set to use.

    Returns
    -------
    float or ndarray
        Ionisation rate coefficient [m^3/s].
    """
    return _eval_rate(isotope, 'scd', ne_m3, Te_eV)


def scx_radas(ne_m3, Te_eV, isotope='D'):
    """
    Charge-exchange rate coefficient from RADAS CCD (cross-coupling) data.

    Parameters
    ----------
    ne_m3 : array-like
        Electron density [m^-3] (ADF11 CCD is tabulated vs ne, Te).
    Te_eV : array-like
        Electron temperature [eV].
    isotope : {'D', 'T'}
        Hydrogen isotope file set to use.

    Returns
    -------
    float or ndarray
        CX rate coefficient [m^3/s].
    """
    return _eval_rate(isotope, 'ccd', ne_m3, Te_eV)
