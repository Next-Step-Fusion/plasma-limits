import numpy as np
from scipy.interpolate import interp1d

def OMFITnc_load(self, filename=None):
    """
    Reads an OMFITnc file, averages T_e and n_e across the first dimension,
    and interpolates them onto a common, unified psi_N grid.

    Callable either as a bound method (``self.OMFITnc_load(path)``) or as a
    plain function (``OMFITnc_load(path)``); when only one argument is given it
    is taken as the filepath.

    Parameters:
        filename (str): Path to the NetCDF file.

    Returns:
        tuple: (psi_N_unified, Te_1d, ne_1d) as 1D numpy arrays.
        Te_1d in keV, ne_1d in m^-3.

    # Example usage:
    # psi_grid, Te_profile, ne_profile = OMFITnc_load('my_plasma_data.cdf')
    """
    if filename is None:  # called as OMFITnc_load(path), no instance
        self, filename = None, self

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



def read_pfile(path):
    """Parse a pfile into {key: values, f'{key}_psi': psi_N} numpy arrays.

    Each profile block starts with a '201 ...' header line naming the profile;
    parsing stops at the '3 N Z A' impurity block.
    """
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


def _arr(x):
    """Coerce a profile or grid to a float ndarray."""
    return np.asarray(x, dtype=float)


def kprof_load(kprof_loc='pfile', kprof_fp=None, manual_profs=None, psi_N_sep=1.0):
    """Build the ``kprof_params`` dictionary consumed by
    ``SaarelmaConnorBase.set_kprof_params`` using the method specified by the
    ``kprof_loc`` flag.

    This is a pure loader: it reads/repackages profiles and returns them, and
    never touches a solver instance.  Ion temperature/density defaults
    (``T_i = T_rat * T_e`` and ``n_i = n_e``) are NOT applied here -- they are
    the responsibility of ``set_kprof_params``, which applies them whenever the
    optional ion keys are absent from the returned dictionary.

    Parameters
    ----------
    kprof_loc : string
        Which method to use to load kinetic parameters.  One of
        ``'pfile'``, ``'OMFITnc'``, ``'manual psi_N grid'``,
        ``'manual rho grid'``, ``'manual EPEDNN loop'``, ``'manual profs'``.
    kprof_fp : string
        Filepath to the ``kprof_loc``-type file with kinetic parameters
        (used by ``'pfile'`` and ``'OMFITnc'``).
    manual_profs : dict or None
        Caller-supplied profiles.  With ``kprof_loc='pfile'`` a non-None
        ``manual_profs`` overrides T_e only (n_e still comes from the pfile).
    psi_N_sep : float
        psi_N of the last pressure flux surface, used by ``'manual rho grid'``
        to map rho -> psi_N.  This is ``self.psi_N_pres[-1]``, which is 1.0 by
        construction in ``set_equil_params``; it is exposed here so the loader
        does not need an equilibrium.

    Returns
    -------
    kprof_params : dict
        Always contains:
            'T_e'      : keV, electron temperature evaluated at 'psin_Te'
            'psin_Te'  : psi_N values at which T_e is evaluated
            'n_e'      : m^-3, electron density evaluated at 'psin_ne'
            'psin_ne'  : psi_N values at which n_e is evaluated
        Contains, only when the mode supplies them:
            'T_i'      : keV, ion temperature evaluated at 'psin_Ti'
            'psin_Ti'  : psi_N values at which T_i is evaluated
            'n_i'      : m^-3, ion density evaluated at 'psin_ni'
            'psin_ni'  : psi_N values at which n_i is evaluated
        Contains, for 'manual EPEDNN loop' only (passed through for the
        boundary-condition/initial-guess helpers, ignored by
        ``set_kprof_params``):
            'nCX'      : m^-3, charge-exchange neutral density on 'psin_n'
            'nFC'      : m^-3, Franck-Condon neutral density on 'psin_n'
            'psin_n'   : psi_N values at which n_e, n_CX, n_FC are evaluated
    """

    if kprof_loc == 'pfile' and manual_profs is None:  # true pfile for both n_e and T_e
        pf = read_pfile(kprof_fp)

        kprof_params = {
            'T_e': _arr(pf['te(KeV)']),                      # keV
            'psin_Te': _arr(pf['te(KeV)_psi']),
            'n_e': _arr(pf['ne(10^20/m^3)']) * 1e20,         # 10^20/m^3 -> m^-3
            'psin_ne': _arr(pf['ne(10^20/m^3)_psi']),
        }

    elif kprof_loc == 'pfile' and manual_profs is not None:  # manual T_e, pfile n_e
        pf = read_pfile(kprof_fp)

        kprof_params = {
            'T_e': _arr(manual_profs['Te']),                 # keV
            'psin_Te': _arr(manual_profs['psi_N_Te']),
            'n_e': _arr(pf['ne(10^20/m^3)']) * 1e20,         # 10^20/m^3 -> m^-3
            'psin_ne': _arr(pf['ne(10^20/m^3)_psi']),
        }

    elif kprof_loc == 'OMFITnc':
        psi_N_unified, Te, ne = OMFITnc_load(kprof_fp)

        kprof_params = {
            'T_e': _arr(Te),                                 # keV
            'psin_Te': _arr(psi_N_unified),
            'n_e': _arr(ne),                                 # m^-3
            'psin_ne': _arr(psi_N_unified),
        }

    elif kprof_loc in ('manual psi_N grid', 'manual rho grid'):
        # Profiles supplied directly by the caller, on their own radial grid.
        #   'manual psi_N grid'  the grid IS psi_N and is used as given
        #   'manual rho grid'    legacy grid, rho treated as sqrt(psi_N), so
        #                        psi_N = psi_N_sep * rho**2. Only for inputs
        #                        genuinely on a rho grid (e.g. the digitized ARC
        #                        profiles); anything derived from polflux should
        #                        pass psi_N and use 'manual psi_N grid'.
        assert manual_profs is not None, 'manual_profs must be provided'

        def _psi_grid(psi_key, rho_key, fallback_psi=None):
            """psi_N for one profile, from its psi_N_* key or its legacy rho_* key."""
            if manual_profs.get(psi_key) is not None:
                return _arr(manual_profs[psi_key])
            if manual_profs.get(rho_key) is not None:
                rho = _arr(manual_profs[rho_key])
                return psi_N_sep * rho**2
            if fallback_psi is not None:
                return fallback_psi
            raise KeyError(
                f"manual_profs needs '{psi_key}' (or the legacy '{rho_key}')"
            )

        kprof_params = {
            # Full T_e and corresponding psi_N profile
            'T_e': _arr(manual_profs['Te']),                 # keV
            'psin_Te': _psi_grid('psi_N_Te', 'rho_Te'),
            # n_e here feeds the cross-sections and the n_e boundary conditions
            'n_e': _arr(manual_profs['ne']) * 1e20,          # 10^20/m^3 -> m^-3
            'psin_ne': _psi_grid('psi_N_ne', 'rho_ne'),
        }

        # Optional ion profiles. If omitted, set_kprof_params falls back to
        # Ti = T_rat * Te and ni = ne.
        if manual_profs.get('Ti') is not None:
            kprof_params['T_i'] = _arr(manual_profs['Ti'])   # keV
            kprof_params['psin_Ti'] = _psi_grid(
                'psi_N_Ti', 'rho_Ti', fallback_psi=kprof_params['psin_Te']
            )
        if manual_profs.get('ni') is not None:
            kprof_params['n_i'] = _arr(manual_profs['ni']) * 1e20  # 10^20/m^3 -> m^-3
            kprof_params['psin_ni'] = _psi_grid(
                'psi_N_ni', 'rho_ni', fallback_psi=kprof_params['psin_ne']
            )

    elif kprof_loc == 'manual EPEDNN loop':
        assert manual_profs is not None, 'T and n_e, n_CX, n_FC profiles must be provided if T_e_source is epednn'

        kprof_params = {
            'T_e': _arr(manual_profs['Te']),                 # keV
            'psin_Te': _arr(manual_profs['psi_N_Te']),
            'n_e': _arr(manual_profs['ne']),                 # m^-3
            'psin_ne': _arr(manual_profs['psi_N_n']),
            # Neutral profiles used by the BC/initial-guess helpers; carried
            # through here so this mode keeps all of its inputs in one dict.
            'nCX': _arr(manual_profs['nCX']),                # m^-3, on 'psin_n'
            'nFC': _arr(manual_profs['nFC']),                # m^-3, on 'psin_n'
            'psin_n': _arr(manual_profs['psi_N_n']),
        }

    elif kprof_loc == 'manual profs':
        assert manual_profs is not None, 'T and n_e, profiles must be provided'

        kprof_params = {
            'T_e': _arr(manual_profs['Te']),                 # keV
            'psin_Te': _arr(manual_profs['psi_N_Te']),
            'n_e': _arr(manual_profs['ne']),                 # m^-3
            'psin_ne': _arr(manual_profs['psi_N_ne']),
        }

    else:
        assert False, 'kprof_loc method not supported'

    return kprof_params
