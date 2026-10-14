import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator

ADAS_DIR = os.path.dirname(os.path.abspath(__file__))


def _adas_path(filename):
    """Return the full path to an ADAS data file stored next to this module."""
    return os.path.join(ADAS_DIR, filename)

def load_and_interpolate_scx(filepath):
    """
    Parses an ADAS adf11 CCD file and returns a robust 2D interpolator function
    for the charge-exchange rate coefficient (S_cx) in SI units (m^3/s).
    Includes warnings if queried parameters fall outside the database grid.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 1. Extract grid dimensions from the first line
    header = lines[0].split()
    num_dens = int(header[1])
    num_temps = int(header[2])

    # 2. Extract all numeric data ignoring headers, dividers, and comments
    raw_numbers = []
    data_started = False
    
    for line in lines[1:]:
        if 'IPRT=' in line:
            data_started = True
            continue
        if line.startswith('C') or line.startswith(' ---'):
            if data_started: 
                break
            continue
            
        parts = line.split()
        for part in parts:
            try:
                raw_numbers.append(float(part))
            except ValueError:
                pass

    # 3. Partition the extracted numbers into grids and data
    log10_dens_cm3 = np.array(raw_numbers[:num_dens])
    log10_temp_ev = np.array(raw_numbers[num_dens : num_dens + num_temps])
    
    data_start_idx = num_dens + num_temps
    log10_scx_cm3_s = np.array(raw_numbers[data_start_idx : data_start_idx + (num_dens * num_temps)])
    
    log10_scx_2d = log10_scx_cm3_s.reshape((num_temps, num_dens))

    # 4. Convert units from CGS (log10) to SI (linear)
    dens_m3 = 10**(log10_dens_cm3 + 6)
    temp_ev = 10**(log10_temp_ev)
    scx_m3_s = 10**(log10_scx_2d - 6)

    # 5. Determine grid bounds for warning messages
    min_t, max_t = np.min(temp_ev), np.max(temp_ev)
    min_n, max_n = np.min(dens_m3), np.max(dens_m3)

    # 6. Create the base interpolator (returns NaN if out of bounds)
    base_interpolator = RegularGridInterpolator((temp_ev, dens_m3), scx_m3_s, 
                                                bounds_error=False, fill_value=np.nan)
    # base_interpolator = RegularGridInterpolator((temp_ev, dens_m3), scx_m3_s, 
    #                                               bounds_error=False, fill_value=None)
    
    # 7. Create a wrapper function to catch out-of-bounds inputs
    def scx_wrapper(points):
        # Convert to numpy array to handle both single tuples and arrays of profiles
        pts = np.asarray(points)
        
        # Check if 1D tuple (single point) or 2D array (profile vector)
        if pts.ndim == 1:
            t, n = pts[0], pts[1]
            t_out = (t < min_t) or (t > max_t)
            n_out = (n < min_n) or (n > max_n)
        else:
            t, n = pts[..., 0], pts[..., 1]
            t_out = np.any((t < min_t) | (t > max_t))
            n_out = np.any((n < min_n) | (n > max_n))
            
        # Print warnings
        if t_out:
            print(f"WARNING: Input temperature is out of ADAS grid range ({min_t:.2e} to {max_t:.2e} eV). "
                  "Returned S_cx will contain NaN values.")
        if n_out:
            print(f"WARNING: Input density is out of ADAS grid range ({min_n:.2e} to {max_n:.2e} m^-3). "
                  "Returned S_cx will contain NaN values.")
            
        return base_interpolator(points)

    return scx_wrapper


# Build interpolator once at import time (same pattern as adas_ionisation.py).
_get_scx = load_and_interpolate_scx(_adas_path("ccd96_d.dat"))


def scx_adas(ne_m3, Te_eV):
    # 1. Ensure inputs are treated as arrays (even if single numbers are passed)
    Te_arr = np.atleast_1d(Te_eV)
    ne_arr = np.atleast_1d(ne_m3)
    
    # 2. Stack them into the (N, 2) shape expected by the interpolator
    points = np.column_stack((Te_arr, ne_arr))
    
    # 3. Evaluate the interpolation
    result = _get_scx(points)
    
    # 4. Return a single float if the user passed in scalars, otherwise return the array
    return result[0] if result.size == 1 else result