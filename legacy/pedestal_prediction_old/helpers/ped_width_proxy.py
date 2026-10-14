"""
Pedestal-width proxy for a single density profile.

Three independent width estimates, ported from sc_inputs/fit_pedestals.py
(the methods shown in sc_inputs/pedestal_fits_HighPerfHMode.png). Use one
method, the average of any two (tanh+grad by default), or the average of all three:

  1. tanh  - modified tanh (Burrell/Groebner) fit; width = full tanh width.
  2. grad  - Snyder gradient-width method; |dn/dx| falls to 0.7862*max at
             width/4 and to 0.5*max at width/2.27 from the symmetry point.
  3. lin   - two-line corner fit (ASDEX-U style); width = horizontal span of
             the steep outer line from the pedestal-top value down to the
             separatrix value.

Differences from fit_pedestals.py, needed for Saarelma-Connor output:
  * Profiles run from psi_N_inner_boundary to psi_N = 1 (or the equivalent x
    grid ending at x = 0), so there is no SOL and the steepest gradient often
    sits at the last grid point. The tanh symmetry point may therefore lie
    slightly outboard of the domain, and its offset (SOL asymptote) is capped
    at the separatrix value.
  * All fits are done in normalised coordinates s = (x - x[0]) / (x[-1] - x[0])
    and y / max(y), so the same settings work for psi_N or real-space x input.
    Widths are returned in the units of the input coordinate.

Usage:
    from src.ped_width_proxy import ped_width
    width = ped_width(psi_N, n_e)                  # mean of tanh and grad (default)
    width = ped_width(psi_N, n_e, method='all')    # mean of tanh, grad and lin
    width = ped_width(psi_N, n_e, method='grad+lin')  # any two methods
    width = ped_width(psi_N, n_e, method='tanh')   # a single method

    from src.ped_width_proxy import ped_width_all
    res = ped_width_all(psi_N, n_e)           # per-method widths + fit details
"""
import numpy as np
from scipy.optimize import least_squares

# ── Settings (in normalised s units, i.e. fractions of the input domain) ─────
# For a psi_N in [0.85, 1] profile, 1 s-unit = 0.15 psi_N.

WIN_FACTOR     = 2.0    # tanh/grad window: s >= sym - WIN_FACTOR * width
LIN_WIN_FACTOR = 3.0    # two-line window:  s >= sym - LIN_WIN_FACTOR * width
WIN_HALF_MIN   = 0.25   # never use less than this much domain inboard of sym
MIN_EDGE_PTS   = 10     # widen the window until it holds at least this many pts
LIN_FOOT_FACTOR    = 0.6   # two-line fit drops data beyond sym + this*width
LIN_TOP_CAP_FACTOR = 0.35  # two-line corner kept this*width inboard of sym
LIN_MIN_SLOPE_RATIO = 1.2  # two-line fit fails unless |outer slope| >= this*inner
TANH_SYM_OVERSHOOT = 0.35  # tanh symmetry point may sit this far past the edge
WIDTH_MIN, WIDTH_MAX = 0.02, 3.0  # tanh width bounds


# ── Model functions ──────────────────────────────────────────────────────────

def tnh0(c, x):
    """
    Modified tanh (Burrell/Groebner).
      c[0]=symmetry point  c[1]=full width  c[2]=height
      c[3]=offset          c[4]=alpha (inboard linear slope)
    """
    z   = 2.0 * (c[0] - x) / c[1]
    pz  = 1.0 + c[4] * z
    mth = 0.5 * ((pz + 1.0) * np.tanh(z) + pz - 1.0)
    return 0.5 * ((c[2] - c[3]) * mth + c[2] + c[3])


def linfun(c, x):
    """
    Two-line corner fit (ASDEX-U style).
      c[0]=pedestal top location  c[1]=pedestal top value
      c[2]=inner slope            c[3]=outer slope
    """
    y = np.where(x <= c[0], c[2] * (c[0] - x) + c[1], c[3] * (x - c[0]) + c[1])
    return np.where(y < 0.0, 0.0, y)


# ── Method 2: gradient width (also the initial estimate for the others) ──────

def _crossing(x, ag, idx, thresh, side):
    """Location where |grad| first drops below `thresh` moving away from idx."""
    if side == 'in':
        ii = np.arange(idx - 1, -1, -1)
    else:
        ii = np.arange(idx + 1, len(x))
    for k, j in enumerate(ii):
        if ag[j] < thresh:
            prev = idx if k == 0 else ii[k - 1]
            if ag[prev] != ag[j]:
                return x[j] + (thresh - ag[j]) / (ag[prev] - ag[j]) * (x[prev] - x[j])
            return x[j]
    return None


def grad_width(x, y):
    """
    Snyder gradient-width method. Returns dict (sym, width, width_err) or None.
    Width is the mean over all threshold crossings found on either side.
    """
    if len(x) < 6:
        return None
    ag = np.abs(np.gradient(y, x, edge_order=2))
    # Skip the innermost points so a core gradient cannot be picked up.
    search = np.where(x >= x[0] + 0.1 * (x[-1] - x[0]))[0]
    idx = int(search[np.argmax(ag[search])])
    gmax, sym = float(ag[idx]), float(x[idx])
    if gmax == 0.0:
        return None

    widths = []
    for frac, factor in [(0.7862, 4.0), (0.5, 2.27)]:
        for side in ('in', 'out'):
            xc = _crossing(x, ag, idx, frac * gmax, side)
            if xc is not None and abs(xc - sym) > 0:
                widths.append(abs(xc - sym) * factor)
    if not widths:
        return None
    return {
        'sym': sym, 'width': float(np.mean(widths)),
        'width_err': float(np.std(widths)) if len(widths) > 1 else np.nan,
        'n_crossings': len(widths),
    }


# ── Method 1: tanh fit ───────────────────────────────────────────────────────

def fit_tanh(x, y, init=None):
    if len(x) < 6:
        return None
    y_max, y_sep = float(np.max(y)), float(max(y[-1], 0.0))
    sym0 = float(init['sym']) if init else float(x[np.argmax(np.abs(np.gradient(y, x)))])
    wid0 = float(np.clip(init['width'] if init else 0.3, WIDTH_MIN, WIDTH_MAX))

    lo = [float(x[0]),                        WIDTH_MIN, 0.0,         0.0,   -3.0]
    hi = [float(x[-1]) + TANH_SYM_OVERSHOOT,  WIDTH_MAX, y_max * 2.5, y_sep,  3.0]
    inits = [
        [sym0,               wid0,       y_max,       0.5 * y_sep, 0.0],
        [sym0 - 0.5 * wid0,  wid0 * 1.5, y_max * 1.1, 0.5 * y_sep, 0.0],
        [sym0,               wid0 * 0.5, y_max,       0.0,        -0.5],
        [float(x[-1]),       wid0,       y_max,       0.0,         0.0],
    ]
    best = None
    for c0 in inits:
        c0 = np.clip(c0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        try:
            res = least_squares(lambda c: tnh0(c, x) - y, c0, bounds=(lo, hi),
                                method='trf', ftol=1e-12, xtol=1e-12, max_nfev=10000)
        except Exception:
            continue
        if not res.success or not np.all(np.isfinite(res.x)):
            continue
        rms = float(np.sqrt(np.mean(res.fun ** 2)))
        if best is None or rms < best['rms']:
            c = res.x
            best = {'params': c, 'rms': rms, 'sym': c[0], 'width': c[1],
                    'height': c[2], 'offset': c[3], 'alpha': c[4]}
    if best is None:
        return None
    # A width pinned at its bound, or a symmetry point pushed to the outboard
    # limit, means the data do not constrain the tanh: treat as failed.
    tol = 1e-3
    if (best['width'] <= WIDTH_MIN * (1 + tol) or best['width'] >= WIDTH_MAX * (1 - tol)
            or best['sym'] >= hi[0] - tol * (hi[0] - lo[0])):
        return None
    return best


# ── Method 3: two-line fit ───────────────────────────────────────────────────

def fit_linfun(x, y, init=None):
    if len(x) < 6:
        return None
    sol = float(np.min(y))   # separatrix / foot floor
    sym = float(init['sym']) if init else float(x[np.argmax(np.abs(np.gradient(y, x)))])
    width = float(init['width']) if init else 0.3

    # Drop any flat foot beyond the pedestal so the outer line keeps the steep slope.
    fmask = x <= sym + LIN_FOOT_FACTOR * width
    if int(fmask.sum()) >= 6 and int((x[fmask] > sym).sum()) >= 2:
        x, y = x[fmask], y[fmask]

    y_max = float(np.max(y))
    cap  = float(np.clip(sym - LIN_TOP_CAP_FACTOR * width, x[0] + 1e-6, x[-1]))
    top0 = float(np.clip(sym - 0.5 * width, x[0], cap))
    val0 = float(np.interp(top0, x, y))
    sl_in0 = -(val0 - y[0]) / (top0 - x[0]) if top0 > x[0] else 0.0
    foot = min(sym + 0.5 * width, float(x[-1]))
    sl_out0 = (np.interp(foot, x, y) - val0) / (foot - top0) if foot > top0 else -y_max / 0.3

    lo = [float(x[0]), 0.0,         0.0,            -y_max * 2000.0]
    hi = [cap,         y_max * 2.0, y_max * 2000.0,  0.0]
    inits = [
        [top0,               val0, max(sl_in0, 0.0),       min(sl_out0, 0.0)],
        [sym - width,        val0, max(sl_in0, 0.0),       min(sl_out0, 0.0)],
        [sym - 0.25 * width, val0, max(sl_in0 * 0.5, 0.0), min(sl_out0 * 1.5, 0.0)],
    ]
    best = None
    for c0 in inits:
        c0 = np.clip(c0, lo, hi)
        try:
            res = least_squares(lambda c: linfun(c, x) - y, c0, bounds=(lo, hi),
                                method='trf', ftol=1e-12, xtol=1e-12, max_nfev=10000)
        except Exception:
            continue
        if not res.success or not np.all(np.isfinite(res.x)):
            continue
        rms = float(np.sqrt(np.mean(res.fun ** 2)))
        if best is None or rms < best['rms']:
            c = res.x
            w = float((c[1] - sol) / abs(c[3])) if c[3] != 0 else np.nan
            best = {'params': c, 'rms': rms, 'top': c[0], 'val': c[1],
                    'slope_in': c[2], 'slope_out': c[3], 'width': w,
                    'fit_hi': float(x[-1])}
    if best is None or not (np.isfinite(best['width']) and best['width'] > 0):
        return None
    # No corner (outer line not clearly steeper than the inner one): the
    # two-line model does not describe a pedestal, so treat as failed.
    if abs(best['slope_out']) < LIN_MIN_SLOPE_RATIO * best['slope_in']:
        return None
    return best


# ── Driver ───────────────────────────────────────────────────────────────────

def _window_lo(s, sym, width, factor):
    """Inboard edge of the fit window, widened until it holds MIN_EDGE_PTS points."""
    lo = max(sym - max(factor * width, WIN_HALF_MIN), s[0])
    while int((s >= lo).sum()) < MIN_EDGE_PTS and lo > s[0]:
        lo = max(lo - 0.05, s[0])
    return lo


METHODS = ('tanh', 'grad', 'lin')
DEFAULT_METHOD = 'tanh+grad'


def parse_method(method):
    """
    Turn a method selection into a tuple of method names.

    Accepts 'all' (or 'mean'), a single name ('tanh', 'grad', 'lin'), names
    joined by '+' (e.g. 'tanh+grad'), or a list/tuple of names.
    """
    if isinstance(method, str):
        if method in ('all', 'mean'):
            return METHODS
        names = [m.strip() for m in method.split('+')]
    else:
        names = list(method)
    bad = [m for m in names if m not in METHODS]
    if not names or bad or len(set(names)) != len(names):
        raise ValueError(f"method must be 'all', one of {METHODS}, or a '+'-joined "
                         f"combination such as 'tanh+grad'; got {method!r}")
    return tuple(m for m in METHODS if m in names)


def ped_width_all(x, n, method=DEFAULT_METHOD):
    """
    Pedestal width of one density profile by the tanh, gradient or two-line
    method, or the average of two or all three of them.

    Parameters
    ----------
    x : array
        Radial coordinate (psi_N, or the x grid), from the inner boundary to the
        separatrix. Any order; the profile is assumed to fall toward the edge.
    n : array
        Density on `x`.
    method : str or sequence of str
        Which methods to run and average:
          'tanh', 'grad', 'lin'          - one method
          'tanh+grad' (default), 'tanh+lin', 'grad+lin' - average of two
          'all'                          - average of all three
        A list such as ['tanh', 'lin'] also works. Selected methods that fail
        are dropped from the average; RuntimeError is raised only if every
        selected method fails.

    Returns
    -------
    dict with keys
        'width'       : selected width (input x units)
        'methods'     : the selected methods
        'width_tanh', 'width_grad', 'width_lin' : per-method widths (NaN if that
                        method failed or was not run)
        'methods_used': names of the methods included in 'width'
        'width_std'   : spread of the averaged widths (NaN if only one used)
        'tanh', 'grad', 'lin' : fit details in normalised units (or None)
        'norm'        : (x0, L, sign, n_scale) mapping s -> x = x0 + sign*L*s
    """
    run = parse_method(method)
    x = np.asarray(x, dtype=float).ravel()
    n = np.asarray(n, dtype=float).ravel()
    if x.shape != n.shape:
        raise ValueError(f"x and n must have the same shape, got {x.shape} and {n.shape}")
    ok = np.isfinite(x) & np.isfinite(n)
    x, n = x[ok], n[ok]
    order = np.argsort(x)
    x, n = x[order], n[order]
    if len(x) < 6:
        raise ValueError("need at least 6 finite points")

    # Orient so the profile falls with increasing s, then normalise.
    sign = 1.0 if n[0] >= n[-1] else -1.0
    if sign < 0:
        x, n = -x[::-1], n[::-1]
    x0, L = float(x[0]), float(x[-1] - x[0])
    n_scale = float(np.max(np.abs(n)))
    if L <= 0 or n_scale == 0:
        raise ValueError("degenerate profile (zero coordinate span or all-zero density)")
    s, y = (x - x0) / L, n / n_scale

    # The broad-window gradient estimate only seeds the window and the initial
    # guesses; if it fails, start the tanh/two-line fits from the edge instead.
    init = grad_width(s, y)
    if init is None:
        init = {'sym': 1.0, 'width': 0.3}

    lo = _window_lo(s, init['sym'], init['width'], WIN_FACTOR)
    lin_lo = min(_window_lo(s, init['sym'], init['width'], LIN_WIN_FACTOR), lo)
    win, lin_win = s >= lo, s >= lin_lo

    tanh = fit_tanh(s[win], y[win], init) if 'tanh' in run else None
    grad = grad_width(s[win], y[win]) if 'grad' in run else None
    lin  = fit_linfun(s[lin_win], y[lin_win], init) if 'lin' in run else None

    # Failed methods are dropped; the width uses whichever selected methods succeeded.
    widths = {}
    for name, r in (('tanh', tanh), ('grad', grad), ('lin', lin)):
        ok = r is not None and np.isfinite(r['width']) and r['width'] > 0
        widths[name] = r['width'] * L if ok else np.nan
    used = [k for k in run if np.isfinite(widths[k])]
    if not used:
        raise RuntimeError(f"pedestal-width method(s) {', '.join(run)} failed")
    good = [widths[k] for k in used]

    return {
        'width': float(np.mean(good)),
        'methods': run,
        'methods_used': used,
        'width_tanh': widths['tanh'], 'width_grad': widths['grad'], 'width_lin': widths['lin'],
        'width_std': float(np.std(good)) if len(good) > 1 else np.nan,
        'tanh': tanh, 'grad': grad, 'lin': lin,
        'fit_lo': lo, 'lin_fit_lo': lin_lo,
        'norm': (x0, L, sign, n_scale),
    }


def ped_width(x, n, method=DEFAULT_METHOD):
    """
    Pedestal width in x units. `method` selects one method ('tanh', 'grad',
    'lin'), two to average ('tanh+grad' default, 'tanh+lin', 'grad+lin'), or
    'all'. See ped_width_all.
    """
    return ped_width_all(x, n, method=method)['width']


def plot_ped_width(x, n, ax=None, res=None, method=DEFAULT_METHOD):
    """Debug plot of the profile with the fits that were run (in the input coordinates)."""
    import matplotlib.pyplot as plt
    if res is None:
        res = ped_width_all(x, n, method=method)
    if ax is None:
        _, ax = plt.subplots()
    x0, L, sign, ns = res['norm']
    to_x = lambda s: sign * (x0 + L * np.asarray(s))

    ax.plot(x, n, 'k.', ms=3, label='data')
    s_fine = np.linspace(0.0, 1.0, 400)
    if res['tanh'] is not None:
        sf = s_fine[s_fine >= res['fit_lo']]
        ax.plot(to_x(sf), tnh0(res['tanh']['params'], sf) * ns, color='#1f77b4',
                label=f"tanh $\\Delta$={res['width_tanh']:.4f}")
    if res['grad'] is not None:
        g = res['grad']
        for v in (g['sym'], g['sym'] - 0.5 * g['width']):
            ax.axvline(to_x(v), color='#2ca02c', ls='--', lw=0.8)
        ax.plot([], [], color='#2ca02c', ls='--', label=f"grad $\\Delta$={res['width_grad']:.4f}")
    if res['lin'] is not None:
        sf = s_fine[(s_fine >= res['lin_fit_lo']) & (s_fine <= res['lin']['fit_hi'])]
        ax.plot(to_x(sf), linfun(res['lin']['params'], sf) * ns, color='#d62728',
                label=f"lin $\\Delta$={res['width_lin']:.4f}")
    ax.set_title(f"{'+'.join(res['methods_used'])} $\\Delta$ = {res['width']:.4f}", fontsize=9)
    ax.legend(fontsize=7)
    return ax


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Pedestal width of a saved SC density profile "
                                            "(ne_and_Te_iter_*.npy).")
    p.add_argument('npy')
    p.add_argument('--coord', choices=['psi_N', 'x'], default='psi_N')
    p.add_argument('--method', default=DEFAULT_METHOD,
                   help="tanh, grad, lin, a '+'-joined pair such as tanh+grad (default), or all")
    p.add_argument('--plot', help="save a debug plot to this path")
    a = p.parse_args()
    d = np.load(a.npy, allow_pickle=True).item()
    xx = d['psi_N_ne'] if a.coord == 'psi_N' else d['x']
    r = ped_width_all(xx, d['y'], method=a.method)
    print(f"tanh {r['width_tanh']:.5f}  grad {r['width_grad']:.5f}  "
          f"lin {r['width_lin']:.5f}  ->  {a.method} {r['width']:.5f} ({a.coord}; "
          f"used {', '.join(r['methods_used'])})")
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plot_ped_width(xx, d['y'], res=r)
        plt.savefig(a.plot, dpi=120)
