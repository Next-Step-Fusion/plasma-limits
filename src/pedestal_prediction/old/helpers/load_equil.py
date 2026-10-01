# Load in equilibria
import re
from collections import defaultdict
from pathlib import Path


def initialize_inputs(equil_num, geqdsk_dir=None, pfile_dir=None, p_filetype="pfile",
                      select_equil=None):
    """Select equil_num g/profile file pairs from the input directories.

    Parameters
    ----------
    equil_num : int
        Number of matched equilibria to return. Ignored when
        ``select_equil`` is given.
    geqdsk_dir : path-like
        Directory of GEQDSK files named g{shot}.{time}.
    pfile_dir : path-like
        Directory of kinetic profiles.
    p_filetype : {"pfile", "OMFITnc"}
        Profile file type. ``"pfile"`` expects files named p{shot}.{time}.
        ``"OMFITnc"`` expects ``*.cdf`` files named like
        ``IDA_{shot}_{t0}_{t1}_.cdf``, matched to g-files when the g-file
        time (ms→s) lies in ``[t0, t1]``.
    select_equil : str or sequence of str, optional
        If given, switches to user-selected mode: only the named equilibria
        are returned, in the order given, and ``equil_num`` is ignored. Each
        name is an equilibrium identifier ``"{shot}.{time}"`` as it appears in
        the g-file name (a leading ``"g"`` is accepted, e.g. ``"g153523.03745"``
        or ``"153523.03745"``). A name that does not exactly match a matched
        equilibrium raises ``ValueError``.

    Selection prioritizes one equilibrium per shot number before adding
    additional times from shots that already have a selected equilibrium.
    Unmatched g or profile files are discarded with a printed warning.
    """

    if geqdsk_dir is None:
        raise ValueError("geqdsk_dir is required")
    if pfile_dir is None:
        raise ValueError("pfile_dir is required")
    if p_filetype not in ("pfile", "OMFITnc"):
        raise ValueError('p_filetype must be "pfile" or "OMFITnc"')

    geqdsk_dir = Path(geqdsk_dir)
    pfile_dir = Path(pfile_dir)

    g_files = sorted(f for f in geqdsk_dir.glob("g*") if f.is_file())

    if p_filetype == "pfile":
        g_by_key, p_by_key = _match_pfiles(g_files, pfile_dir)
    elif p_filetype == "OMFITnc":
        g_by_key, p_by_key = _match_omfitnc(g_files, pfile_dir)

    shared_keys = sorted(set(g_by_key) & set(p_by_key))

    if select_equil is not None:
        selected = _select_named(select_equil, shared_keys, g_by_key, p_by_key,
                                 geqdsk_dir, pfile_dir, p_filetype)
        print(f"Selected {len(selected)} user-specified g/{p_filetype} file pairs:")
        for mhd_fp, kprof_fp in selected:
            print(f"  g: {mhd_fp}\n  p: {kprof_fp}")
        return selected

    by_shot = defaultdict(list)
    for key in shared_keys:
        shot, time = key.split(".", 1)
        by_shot[shot].append((time, key))
    for shot in by_shot:
        by_shot[shot].sort()

    shots = sorted(by_shot)
    selected = []
    time_idx = 0
    while len(selected) < equil_num:
        added_this_round = False
        for shot in shots:
            if len(selected) >= equil_num:
                break
            entries = by_shot[shot]
            if time_idx < len(entries):
                key = entries[time_idx][1]
                selected.append((str(g_by_key[key]), str(p_by_key[key])))
                added_this_round = True
        if not added_this_round:
            break
        time_idx += 1

    if len(selected) < equil_num:
        raise ValueError(
            f"Requested {equil_num} equilibria but only found {len(selected)} "
            f"matching g/{p_filetype} pairs in {geqdsk_dir} and {pfile_dir}"
        )

    print(f"Selected {len(selected)} g/{p_filetype} file pairs:")
    for mhd_fp, kprof_fp in selected:
        print(f"  g: {mhd_fp}\n  p: {kprof_fp}")
    return selected


def _select_named(select_equil, shared_keys, g_by_key, p_by_key,
                  geqdsk_dir, pfile_dir, p_filetype):
    """Return the pairs named in select_equil, requiring exact matches."""
    if isinstance(select_equil, (str, Path)):
        names = [select_equil]
    else:
        names = list(select_equil)
    if not names:
        raise ValueError("select_equil was given but is empty")

    available = set(shared_keys)
    selected = []
    for name in names:
        key = str(name)
        if key.startswith("g"):
            key = key[1:]
        if key not in available:
            raise ValueError(
                f"Requested equilibrium {name!r} was not found among the "
                f"matched g/{p_filetype} pairs in {geqdsk_dir} and {pfile_dir}. "
                f"Available equilibria: {sorted(available)}"
            )
        selected.append((str(g_by_key[key]), str(p_by_key[key])))
    return selected


def _match_pfiles(g_files, pfile_dir):
    """Match g{shot}.{time} to p{shot}.{time} by shared suffix."""
    g_by_key = {f.name[1:]: f for f in g_files}
    p_files = sorted(f for f in pfile_dir.glob("p*") if f.is_file())
    p_by_key = {f.name[1:]: f for f in p_files}

    for key in sorted(set(g_by_key) - set(p_by_key)):
        print("Equilibrium found without match")
        del g_by_key[key]
    for key in sorted(set(p_by_key) - set(g_by_key)):
        print("Equilibrium found without match")
        del p_by_key[key]

    return g_by_key, p_by_key


def _match_omfitnc(g_files, pfile_dir):
    """Match g{shot}.{time} to IDA_{shot}_{t0}_{t1}_.cdf when time∈[t0,t1]."""
    cdf_re = re.compile(r"^IDA_(\d+)_([0-9.]+)_([0-9.]+)_?\.cdf$")

    g_entries = []
    for f in g_files:
        try:
            shot, tstr = f.name[1:].split(".", 1)
            time_s = int(tstr) / 1000.0
        except (ValueError, IndexError):
            print("Equilibrium found without match")
            continue
        g_entries.append((shot, time_s, tstr, f))

    cdf_entries = []
    for f in sorted(pfile_dir.glob("*.cdf")):
        if not f.is_file():
            continue
        m = cdf_re.match(f.name)
        if m is None:
            print("Equilibrium found without match")
            continue
        shot, t0, t1 = m.group(1), float(m.group(2)), float(m.group(3))
        cdf_entries.append((shot, t0, t1, f))

    # Candidates: (distance_to_window_midpoint, g_idx, cdf_idx)
    candidates = []
    for ig, (shot, time_s, tstr, g_fp) in enumerate(g_entries):
        for ic, (cshot, t0, t1, c_fp) in enumerate(cdf_entries):
            if shot != cshot or not (t0 <= time_s <= t1):
                continue
            mid = 0.5 * (t0 + t1)
            candidates.append((abs(time_s - mid), ig, ic))
    candidates.sort()

    used_g = set()
    used_c = set()
    g_by_key = {}
    p_by_key = {}
    for _, ig, ic in candidates:
        if ig in used_g or ic in used_c:
            continue
        shot, time_s, tstr, g_fp = g_entries[ig]
        _, _, _, c_fp = cdf_entries[ic]
        key = f"{shot}.{tstr}"
        g_by_key[key] = g_fp
        p_by_key[key] = c_fp
        used_g.add(ig)
        used_c.add(ic)

    for ig, _ in enumerate(g_entries):
        if ig not in used_g:
            print("Equilibrium found without match")
    for ic, _ in enumerate(cdf_entries):
        if ic not in used_c:
            print("Equilibrium found without match")

    return g_by_key, p_by_key
