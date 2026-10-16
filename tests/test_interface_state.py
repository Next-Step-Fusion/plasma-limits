"""PedestalPlasmaState: construction and array ownership."""
import dataclasses

import numpy as np
import pytest

from plasma_limits.pedestal_prediction.ESCAPE_interface import PedestalPlasmaState

REQUIRED = {"R", "Z", "psi", "psi_axis", "psi_boundary", "R_axis", "Z_axis", "Ip",
            "F", "boundary", "psi_N_Te", "Te", "psi_N_ne", "ne", "Z_eff", "P_tot_e"}
DEFAULTS = {"pressure": None, "psi_N_Ti": None, "Ti": None, "psi_N_ni": None, "ni": None,
            "species": "D", "Z_i": 1, "T_rat": 1.0, "Ti_follows_Te": True,
            "quasineutral": True, "regime": "PT H-mode", "ne_sep": None, "pol_norm": False}


def _array_fields(state):
    return {f.name: getattr(state, f.name) for f in dataclasses.fields(state)
            if isinstance(getattr(state, f.name), np.ndarray)}


def test_fields_match_appendix_a():
    fields = {f.name: f for f in dataclasses.fields(PedestalPlasmaState)}
    assert set(fields) == REQUIRED | set(DEFAULTS)
    for name in REQUIRED:
        assert fields[name].default is dataclasses.MISSING, name
    for name, default in DEFAULTS.items():
        assert fields[name].default == default, name


def test_constructs_with_defaults(synthetic_inputs):
    state = PedestalPlasmaState(**synthetic_inputs)
    for name, default in DEFAULTS.items():
        assert getattr(state, name) == default, name
    for name, value in synthetic_inputs.items():
        np.testing.assert_array_equal(getattr(state, name), value)


def test_positional_arguments_rejected(synthetic_inputs):
    with pytest.raises(TypeError):
        PedestalPlasmaState(*synthetic_inputs.values())


def test_frozen(synthetic_inputs):
    state = PedestalPlasmaState(**synthetic_inputs)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.Ip = 2e6


def test_arrays_are_float64_and_read_only(synthetic_inputs):
    inputs = dict(synthetic_inputs, pressure=[0, 1, 2, 3, 4, 5],
                  psi_N_Ti=synthetic_inputs["psi_N_Te"], Ti=synthetic_inputs["Te"])
    state = PedestalPlasmaState(**inputs)
    arrays = _array_fields(state)
    assert set(arrays) == {"R", "Z", "psi", "F", "pressure", "boundary",
                           "psi_N_Te", "Te", "psi_N_ne", "ne", "psi_N_Ti", "Ti"}
    for name, arr in arrays.items():
        assert arr.dtype == np.float64, name
        assert not arr.flags.writeable, name
        with pytest.raises(ValueError):
            arr[...] = 0.0


def test_caller_arrays_are_copied(synthetic_inputs):
    state = PedestalPlasmaState(**synthetic_inputs)
    expected = {name: arr.copy() for name, arr in _array_fields(state).items()}
    for name, value in synthetic_inputs.items():
        if isinstance(value, np.ndarray):
            assert not np.shares_memory(getattr(state, name), value), name
            value[...] = -1.0
    for name, arr in expected.items():
        np.testing.assert_array_equal(getattr(state, name), arr)


def _set(**changes):
    return lambda inputs: changes


INVALID = {
    "R_2d": (lambda d: {"R": d["R"].reshape(1, -1)}, "R must be a 1D array"),
    "R_one_point": (_set(R=[2.0]), "R must be a 1D array with at least 2 points"),
    "R_decreasing": (lambda d: {"R": d["R"][::-1]}, "R must increase strictly"),
    "R_not_uniform": (_set(R=[1.0, 1.4, 2.0, 2.5, 3.0]), "R must be uniform"),
    "Z_not_uniform": (_set(Z=[-1.0, -0.6, 0.0, 0.5, 1.0]), "Z must be uniform"),
    "nr_not_nz": (lambda d: {"R": np.linspace(1.0, 3.0, 6), "psi": np.zeros((5, 6))}, "nr == nz"),
    "psi_shape": (lambda d: {"psi": d["psi"][:4]}, r"psi must have shape"),
    "F_2d": (lambda d: {"F": d["F"].reshape(2, 3)}, "F must be a 1D array"),
    "pressure_length": (_set(pressure=np.zeros(5)), "pressure must have the shape of F"),
    "boundary_columns": (lambda d: {"boundary": d["boundary"][:, :1]}, r"boundary must have shape \(N, 2\)"),
    "boundary_transposed": (lambda d: {"boundary": d["boundary"].T}, r"boundary must have shape \(N, 2\)"),
    "boundary_short": (lambda d: {"boundary": d["boundary"][:2]}, "N >= 3"),
    "profile_2d": (lambda d: {"Te": d["Te"].reshape(2, 3)}, "psi_N_Te and Te must be 1D"),
    "profile_length": (lambda d: {"Te": d["Te"][:5]}, r"len\(psi_N_Te\) = 6 != len\(Te\) = 5"),
    "profile_one_point": (_set(psi_N_ne=[1.0], ne=[1e20]), "psi_N_ne must have at least 2 points"),
    "psi_N_not_increasing": (_set(psi_N_ne=[0.0, 0.2, 0.2, 0.6, 0.8, 1.0]), "psi_N_ne must increase strictly"),
    "psi_N_not_at_1": (lambda d: {"psi_N_Te": 0.9 * d["psi_N_Te"]}, "psi_N_Te must end at the separatrix"),
    "Ti_without_grid": (lambda d: {"Ti": d["Te"]}, "give both psi_N_Ti and Ti"),
    "ni_grid_only": (lambda d: {"psi_N_ni": d["psi_N_ne"]}, "give both psi_N_ni and ni"),
    "Ti_bad_profile": (lambda d: {"psi_N_Ti": d["psi_N_Te"][::-1], "Ti": d["Te"]}, "psi_N_Ti must increase"),
    "no_Ti_follows_Te": (_set(Ti_follows_Te=False), "Ti_follows_Te=False needs Ti"),
    "no_ni_quasineutral": (_set(quasineutral=False), "quasineutral=False needs ni"),
    "species": (_set(species="T"), "species must be one of"),
}


@pytest.mark.parametrize("change, match", INVALID.values(), ids=INVALID.keys())
def test_invalid_input_raises(synthetic_inputs, change, match):
    with pytest.raises(ValueError, match=match):
        PedestalPlasmaState(**{**synthetic_inputs, **change(synthetic_inputs)})


def test_valid_input_passes(synthetic_inputs):
    d = synthetic_inputs
    PedestalPlasmaState(**d)
    PedestalPlasmaState(**{**d, "pressure": np.zeros(6), "species": "D-T",
                           "psi_N_Ti": d["psi_N_Te"], "Ti": d["Te"], "Ti_follows_Te": False,
                           "psi_N_ni": d["psi_N_ne"], "ni": d["ne"], "quasineutral": False})
    PedestalPlasmaState(**{**d, "psi_N_Te": np.append(d["psi_N_Te"][:-1], 1.0 - 1e-13),
                           "R": d["R"] * (1.0 + 1e-12 * np.arange(5))})


ESCAPE_EQ_KEYS = {"nr", "nz", "rleft", "rdim", "zmid", "zdim", "psirz", "raxis", "zaxis",
                  "psimag", "psibry", "fpol", "pres", "ip", "rzout"}
GRID_SCALARS = ("rleft", "rdim", "zmid", "zdim")


def test_escape_dict_reproduces_gfile(arc_state, ref_eq):
    eq = arc_state._to_escape_dict()
    assert set(eq) == ESCAPE_EQ_KEYS
    for key, value in eq.items():
        ref = ref_eq[key]
        if isinstance(value, np.ndarray):
            assert value.dtype == ref.dtype, key
            np.testing.assert_array_equal(value, ref, err_msg=key)
        elif key in GRID_SCALARS:
            assert abs(value - ref) <= np.spacing(ref), key
        else:
            assert value == ref and type(value) is type(ref), key


def test_escape_dict_arrays_are_fresh(arc_state):
    eq, eq_again = arc_state._to_escape_dict(), arc_state._to_escape_dict()
    state_arrays = _array_fields(arc_state).values()
    for key, value in eq.items():
        if isinstance(value, np.ndarray):
            assert value.flags.writeable, key
            assert not np.shares_memory(value, eq_again[key]), key
            assert not any(np.shares_memory(value, arr) for arr in state_arrays), key


def test_escape_dict_pressure(synthetic_inputs):
    eq = PedestalPlasmaState(**synthetic_inputs)._to_escape_dict()
    np.testing.assert_array_equal(eq["pres"], np.zeros(6))
    assert eq["pres"].flags.writeable
    pressure = np.linspace(1e5, 1e3, 6)
    eq = PedestalPlasmaState(**synthetic_inputs, pressure=pressure)._to_escape_dict()
    np.testing.assert_array_equal(eq["pres"], pressure)


KPROF_KEYS = {"T_e", "psin_Te", "n_e", "psin_ne"}
KPROF_FIELDS = {"T_e": "Te", "psin_Te": "psi_N_Te", "n_e": "ne", "psin_ne": "psi_N_ne",
                "T_i": "Ti", "psin_Ti": "psi_N_Ti", "n_i": "ni", "psin_ni": "psi_N_ni"}


@pytest.mark.parametrize("with_Ti, with_ni", [(False, False), (True, False), (False, True), (True, True)])
def test_kprof_params_keys_and_values(synthetic_inputs, with_Ti, with_ni):
    d = dict(synthetic_inputs)
    expected = set(KPROF_KEYS)
    if with_Ti:
        d.update(psi_N_Ti=d["psi_N_Te"], Ti=0.8 * d["Te"])
        expected |= {"T_i", "psin_Ti"}
    if with_ni:
        d.update(psi_N_ni=d["psi_N_ne"], ni=0.9 * d["ne"])
        expected |= {"n_i", "psin_ni"}
    state = PedestalPlasmaState(**d)
    kprof = state._to_kprof_params()
    assert set(kprof) == expected
    for key, value in kprof.items():
        np.testing.assert_array_equal(value, getattr(state, KPROF_FIELDS[key]), err_msg=key)


def test_kprof_params_arrays_are_fresh(arc_state):
    kprof, kprof_again = arc_state._to_kprof_params(), arc_state._to_kprof_params()
    state_arrays = _array_fields(arc_state).values()
    for key, value in kprof.items():
        assert value.flags.writeable, key
        assert not np.shares_memory(value, kprof_again[key]), key
        assert not any(np.shares_memory(value, arr) for arr in state_arrays), key


def test_kprof_params_reproduce_reference(arc_state, ref_profiles):
    kprof = arc_state._to_kprof_params()
    assert set(kprof) == set(ref_profiles)
    for key, value in kprof.items():
        np.testing.assert_array_equal(value, ref_profiles[key], err_msg=key)
