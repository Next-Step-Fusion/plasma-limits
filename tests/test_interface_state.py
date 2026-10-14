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
