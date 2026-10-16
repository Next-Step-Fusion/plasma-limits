"""FreeParams and SCConfig: defaults, to_params() and ownership of the returned dicts."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from plasma_limits.pedestal_prediction.ESCAPE_interface import FreeParams, SCConfig

DATA_DIR = Path(__file__).parent / "data"


def _baseline_solve_kw(name):
    """SOLVE_KW of iteration 0 of an ARC integration baseline (ESCAPE's ARC sc_params)."""
    baseline = np.load(DATA_DIR / name, allow_pickle=True).item()
    return baseline["iterations"][0]["SOLVE_KW"]


def _assert_same_values_and_types(actual, expected):
    assert actual == expected
    for key, value in expected.items():
        assert type(actual[key]) is type(value), key
        if isinstance(value, dict):
            _assert_same_values_and_types(actual[key], value)


@pytest.mark.parametrize("baseline, settings", [
    ("arc_1d_baseline.npy", {}),
    ("arc_3d_baseline.npy", {"model": "3D", "implementation": "firedrake"}),
])
def test_params_equal_arc_baseline(baseline, settings):
    params = SCConfig(**settings).to_params()
    _assert_same_values_and_types(params, _baseline_solve_kw(baseline))


def test_solve_keywords_are_not_params():
    config = SCConfig(tanh_center_psiN=0.95, psiN_pwg=0.9)
    assert config._solve_keywords() == {"tanh_center_psiN": 0.95, "psiN_pwg": 0.9}
    assert SCConfig()._solve_keywords() == {"tanh_center_psiN": None, "psiN_pwg": None}
    params = config.to_params()
    assert not {"tanh_center_psiN", "psiN_pwg", "solver_kwargs"} & set(params)


def test_solver_kwargs_are_merged():
    params = SCConfig(solver_kwargs={"dne_method": "state", "ivp_rtol": 1e-9}).to_params()
    assert params["dne_method"] == "state"
    assert params["ivp_rtol"] == 1e-9
    assert set(params) == set(SCConfig().to_params()) | {"dne_method", "ivp_rtol"}


def test_each_call_returns_new_dicts():
    config = SCConfig(solver_kwargs={"ne_inner": [1.0]})
    first, second = config.to_params(), config.to_params()
    assert first is not second
    assert first["free_params"] is not second["free_params"]
    assert first["ne_inner"] is not second["ne_inner"]
    assert first["ne_inner"] is not config.solver_kwargs["ne_inner"]

    first["initial_guess"] = "state"
    first["free_params"]["C_KBM"] = 1.0
    first["ne_inner"].append(2.0)
    assert config.to_params() == second


def test_solver_kwargs_are_copied_and_read_only():
    caller_kwargs = {"ne_inner": [1.0]}
    config = SCConfig(solver_kwargs=caller_kwargs)
    caller_kwargs["dne_method"] = "state"
    caller_kwargs["ne_inner"].append(2.0)
    assert dict(config.solver_kwargs) == {"ne_inner": [1.0]}
    with pytest.raises(TypeError):
        config.solver_kwargs["dne_method"] = "state"


@pytest.mark.parametrize("cls, name", [(FreeParams, "C_KBM"), (SCConfig, "model")])
def test_frozen_and_keyword_only(cls, name):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(cls(), name, None)
    with pytest.raises(TypeError):
        cls(0.5)


@pytest.mark.parametrize("settings, message", [
    ({"model": "3d"}, "model must be one of"),
    ({"model": "coupled"}, "model must be one of"),
    ({"implementation": "fem"}, "implementation must be one of"),
    ({"model": "3D", "implementation": "firedrake", "psi_N_inner_boundary": None,
      "nFC_threshold": 0.01, "nCX_threshold": 0.01}, "solve_coupled_nondim"),
    ({"psi_N_inner_boundary": None}, "init_saarelmaconnor"),
    ({"psi_N_inner_boundary": None, "nFC_threshold": 0.01}, "init_saarelmaconnor"),
    ({"psi_N_inner_boundary": None, "nCX_threshold": 0.01}, "init_saarelmaconnor"),
    ({"model": "3D", "nFC_ic": "state"}, "nFC_ic='state'"),
    ({"model": "3D", "nCX_ic": "state"}, "nCX_ic='state'"),
    ({"initial_guess": "state"}, "initial_guess must be one of"),
    ({"initial_guess": "tanh_linear"}, "build_ne_initial_guess"),
    ({"solver_kwargs": {"x_res": 80}}, r"\['x_res'\]"),
    ({"solver_kwargs": {"initial_guess": "tanh"}}, r"\['initial_guess'\]"),
    ({"solver_kwargs": {"free_params": {}}}, r"\['free_params'\]"),
    ({"solver_kwargs": {"tanh_center": -0.01, "x_pwg": -0.02}}, r"\['tanh_center', 'x_pwg'\]"),
])
def test_invalid_settings_rejected(settings, message):
    with pytest.raises(ValueError, match=message):
        SCConfig(**settings)


@pytest.mark.parametrize("settings", [
    {},
    {"model": "3D", "implementation": "firedrake"},
    {"model": "3D", "implementation": "scipy"},
    {"psi_N_inner_boundary": None, "nFC_threshold": 0.01, "nCX_threshold": 0.01},
    {"nFC_ic": "state", "nCX_ic": "state"},
    {"initial_guess": "tanh_linear", "psiN_pwg": 0.9, "tanh_center_psiN": 0.95},
    {"solver_kwargs": {"dne_method": "state", "ne_inner_bc": "neumann"}},
])
def test_valid_settings_accepted(settings):
    SCConfig(**settings)


@pytest.mark.parametrize("settings", [
    {"model": "3D", "implementation": "firedrake"},
    {"model": "3D", "implementation": "scipy"},
    {"model": "1D", "implementation": "firedrake"},
])
def test_firedrake_needed(settings):
    config = SCConfig(**settings)
    error = ModuleNotFoundError("No module named 'firedrake'")
    with pytest.raises(ImportError, match="needs Firedrake") as info:
        config._check_firedrake(False, error)
    assert info.value.__cause__ is error
    config._check_firedrake(True, None)


def test_firedrake_not_needed_for_1d_scipy():
    SCConfig()._check_firedrake(False, ModuleNotFoundError("No module named 'firedrake'"))


def test_replace_for_scans():
    config = SCConfig()
    changed = dataclasses.replace(config, free_params=dataclasses.replace(config.free_params, C_KBM=0.5))
    assert changed.to_params()["free_params"]["C_KBM"] == 0.5
    assert config.free_params.C_KBM == 0.3
