"""run_escape: the checks before the run (fast), and real runs on arc_state with real EPED1 (slow).

The slow runs use nothing fake: real EPED1 through juliacall and the real
ESCAPE_solve with the 1D scipy defaults. Their records must equal
arc_1d_baseline.npy, which ESCAPE's own file path wrote from the same inputs.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from plasma_limits.pedestal_prediction.ESCAPE_interface import (EPEDConfig, EPEDModel, ESCAPERunError,
                                                              PedestalPlasmaState, SCConfig, load_eped_model,
                                                              run_escape)

TESTS_DIR = Path(__file__).parent
N_BASELINE_ITER = 5


def _load_tests_module(name):
    spec = importlib.util.spec_from_file_location(name, TESTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def diff():
    return _load_tests_module("test_arc_integration")._diff


@pytest.fixture(scope="module")
def sol_keys():
    """IterationRecord field -> key of ESCAPE's iteration dict (Appendix A of TASK.md)."""
    module = _load_tests_module("test_interface_result")
    return {**module.ARRAY_KEYS, **module.VALUE_KEYS, "diagnostics": "diagnostics"}


# Checks before the run. The EPED model is never called.

UNUSED_EPED = EPEDModel(config=EPEDConfig(), model=None)


def test_rejects_eped_config_instead_of_model(arc_state):
    with pytest.raises(TypeError, match="EPEDModel"):
        run_escape(arc_state, EPEDConfig())


def test_pres_gfile_needs_pressure(synthetic_inputs):
    eped = EPEDModel(config=EPEDConfig(pres_gfile=True), model=None)
    with pytest.raises(ValueError, match="pressure"):
        run_escape(PedestalPlasmaState(**synthetic_inputs), eped)


def test_max_iter_at_least_one(arc_state):
    with pytest.raises(ValueError, match="max_iter"):
        run_escape(arc_state, UNUSED_EPED, max_iter=0)


def test_firedrake_checked_before_run(arc_state, monkeypatch):
    from escape.saarelma_connor import solver_3d

    error = ImportError("no firedrake")
    monkeypatch.setattr(solver_3d, "_FIREDRAKE_AVAILABLE", False)
    monkeypatch.setattr(solver_3d, "_FIREDRAKE_IMPORT_ERR", error)
    with pytest.raises(ImportError, match="needs Firedrake") as excinfo:
        run_escape(arc_state, UNUSED_EPED, SCConfig(model="3D", implementation="firedrake"))
    assert excinfo.value.__cause__ is error


# Real runs. Julia starts on the first load of EPED1 in the session.

@pytest.fixture(scope="module")
def eped1():
    return load_eped_model()


@pytest.fixture(scope="module")
def baseline_iterations():
    return np.load(TESTS_DIR / "data" / "arc_1d_baseline.npy", allow_pickle=True).item()["iterations"]


def _as_sol(record, sol_keys):
    """The record with ESCAPE's iteration-dict keys, as in the baseline file."""
    sol = {key: getattr(record, name) for name, key in sol_keys.items() if key != "T_e_pres"}
    if record.loop_tol is not None:
        sol["loop_tol"] = record.loop_tol
    return sol


def _assert_records_equal_baseline(history, baseline_iterations, sol_keys, diff):
    for record, expected in zip(history, baseline_iterations):
        assert diff(_as_sol(record, sol_keys), expected, f"iteration {record.iteration}") == []
        assert record.T_e_pres.shape == record.psi_N_pres.shape
        assert np.all(np.isfinite(record.T_e_pres))
        assert np.isfinite(record.betan)


@pytest.fixture(scope="module")
def full_run(arc_state, eped1, tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("full_run")
    sc = SCConfig()
    params_before = sc.to_params()
    received = []
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(work_dir)
        result = run_escape(arc_state, eped1, sc, max_iter=N_BASELINE_ITER, on_iteration=received.append)
    return dict(result=result, received=received, work_dir=work_dir, sc=sc, params_before=params_before)


@pytest.mark.slow
def test_run_equals_file_baseline(full_run, baseline_iterations, sol_keys, diff):
    result = full_run["result"]
    assert result.n_iter == N_BASELINE_ITER == len(baseline_iterations)
    assert [record.iteration for record in result.history] == list(range(N_BASELINE_ITER))
    _assert_records_equal_baseline(result.history, baseline_iterations, sol_keys, diff)
    assert result.converged is False
    assert result.state is None


@pytest.mark.slow
def test_run_callback_and_no_side_effects(full_run):
    result = full_run["result"]
    assert len(full_run["received"]) == N_BASELINE_ITER
    assert all(got is record for got, record in zip(full_run["received"], result.history))
    assert list(full_run["work_dir"].iterdir()) == []
    assert full_run["sc"].to_params() == full_run["params_before"]


@pytest.mark.slow
def test_run_stops_on_tol(arc_state, eped1, baseline_iterations, sol_keys, diff, tmp_path, monkeypatch):
    """Baseline loop_tol of iteration 1 is 0.272, so tol=0.3 stops there."""
    monkeypatch.chdir(tmp_path)
    result = run_escape(arc_state, eped1, max_iter=N_BASELINE_ITER, tol=0.3)
    assert result.n_iter == 2
    assert result.converged is True
    _assert_records_equal_baseline(result.history, baseline_iterations, sol_keys, diff)


@pytest.mark.slow
def test_run_error_keeps_history(arc_state, eped1, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    failure = RuntimeError("stop at iteration 1")

    def on_iteration(record):
        if record.iteration == 1:
            raise failure

    with pytest.raises(ESCAPERunError) as excinfo:
        run_escape(arc_state, eped1, max_iter=N_BASELINE_ITER, on_iteration=on_iteration)
    assert [record.iteration for record in excinfo.value.history] == [0, 1]
    assert excinfo.value.__cause__ is failure
