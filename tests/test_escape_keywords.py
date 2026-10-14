"""Keywords of ESCAPE_solve for in-memory use, with fake EPEDNN models (no Julia).

The runs use the ARC inputs of tests/arc_integration_driver.py with the real
1D scipy solver. feed_epednn is replaced by a fake, so no EPEDNN model is
evaluated.
"""
import copy
import importlib.util

import numpy as np
import pytest

from pedestal_prediction.ESCAPE_interface import ESCAPE_ROOT

TESTS_DIR = ESCAPE_ROOT.parents[1] / "tests"
SOLVE_ONLY_KEYS = ("sc_params", "epednn_params", "ESCAPE_iter_max", "ESCAPE_tol_max")
CALLBACK_ONLY_KEYS = ("betan", "T_e_pres")


def _load_tests_module(name):
    spec = importlib.util.spec_from_file_location(name, TESTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load_tests_module("arc_integration_driver")


@pytest.fixture(scope="module")
def diff():
    return _load_tests_module("test_arc_integration")._diff


@pytest.fixture
def load_calls(monkeypatch):
    """Replaces load_epednn_model; the list records each requested model name."""
    from escape.epednn import epednn_call

    calls = []

    def fake_load(model='EPED1'):
        calls.append(model)
        return ("loaded", model)

    monkeypatch.setattr(epednn_call, "load_epednn_model", fake_load)
    return calls


@pytest.fixture
def fake_eped(monkeypatch):
    """Replaces feed_epednn with a fixed pedestal (height 0.09 MPa, width 0.03)."""
    from escape.ESCAPE.ESCAPE_api import ESCAPE

    def fake_feed(self, model='EPED1', pres_gfile=False, ne_ped_h=None, psin_ped=None):
        self.betan = np.array([1.0])
        return 0.09, 0.03, self.betan

    monkeypatch.setattr(ESCAPE, "feed_epednn", fake_feed)


def _state(driver):
    from escape.ESCAPE.ESCAPE_api import ESCAPE

    inputs = driver.arc_inputs()
    return ESCAPE(**{k: v for k, v in inputs.items() if k not in SOLVE_ONLY_KEYS})


def test_setup_epednn_uses_injected_model(driver, load_calls):
    sentinel = object()
    injected = _state(driver)
    injected.setup_epednn(model='EPED1', epednn_model=sentinel)
    assert load_calls == []
    assert injected.epednn_model is sentinel

    loaded = _state(driver)
    loaded.setup_epednn(model='EPED1')
    assert load_calls == ['EPED1']
    assert loaded.epednn_model == ("loaded", 'EPED1')

    assert np.array_equal(injected.bt, loaded.bt)
    assert np.array_equal(injected.B, loaded.B)


def test_load_epednn_model_rejects_unknown_name():
    from escape.epednn.epednn_call import load_epednn_model

    with pytest.raises(ValueError, match="'nope'"):
        load_epednn_model('nope')


def test_escape_solve_passes_injected_model(driver, load_calls, fake_eped, tmp_path):
    from escape.ESCAPE.ESCAPE_solve import ESCAPE_solve

    sentinel = object()
    inputs = dict(driver.arc_inputs(), ESCAPE_iter_max=1)
    state = ESCAPE_solve(**inputs, out_dir=str(tmp_path), epednn_model=sentinel)
    assert load_calls == []
    assert state.epednn_model is sentinel
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ne_and_Te_iter_0.npy"]


@pytest.fixture(scope="module")
def callback_run(driver, tmp_path_factory):
    """One ESCAPE_solve run of 2 iterations with out_dir and on_iteration.

    The fake EPED returns a fixed pedestal, so the run stops after iteration 1
    (loop_tol = 0), and betan = 1, 2 in iterations 0, 1.
    """
    from escape.ESCAPE.ESCAPE_api import ESCAPE
    from escape.ESCAPE.ESCAPE_solve import ESCAPE_solve

    out_dir = tmp_path_factory.mktemp("out")
    calls, solver_T_e_pres = [], []
    solve_scmodel = ESCAPE.solve_scmodel

    def recording_solve(self, *args, **kwargs):
        res = solve_scmodel(self, *args, **kwargs)
        solver_T_e_pres.append(np.array(res['T_e_pres']))
        return res

    def fake_feed(self, model='EPED1', pres_gfile=False, ne_ped_h=None, psin_ped=None):
        self.betan = np.array([1.0 + len(calls)])
        return 0.09, 0.03, self.betan

    def on_iteration(k, sol):
        calls.append((k, copy.deepcopy(driver.plain(sol))))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ESCAPE, "solve_scmodel", recording_solve)
        mp.setattr(ESCAPE, "feed_epednn", fake_feed)
        inputs = dict(driver.arc_inputs(), ESCAPE_iter_max=2)
        ESCAPE_solve(**inputs, out_dir=str(out_dir), epednn_model=object(), on_iteration=on_iteration)

    saved = [np.load(out_dir / f"ne_and_Te_iter_{k}.npy", allow_pickle=True).item() for k in range(2)]
    return dict(calls=calls, saved=saved, solver_T_e_pres=solver_T_e_pres, out_dir=out_dir)


def test_on_iteration_called_once_per_iteration(callback_run):
    assert [k for k, _ in callback_run["calls"]] == [0, 1]
    assert sorted(p.name for p in callback_run["out_dir"].iterdir()) == [
        "ne_and_Te_iter_0.npy", "ne_and_Te_iter_1.npy"]


def test_on_iteration_sol_equals_saved_file(callback_run, driver, diff):
    for (k, sol), saved in zip(callback_run["calls"], callback_run["saved"]):
        assert not set(CALLBACK_ONLY_KEYS) & saved.keys()
        without_extras = {key: v for key, v in sol.items() if key not in CALLBACK_ONLY_KEYS}
        assert diff(without_extras, driver.plain(saved), f"iteration {k}") == []


def test_on_iteration_extra_keys(callback_run):
    for (k, sol), solver_T_e_pres, saved in zip(
            callback_run["calls"], callback_run["solver_T_e_pres"], callback_run["saved"]):
        assert np.array_equal(sol["betan"], [1.0 + k])
        assert np.array_equal(sol["T_e_pres"], solver_T_e_pres)
        assert sol["T_e_pres"].shape == saved["psi_N"].shape
        assert not np.array_equal(sol["T_e_pres"], saved["T_e"])
