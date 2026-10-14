"""Keywords of ESCAPE_solve for in-memory use, with fake EPEDNN models (no Julia).

The runs use the ARC inputs of tests/arc_integration_driver.py with the real
1D scipy solver. feed_epednn is replaced by a fake, so no EPEDNN model is
evaluated.
"""
import importlib.util

import numpy as np
import pytest

from pedestal_prediction.ESCAPE_interface import ESCAPE_ROOT

DRIVER_FILE = ESCAPE_ROOT.parents[1] / "tests" / "arc_integration_driver.py"
SOLVE_ONLY_KEYS = ("sc_params", "epednn_params", "ESCAPE_iter_max", "ESCAPE_tol_max")


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("arc_integration_driver", DRIVER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
