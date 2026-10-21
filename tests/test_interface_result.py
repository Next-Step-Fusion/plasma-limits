"""IterationRecord, ESCAPEResult and ESCAPERunError, on the iteration dicts of arc_1d_baseline.npy."""
import copy
import dataclasses
import json
import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from plasma_limits.pedestal_prediction.ESCAPE_interface import (ESCAPEResult, ESCAPERunError, IterationRecord,
                                                              _plain)


@pytest.fixture
def sols(arc_1d_iterations):
    """The baseline iteration dicts with the two keys that only on_iteration gets."""
    return [{**copy.deepcopy(it), "betan": np.array([1.5 + k]), "T_e_pres": np.full_like(it["psi_N"], 100.0 + k)}
            for k, it in enumerate(arc_1d_iterations)]


@pytest.fixture
def records(sols):
    return [IterationRecord._from_sol(k, sol) for k, sol in enumerate(sols)]


def test_record_fields_are_the_mapping(record_array_keys, record_value_keys):
    names = {f.name for f in dataclasses.fields(IterationRecord)}
    assert names == {"iteration", "betan", "loop_tol", "diagnostics"} | record_array_keys.keys() | record_value_keys.keys()


def test_record_from_sol_maps_every_key(records, sols, diff, record_array_keys, record_value_keys):
    for k, (record, sol) in enumerate(zip(records, sols)):
        assert record.iteration == k
        for name, key in {**record_array_keys, **record_value_keys}.items():
            assert diff(getattr(record, name), sol[key]) == [], name
        assert diff(record.diagnostics, sol["diagnostics"]) == []
        assert type(record.betan) is float and record.betan == 1.5 + k
    assert records[0].loop_tol is None
    for record, sol in zip(records[1:], sols[1:]):
        assert diff(record.loop_tol, sol["loop_tol"]) == []


def test_record_keeps_read_only_copies(records, sols, record_array_keys):
    for record, sol in zip(records, sols):
        for name, key in record_array_keys.items():
            arr = getattr(record, name)
            assert not arr.flags.writeable, name
            assert not np.shares_memory(arr, sol[key]), name


def test_record_does_not_follow_later_changes_of_sol(records, sols, arc_1d_iterations, diff):
    sol = sols[1]
    sol["SOLVE_KW"]["initial_guess"] = "changed"
    sol["SOLVE_KW"]["free_params"]["C_KBM"] = -1.0
    sol["diagnostics"]["picard_info"]["history"][0]["alpha_bar"] = -1.0
    sol["diagnostics"]["hat_x_sol"][0] = -1.0
    expected = arc_1d_iterations[1]
    assert diff(records[1].params, expected["SOLVE_KW"]) == []
    assert diff(records[1].diagnostics, expected["diagnostics"]) == []


def test_record_is_frozen_and_keyword_only(sols):
    with pytest.raises(TypeError):
        IterationRecord(0)
    record = IterationRecord._from_sol(0, sols[0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.betan = 0.0


def test_record_to_dict(records, diff):
    record = records[1]
    d = record.to_dict()
    assert d.keys() == {f.name for f in dataclasses.fields(IterationRecord)}
    for name, value in d.items():
        assert diff(value, getattr(record, name)) == [], name
    d["x"][0] = -1.0
    d["params"]["model"] = "changed"
    assert record.x[0] != -1.0 and record.params["model"] == "1D"


class _Live:
    """Stands for a live object (for example a Julia object) in the diagnostics."""


def test_plain_drops_live_objects():
    obj = {"a": 1.0, "live": _Live(), "nested": {"live": _Live(), "b": "s"},
           "list": [1, _Live(), {"live": _Live()}], "tuple": (np.float64(2.0), _Live()),
           "objects": np.array([_Live()], dtype=object), "none": None}
    assert _plain(obj) == {"a": 1.0, "nested": {"b": "s"}, "list": [1, {}], "tuple": (np.float64(2.0),),
                           "none": None}


def test_plain_keeps_baseline_diagnostics_as_copies(arc_1d_iterations, diff):
    for it in arc_1d_iterations:
        plain = _plain(it["diagnostics"])
        assert diff(plain, it["diagnostics"]) == []
        assert not np.shares_memory(plain["hat_x_sol"], it["diagnostics"]["hat_x_sol"])
        assert plain["picard_info"]["history"] is not it["diagnostics"]["picard_info"]["history"]


def test_result_properties(records):
    result = ESCAPEResult(converged=False, history=records)
    assert result.history == tuple(records)
    assert result.n_iter == len(records)
    assert result.final is records[-1]
    assert result.state is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.converged = True
    with pytest.raises(TypeError):
        ESCAPEResult(False, records)


def test_result_to_dict_has_no_state(records, diff):
    result = ESCAPEResult(converged=True, history=records, state=_Live())
    d = result.to_dict()
    assert d.keys() == {"converged", "n_iter", "history"}
    assert (d["converged"], d["n_iter"]) == (True, len(records))
    assert diff(d["history"], [record.to_dict() for record in records]) == []
    assert diff(pickle.loads(pickle.dumps(d)), d) == []


def test_result_to_dict_loads_with_numpy_alone(records):
    """The pickled dict loads in a fresh interpreter that imports neither plasma_limits nor escape."""
    payload = pickle.dumps(ESCAPEResult(converged=False, history=records).to_dict())
    code = textwrap.dedent("""
        import json, pickle, sys
        d = pickle.load(sys.stdin.buffer)
        roots = {name.split(".")[0] for name in sys.modules}
        print(json.dumps({"n_iter": d["n_iter"], "n_history": len(d["history"]),
                          "loaded": sorted(roots & {"plasma_limits", "escape", "juliacall", "firedrake"})}))
    """)
    out = subprocess.run([sys.executable, "-c", code], input=payload, capture_output=True, check=True)
    result = json.loads(out.stdout)
    assert result == {"n_iter": len(records), "n_history": len(records), "loaded": []}


def test_run_error_keeps_history_through_pickle(records):
    err = ESCAPERunError("iteration 2 failed", records[:2])
    assert isinstance(err, RuntimeError)
    assert str(err) == "iteration 2 failed"
    assert err.history == tuple(records[:2])
    copy_ = pickle.loads(pickle.dumps(err))
    assert str(copy_) == "iteration 2 failed"
    assert [r.iteration for r in copy_.history] == [0, 1]
    assert ESCAPERunError("no history").history == ()
