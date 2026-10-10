"""Smoke tests for the reference-data fixtures in conftest.py."""

ITER_KEYS = {
    "x", "ne", "nFC", "nCX", "T_e", "psi_N", "psi_Te_eval", "psi_ne_eval", "n_e",
    "Te_ped_eV", "ne_ped_val", "psi_ped", "pedestal_height", "pedestal_width",
    "SOLVE_KW", "model", "implementation", "diagnostics",
}


def test_ref_iters(ref_iters):
    assert len(ref_iters) == 5
    for k, it in enumerate(ref_iters):
        assert ITER_KEYS <= it.keys(), f"iteration {k}"
        assert ("loop_tol" in it) == (k > 0), f"iteration {k}"


def test_ref_eq_grid(ref_eq):
    assert ref_eq["psirz"].shape == (129, 129)
    assert ref_eq["rzout"].shape == (102, 2)


def test_julia_model_rebuilt(ref_out_dict):
    import juliacall  # imported here: importing juliacall starts Julia
    assert isinstance(ref_out_dict["state"]["epednn_model"], juliacall.AnyValue)


def test_ne_bcs_rebuilt(ref_out_dict):
    from src.saarelma_connor.bc_ig_helpers import NeBCs  # importable once ref_out_dict ran
    assert isinstance(ref_out_dict["state"]["ne_bcs"], NeBCs)
