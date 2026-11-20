"""Extract a bracketed n=1 no-wall threshold from completed GPEC results."""

import math


def estimate_limit(rows):
    rows = sorted(rows, key=lambda r: r["multiplier"])
    if len(rows) < 2:
        return {"status": "insufficient_points"}
    previous_beta = -math.inf
    for i, row in enumerate(rows):
        if row.get("status") != "completed" or not row.get("screening_pass", False):
            return {"status": "blocked_by_failed_screening", "point": row["point"]}
        beta, energy = (row["beta_n"], row["delta_w"])
        if (
            not math.isfinite(beta)
            or not math.isfinite(energy)
            or beta <= previous_beta
        ):
            return {"status": "invalid_beta_order", "point": row["point"]}
        previous_beta = beta
        if i == 0 and energy <= 0:
            return {"status": "lowest_point_already_unstable", "point": row["point"]}
        if i and energy <= 0 < rows[i - 1]["delta_w"]:
            a = rows[i - 1]
            root = a["beta_n"] - a["delta_w"] * (beta - a["beta_n"]) / (
                energy - a["delta_w"]
            )
            return {
                "status": "bracketed",
                "beta_n_estimate": root,
                "beta_n_bracket": [a["beta_n"], beta],
                "bracket_points": [a["point"], row["point"]],
                "method": "linear interpolation within the first screened stable-to-unstable bracket",
                "precision_note": "The bracket is the data resolution; interpolation is not an additional equilibrium solve.",
            }
    return {"status": "no_unstable_point_in_scan"}


def compare_estimates(base, checks, relative_tolerance=0.01):
    if base.get("status") != "bracketed":
        return {"status": "no_baseline_bracket"}
    values = [base["beta_n_estimate"]]
    for name in ["fine", "core", "riccati"]:
        r = checks.get(name)
        if r is None or r.get("status") != "bracketed":
            return {"status": "incomplete_or_failed_checks", "check": name}
        values.append(r["beta_n_estimate"])
    spread = (max(values) - min(values)) / abs(base["beta_n_estimate"])
    return {
        "status": "passed_demonstration_gate"
        if spread <= relative_tolerance
        else "resolution_sensitive",
        "relative_estimate_spread": spread,
        "relative_tolerance": relative_tolerance,
        "note": "Checks recompute the bracket endpoints. This is not all-mode or experimental validation.",
    }
