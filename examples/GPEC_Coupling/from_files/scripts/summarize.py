"""Summarize complete prepared-equilibrium scans without an equilibrium solver."""

import argparse
import json
from pathlib import Path
import tomllib

from beta_limit import compare_estimates, estimate_limit

ROOT = Path(__file__).resolve().parent.parent


def summarize(rows, datasets):
    report = {
        "scope": "DIII-D, n=1, no conducting wall, qlow=1.02, fixed inner start; conditional example threshold.",
        "beta_definition": "GPEC normalized beta for both producers, not ordinary beta percent.",
        "datasets": {},
    }
    for name in datasets:
        metadata = tomllib.loads((ROOT / "inputs" / f"{name}.toml").read_text())
        selected = [row for row in rows if row["dataset"] == name]
        baseline_rows = [row for row in selected if row["preset"] == "baseline"]
        expected = {point["id"] for point in metadata["points"]}
        present = {row["point"] for row in baseline_rows}
        keys = [(row["point"], row["preset"]) for row in selected]
        complete = present == expected and len(baseline_rows) == len(expected)
        unique = len(keys) == len(set(keys))
        baseline = estimate_limit(baseline_rows)
        checks = {
            preset: estimate_limit([row for row in selected if row["preset"] == preset])
            for preset in ["fine", "core", "riccati"]
        }
        convergence = compare_estimates(baseline, checks)
        failed = sum(row["status"] != "completed" for row in selected)
        if not complete or not unique:
            convergence = {
                "status": "incomplete_or_duplicate_scan",
                "missing_points": sorted(expected - present),
            }
        elif failed:
            convergence = {"status": "solver_failures", "count": failed}
        report["datasets"][name] = {
            "source": metadata["source"],
            "constraints": metadata["constraints"],
            "baseline_complete": complete,
            "solver_warnings": [
                {
                    "point": row["point"],
                    "preset": row["preset"],
                    "messages": row["warnings"],
                }
                for row in selected
                if row.get("warnings")
            ],
            "baseline": baseline,
            "bracket_checks": checks,
            "convergence": convergence,
            "completed": len(selected) - failed,
            "failed": failed,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require complete scans and passing bracket comparisons",
    )
    args = parser.parse_args()
    rows = tomllib.loads((args.output / "results.toml").read_text())["results"]
    runtime = tomllib.loads((args.output / "runtime.toml").read_text())
    report = summarize(rows, runtime["datasets"])
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2))
    if args.strict and (
        not report["datasets"]
        or any(
            dataset["convergence"]["status"] != "passed_demonstration_gate"
            for dataset in report["datasets"].values()
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
