"""Check the two full scans against the documented reference threshold estimates."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
actual = json.loads((args.output / "summary.json").read_text())["datasets"]
reference = json.loads(
    (Path(__file__).resolve().parent.parent / "reference" / "summary.json").read_text()
)["datasets"]
for name, expected in reference.items():
    result = actual[name]
    if result["convergence"]["status"] != "passed_demonstration_gate":
        raise SystemExit(f"{name}: comparison gate failed")
    baseline = result["baseline"]
    target = expected["baseline"]
    error = abs(baseline["beta_n_estimate"] / target["beta_n_estimate"] - 1)
    if baseline["bracket_points"] != target["bracket_points"] or error > 0.01:
        raise SystemExit(
            f"{name}: reference changed (relative threshold error {error:.3%})"
        )
    print(f"{name}: same bracket, reference threshold difference {error:.3%}")
