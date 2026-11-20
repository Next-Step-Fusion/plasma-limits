"""Checks for input integrity and incomplete/failed scan reporting."""

import copy
import hashlib
import tomllib
import unittest

from summarize import ROOT, summarize


def complete_rows(name):
    metadata = tomllib.loads((ROOT / "inputs" / f"{name}.toml").read_text())
    rows = [
        dict(
            dataset=name,
            point=point["id"],
            preset="baseline",
            multiplier=i,
            beta_n=1.0 + i,
            delta_w=1.0 - i,
            status="completed",
            screening_pass=True,
        )
        for i, point in enumerate(metadata["points"])
    ]
    for preset in ["fine", "core", "riccati"]:
        for row in rows[:2]:
            rows.append(dict(row, preset=preset))
    return rows


class ScanContractTests(unittest.TestCase):
    def test_prepared_equilibrium_checksums(self):
        for name in ["tokamaker", "nsfsim"]:
            metadata = tomllib.loads((ROOT / "inputs" / f"{name}.toml").read_text())
            for point in metadata["points"]:
                with self.subTest(dataset=name, point=point["id"]):
                    digest = hashlib.sha256(
                        (ROOT / point["equilibrium"]).read_bytes()
                    ).hexdigest()
                    self.assertEqual(digest, point["sha256"])
                    self.assertTrue((ROOT / point["kinetics"]).is_file())

    def test_whole_dataset_cannot_silently_disappear(self):
        report = summarize(complete_rows("tokamaker"), ["tokamaker", "nsfsim"])
        self.assertEqual(
            report["datasets"]["nsfsim"]["convergence"]["status"],
            "incomplete_or_duplicate_scan",
        )

    def test_missing_high_pressure_point_blocks_strict_success(self):
        rows = complete_rows("tokamaker")
        rows = [row for row in rows if row["point"] != "case_08"]
        report = summarize(rows, ["tokamaker"])
        self.assertFalse(report["datasets"]["tokamaker"]["baseline_complete"])

    def test_duplicate_results_are_rejected(self):
        rows = complete_rows("tokamaker")
        rows.append(copy.copy(rows[0]))
        report = summarize(rows, ["tokamaker"])
        self.assertEqual(
            report["datasets"]["tokamaker"]["convergence"]["status"],
            "incomplete_or_duplicate_scan",
        )

    def test_solver_failure_after_threshold_is_not_a_passing_run(self):
        rows = complete_rows("tokamaker")
        rows[7]["status"] = "failed"
        report = summarize(rows, ["tokamaker"])
        self.assertEqual(
            report["datasets"]["tokamaker"]["convergence"]["status"], "solver_failures"
        )

    def test_internal_instability_after_first_threshold_remains_visible(self):
        rows = complete_rows("tokamaker")
        rows[7]["screening_pass"] = False
        rows[7]["warnings"] = ["Fixed-boundary mode unstable for n = 1"]
        report = summarize(rows, ["tokamaker"])["datasets"]["tokamaker"]
        self.assertEqual(report["convergence"]["status"], "passed_demonstration_gate")
        self.assertEqual(report["solver_warnings"][0]["point"], "case_08")


if __name__ == "__main__":
    unittest.main()
