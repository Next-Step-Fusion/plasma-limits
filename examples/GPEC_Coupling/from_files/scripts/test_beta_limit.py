import unittest
from beta_limit import estimate_limit, compare_estimates


def point(i, beta, energy, valid=True):
    return dict(
        point=str(i),
        multiplier=i,
        beta_n=beta,
        delta_w=energy,
        status="completed",
        screening_pass=valid,
    )


class LimitTests(unittest.TestCase):
    def test_linear_threshold(self):
        result = estimate_limit([point(1, 2.0, 0.4), point(2, 3.0, -0.4)])
        self.assertEqual(result["beta_n_estimate"], 2.5)
        self.assertEqual(result["beta_n_bracket"], [2.0, 3.0])

    def test_pole_or_internal_instability_cannot_create_threshold(self):
        rows = [point(1, 2.0, 0.4), point(2, 3.0, -0.4, False), point(3, 4.0, 0.2)]
        self.assertEqual(estimate_limit(rows)["status"], "blocked_by_failed_screening")

    def test_missing_low_beta_state(self):
        self.assertEqual(
            estimate_limit([point(1, 2.0, -0.1), point(2, 3.0, -0.2)])["status"],
            "lowest_point_already_unstable",
        )

    def test_nonmonotonic_beta_is_not_sorted_into_false_scan(self):
        self.assertEqual(
            estimate_limit([point(1, 3.0, 0.1), point(2, 2.0, -0.1)])["status"],
            "invalid_beta_order",
        )

    def test_exact_endpoint_zero(self):
        self.assertEqual(
            estimate_limit([point(1, 2.0, 0.1), point(2, 3.0, 0.0)])["beta_n_estimate"],
            3.0,
        )

    def test_later_internal_instability_does_not_replace_first_threshold(self):
        rows = [point(1, 2.0, 0.4), point(2, 3.0, -0.4), point(3, 4.0, 0.2, False)]
        self.assertEqual(estimate_limit(rows)["beta_n_estimate"], 2.5)

    def test_unchecked_result_is_not_accepted(self):
        base = estimate_limit([point(1, 2.0, 0.4), point(2, 3.0, -0.4)])
        self.assertEqual(
            compare_estimates(base, {})["status"], "incomplete_or_failed_checks"
        )
        self.assertEqual(
            compare_estimates(base, dict.fromkeys(["fine", "core", "riccati"], base))[
                "status"
            ],
            "passed_demonstration_gate",
        )


if __name__ == "__main__":
    unittest.main()
