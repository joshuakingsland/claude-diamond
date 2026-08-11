import unittest

import numpy as np
import pandas as pd

from market_offset import (_walk_forward_movement, apply, blend_probability,
                           fit_outcome_weight)


class OffsetTests(unittest.TestCase):
    def test_zero_weight_is_exactly_the_market(self):
        market = np.array([0.35, 0.5, 0.72])
        model = np.array([0.7, 0.2, 0.4])
        np.testing.assert_allclose(blend_probability(market, model, 0), market)

    def test_a_worse_residual_is_constrained_back_to_market(self):
        market = np.tile([0.35, 0.65], 200)
        outcome = np.tile([0.0, 1.0], 200)
        # Standalone pushes with confidence in exactly the wrong direction.
        model = 1.0 - market
        frame = pd.DataFrame({"market_prob": market, "model_prob": model,
                              "outcome": outcome})
        fitted = fit_outcome_weight(frame)
        self.assertLess(fitted["weight"], 0.001)
        self.assertLessEqual(fitted["log_loss_offset"],
                             fitted["log_loss_standalone"])

    def test_live_application_keeps_outcome_and_clv_targets_separate(self):
        artifact = {"version": "x", "outcome": {"h2h": {"weight": 0.0}},
                    "movement": {"h2h": {"weight": 1.0}}}
        result = apply(0.60, 0.50, "h2h", artifact)
        self.assertAlmostEqual(result["fair_prob_home"], 0.50)
        self.assertAlmostEqual(result["predicted_close_prob_home"], 0.60)

    def test_market_leader_signal_can_move_the_close_without_moving_fair(self):
        artifact = {
            "version": "x", "outcome": {"h2h": {"weight": 0.0}},
            "movement": {"h2h": {"model_weight": 0.0,
                                   "leader_weight": 1.0}},
        }
        result = apply(0.50, 0.50, "h2h", artifact,
                       leader_probability=0.58)
        self.assertAlmostEqual(result["fair_prob_home"], 0.50)
        self.assertAlmostEqual(result["predicted_close_prob_home"], 0.58)

    def test_movement_walk_forward_is_strictly_later_and_reports_uncertainty(self):
        rows = 600
        dates = [f"2025-{4 + (i // 150):02d}-{i % 25 + 1:02d}"
                 for i in range(rows)]
        entry = np.tile([0.45, 0.55], rows // 2)
        leader = entry + np.tile([-0.02, 0.02], rows // 2)
        block = pd.DataFrame({
            "official_date": dates, "entry_prob": entry,
            "close_prob": leader, "model_prob": entry,
            "entry_leader_prob": leader,
        })
        report = _walk_forward_movement(block)
        self.assertGreater(report["rows"], 0)
        self.assertLess(report["rmse_offset_logit"],
                        report["rmse_entry_logit"])
        self.assertIsNotNone(report["improvement_ci95_date_clustered"])


if __name__ == "__main__":
    unittest.main()
