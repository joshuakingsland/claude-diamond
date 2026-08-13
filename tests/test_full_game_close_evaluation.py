import unittest

import pandas as pd

from full_game_close_evaluation import (_metric_block,
                                        select_market_anchor_weight)


class MetricBlockTests(unittest.TestCase):
    def test_report_reads_interval_not_only_point_estimate(self):
        rows = []
        for index in range(120):
            outcome = float(index % 2)
            rows.append({
                "official_date": f"2024-{1 + index // 30:02d}-{1 + index % 28:02d}",
                "outcome": outcome,
                "close_prob": 0.5,
                "model_prob": 0.75 if outcome else 0.25,
            })
        report = _metric_block(pd.DataFrame(rows), "model_prob")
        self.assertLess(report["log_loss_delta_model_minus_market"], 0)
        self.assertEqual(report["verdict"],
                         "model better; interval excludes zero")

    def test_market_anchor_rejects_a_model_with_no_incremental_signal(self):
        frame = pd.DataFrame({
            "close_prob": [0.8, 0.2] * 60,
            "model_prob": [0.2, 0.8] * 60,
            "outcome": [1.0, 0.0] * 60,
        })
        self.assertEqual(select_market_anchor_weight(frame, "model_prob"), 0.0)

    def test_market_anchor_accepts_a_stronger_model(self):
        frame = pd.DataFrame({
            "close_prob": [0.5, 0.5] * 60,
            "model_prob": [0.8, 0.2] * 60,
            "outcome": [1.0, 0.0] * 60,
        })
        self.assertEqual(select_market_anchor_weight(frame, "model_prob"), 1.0)


if __name__ == "__main__":
    unittest.main()
