import unittest

import numpy as np
import pandas as pd

from full_game_movement_evaluation import (
    CONFIRMATION_YEAR, ROW_COLUMNS, SELECTION_YEAR, TRAIN_YEAR, audit_coverage,
    evaluate_market,
)


def _coverage(complete=True):
    return {str(year): {"complete": complete}
            for year in (TRAIN_YEAR, SELECTION_YEAR, CONFIRMATION_YEAR)}


def _rows(signal=True, rows_per_year=600):
    rows = []
    rng = np.random.default_rng(4)
    for year in (TRAIN_YEAR, SELECTION_YEAR, CONFIRMATION_YEAR):
        for index in range(rows_per_year):
            leader_gap = rng.normal(0, 0.04)
            noise = rng.normal(0, 0.004)
            move = 0.7 * leader_gap + noise if signal else noise
            rows.append({
                "game_pk": year * 10000 + index,
                "official_date": f"{year}-{4 + index // 200:02d}-{index % 28 + 1:02d}",
                "season": year,
                "market": "h2h",
                "move_logit": move,
                "entry_logit": rng.normal(0, 0.3),
                "abs_entry_logit": 0.2,
                "leader_gap_logit": leader_gap,
                "follower_gap_logit": 0.0,
                "leader_available": 1.0,
                "follower_available": 1.0,
                "entry_market_spread": abs(leader_gap),
                "entry_books": 6.0,
                "entry_lead_hours": 24.0,
                "point": 0.0,
                "month_sin": 0.5,
                "month_cos": -0.5,
                "model_gap_logit": 0.0,
            })
    return pd.DataFrame(rows)


class MovementEvaluationTests(unittest.TestCase):
    def test_empty_progress_table_has_a_stable_schema(self):
        empty = pd.DataFrame(columns=ROW_COLUMNS)
        report = evaluate_market(empty, _coverage(False), "h2h", draws=10)
        self.assertEqual(report["rows_by_season"][str(TRAIN_YEAR)], 0)
        self.assertIn("awaiting", report["status"])

    def test_confirmation_stays_sealed_until_archive_is_complete(self):
        coverage = _coverage()
        coverage[str(CONFIRMATION_YEAR)]["complete"] = False
        report = evaluate_market(_rows(), coverage, "h2h", draws=100)
        self.assertIn("remains sealed", report["status"])
        self.assertNotIn("confirmation_2024", report)

    def test_real_forward_signal_must_survive_held_out_year(self):
        report = evaluate_market(_rows(), _coverage(), "h2h", draws=200)
        self.assertTrue(report["development_signal"])
        self.assertTrue(report["confirmation_signal"])
        self.assertGreater(
            report["confirmation_2024"]["improvement_ci95_date_clustered"][0], 0)

    def test_audit_completion_uses_attempts_not_only_offers(self):
        rows = []
        for role, statuses in (("close", ["offered"] * 3),
                               ("early", ["offered", "failed", "no_offer"])):
            for status in statuses:
                rows.append({"commence_time": "2024-05-01T20:00:00Z",
                             "snapshot_role": role, "status": status})
        coverage = audit_coverage(pd.DataFrame(rows))
        self.assertTrue(coverage["2024"]["complete"])
        self.assertEqual(coverage["2024"]["early_offered"], 1)


if __name__ == "__main__":
    unittest.main()
