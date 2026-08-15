import unittest

import numpy as np
import pandas as pd

from first_inning_model_evaluation import (
    CONFIRMATION_YEAR, EXCLUDED_YEAR, SELECTION_YEAR, TRAIN_YEAR,
    build_evaluation_rows, evaluate,
)


class FirstInningModelRowsTests(unittest.TestCase):
    def test_rows_require_two_books_and_collapse_duplicate_game_events(self):
        quotes = pd.DataFrame([
            {"event_id": "a", "market": "totals_1st_1_innings", "point": 0.5,
             "book_key": "one", "devig_prob_home": 0.45},
            {"event_id": "a", "market": "totals_1st_1_innings", "point": 0.5,
             "book_key": "two", "devig_prob_home": 0.55},
            {"event_id": "b", "market": "totals_1st_1_innings", "point": 0.5,
             "book_key": "one", "devig_prob_home": 0.47},
            {"event_id": "b", "market": "totals_1st_1_innings", "point": 0.5,
             "book_key": "two", "devig_prob_home": 0.49},
            {"event_id": "thin", "market": "totals_1st_1_innings", "point": 0.5,
             "book_key": "one", "devig_prob_home": 0.90},
        ])
        result = {"game_pk": 10, "official_date": "2025-05-01", "yrfi": 1,
                  "result_status": "final", "game_type": "R",
                  "commence_time": "2025-05-01T20:00:00Z",
                  "home_team": "Home", "away_team": "Away"}
        results = pd.DataFrame([
            {"event_id": "a", **result}, {"event_id": "b", **result},
            {"event_id": "thin", **{**result, "game_pk": 11}},
        ])
        features = pd.DataFrame([
            {"game_pk": 10, "official_date": "2025-05-01",
             "expected_home_runs_prior": 4.5},
            {"game_pk": 11, "official_date": "2025-05-01",
             "expected_home_runs_prior": 4.5},
        ])
        rows, integrity = build_evaluation_rows(quotes, results, features)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0]["game_pk"], 10)
        self.assertAlmostEqual(rows.iloc[0]["market_prob_yrfi"], 0.49)
        self.assertEqual(integrity["duplicate_provider_event_rows_collapsed"], 1)


class FirstInningTemporalEvaluationTests(unittest.TestCase):
    def _rows(self):
        rng = np.random.default_rng(7)
        rows = []
        for year in (TRAIN_YEAR, SELECTION_YEAR, CONFIRMATION_YEAR,
                     EXCLUDED_YEAR):
            count = 500 if year != EXCLUDED_YEAR else 50
            for index in range(count):
                signal = rng.normal()
                probability = 1 / (1 + np.exp(-1.4 * signal))
                outcome = float(rng.random() < probability)
                rows.append({
                    "game_pk": year * 10000 + index,
                    "official_date": f"{year}-{4 + index // 180:02d}-{index % 28 + 1:02d}",
                    "season": year,
                    "yrfi": outcome,
                    "market_prob_yrfi": 0.5,
                    "market_logit": 0.0,
                    "signal": signal,
                })
        return pd.DataFrame(rows)

    def test_real_incremental_signal_survives_confirmation(self):
        report = evaluate(
            self._rows(), {"qualified_unique_games": 1550}, draws=300,
            feature_families={"signal": ("market_logit", "signal")},
            c_values=(0.2, 1.0),
        )
        self.assertTrue(report["confirmation_2025"][
            "confirmed_incremental_baseball_signal"])
        self.assertEqual(report["excluded_2026"]["eligible_rows"], 50)
        self.assertFalse(report["excluded_2026"]["outcomes_evaluated"])
        self.assertNotIn("yrfi_rate", report["excluded_2026"])


if __name__ == "__main__":
    unittest.main()
