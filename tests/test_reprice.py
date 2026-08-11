import unittest

import pandas as pd

from models import reprice_requests


class RepriceTests(unittest.TestCase):
    def setUp(self):
        self.predictions = pd.DataFrame([{
            "game_pk": 1, "expected_home_runs": 4.6,
            "expected_away_runs": 4.1, "scheduled_innings": 9,
            "dispersion_home": 3.5, "dispersion_away": 4.5,
            "inning_scoreless": 0.73, "inning_tail": 0.9,
            "extra_home_edge": 0.52,
            "extra_margin_1": 0.688, "extra_margin_2": 0.159,
            "extra_margin_3": 0.086, "extra_margin_4": 0.067,
            "walk_off_margin_1": 0.872, "walk_off_margin_2": 0.075,
            "walk_off_margin_3": 0.037, "walk_off_margin_4": 0.016,
        }])

    def test_arbitrary_points_are_priced_from_one_stored_distribution(self):
        requests = pd.DataFrame([
            {"game_pk": 1, "official_date": "2026-08-01",
             "market": "spreads", "point": -2.5},
            {"game_pk": 1, "official_date": "2026-08-01",
             "market": "totals", "point": 9.0},
        ])
        result = reprice_requests(requests, self.predictions)
        self.assertEqual(len(result), 2)
        self.assertTrue(result["model_prob_home"].between(0, 1).all())
        self.assertEqual(result.iloc[0]["model_push_prob"], 0.0)
        self.assertGreater(result.iloc[1]["model_push_prob"], 0.0)


if __name__ == "__main__":
    unittest.main()
