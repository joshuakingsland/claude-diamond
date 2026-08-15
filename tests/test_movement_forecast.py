import unittest

import numpy as np
import pandas as pd

from movement_forecast import apply, fit


class MovementForecastTests(unittest.TestCase):
    def _artifact(self):
        rows = []
        for year in (2022, 2023):
            for index in range(300):
                value = -0.5 + index / 299
                rows.append({
                    "official_date": f"{year}-05-{index % 28 + 1:02d}",
                    "season": year, "market": "h2h",
                    "entry_logit": value, "move_logit": 0.4 * value,
                })
        evaluation = {
            "protocol": {"version": "x", "protocol_hash": "hash"},
            "markets": {"h2h": {
                "candidate_selection": {
                    "selected_features": ["entry_logit"],
                    "selected_alpha": 0.1,
                },
                "confirmation_signal": True,
                "confirmation_2024": {"rows": 500},
            }},
        }
        return fit(pd.DataFrame(rows), evaluation)

    def test_supported_24_hour_row_gets_a_close_prediction(self):
        result = apply(
            0.55, 0.55, "h2h", self._artifact(), market_books=5,
            lead_minutes=1440, official_date="2026-08-15")
        self.assertTrue(result["eligible"])
        self.assertGreater(result["predicted_close_prob_home"], 0.55)

    def test_lock_window_is_not_mistaken_for_24_hour_entry(self):
        result = apply(
            0.55, 0.55, "h2h", self._artifact(), market_books=5,
            lead_minutes=180, official_date="2026-08-15")
        self.assertFalse(result["eligible"])
        self.assertEqual(result["predicted_clv"], 0.0)


if __name__ == "__main__":
    unittest.main()
