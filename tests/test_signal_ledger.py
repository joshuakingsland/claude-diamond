import unittest
from datetime import datetime, timezone

import pandas as pd

from signal_ledger import screen


NOW = datetime(2026, 8, 11, 18, 0, tzinfo=timezone.utc)


def card(**overrides):
    row = {
        "event_id": "e", "game_pk": 1, "official_date": "2026-08-11",
        "commence_time": "2026-08-11T20:00:00Z", "home_team": "H",
        "away_team": "A", "market": "h2h", "point": "",
        "predicted_clv": 0.004, "lineups_confirmed": 1,
        "market_books": 5, "lead_minutes": 120,
        "odds_fetched_at": "2026-08-11T17:55:00Z",
        "best_price_home_updated_at": "2026-08-11T17:55:00Z",
        "best_price_away_updated_at": "2026-08-11T17:55:00Z",
        "best_price_home": -105, "best_price_away": -105,
        "best_book_home": "B", "best_book_away": "B",
        "model_version": "m", "market_offset_version": "o",
    }
    row.update(overrides)
    return pd.DataFrame([row])


class SignalTests(unittest.TestCase):
    def test_a_fixed_forward_signal_is_recorded_without_calling_it_a_fill(self):
        rows = screen(card(), now=NOW)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["execution_status"], "paper_quote")
        self.assertEqual(rows[0]["side"], "home")

    def test_unconfirmed_lineups_never_enter_the_probe(self):
        self.assertEqual(screen(card(lineups_confirmed=0), now=NOW), [])

    def test_h2h_and_runline_compete_for_one_side_signal(self):
        runline = card(market="spreads", point=-1.5, predicted_clv=0.008)
        rows = screen(pd.concat([card(), runline], ignore_index=True), now=NOW)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["market"], "spreads")


if __name__ == "__main__":
    unittest.main()
