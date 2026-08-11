import unittest

import pandas as pd

from config import MIN_FORWARD_INDEPENDENT_GAMES
from forward_evidence import evaluate, sharp_closes


class SharpCloseTests(unittest.TestCase):
    def test_latest_pregame_leader_quote_prices_the_backed_side(self):
        ledger = pd.DataFrame([{
            "wager_id": "w", "event_id": "e", "game_pk": 1,
            "official_date": "2026-08-01", "market": "h2h", "point": "",
            "side": "away", "price": 100,
        }])
        quotes = pd.DataFrame([
            {"event_id": "e", "market": "h2h", "point": "",
             "book_key": "pinnacle", "fetched_at": "2026-08-01T17:00:00Z",
             "commence_time": "2026-08-01T20:00:00Z",
             "devig_prob_home": 0.55},
            {"event_id": "e", "market": "h2h", "point": "",
             "book_key": "pinnacle", "fetched_at": "2026-08-01T19:59:00Z",
             "commence_time": "2026-08-01T20:00:00Z",
             "devig_prob_home": 0.48},
        ])
        close = sharp_closes(ledger, quotes).iloc[0]
        self.assertAlmostEqual(close["sharp_close_side"], 0.52)
        self.assertAlmostEqual(close["clv_probability"], 0.02)

    def test_promotion_counts_games_and_requires_accepted_fills(self):
        n = MIN_FORWARD_INDEPENDENT_GAMES
        ledger = pd.DataFrame({
            "game_pk": range(n),
            "execution_status": ["accepted"] * n,
            "official_date": [f"2026-07-{index % 20 + 1:02d}"
                              for index in range(n)],
            "profit": [None] * n,
        })
        closes = pd.DataFrame({
            "official_date": ledger["official_date"],
            "clv_probability": [0.02] * n,
        })
        report = evaluate(ledger, closes)
        self.assertEqual(report["promotion_status"], "eligible_for_review")
        paper = ledger.copy()
        paper["execution_status"] = "paper"
        self.assertEqual(evaluate(paper, closes)["promotion_status"],
                         "research_only")


if __name__ == "__main__":
    unittest.main()
