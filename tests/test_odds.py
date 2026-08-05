import unittest
from datetime import datetime, timedelta, timezone

from odds import _is_future, american_to_prob, consensus_lines, leader_split, paired_book_quotes


def _event(books, home="Home Nine", away="Away Nine"):
    return {"id": "e1", "commence_time": "2099-01-02T00:00:00Z",
            "home_team": home, "away_team": away,
            "bookmakers": [{"key": k, "title": t, "last_update": "2099-01-01T00:00:00Z",
                            "markets": m} for k, t, m in books]}


def _h2h(home_price, away_price, home="Home Nine", away="Away Nine"):
    return [{"key": "h2h", "outcomes": [{"name": home, "price": home_price},
                                        {"name": away, "price": away_price}]}]


class InPlayRejectionTests(unittest.TestCase):
    """In-play prices reflect the current score; the model cannot see it.

    The odds endpoint keeps returning a game after first pitch. In the first
    live capture those rows priced at 0.96, 0.97 and 0.13 home win
    probability — values a pre-game baseball market never produces, and which
    would read as enormous edges.
    """

    def test_future_start_is_accepted(self):
        ahead = datetime.now(timezone.utc) + timedelta(hours=3)
        self.assertTrue(_is_future(ahead.strftime("%Y-%m-%dT%H:%M:%SZ")))

    def test_started_game_is_rejected(self):
        behind = datetime.now(timezone.utc) - timedelta(minutes=1)
        self.assertFalse(_is_future(behind.strftime("%Y-%m-%dT%H:%M:%SZ")))

    def test_missing_or_malformed_start_is_rejected(self):
        self.assertFalse(_is_future(""))
        self.assertFalse(_is_future(None))
        self.assertFalse(_is_future("not a timestamp"))


class MarketPairingTests(unittest.TestCase):
    def test_totals_pair_over_and_under_at_the_same_point(self):
        markets = [{"key": "totals", "outcomes": [
            {"name": "Over", "price": -110, "point": 8.5},
            {"name": "Under", "price": -110, "point": 8.5},
            {"name": "Over", "price": 120, "point": 9.5},
            {"name": "Under", "price": -140, "point": 9.5},
        ]}]
        quotes = paired_book_quotes(_event([("dk", "DraftKings", markets)]))
        self.assertEqual({q["point"] for q in quotes}, {8.5, 9.5})

    def test_run_line_groups_both_sides_onto_the_home_point(self):
        markets = [{"key": "spreads", "outcomes": [
            {"name": "Home Nine", "price": 130, "point": -1.5},
            {"name": "Away Nine", "price": -150, "point": 1.5},
        ]}]
        quotes = paired_book_quotes(_event([("dk", "DraftKings", markets)]))
        self.assertEqual(len(quotes), 1)
        self.assertEqual(quotes[0]["point"], -1.5)

    def test_one_sided_market_is_not_invented_into_a_price(self):
        markets = [{"key": "h2h", "outcomes": [{"name": "Home Nine", "price": -150}]}]
        self.assertEqual(paired_book_quotes(_event([("dk", "DraftKings", markets)])), [])

    def test_unpriced_region_never_moves_the_consensus(self):
        us = paired_book_quotes(_event([("dk", "DraftKings", _h2h(-150, 130)),
                                        ("fd", "FanDuel", _h2h(-155, 135)),
                                        ("betonlineag", "BetOnline.ag", _h2h(-145, 125))]), "us")
        eu = paired_book_quotes(_event([("pinnacle", "Pinnacle", _h2h(-400, 320))]), "eu")
        with_eu = consensus_lines(None, us + eu)
        without = consensus_lines(None, us)
        self.assertEqual(with_eu[0]["consensus_prob_home"],
                         without[0]["consensus_prob_home"])
        self.assertEqual(with_eu[0]["market_books"], 3)
        # Pinnacle offers the best away price but must not be executable.
        self.assertNotEqual(with_eu[0]["best_book_away"], "Pinnacle")
        # It does count as a market leader, which is why eu is captured.
        self.assertEqual(with_eu[0]["leader_books"], 2)


if __name__ == "__main__":
    unittest.main()
