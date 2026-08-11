import unittest
from datetime import datetime, timedelta, timezone

from odds import (_is_future, american_to_prob, consensus_lines,
                  leader_split, main_line_points, paired_book_quotes)


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

    def test_first_inning_total_is_opt_in_and_pairs_like_a_total(self):
        market = [{"key": "totals_1st_1_innings", "outcomes": [
            {"name": "Over", "price": -105, "point": 0.5},
            {"name": "Under", "price": -115, "point": 0.5},
        ]}]
        event = _event([("dk", "DraftKings", market)])
        # A period total cannot accidentally enter the full-game path.
        self.assertEqual(paired_book_quotes(event), [])
        quote = paired_book_quotes(
            event, accepted_markets=("totals_1st_1_innings",))[0]
        self.assertEqual(quote["market"], "totals_1st_1_innings")
        self.assertEqual(quote["point"], 0.5)

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


class MainLineTests(unittest.TestCase):
    def _totals(self, point, home=-110, away=-110):
        return [{"key": "totals", "outcomes": [
            {"name": "Over", "price": home, "point": point},
            {"name": "Under", "price": away, "point": point},
        ]}]

    def test_broadest_book_point_is_the_only_executable_line(self):
        books = [
            ("a", "A", self._totals(8.5)),
            ("b", "B", self._totals(8.5)),
            ("c", "C", self._totals(8.5)),
            ("d", "D", self._totals(9.5)),
        ]
        event = _event(books)
        paired = paired_book_quotes(event)
        self.assertEqual(main_line_points(paired)["totals"], 8.5)
        rows = consensus_lines(event, paired)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["point"], 8.5)
        self.assertEqual(rows[0]["line_role"], "main")

    def test_best_price_carries_that_books_own_update_time(self):
        event = _event([
            ("a", "A", self._totals(8.5, -110, -110)),
            ("b", "B", self._totals(8.5, 105, -125)),
            ("c", "C", self._totals(8.5, -105, -115)),
        ])
        event["bookmakers"][0]["last_update"] = "2099-01-01T00:00:00Z"
        event["bookmakers"][1]["last_update"] = "2099-01-01T00:02:00Z"
        event["bookmakers"][2]["last_update"] = "2099-01-01T00:01:00Z"
        row = consensus_lines(event)[0]
        self.assertEqual(row["best_book_home"], "B")
        self.assertEqual(row["best_price_home_updated_at"],
                         "2099-01-01T00:02:00Z")
        self.assertEqual(row["consensus_oldest_updated_at"],
                         "2099-01-01T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
