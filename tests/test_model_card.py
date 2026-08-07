"""Labelling on the public page, which is where a silent inversion would land.

The model prices everything home-oriented, and on a total "home" is the Over.
Every row on the page has to be turned back into the side a reader would
actually back. Getting that backwards does not raise: it publishes a
confident, well-formatted recommendation for the opposite side of the bet.
"""

import unittest

from model_card import build_board, build_verdict, market_label, side_label

HOME, AWAY = "Kansas City Royals", "Minnesota Twins"


class SideLabelTests(unittest.TestCase):
    def test_moneyline_names_the_team_backed(self):
        self.assertEqual(side_label("h2h", "home", HOME, AWAY, None),
                         f"{HOME} ML")
        self.assertEqual(side_label("h2h", "away", HOME, AWAY, None),
                         f"{AWAY} ML")

    def test_the_away_run_line_is_the_negation_of_the_home_point(self):
        """A home team at +1.5 is the away team at -1.5."""
        self.assertEqual(side_label("spreads", "away", HOME, AWAY, 1.5),
                         f"{AWAY} -1.5")
        self.assertEqual(side_label("spreads", "home", HOME, AWAY, -1.5),
                         f"{HOME} -1.5")

    def test_the_over_is_the_home_side_of_a_total(self):
        self.assertEqual(side_label("totals", "home", HOME, AWAY, 8.5),
                         "Over 8.5")
        self.assertEqual(side_label("totals", "away", HOME, AWAY, 8.5),
                         "Under 8.5")

    def test_the_run_line_point_is_not_restated_against_the_pick(self):
        self.assertEqual(market_label("spreads", 1.5), "run line")
        self.assertEqual(market_label("totals", 8.5), "total 8.5")


def _card_row(**overrides):
    row = {
        "game_pk": "1", "official_date": "2026-08-05",
        "commence_time": "2026-08-05T23:00:00Z",
        "home_team": HOME, "away_team": AWAY, "market": "h2h", "point": "",
        "model_prob_home": "0.60", "market_prob_home": "0.50",
        "disagreement": "0.10", "market_books": "8", "market_spread": "0.01",
        "consensus_price_home": "-110", "consensus_price_away": "-110",
        "best_price_home": "-105", "best_book_home": "Book A",
        "best_price_away": "120", "best_book_away": "Book B",
        "expected_home_runs": "4.5", "expected_away_runs": "4.2",
        "lead_minutes": "120", "odds_fetched_at": "2026-08-05T22:00:00Z",
        "model_version": "diamond-v0", "model_kind": "glm",
        "priced_at": "2026-08-05T22:01:00Z",
    }
    row.update(overrides)
    return row


class BoardTests(unittest.TestCase):
    def test_the_leaning_side_carries_its_own_price_and_book(self):
        board = build_board([_card_row()], [], [])
        self.assertEqual(board[0]["pick"], f"{HOME} ML")
        self.assertEqual(board[0]["price"], -105.0)
        self.assertEqual(board[0]["book"], "Book A")

    def test_a_model_below_the_market_flips_to_the_away_side(self):
        board = build_board([_card_row(model_prob_home="0.40",
                                       disagreement="-0.10")], [], [])
        self.assertEqual(board[0]["pick"], f"{AWAY} ML")
        self.assertEqual(board[0]["price"], 120.0)
        self.assertEqual(board[0]["model"], 60.0)
        self.assertEqual(board[0]["consensus"], 50.0)

    def test_the_gap_is_reported_unsigned(self):
        board = build_board([_card_row(model_prob_home="0.40",
                                       disagreement="-0.10")], [], [])
        self.assertEqual(board[0]["gap"], 10.0)

    def test_a_wager_in_the_ledger_marks_the_row(self):
        wager = {"game_pk": "1", "market": "h2h", "point": "", "stake": "1.0",
                 "official_date": "2026-08-05", "home_team": HOME,
                 "away_team": AWAY, "side": "home", "model_prob": "0.6",
                 "market_prob": "0.5", "disagreement": "0.1", "price": "-105",
                 "outcome": "", "profit": ""}
        board = build_board([_card_row()], [wager], [])
        self.assertTrue(board[0]["bet"])
        self.assertEqual(board[0]["stake"], 1.0)

    def test_a_locked_wager_is_shown_as_it_was_struck(self):
        """The board moves after a lock; the recorded quote is what settles.

        A wager taken at five books was displaying the one book still quoting
        the line, which reads as a wager that breached the book-count gate.
        """
        wager = {"game_pk": "1", "market": "h2h", "point": "", "stake": "1.0",
                 "official_date": "2026-08-05", "home_team": HOME,
                 "away_team": AWAY, "side": "home", "model_prob": "0.61",
                 "market_prob": "0.49", "disagreement": "0.12",
                 "price": "-101", "book": "Locked Book", "market_books": "5",
                 "market_spread": "0.02", "lead_minutes": "45",
                 "outcome": "", "profit": ""}
        thin = _card_row(market_books="1", best_price_home="-300",
                         best_book_home="Last Book")
        row = build_board([thin], [wager], [])[0]
        self.assertEqual(row["books"], 5)
        self.assertEqual(row["price"], -101.0)
        self.assertEqual(row["book"], "Locked Book")
        self.assertEqual(row["model"], 61.0)
        self.assertEqual(row["gap"], 12.0)

    def test_an_unlocked_row_still_reads_the_live_board(self):
        row = build_board([_card_row(market_books="4")], [], [])[0]
        self.assertEqual(row["books"], 4)

    def test_a_rejected_row_shows_the_gate_that_stopped_it(self):
        rejection = {"game_pk": "1", "market": "h2h", "point": "",
                     "gate": "below_edge_rule"}
        board = build_board([_card_row()], [], [rejection])
        self.assertFalse(board[0]["bet"])
        self.assertEqual(board[0]["reason"], "below rule")


class VerdictTests(unittest.TestCase):
    def test_a_market_the_model_loses_is_not_marked_beaten(self):
        comparison = {"close_prob": {"h2h": {
            "games": 100, "delta": 0.005,
            "delta_ci90_date_clustered": [0.001, 0.01],
            "verdict": "market better; interval excludes zero"}}}
        row = build_verdict(comparison)[0]
        self.assertFalse(row["beaten"])

    def test_beaten_requires_the_whole_interval_below_zero(self):
        """A negative point estimate whose interval spans zero is undecided."""
        spans = {"close_prob": {"h2h": {
            "games": 100, "delta": -0.0001,
            "delta_ci90_date_clustered": [-0.007, 0.007], "verdict": "undecided"}}}
        self.assertFalse(build_verdict(spans)[0]["beaten"])
        clear = {"close_prob": {"h2h": {
            "games": 100, "delta": -0.01,
            "delta_ci90_date_clustered": [-0.02, -0.002], "verdict": "model better"}}}
        self.assertTrue(build_verdict(clear)[0]["beaten"])


if __name__ == "__main__":
    unittest.main()
