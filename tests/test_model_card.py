"""Labelling on the public page, which is where a silent inversion would land.

The model prices everything home-oriented, and on a total "home" is the Over.
Every row on the page has to be turned back into the side a reader would
actually back. Getting that backwards does not raise: it publishes a
confident, well-formatted recommendation for the opposite side of the bet.
"""

import unittest
from datetime import datetime, timedelta, timezone

from model_card import (SHOP_SURVIVAL, build_board, build_shop,
                        build_shop_record, build_verdict,
                        market_label, side_label)

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

    def test_market_only_fair_does_not_hide_the_standalone_diagnostic(self):
        row = build_board([_card_row(
            model_prob_home="0.60", fair_prob_home="0.50",
            predicted_close_prob_home="0.505", predicted_clv="0.005",
            movement_weight="0.03")], [], [])[0]
        self.assertEqual(row["standalone"], 60.0)
        self.assertEqual(row["consensus"], 50.0)
        self.assertEqual(row["fair"], 50.0)
        self.assertEqual(row["gap"], 0.0)
        self.assertEqual(row["raw_gap"], 10.0)
        self.assertEqual(row["projected_move"], 0.5)
        self.assertTrue(row["movement_supported"])

    def test_exact_zero_does_not_invent_an_away_or_under_pick(self):
        row = build_board([_card_row(
            model_prob_home="0.50", fair_prob_home="0.50",
            predicted_close_prob_home="0.50", predicted_clv="0",
            movement_weight="0")], [], [])[0]
        self.assertEqual(row["pick"], "No directional signal")
        self.assertEqual(row["signal_kind"], "no directional signal")
        self.assertIsNone(row["price"])

    def test_timing_is_shown_even_when_an_earlier_gate_rejected_the_row(self):
        rejection = {"game_pk": "1", "market": "h2h", "point": "",
                     "gate": "below_edge_rule"}
        row = build_board([_card_row(lead_minutes="340")], [],
                          [rejection])[0]
        self.assertEqual(row["reason"], "below rule")
        self.assertEqual(row["timing"], "eligible window in 100 min")

    def test_a_frozen_clv_probe_is_labelled_as_a_non_wager_quote(self):
        card = _card_row(
            model_prob_home="0.60", fair_prob_home="0.50",
            predicted_close_prob_home="0.505", predicted_clv="0.005",
            movement_weight="0.03")
        signal = {"game_pk": "1", "market": "h2h", "point": "",
                  "side": "home", "predicted_clv": "0.004",
                  "price": "115", "book": "Frozen Book"}
        row = build_board([card], [], [], [signal])[0]
        self.assertTrue(row["probe"])
        self.assertFalse(row["bet"])
        self.assertEqual(row["projected_move"], 0.4)
        self.assertEqual(row["book"], "Frozen Book")
        self.assertEqual(row["reason"], "paper quote captured")


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


class ShopPanelTests(unittest.TestCase):
    """The panel is the only time-critical thing on a page written hourly."""

    def _alert(self, minutes_ago, deviation="1.50"):
        when = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
        return {"selection": "Chicago Cubs ML", "book_key": "betrivers",
                "price": "-110", "market": "h2h",
                "deviation_points": deviation, "books": "11",
                "fetched_at": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "commence_time": (when + timedelta(minutes=90)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ")}

    def test_an_alert_older_than_a_day_falls_off(self):
        rows = build_shop([self._alert(30), self._alert(60 * 25)])
        self.assertEqual(len(rows), 1)

    def test_newest_first_so_the_page_leads_with_the_live_one(self):
        rows = build_shop([self._alert(120), self._alert(2)])
        self.assertGreater(rows[0]["captured"], rows[1]["captured"])

    def test_the_price_is_a_number_not_the_logged_string(self):
        # The page prints a leading plus by sign. A string would compare by
        # coercion and a change in how the log is written could flip it.
        row = build_shop([self._alert(5)])[0]
        self.assertIsInstance(row["price"], float)
        self.assertEqual(row["price"], -110.0)

    def test_no_freshness_verdict_is_baked_in_at_render_time(self):
        # The page is written hourly and read whenever, so every judgement
        # about whether a price is still up belongs to the reader's clock.
        row = build_shop([self._alert(1)])[0]
        self.assertNotIn("live", row)
        self.assertNotIn("verdict", row)
        self.assertIn("captured", row)

    def test_the_survival_curve_never_climbs(self):
        shares = [step["live"] for step in SHOP_SURVIVAL]
        self.assertEqual(shares, sorted(shares, reverse=True))
        self.assertEqual(shares[0], 1.0)

    def test_an_unparseable_stamp_is_dropped_rather_than_shown_as_now(self):
        broken = self._alert(5)
        broken["fetched_at"] = "not a date"
        self.assertEqual(build_shop([broken]), [])


class ShopRecordTests(unittest.TestCase):
    """The line saying whether the alerts have actually been right."""

    def test_no_evidence_file_says_so_plainly(self):
        self.assertIn("No alerts have fired", build_shop_record({}))

    def test_logged_but_unscored_explains_why(self):
        line = build_shop_record({"alerts_logged": 3,
                                  "against_sharp_close": {"alerts": 0}})
        self.assertIn("3 alerts logged", line)
        self.assertIn("none scored", line)

    def test_the_record_is_read_off_the_sharp_close_not_the_panel(self):
        # The panel median is built from the quotes each alert deviated from,
        # so a record read off it would flatter itself.
        line = build_shop_record({
            "alerts_logged": 9, "forward_status": "research_only",
            "against_panel_median": {"alerts": 9, "dates": 4,
                                     "mean_clv_probability_points": 9.99,
                                     "share_beating_close": 1.0},
            "against_sharp_close": {"alerts": 9, "dates": 4,
                                    "mean_clv_probability_points": 0.31,
                                    "ci90_date_clustered_points": [0.07, 0.49],
                                    "share_beating_close": 0.6}})
        self.assertIn("+0.31", line)
        self.assertNotIn("9.99", line)
        self.assertIn("Pinnacle", line)

    def test_a_missing_interval_is_stated_not_omitted(self):
        line = build_shop_record({
            "alerts_logged": 5, "forward_status": "research_only",
            "against_sharp_close": {"alerts": 5, "dates": 1,
                                    "mean_clv_probability_points": 0.13,
                                    "ci90_date_clustered_points": None,
                                    "share_beating_close": 0.6}})
        self.assertIn("too few dates", line)


if __name__ == "__main__":
    unittest.main()
