"""The ways a forward test can quietly become a backward one.

This file scores a record that cannot be re-selected, which is its whole
value. The failures worth guarding are the ones that would let it report
evidence it does not have: scoring a game that has not started, scoring
against a close that is the alert's own capture, and calling a degenerate
bootstrap an interval.
"""

import unittest
from datetime import timedelta

import numpy as np
import pandas as pd

import alert_evidence as ae

NOW = pd.Timestamp("2026-08-06T02:00:00Z")


def alert(event="g1", side="home", market="h2h", point=np.nan,
          break_even=0.45, captured="2026-08-05T21:00:00Z",
          commence="2026-08-05T22:00:00Z", book="betrivers", deviation=0.6):
    return {"alerted_at": captured, "fetched_at": captured,
            "event_id": event, "commence_time": commence,
            "home_team": "Chicago Cubs", "away_team": "St. Louis Cardinals",
            "market": market, "point": point, "side": side,
            "selection": "Chicago Cubs ML", "book_key": book, "price": 120,
            "deviation_points": deviation, "consensus_probability": 0.5,
            "break_even": break_even, "books": 11, "lead_minutes": 60,
            "threshold": 0.005}


def alerts(rows):
    frame = pd.DataFrame(rows)
    for column in ("commence_time", "fetched_at", "alerted_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def close(event="g1", market="h2h", point=np.nan, fair_home=0.5,
          captured="2026-08-05T21:45:00Z", books=11):
    return {"event_id": event, "market": market, "point": point,
            "close_fair_home": fair_home, "close_books": books,
            "close_captured": pd.Timestamp(captured),
            "close_lead": 15.0}


class ScoringTests(unittest.TestCase):
    def test_closing_line_value_uses_the_raw_break_even(self):
        # Took a price implying 0.45 into a close of 0.50: five points, and
        # the vig paid stays in the number, matching line_shopping.py.
        scored = ae.score(alerts([alert()]), pd.DataFrame([close()]), now=NOW)
        self.assertEqual(len(scored), 1)
        self.assertAlmostEqual(float(scored.iloc[0]["clv"]), 0.05, places=9)
        self.assertEqual(float(scored.iloc[0]["beat_close"]), 1.0)

    def test_the_away_side_is_scored_against_the_other_half(self):
        scored = ae.score(alerts([alert(side="away", break_even=0.45)]),
                          pd.DataFrame([close(fair_home=0.6)]), now=NOW)
        self.assertAlmostEqual(float(scored.iloc[0]["close_probability"]),
                               0.4, places=9)
        self.assertLess(float(scored.iloc[0]["clv"]), 0.0)

    def test_a_game_that_has_not_started_is_carried_not_scored(self):
        # The newest quote in the log is not a closing quote merely because
        # nothing newer exists yet.
        early = NOW - timedelta(days=1)
        self.assertEqual(
            len(ae.score(alerts([alert()]), pd.DataFrame([close()]),
                         now=early)), 0)

    def test_an_alert_at_the_close_itself_is_dropped(self):
        # Its consensus and its close come from one capture, so its value
        # would be the deviation it was selected on, read back.
        scored = ae.score(alerts([alert(captured="2026-08-05T21:45:00Z")]),
                          pd.DataFrame([close(captured="2026-08-05T21:45:00Z")]),
                          now=NOW)
        self.assertEqual(len(scored), 0)

    def test_a_moneyline_alert_survives_the_empty_point_spelling(self):
        # The log writes an absent point as an empty cell and the quote log
        # carries NaN. A mismatch here drops every moneyline row silently,
        # which is the single most likely way this file could report nothing
        # while looking healthy.
        rows = alerts([alert()])
        rows["point"] = ""
        self.assertEqual(len(ae.score(rows, pd.DataFrame([close()]),
                                      now=NOW)), 1)


class IntervalTests(unittest.TestCase):
    """One date resampled is the same number every draw, not an interval."""

    def _spread(self, dates):
        rows = [alert(event=f"g{i}", commence=f"2026-08-0{i+1}T22:00:00Z",
                      captured=f"2026-08-0{i+1}T21:00:00Z")
                for i in range(dates)]
        closes = pd.DataFrame([close(event=f"g{i}",
                                     captured=f"2026-08-0{i+1}T21:45:00Z")
                               for i in range(dates)])
        return ae.score(alerts(rows), closes, now=pd.Timestamp("2026-09-01T00:00:00Z"))

    def test_too_few_dates_yields_no_interval_at_all(self):
        for dates in (1, 2):
            self.assertIsNone(ae.interval(self._spread(dates)),
                              f"{dates} date(s) must not produce an interval")

    def test_enough_dates_yields_one(self):
        bounds = ae.interval(self._spread(4))
        self.assertIsNotNone(bounds)
        self.assertLessEqual(bounds[0], bounds[1])

    def test_the_summary_says_why_an_interval_is_missing(self):
        block = ae.summarise(self._spread(1), "panel median")
        self.assertIsNone(block["ci90_date_clustered_points"])
        self.assertIn("no interval", block["interval_note"])


class GateTests(unittest.TestCase):
    """The gate must not be passable by a flattering reference or by luck."""

    def _scored(self, dates):
        rows = [alert(event=f"g{i}", commence=f"2026-08-0{i+1}T22:00:00Z",
                      captured=f"2026-08-0{i+1}T21:00:00Z")
                for i in range(dates)]
        closes = pd.DataFrame([close(event=f"g{i}",
                                     captured=f"2026-08-0{i+1}T21:45:00Z")
                               for i in range(dates)])
        return ae.score(alerts(rows), closes, now=pd.Timestamp("2026-09-01T00:00:00Z"))

    def test_a_single_date_cannot_establish_anything(self):
        scored = self._scored(1)
        report = ae.evaluate(alerts([alert()]), scored, scored)
        self.assertEqual(report["forward_status"], "research_only")
        self.assertTrue(any("not established" in f
                            for f in report["forward_failures"]))

    def test_the_panel_median_alone_cannot_pass_the_gate(self):
        # Every alert deviates from the panel median by construction, so a
        # gate readable off it would be marking its own homework.
        strong = self._scored(5)
        report = ae.evaluate(alerts([alert()]), strong, pd.DataFrame())
        self.assertEqual(report["forward_status"], "research_only")

    def test_nothing_here_ever_recommends_a_stake(self):
        scored = self._scored(5)
        report = ae.evaluate(alerts([alert()]), scored, scored)
        self.assertEqual(report["stake_recommendation"], "none")


class EmptyTests(unittest.TestCase):
    def test_no_alerts_is_reported_rather_than_crashing(self):
        self.assertEqual(len(ae.score(pd.DataFrame(), pd.DataFrame())), 0)
        self.assertEqual(ae.summarise(pd.DataFrame(), "x")["alerts"], 0)


if __name__ == "__main__":
    unittest.main()
