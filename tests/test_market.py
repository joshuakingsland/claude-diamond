"""The join is where a market comparison quietly goes wrong.

None of these failures announce themselves. A dropped game leaves a smaller
sample that still looks clean, and a misjoined game attaches one game's price
to another game's result and reads as edge. Both were live in this repository:
the Athletics rename removed 129 game-keys and the UTC date rollover removed
another 68, together 9.5% of the priced card, and losing late West Coast games
is not losing games at random.
"""

import unittest

import numpy as np
import pandas as pd

from market import (MAX_START_DRIFT_HOURS, build_priced_games, delta_interval,
                    match_events_to_games, normalise, verdict)


def _events(rows):
    frame = pd.DataFrame(rows, columns=["event_id", "home_key", "away_key",
                                        "commence"])
    frame["commence"] = pd.to_datetime(frame["commence"], utc=True)
    return frame


def _games(rows):
    frame = pd.DataFrame(rows, columns=["game_pk", "home_key", "away_key",
                                        "start", "official_date"])
    frame["start"] = pd.to_datetime(frame["start"], utc=True)
    return frame


class TeamKeyTests(unittest.TestCase):
    def test_books_keep_oakland_after_statsapi_drops_it(self):
        self.assertEqual(normalise("Oakland Athletics"), normalise("Athletics"))

    def test_cleveland_spans_its_rename(self):
        self.assertEqual(normalise("Cleveland Indians"),
                         normalise("Cleveland Guardians"))

    def test_unrelated_teams_stay_distinct(self):
        self.assertNotEqual(normalise("New York Yankees"),
                            normalise("New York Mets"))


class StartTimeMatchTests(unittest.TestCase):
    def test_late_start_matches_across_the_utc_date_boundary(self):
        """A 19:10 Pacific first pitch is 02:10 UTC the next calendar day."""
        events = _events([("e1", "los angeles dodgers", "colorado rockies",
                           "2025-04-17T02:10:00Z")])
        games = _games([(1, "los angeles dodgers", "colorado rockies",
                         "2025-04-17T02:10:00Z", "2025-04-16")])
        matched, unmatched = match_events_to_games(events, games)
        self.assertEqual(unmatched, [])
        self.assertEqual(matched["e1"], (1, "2025-04-16"))

    def test_doubleheader_halves_do_not_collapse_onto_one_game(self):
        events = _events([
            ("e1", "home nine", "away nine", "2025-05-01T17:10:00Z"),
            ("e2", "home nine", "away nine", "2025-05-01T21:40:00Z"),
        ])
        games = _games([
            (1, "home nine", "away nine", "2025-05-01T17:10:00Z", "2025-05-01"),
            (2, "home nine", "away nine", "2025-05-01T21:40:00Z", "2025-05-01"),
        ])
        matched, _ = match_events_to_games(events, games)
        self.assertEqual({matched["e1"][0], matched["e2"][0]}, {1, 2})

    def test_a_game_is_claimed_only_once(self):
        events = _events([
            ("e1", "home nine", "away nine", "2025-05-01T17:10:00Z"),
            ("e2", "home nine", "away nine", "2025-05-01T17:40:00Z"),
        ])
        games = _games([(1, "home nine", "away nine",
                         "2025-05-01T17:10:00Z", "2025-05-01")])
        matched, unmatched = match_events_to_games(events, games)
        self.assertEqual(len(matched), 1)
        self.assertEqual(len(unmatched), 1)

    def test_a_postponed_game_is_dropped_not_stretched(self):
        """Prices struck for Tuesday do not belong to Thursday's makeup."""
        events = _events([("e1", "home nine", "away nine",
                           "2025-05-01T17:10:00Z")])
        games = _games([(1, "home nine", "away nine",
                         "2025-05-03T17:10:00Z", "2025-05-03")])
        matched, unmatched = match_events_to_games(events, games)
        self.assertEqual(matched, {})
        self.assertEqual(len(unmatched), 1)

    def test_drift_inside_the_tolerance_still_matches(self):
        start = pd.Timestamp("2025-05-01T17:10:00Z")
        events = _events([("e1", "home nine", "away nine", start)])
        late = start + pd.Timedelta(hours=MAX_START_DRIFT_HOURS - 1)
        games = _games([(1, "home nine", "away nine", late, "2025-05-01")])
        matched, _ = match_events_to_games(events, games)
        self.assertIn("e1", matched)


def _quote(event_id, book, market, price_home, price_away, taken,
           commence="2025-05-01T23:10:00Z", point=""):
    return {"event_id": event_id, "fetched_at": taken,
            "commence_time": commence, "home_team": "Home Nine",
            "away_team": "Away Nine", "market": market, "point": point,
            "book_key": book, "price_home": price_home,
            "price_away": price_away,
            "devig_prob_home": price_home / (price_home + price_away)}


class BookGateTests(unittest.TestCase):
    """A one-book close is not a market price, even beside a full entry."""

    def _frame(self, close_books):
        rows = []
        for book in ("a", "b", "c"):
            rows.append(_quote("e1", book, "h2h", 0.5, 0.5,
                               "2025-04-30T20:00:00Z"))
        for book in close_books:
            rows.append(_quote("e1", book, "h2h", 0.5, 0.5,
                               "2025-05-01T21:00:00Z"))
        quotes = pd.DataFrame(rows)
        games = pd.DataFrame([{
            "game_pk": 1, "home_team_name": "Home Nine",
            "away_team_name": "Away Nine",
            "game_date_utc": "2025-05-01T23:10:00Z",
            "official_date": "2025-05-01"}])
        priced, _ = build_priced_games(quotes, games)
        return priced[priced["market"] == "h2h"].iloc[0]

    def test_a_thin_close_is_dropped_while_the_entry_survives(self):
        row = self._frame(close_books=("a",))
        self.assertIsNone(row["close_prob"])
        self.assertIsNotNone(row["entry_prob"])

    def test_a_full_close_is_kept(self):
        row = self._frame(close_books=("a", "b", "c"))
        self.assertIsNotNone(row["close_prob"])


class IntervalTests(unittest.TestCase):
    """The interval, not the sign of the point estimate, is the finding."""

    def test_a_hair_thin_gap_reads_as_undecided(self):
        """Two equally uninformed forecasts, one arbitrarily ahead on points.

        Whichever of the two lands closer on a given sample, the honest
        reading is that nothing has been established. This is the totals
        result: a delta of -0.00007 that flips `model_beats_market` to true.
        """
        rng = np.random.default_rng(0)
        size = 600
        frame = pd.DataFrame({
            "official_date": [f"2025-05-{1 + i % 28:02d}" for i in range(size)],
            "model": 0.5 + rng.normal(0, 0.01, size),
            "market": 0.5 + rng.normal(0, 0.01, size),
        })
        outcome = rng.integers(0, 2, size)
        interval = delta_interval(frame, "model", "market", outcome, draws=300)
        self.assertEqual(verdict(interval), "undecided; interval spans zero")

    def test_too_few_slates_yields_no_interval(self):
        frame = pd.DataFrame({"official_date": ["2025-05-01"] * 20,
                              "model": [0.5] * 20, "market": [0.5] * 20})
        self.assertIsNone(
            delta_interval(frame, "model", "market", [0, 1] * 10))
        self.assertEqual(verdict(None), "insufficient sample for an interval")


if __name__ == "__main__":
    unittest.main()
