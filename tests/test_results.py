"""A postponed game is returned twice, and both copies used to survive.

This is not a cosmetic duplicate. `features.py` folds each row's result into
team state as it walks the table, so a duplicated game counted twice toward
Elo, run rates and park factors, and `weather.py` fetched conditions for a
date the game was not played on. The committed dataset carried 274 such games
and every season came out above the 2,430 a regular season contains.
"""

import unittest

from results import deduplicate, parse_game


def _row(game_pk, official_date, start, home_score=None, away_score=None):
    return {"game_pk": game_pk, "official_date": official_date,
            "game_date_utc": start, "home_score": home_score,
            "away_score": away_score, "status": "Final"}


class DeduplicateTests(unittest.TestCase):
    def test_the_self_consistent_row_wins(self):
        """Keep the row whose UTC start agrees with its own official date.

        `features.py` keys off official_date and `market.py` matches on the
        UTC start; a row where the two disagree sends them to different games.
        """
        kept = deduplicate([
            _row(1, "2021-04-11", "2021-08-31T17:10:00Z", 6, 5),
            _row(1, "2021-04-11", "2021-04-11T17:10:00Z", 6, 5),
        ])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["game_date_utc"], "2021-04-11T17:10:00Z")

    def test_a_played_row_beats_an_unplayed_one(self):
        kept = deduplicate([
            _row(2, "2021-08-10", "2021-04-11T17:07:00Z"),
            _row(2, "2021-08-10", "2021-08-10T22:07:00Z", 3, 6),
        ])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["home_score"], 3)

    def test_distinct_games_are_untouched(self):
        """A doubleheader is two game_pks and must survive as two rows."""
        kept = deduplicate([
            _row(10, "2021-05-01", "2021-05-01T17:10:00Z", 1, 0),
            _row(11, "2021-05-01", "2021-05-01T21:40:00Z", 2, 3),
        ])
        self.assertEqual(len(kept), 2)

    def test_a_night_game_crossing_midnight_utc_is_not_discarded(self):
        """The only row present wins even though its dates disagree.

        A 19:10 Pacific start is 02:10 UTC the next day, which is normal and
        must not be treated as the wrong half of a postponement.
        """
        kept = deduplicate([_row(3, "2021-04-16", "2021-04-17T02:10:00Z", 4, 2)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["game_date_utc"], "2021-04-17T02:10:00Z")


def _payload(state, home_score, away_score):
    return {
        "gamePk": 1, "officialDate": "2026-08-05",
        "gameDate": "2026-08-05T23:05:00Z", "season": "2026",
        "status": {"abstractGameState": state},
        "teams": {
            "home": {"team": {"id": 1, "name": "Home Nine"}, "score": home_score},
            "away": {"team": {"id": 2, "name": "Away Nine"}, "score": away_score},
        },
        "venue": {"id": 3313, "name": "Somewhere"},
    }


class UnplayedScoreTests(unittest.TestCase):
    """StatsAPI opens a linescore at 0-0 before a game starts.

    Everything downstream reads `home_score.notna()` as "this game happened",
    so a scheduled game carrying a real-looking nil-nil would be trained on as
    a genuine shutout and would make tonight's card look already played.
    """

    def test_a_scheduled_game_reports_no_score(self):
        row = parse_game(_payload("Preview", 0, 0))
        self.assertIsNone(row["home_score"])
        self.assertIsNone(row["away_score"])
        self.assertIsNone(row["home_win"])

    def test_a_game_under_way_reports_no_score(self):
        row = parse_game(_payload("Live", 2, 1))
        self.assertIsNone(row["home_score"])

    def test_a_final_game_keeps_its_score(self):
        row = parse_game(_payload("Final", 5, 3))
        self.assertEqual(row["home_score"], 5)
        self.assertEqual(row["home_win"], 1)
        self.assertEqual(row["total_runs"], 8)

    def test_a_real_final_shutout_survives(self):
        row = parse_game(_payload("Final", 0, 0))
        self.assertEqual(row["home_score"], 0)
        self.assertEqual(row["total_runs"], 0)


if __name__ == "__main__":
    unittest.main()
