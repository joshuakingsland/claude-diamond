import unittest

import pandas as pd

from lineup_snapshots import confirmed_games, parse


class LineupTests(unittest.TestCase):
    def _payload(self, count=9):
        home = list(range(1, count + 1))
        away = list(range(11, 11 + count))
        return {
            "gameData": {
                "game": {"pk": 7},
                "datetime": {"dateTime": "2026-08-11T23:00:00Z"},
                "status": {"detailedState": "Scheduled"},
                "players": {},
            },
            "liveData": {"boxscore": {"teams": {
                "home": {"battingOrder": home},
                "away": {"battingOrder": away},
            }}},
        }

    def test_both_complete_orders_are_required(self):
        self.assertIsNotNone(parse(self._payload(),
                                   "2026-08-11T18:00:00Z"))
        self.assertIsNone(parse(self._payload(8),
                                "2026-08-11T18:00:00Z"))

    def test_confirmation_is_point_in_time(self):
        frame = pd.DataFrame([{
            "game_pk": 7, "confirmed": 1,
            "captured_at": "2026-08-11T19:00:00Z",
        }])
        self.assertEqual(confirmed_games(
            frame, as_of="2026-08-11T18:00:00Z"), set())
        self.assertEqual(confirmed_games(
            frame, as_of="2026-08-11T20:00:00Z"), {"7"})


if __name__ == "__main__":
    unittest.main()
