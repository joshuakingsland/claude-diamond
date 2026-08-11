import unittest

import pandas as pd

from schedule_snapshots import apply_probable_snapshots, snapshot_row


class SnapshotTests(unittest.TestCase):
    def _raw(self):
        return {
            "gamePk": 7, "officialDate": "2026-08-11",
            "gameDate": "2026-08-11T23:00:00Z",
            "status": {"detailedState": "Scheduled"},
            "teams": {
                "home": {"team": {"id": 1},
                         "probablePitcher": {"id": 10, "fullName": "H One"}},
                "away": {"team": {"id": 2},
                         "probablePitcher": {"id": 20, "fullName": "A One"}},
            },
        }

    def test_snapshot_has_stable_provenance(self):
        row = snapshot_row(self._raw(), "2026-08-11T18:00:00Z")
        self.assertEqual(row["home_sp_id"], 10)
        self.assertEqual(len(row["snapshot_id"]), 20)
        self.assertEqual(row, snapshot_row(self._raw(),
                                           "2026-08-11T18:00:00Z"))

    def test_latest_snapshot_before_decision_wins(self):
        games = pd.DataFrame([{
            "game_pk": 7, "home_score": None,
            "home_sp_id": 99, "home_sp_name": "latest result-table answer",
            "away_sp_id": 98, "away_sp_name": "latest result-table answer",
        }])
        snapshots = pd.DataFrame([
            {"game_pk": 7, "captured_at": "2026-08-11T17:00:00Z",
             "home_sp_id": 10, "home_sp_name": "H One",
             "away_sp_id": 20, "away_sp_name": "A One"},
            {"game_pk": 7, "captured_at": "2026-08-11T19:00:00Z",
             "home_sp_id": 11, "home_sp_name": "H Scratch",
             "away_sp_id": 21, "away_sp_name": "A Scratch"},
        ])
        out = apply_probable_snapshots(
            games, snapshots, as_of="2026-08-11T18:00:00Z")
        self.assertEqual(out.iloc[0]["home_sp_id"], 10)
        self.assertEqual(out.iloc[0]["away_sp_id"], 20)

    def test_final_games_are_never_rewritten(self):
        games = pd.DataFrame([{
            "game_pk": 7, "home_score": 4,
            "home_sp_id": 99, "away_sp_id": 98,
        }])
        snapshots = pd.DataFrame([{
            "game_pk": 7, "captured_at": "2026-08-11T17:00:00Z",
            "home_sp_id": 10, "away_sp_id": 20,
        }])
        out = apply_probable_snapshots(games, snapshots,
                                       as_of="2026-08-11T18:00:00Z")
        self.assertEqual(out.iloc[0]["home_sp_id"], 99)


if __name__ == "__main__":
    unittest.main()
