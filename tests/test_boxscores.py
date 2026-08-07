"""Repairing a pitching log that was damaged from outside.

This file is appended to over a long run, which makes it easy to corrupt: a
`git stash` of it mid-run left the writer appending at a stale offset, costing
4,479 games and duplicating 1,552 lines. Duplicates are the dangerous half,
because `features.py` folds every line it is given — a repeated start counts a
pitcher's strikeouts and home runs twice and produces a rate that looks
entirely reasonable.
"""

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from boxscores import PITCHER_FIELDS, deduplicate


def _write(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PITCHER_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _line(game_pk, player_id, strike_outs=5):
    return {"game_pk": game_pk, "official_date": "2025-05-01", "team_id": 147,
            "player_id": player_id, "is_starter": 1, "outs": 18,
            "batters_faced": 24, "hits": 5, "runs": 2, "earned_runs": 2,
            "home_runs": 1, "walks": 2, "hit_batsmen": 0,
            "strike_outs": strike_outs, "pitches": 90, "strikes": 60}


class DeduplicateTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.path = self.dir / "pitching.csv"

    def test_a_repeated_start_is_removed(self):
        _write(self.path, [_line(1, 900), _line(1, 900), _line(1, 901)])
        removed = deduplicate(self.path, verbose=False)
        self.assertEqual(removed, 1)
        rows = _read(self.path)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["player_id"] for r in rows}, {"900", "901"})

    def test_the_same_pitcher_in_different_games_is_kept(self):
        _write(self.path, [_line(1, 900), _line(2, 900)])
        self.assertEqual(deduplicate(self.path, verbose=False), 0)
        self.assertEqual(len(_read(self.path)), 2)

    def test_a_clean_file_is_left_untouched(self):
        _write(self.path, [_line(1, 900), _line(1, 901), _line(2, 900)])
        before = self.path.read_bytes()
        self.assertEqual(deduplicate(self.path, verbose=False), 0)
        self.assertEqual(self.path.read_bytes(), before)

    def test_the_first_copy_wins_so_a_partial_rewrite_cannot_win(self):
        _write(self.path, [_line(1, 900, strike_outs=7),
                           _line(1, 900, strike_outs=99)])
        deduplicate(self.path, verbose=False)
        self.assertEqual(_read(self.path)[0]["strike_outs"], "7")

    def test_a_missing_file_is_not_an_error(self):
        self.assertEqual(deduplicate(self.dir / "absent.csv", verbose=False), 0)


if __name__ == "__main__":
    unittest.main()
