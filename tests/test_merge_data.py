"""Merging two concurrent captures without losing either one's rows.

Concurrency was previously handled by refusing to run two captures at once,
which cost two snapshots outright: GitHub holds only one pending run per
concurrency group and cancels the older one when a third arrives. Allowing the
overlap is only safe if the commit can merge, and only the append-only logs
may be merged — splicing two regenerated snapshots would build a board state
that never existed.
"""

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from merge_data import merge_csv, merge_tree, union_key


def _write(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class KeyTests(unittest.TestCase):
    def test_append_only_logs_are_unioned(self):
        self.assertEqual(union_key("data/paper_ledger.csv"), ("wager_id",))
        self.assertEqual(union_key("data/market_quotes/quotes_2026-08.csv"),
                         ("snapshot_id",))

    def test_regenerated_snapshots_are_replaced(self):
        for path in ("data/lines_upcoming.csv", "docs/index.html",
                     "data/predictions_upcoming.csv"):
            self.assertIsNone(union_key(path), path)


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_neither_run_loses_a_quote(self):
        theirs = self.dir / "theirs.csv"
        ours = self.dir / "ours.csv"
        _write(theirs, ["snapshot_id", "price"],
               [{"snapshot_id": "a", "price": "-110"}])
        _write(ours, ["snapshot_id", "price"],
               [{"snapshot_id": "b", "price": "-120"}])
        (fields, rows), added = merge_csv(ours, theirs, ("snapshot_id",))
        self.assertEqual(added, 1)
        self.assertEqual([r["snapshot_id"] for r in rows], ["a", "b"])

    def test_our_version_wins_a_collision(self):
        """Ours is the newer read: a wager they wrote open may now be settled."""
        theirs = self.dir / "theirs.csv"
        ours = self.dir / "ours.csv"
        _write(theirs, ["wager_id", "outcome"],
               [{"wager_id": "w1", "outcome": ""}])
        _write(ours, ["wager_id", "outcome"],
               [{"wager_id": "w1", "outcome": "win"}])
        (fields, rows), added = merge_csv(ours, theirs, ("wager_id",))
        self.assertEqual(added, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["outcome"], "win")

    def test_a_column_added_by_a_newer_writer_survives(self):
        theirs = self.dir / "theirs.csv"
        ours = self.dir / "ours.csv"
        _write(theirs, ["wager_id"], [{"wager_id": "w1"}])
        _write(ours, ["wager_id", "note"], [{"wager_id": "w2", "note": "x"}])
        (fields, rows), _ = merge_csv(ours, theirs, ("wager_id",))
        self.assertIn("note", fields)

    def test_a_missing_remote_file_is_taken_wholesale(self):
        ours = self.dir / "ours.csv"
        _write(ours, ["wager_id"], [{"wager_id": "w1"}])
        (fields, rows), added = merge_csv(ours, self.dir / "absent.csv",
                                          ("wager_id",))
        self.assertEqual(added, 1)
        self.assertEqual(len(rows), 1)


class TreeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.cwd = Path.cwd()
        self.addCleanup(lambda: __import__("os").chdir(self.cwd))
        __import__("os").chdir(self.root)

    def test_logs_union_while_snapshots_replace(self):
        # The working tree stands in for what the other run already pushed.
        _write("data/paper_ledger.csv", ["wager_id", "stake"],
               [{"wager_id": "theirs", "stake": "1"}])
        _write("data/lines_upcoming.csv", ["event_id"],
               [{"event_id": "old"}])
        ours = self.root / "ours"
        _write(ours / "data/paper_ledger.csv", ["wager_id", "stake"],
               [{"wager_id": "ours", "stake": "1"}])
        _write(ours / "data/lines_upcoming.csv", ["event_id"],
               [{"event_id": "new"}])

        merge_tree(ours, ["data"], verbose=False)

        ledger = _read("data/paper_ledger.csv")
        self.assertEqual({r["wager_id"] for r in ledger}, {"theirs", "ours"})
        board = _read("data/lines_upcoming.csv")
        self.assertEqual([r["event_id"] for r in board], ["new"])

    def test_a_file_only_the_other_run_has_is_left_alone(self):
        _write("data/credit_log.csv", ["fetched_at", "region"],
               [{"fetched_at": "t1", "region": "us"}])
        ours = self.root / "ours"
        _write(ours / "data/paper_ledger.csv", ["wager_id"],
               [{"wager_id": "w"}])
        merge_tree(ours, ["data"], verbose=False)
        self.assertEqual(len(_read("data/credit_log.csv")), 1)


if __name__ == "__main__":
    unittest.main()
