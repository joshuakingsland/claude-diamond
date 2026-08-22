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
import os
import pathlib
import unittest

import merge_data
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
        self.assertEqual(
            union_key("data/full_game_event_quotes/quotes_2024.csv"),
            ("snapshot_id",))
        self.assertEqual(
            union_key("data/full_game_event_audit.csv"), ("audit_id",))

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

    def test_stale_full_game_audit_cannot_delete_remote_rows(self):
        """A live capture started earlier must retain newer backfill attempts."""
        theirs = self.dir / "theirs.csv"
        ours = self.dir / "ours.csv"
        _write(theirs, ["audit_id", "status"],
               [{"audit_id": "old", "status": "offered"},
                {"audit_id": "new-checkpoint", "status": "offered"}])
        _write(ours, ["audit_id", "status"],
               [{"audit_id": "old", "status": "offered"}])
        (fields, rows), added = merge_csv(ours, theirs, ("audit_id",))
        self.assertEqual(added, 0)
        self.assertEqual({row["audit_id"] for row in rows},
                         {"old", "new-checkpoint"})

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


class NoOpMergeTests(unittest.TestCase):
    """A merge that changes nothing must not touch the file.

    Every burst commit rewrote all nineteen quote shards -- about 318 MB --
    because the write was unconditional. Git stored a fresh blob for each and
    the repository grew roughly 50 MB a day for data nobody had changed.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = pathlib.Path(self.dir.name)
        self.cwd = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.cwd)
        (self.root / "data" / "market_quotes").mkdir(parents=True)
        (self.root / "ours" / "data" / "market_quotes").mkdir(parents=True)

    def write(self, path, ids):
        rows = ["snapshot_id,fetched_at,price_home"]
        rows += [f"{i},2026-08-21T20:00:00Z,-110" for i in ids]
        pathlib.Path(path).write_text("\r\n".join(rows) + "\r\n",
                                      encoding="utf-8", newline="")

    def test_identical_content_leaves_the_file_untouched(self):
        dest = "data/market_quotes/quotes_2026-08-21T18.csv"
        ours = "ours/" + dest
        self.write(dest, ["a", "b", "c"])
        self.write(ours, ["a", "b", "c"])
        before = os.stat(dest).st_mtime_ns
        merge_data.merge_tree("ours", ["data"], verbose=False)
        self.assertEqual(os.stat(dest).st_mtime_ns, before,
                         "a no-op merge rewrote the file")

    def test_new_rows_still_get_written(self):
        dest = "data/market_quotes/quotes_2026-08-21T18.csv"
        self.write(dest, ["a", "b"])
        self.write("ours/" + dest, ["b", "c"])
        merge_data.merge_tree("ours", ["data"], verbose=False)
        text = pathlib.Path(dest).read_text(encoding="utf-8")
        for marker in ("a", "b", "c"):
            self.assertIn(f"{marker},2026-08-21", text)

    def test_a_second_identical_merge_is_also_a_no_op(self):
        # The first merge may legitimately reformat. The second must not.
        dest = "data/market_quotes/quotes_2026-08-21T18.csv"
        self.write(dest, ["a"])
        self.write("ours/" + dest, ["a", "b"])
        merge_data.merge_tree("ours", ["data"], verbose=False)
        settled = os.stat(dest).st_mtime_ns
        merge_data.merge_tree("ours", ["data"], verbose=False)
        self.assertEqual(os.stat(dest).st_mtime_ns, settled)


if __name__ == "__main__":
    unittest.main()
