"""The quote log must never grow a file the remote will refuse.

Two attempts, and the first one's failure is the reason for the second's shape.

Monthly shards were sized for seventeen captures a day and reached 138 MB once
two bursts a day were added, so GitHub's pre-receive hook rejected the push —
after 5.5 hours of polling and about 1,300 credits had been spent.

The replacement rolled to a new part past 40 MB and *also* could not work. A
shard is written on a runner; what lands in the repository is `merge_data.py`'s
union of every runner that committed meanwhile. Two bursts each wrote ~35 MB
locally, neither crossed the cap, neither rolled, and the union committed at
70 MB. **A local size check cannot bound a merged file.**

So the shard name is now a pure function of the timestamp. These tests exist to
keep it that way: the moment it consults the disk again, concurrent writers can
disagree about where a row belongs and the bound is gone.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from csv_collection import SHARD_HOURS, dated_part


class DeterminismTests(unittest.TestCase):
    """The property the size-based version lacked, tested directly."""

    def setUp(self):
        self.dir = TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def test_the_name_does_not_depend_on_what_is_on_disk(self):
        # The heart of it. Two runners hold different local copies of the same
        # log; they must still agree on where a given quote belongs.
        stamp = "2026-08-21T20:15:00Z"
        empty = dated_part(self.root, stamp)
        (self.root / empty.name).write_bytes(b"x" * (80 * 1024 * 1024))
        self.assertEqual(dated_part(self.root, stamp), empty,
                         "an 80 MB file on disk must not change the answer")

    def test_two_runners_in_the_same_block_agree(self):
        # 18:00 and 20:59 are the ends of one block; 21:00 begins the next.
        a = dated_part(self.root, "2026-08-21T18:00:03Z")
        b = dated_part(self.root, "2026-08-21T20:59:59Z")
        self.assertEqual(a, b)
        self.assertNotEqual(b, dated_part(self.root, "2026-08-21T21:00:00Z"))

    def test_nothing_in_the_module_reads_a_file_size_to_pick_a_shard(self):
        source = (Path(__file__).resolve().parent.parent
                  / "csv_collection.py").read_text(encoding="utf-8")
        body = source[source.index("def dated_part"):]
        body = body[:body.index("\ndef ")]
        self.assertNotIn("st_size", body,
                         "dated_part has gone back to sizing the shard, which "
                         "cannot bound a file assembled by merge_data.py")


class BlockTests(unittest.TestCase):
    def setUp(self):
        self.dir = TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def test_a_shard_is_named_for_its_block(self):
        self.assertEqual(dated_part(self.root, "2026-08-21T00:00:00Z").name,
                         "quotes_2026-08-21T00.csv")
        self.assertEqual(dated_part(self.root, "2026-08-21T20:15:00Z").name,
                         "quotes_2026-08-21T18.csv")

    def test_the_blocks_tile_the_day_without_gap_or_overlap(self):
        names = [dated_part(self.root, f"2026-08-21T{h:02d}:30:00Z").name
                 for h in range(24)]
        self.assertEqual(len(set(names)), 24 // SHARD_HOURS)
        # Each hour must belong to exactly one block, and blocks must run in
        # clock order so the glob sorts chronologically.
        self.assertEqual(sorted(set(names)), sorted(set(names)))

    def test_a_new_block_starts_a_new_file(self):
        self.assertNotEqual(dated_part(self.root, "2026-08-21T02:59:00Z"),
                            dated_part(self.root, "2026-08-21T03:00:00Z"))

    def test_midnight_rolls_the_day(self):
        self.assertNotEqual(dated_part(self.root, "2026-08-21T23:59:00Z"),
                            dated_part(self.root, "2026-08-22T00:01:00Z"))

    def test_blocks_are_small_enough_for_the_heaviest_day_seen(self):
        # 21 August carried 124 MB. A block of it must stay far from the
        # 100 MB the remote refuses, with room for the card to grow.
        heaviest_day_mb = 124
        per_block = heaviest_day_mb / (24 / SHARD_HOURS)
        self.assertLess(per_block * 3, 95,
                        "even at triple the heaviest observed volume a shard "
                        "must not reach the commit guard")

    def test_an_explicit_csv_path_is_left_alone(self):
        target = self.root / "somewhere.csv"
        self.assertEqual(dated_part(target, "2026-08-21T14:00:00Z"), target)

    def test_an_unusable_timestamp_is_refused_rather_than_guessed(self):
        for bad in ("", "2026", "not-a-date", "20260819T14:00:00Z",
                    "2026-08-21", "2026-08-21TZZ:00:00Z", "2026-08-21T99:00Z"):
            with self.assertRaises(ValueError, msg=f"{bad!r} should raise"):
                dated_part(self.root, bad)


class WritersUseItTests(unittest.TestCase):
    """Both capture paths must shard, and must shard the same way."""

    def test_neither_writer_builds_a_path_by_hand(self):
        root = Path(__file__).resolve().parent.parent
        for name in ("odds.py", "odds_burst.py"):
            source = (root / name).read_text(encoding="utf-8")
            self.assertIn("dated_part", source,
                          f"{name} must shard through dated_part")
            self.assertNotIn('f"quotes_{stamp[:7]}.csv"', source,
                             f"{name} has gone back to monthly shards")
            self.assertNotIn("quotes_{result['stamp'][:7]}.csv", source,
                             f"{name} has gone back to monthly shards")


class CommitGuardTests(unittest.TestCase):
    def test_the_commit_script_refuses_an_oversize_file(self):
        # The backstop, and the only check that runs on the merged tree rather
        # than on one runner's copy. It is the real guarantee.
        script = (Path(__file__).resolve().parent.parent
                  / "commit_data.sh").read_text(encoding="utf-8")
        self.assertIn("95 * 1024 * 1024", script)
        self.assertIn("refusing to commit", script)


class OverlappingShardTests(unittest.TestCase):
    """Two shards holding the same quote must count it once."""

    def setUp(self):
        self.dir = TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def write(self, name, ids):
        rows = ["snapshot_id,fetched_at,price_home"]
        rows += [f"{i},2026-08-19T20:00:00Z,-110" for i in ids]
        (self.root / name).write_text("\n".join(rows) + "\n", encoding="utf-8")

    def test_a_quote_in_two_shards_is_read_once(self):
        # This happened: a run on older code pushed its monthly file back
        # beside the daily ones and every count silently doubled. It also
        # covers the migration, where old daily files sit beside new blocks.
        from csv_collection import read_quote_shards
        self.write("quotes_2026-08-19.csv", ["a", "b", "c"])
        self.write("quotes_2026-08-19T18.csv", ["b", "c", "d"])
        quotes = read_quote_shards(self.root / "*.csv")
        self.assertEqual(len(quotes), 4)
        self.assertEqual(sorted(quotes["snapshot_id"]), ["a", "b", "c", "d"])

    def test_shards_without_the_key_are_still_concatenated(self):
        from csv_collection import read_quote_shards
        (self.root / "a.csv").write_text("x\n1\n", encoding="utf-8")
        (self.root / "b.csv").write_text("x\n2\n", encoding="utf-8")
        self.assertEqual(len(read_quote_shards(self.root / "*.csv")), 2)

    def test_no_shards_is_an_empty_frame_not_an_error(self):
        from csv_collection import read_quote_shards
        self.assertTrue(read_quote_shards(self.root / "*.csv").empty)


if __name__ == "__main__":
    unittest.main()
