"""The quote log must never grow a file the remote will refuse.

Found the expensive way. Monthly shards were fine at seventeen captures a day
and became 138 MB once two bursts a day were added, so GitHub's pre-receive
hook rejected the push — after 5.5 hours of polling and about 1,300 credits had
already been spent, and after `commit_data.sh` had retried the merge four
times. The capture cannot be re-bought at the live price.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from csv_collection import MAX_SHARD_BYTES, dated_part


class DatedPartTests(unittest.TestCase):
    def setUp(self):
        self.dir = TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def fill(self, path, size):
        path.write_bytes(b"x" * size)

    def test_a_shard_is_named_for_the_day_not_the_month(self):
        # The whole defect in one assertion: a month of bursts in one file.
        shard = dated_part(self.root, "2026-08-19T21:00:00Z")
        self.assertEqual(shard.name, "quotes_2026-08-19.csv")

    def test_two_captures_in_one_day_share_a_shard(self):
        first = dated_part(self.root, "2026-08-19T14:00:00Z")
        self.fill(first, 1024)
        self.assertEqual(dated_part(self.root, "2026-08-19T21:00:00Z"), first)

    def test_different_days_do_not(self):
        self.assertNotEqual(dated_part(self.root, "2026-08-19T23:59:00Z"),
                            dated_part(self.root, "2026-08-20T00:01:00Z"))

    def test_a_full_shard_rolls_to_a_new_part(self):
        first = dated_part(self.root, "2026-08-19T14:00:00Z")
        self.fill(first, MAX_SHARD_BYTES)
        second = dated_part(self.root, "2026-08-19T15:00:00Z")
        self.assertEqual(second.name, "quotes_2026-08-19.p2.csv")
        self.fill(second, MAX_SHARD_BYTES)
        self.assertEqual(dated_part(self.root, "2026-08-19T16:00:00Z").name,
                         "quotes_2026-08-19.p3.csv")

    def test_no_shard_can_reach_the_size_the_remote_refuses(self):
        # 40 MB rolled, 100 MB rejected: the margin is the point.
        self.assertLess(MAX_SHARD_BYTES, 100 * 1024 * 1024 / 2)

    def test_a_shard_still_below_the_cap_is_reused(self):
        first = dated_part(self.root, "2026-08-19T14:00:00Z")
        self.fill(first, MAX_SHARD_BYTES - 1)
        self.assertEqual(dated_part(self.root, "2026-08-19T15:00:00Z"), first)

    def test_the_name_depends_only_on_the_day_and_what_is_on_disk(self):
        # Two runs appending in the same second must land in the same file, or
        # merge_data.py cannot union them.
        stamp = "2026-08-19T14:00:00Z"
        self.assertEqual(dated_part(self.root, stamp),
                         dated_part(self.root, stamp))

    def test_an_explicit_csv_path_is_left_alone(self):
        target = self.root / "somewhere.csv"
        self.assertEqual(dated_part(target, "2026-08-19T14:00:00Z"), target)

    def test_an_unusable_timestamp_is_refused_rather_than_guessed(self):
        for bad in ("", "2026", "not-a-date", "20260819T14:00:00Z"):
            with self.assertRaises(ValueError):
                dated_part(self.root, bad)


class WritersUseItTests(unittest.TestCase):
    """Both capture paths must shard, and must shard the same way."""

    def test_neither_writer_builds_a_monthly_path_by_hand(self):
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
        # The backstop for whatever the next unforeseen growth is. Without it
        # the failure surfaces at the remote, after the data is paid for.
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
        # This actually happened: a run on older code pushed its monthly file
        # back beside the new daily ones and every count silently doubled.
        from csv_collection import read_quote_shards
        self.write("quotes_2026-08.csv", ["a", "b", "c"])
        self.write("quotes_2026-08-19.csv", ["b", "c", "d"])
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
