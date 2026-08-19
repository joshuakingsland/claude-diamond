"""A cheap run must not be able to delete what an expensive one produced.

Found by tripping it. Running `stationarity.py` and `mean_calibration.py`
without their optional flags during an audit replaced two complete reports with
partial ones — silently, exit code zero, 212 and 85 lines of results gone. It
is the same shape as the stale `--seasons` argument that once deleted a whole
season from the results table, and it gets the same answer: a pass that did not
run cannot delete what an earlier one wrote.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from provenance import merge_report


class MergeReportTests(unittest.TestCase):
    def setUp(self):
        self.dir = TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "report.json"

    def write(self, document):
        self.path.write_text(json.dumps(document), encoding="utf-8")

    def read(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_a_block_this_run_did_not_compute_survives(self):
        self.write({"drift": {"old": 1}, "walk_forward": {"result": 42}})
        merge_report(self.path, {"drift": {"new": 2}})
        after = self.read()
        self.assertEqual(after["drift"], {"new": 2})
        self.assertEqual(after["walk_forward"]["result"], 42)

    def test_a_carried_block_is_labelled_not_passed_off_as_fresh(self):
        self.write({"walk_forward": {"result": 42}})
        merge_report(self.path, {"drift": {"new": 2}})
        after = self.read()
        self.assertIn("carried_over", after["walk_forward"])
        self.assertEqual(after["carried_over_blocks"], ["walk_forward"])

    def test_a_recomputed_block_is_replaced_outright(self):
        # Merging must not resurrect keys inside a block the run did compute;
        # a stale sub-key masquerading as current is the failure being avoided.
        self.write({"walk_forward": {"stale": 1, "result": 42}})
        merge_report(self.path, {"walk_forward": {"result": 43}})
        after = self.read()
        self.assertEqual(after["walk_forward"], {"result": 43})
        self.assertNotIn("carried_over_blocks", after)

    def test_a_full_run_leaves_no_carried_marker(self):
        self.write({"drift": {"a": 1}, "walk_forward": {"b": 2}})
        merge_report(self.path, {"drift": {"a": 9}, "walk_forward": {"b": 9}})
        self.assertNotIn("carried_over_blocks", self.read())

    def test_a_missing_file_is_simply_written(self):
        merge_report(self.path, {"drift": {"a": 1}})
        self.assertEqual(self.read(), {"drift": {"a": 1}})

    def test_corrupt_json_on_disk_does_not_stop_the_write(self):
        # Better to lose an unreadable report than to fail the run that would
        # have replaced it.
        self.path.write_text("{not json", encoding="utf-8")
        merge_report(self.path, {"drift": {"a": 1}})
        self.assertEqual(self.read(), {"drift": {"a": 1}})

    def test_a_non_dict_block_is_carried_without_a_marker(self):
        self.write({"notes": ["a", "b"]})
        merge_report(self.path, {"drift": {}})
        self.assertEqual(self.read()["notes"], ["a", "b"])


class ReportWritersUseItTests(unittest.TestCase):
    """The two scripts that lost blocks must not go back to overwriting."""

    def test_the_scripts_that_were_bitten_call_merge_report(self):
        for name in ("stationarity.py", "mean_calibration.py"):
            source = (Path(__file__).resolve().parent.parent / name).read_text(
                encoding="utf-8")
            self.assertIn("merge_report(args.report", source,
                          f"{name} must merge its report, not overwrite it")
            self.assertNotIn("Path(args.report).write_text", source,
                             f"{name} has gone back to overwriting")


if __name__ == "__main__":
    unittest.main()
