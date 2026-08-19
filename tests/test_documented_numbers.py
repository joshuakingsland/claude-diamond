"""The README must not drift away from the reports it quotes.

This repository's stated discipline is that a number disagreeing with the repo
means the document is stale rather than right. That was enforced for the public
page — `model_card.py` recomputes nothing — and not at all for the README, which
is where the findings are actually argued. Within two days of writing the
line-shopping section its headline had moved from +0.312 to +0.250 and every
sentence around it still said +0.312.

The table is now generated — `line_shopping.py --sync-readme` rewrites those
rows from the report, and `revalidate.yml` runs it weekly — so the exact checks
below hold without anyone transcribing anything. The prose is deliberately not
generated: a changed conclusion needs a person to write it. So the prose checks
are loose enough to survive ordinary accumulation and fail only when a sentence
has stopped describing the data. Widening a tolerance to get green is the one
repair that is never right here.

It skips when the report is absent so a fresh clone without the study still
runs green.
"""

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "line_shopping.json"
README = ROOT / "README.md"
# Rounding in the table is to three decimals; allow the last digit to differ so
# a re-run that changes nothing material does not fail.
TOLERANCE = 0.002

ROW = re.compile(
    r"\|\s*(?P<threshold>[\d.]+) pt\s*\|\s*(?P<bets>\d+)\s*\|\s*(?P<dates>\d+)\s*\|"
    r"\s*\*\*(?P<panel>[+-][\d.]+)\*\*\s*\[(?P<plo>[+-][\d.]+),\s*(?P<phi>[+-][\d.]+)\]\s*\|"
    r"\s*\*\*(?P<sharp>[+-][\d.]+)\*\*\s*\[(?P<slo>[+-][\d.]+),\s*(?P<shi>[+-][\d.]+)\]\s*\|")


class DocumentedNumberTests(unittest.TestCase):
    def setUp(self):
        if not REPORT.exists():
            self.skipTest("line_shopping.json not present")
        self.report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.rows = list(ROW.finditer(README.read_text(encoding="utf-8")))

    def test_the_readme_still_carries_the_table(self):
        self.assertTrue(self.rows,
                        "the line-shopping table has gone from the README")

    def test_every_quoted_row_matches_the_report(self):
        sweep, sharp = (self.report["threshold_sweep"],
                        self.report["threshold_sweep_vs_sharp"])
        for row in self.rows:
            key = f"{float(row['threshold']) / 100:g}"
            self.assertIn(key, sweep, f"README quotes a {row['threshold']}pt "
                                      f"arm the report does not have")
            panel, edge = sweep[key], sharp[key]
            where = f"{row['threshold']}pt row"
            self.assertEqual(int(row["bets"]), panel["picks"],
                             f"{where}: bet count is stale")
            self.assertEqual(int(row["dates"]), panel["dates"],
                             f"{where}: date count is stale")
            for name, quoted, actual in (
                    ("panel CLV", row["panel"],
                     panel["mean_clv_probability_points"]),
                    ("sharp CLV", row["sharp"],
                     edge["mean_clv_probability_points"]),
                    ("panel low", row["plo"],
                     panel["ci90_date_clustered_points"][0]),
                    ("panel high", row["phi"],
                     panel["ci90_date_clustered_points"][1]),
                    ("sharp low", row["slo"],
                     edge["ci90_date_clustered_points"][0]),
                    ("sharp high", row["shi"],
                     edge["ci90_date_clustered_points"][1])):
                self.assertAlmostEqual(
                    float(quoted), actual, delta=TOLERANCE,
                    msg=f"{where}: {name} says {quoted}, report says {actual}")

    def test_the_headline_verdict_is_not_materially_stale(self):
        # The prose is written by a person and is deliberately not generated,
        # so this tolerates the drift of ordinary accumulation and fails only
        # when the sentence has stopped describing the data. A tight tolerance
        # here would turn every weekly re-run into a red build, which teaches
        # people to ignore the check rather than to fix the claim.
        text = README.read_text(encoding="utf-8")
        sharp = self.report["threshold_sweep_vs_sharp"]["0.0025"]
        claimed = re.search(
            r"\*\*([+-][\d.]+) points of CLV against Pinnacle's close\*\*", text)
        self.assertIsNotNone(claimed, "the verdict table no longer states a "
                                      "CLV figure against Pinnacle")
        self.assertAlmostEqual(
            float(claimed.group(1)), sharp["mean_clv_probability_points"],
            delta=0.10,
            msg="the verdict table's headline CLV no longer describes the "
                "report; rewrite the sentence rather than widening this")

    def test_the_readme_does_not_claim_more_than_the_interval_supports(self):
        # The claim that matters is not the digits but the conclusion. If the
        # interval has crossed zero, no sentence may still call this an edge.
        low = self.report["threshold_sweep_vs_sharp"]["0.0025"][
            "ci90_date_clustered_points"][0]
        text = README.read_text(encoding="utf-8")
        asserts_edge = "Yes, but not from the model." in text
        if low <= 0:
            self.assertFalse(
                asserts_edge,
                "the 0.25pt interval now includes zero, so the verdict table "
                "must stop answering 'yes' to whether an edge exists")


if __name__ == "__main__":
    unittest.main()
