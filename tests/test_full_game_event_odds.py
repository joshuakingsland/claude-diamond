import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from full_game_event_odds import FIELDS, _catalog_events, run


def _close_row(event_id="e1"):
    return {
        "audit_id": f"{event_id}|close|2024-05-01T19:50:00Z|us",
        "event_id": event_id,
        "home_team": "Home Nine",
        "away_team": "Away Nine",
        "commence_time": "2024-05-01T20:10:00Z",
        "snapshot_role": "close",
        "requested_snapshot": "2024-05-01T19:50:00Z",
        "returned_snapshot": "2024-05-01T19:50:00Z",
        "status": "offered",
        "quote_count": "2",
        "odds_credits_used": "30",
        "discovery_credits_used": "1",
        "credits_remaining": "1000",
        "error": "",
    }


class FullGameEventOddsTests(unittest.TestCase):
    def test_close_manifest_is_a_zero_credit_early_catalog(self):
        catalog = _catalog_events([_close_row()], "2024-05-01", "2024-05-01")
        self.assertEqual([event["id"] for event in catalog], ["e1"])

    def test_early_run_calls_only_event_odds_not_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "audit.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(_close_row())
            payload = {
                "timestamp": "2024-04-30T20:10:00Z",
                "data": {"id": "e1", "home_team": "Home Nine",
                         "away_team": "Away Nine",
                         "commence_time": "2024-05-01T20:10:00Z",
                         "bookmakers": []},
            }
            with patch("full_game_event_odds._request",
                       return_value=(payload, {"used": "30",
                                               "remaining": "970"})) as request:
                rows = run("key", "2024-05-01", "2024-05-01", 1,
                           lead_minutes=1440, role="early",
                           manifest=manifest, quotes=root / "quotes",
                           dry_run=False)
            self.assertEqual(request.call_count, 1)
            self.assertIn("/e1/odds?", request.call_args.args[0])
            self.assertEqual(rows[0]["discovery_credits_used"], 0)
            self.assertEqual(rows[0]["status"], "no_offer")


if __name__ == "__main__":
    unittest.main()
