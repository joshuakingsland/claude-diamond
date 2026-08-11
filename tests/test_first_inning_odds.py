import csv
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from first_inning_odds import (audit_snapshot, events_on_day, run,
                                response_events)


EVENT = {
    "id": "event-1",
    "commence_time": "2026-08-10T23:10:00Z",
    "home_team": "Home Nine",
    "away_team": "Away Nine",
}


def _priced_event():
    return {**EVENT, "bookmakers": [{
        "key": "draftkings", "title": "DraftKings",
        "last_update": "2026-08-10T22:58:00Z",
        "markets": [{"key": "totals_1st_1_innings", "outcomes": [
            {"name": "Over", "point": 0.5, "price": -105},
            {"name": "Under", "point": 0.5, "price": -115},
        ]}],
    }]}


class FirstInningOddsTests(unittest.TestCase):
    def test_event_odds_response_shape_is_normalized(self):
        self.assertEqual(response_events({"data": EVENT}), [EVENT])
        self.assertEqual(response_events({"data": [EVENT]}), [EVENT])

    def test_only_requested_day_is_selected_and_snapshot_is_pregame(self):
        wrong_day = {**EVENT, "id": "event-2",
                     "commence_time": "2026-08-11T00:10:00Z"}
        events = events_on_day({"data": [wrong_day, EVENT]}, "2026-08-10")
        self.assertEqual([event["id"] for event in events], ["event-1"])
        self.assertEqual(audit_snapshot(EVENT["commence_time"], 10).isoformat(),
                         "2026-08-10T23:00:00+00:00")

    def test_run_records_offer_and_resumes_without_rebuying(self):
        calls = []

        def fake_request(url):
            calls.append(url)
            query = parse_qs(urlparse(url).query)
            if urlparse(url).path.endswith("/events"):
                return {"data": [EVENT]}, {"used": "1", "remaining": "999"}
            self.assertEqual(query["markets"], ["totals_1st_1_innings"])
            return {"timestamp": "2026-08-10T23:00:00Z", "data": _priced_event()}, {
                "used": "10", "remaining": "989"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, quotes = root / "audit.csv", root / "quotes.csv"
            rows = run("key", "2026-08-10", max_events=1, manifest_path=manifest,
                       quotes_path=quotes, request=fake_request)
            self.assertEqual(rows[0]["status"], "offered")
            self.assertEqual(rows[0]["book_count"], 1)
            with quotes.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            self.assertEqual(len(calls), 2)

            run("key", "2026-08-10", max_events=1, manifest_path=manifest,
                quotes_path=quotes, request=fake_request)
            # A repeat needs the cheap event discovery but does not buy the
            # expensive event-odds snapshot again.
            self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
