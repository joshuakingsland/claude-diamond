import csv
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from first_inning_odds import (audit_snapshot, events_on_day, run, run_study,
                                response_events, stratified_days)


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
                     "commence_time": "2026-08-11T18:10:00Z"}
        events = events_on_day({"data": [wrong_day, EVENT]}, "2026-08-10")
        self.assertEqual([event["id"] for event in events], ["event-1"])
        self.assertEqual(audit_snapshot(EVENT["commence_time"], 10).isoformat(),
                         "2026-08-10T23:00:00+00:00")

    def test_late_pacific_game_stays_on_its_mlb_calendar_day(self):
        late = {**EVENT, "commence_time": "2026-08-11T05:10:00Z"}
        events = events_on_day({"data": [late]}, "2026-08-10")
        self.assertEqual(events, [late])

    def test_study_dates_are_deterministic_and_stratified_by_season(self):
        dates = stratified_days("2023-05-03", "2024-09-30", 3)
        self.assertEqual(dates, ["2023-05-03", "2023-07-17", "2023-09-30",
                                 "2024-05-03", "2024-07-17", "2024-09-30"])

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

    def test_study_rotates_across_dates_before_taking_a_second_game(self):
        first = {**EVENT, "id": "first", "commence_time": "2026-05-03T17:00:00Z"}
        second = {**EVENT, "id": "second", "commence_time": "2026-05-03T20:00:00Z"}
        other = {**EVENT, "id": "other", "commence_time": "2026-09-30T20:00:00Z"}
        event_by_id = {"first": first, "second": second, "other": other}
        calls = []

        def fake_request(url):
            calls.append(url)
            path = urlparse(url).path
            query = parse_qs(urlparse(url).query)
            if path.endswith("/events"):
                looked_up = query["date"][0][:10]
                data = [first, second] if looked_up == "2026-05-03" else [other]
                return {"data": data}, {"used": "1", "remaining": "999"}
            event_id = path.split("/")[-2]
            return {"timestamp": "2026-05-03T16:50:00Z",
                    "data": {**_priced_event(), **event_by_id[event_id]}}, {
                        "used": "10", "remaining": "989"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = run_study("key", "2026-05-03", "2026-09-30", max_events=3,
                             days_per_season=2, manifest_path=root / "audit.csv",
                             quotes_path=root / "quotes.csv", request=fake_request)
        self.assertEqual([row["event_id"] for row in rows],
                         ["first", "other", "second"])

    def test_provider_failure_is_recorded_without_discarding_the_study(self):
        def fake_request(url):
            if urlparse(url).path.endswith("/events"):
                return {"data": [EVENT]}, {"used": "1", "remaining": "999"}
            raise OSError("historical event unavailable")

        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "audit.csv"
            rows = run_study("key", "2026-08-10", "2026-08-10", max_events=1,
                             manifest_path=manifest, quotes_path=Path(directory) / "q.csv",
                             request=fake_request)
            self.assertEqual(rows[0]["status"], "failed")
            with manifest.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(list(csv.DictReader(handle))[0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
