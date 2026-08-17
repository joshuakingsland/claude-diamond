import csv
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from first_inning_open_odds import (
    parse_leads, parse_seasons, pending_calls, run, source_events,
    qualified_close_event_ids, stratified_event_order,
)


FIELDS = [
    "audit_id", "requested_date", "event_id", "home_team", "away_team",
    "commence_time", "requested_snapshot", "returned_snapshot", "market",
    "region", "status", "quote_count", "book_count", "book_keys", "points",
    "odds_credits_used", "discovery_credits_used", "credits_remaining",
    "fetched_at", "error",
]


def _write_close(path):
    rows = [
        {"event_id": "2023-a", "requested_date": "2023-06-01",
         "commence_time": "2023-06-01T20:00:00Z", "home_team": "H1",
         "away_team": "A1"},
        {"event_id": "2024-a", "requested_date": "2024-06-01",
         "commence_time": "2024-06-01T20:00:00Z", "home_team": "H2",
         "away_team": "A2"},
        # A duplicate closing row must not create another paid ladder.
        {"event_id": "2023-a", "requested_date": "2023-06-01",
         "commence_time": "2023-06-01T20:00:00Z", "home_team": "H1",
         "away_team": "A1"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _priced_event(event_id):
    return {
        "id": event_id,
        "commence_time": "2023-06-01T20:00:00Z",
        "home_team": "H1", "away_team": "A1",
        "bookmakers": [{
            "key": "draftkings", "title": "DraftKings",
            "last_update": "2023-06-01T08:00:00Z",
            "markets": [{"key": "totals_1st_1_innings", "outcomes": [
                {"name": "Over", "point": 0.5, "price": -105},
                {"name": "Under", "point": 0.5, "price": -115},
            ]}],
        }],
    }


class FirstInningOpenOddsTests(unittest.TestCase):
    def test_leads_are_unique_valid_and_earliest_first(self):
        self.assertEqual(parse_leads("60,1440,360"), (1440, 360, 60))
        with self.assertRaises(ValueError):
            parse_leads("60,60")
        with self.assertRaises(ValueError):
            parse_leads("1441")

    def test_development_seasons_are_validated_and_sorted(self):
        self.assertEqual(parse_seasons("2024,2023"), (2023, 2024))
        with self.assertRaises(ValueError):
            parse_seasons("2022")
        with self.assertRaises(ValueError):
            parse_seasons("2024,2024")

    def test_paid_confirmation_cannot_open_without_locked_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            close = root / "close.csv"
            _write_close(close)
            with self.assertRaisesRegex(RuntimeError, "confirmation is sealed"):
                run("key", max_calls=1, seasons="2025",
                    close_manifest=close, manifest_path=root / "open.csv",
                    quotes_path=root / "quotes.csv",
                    gate_report=root / "missing.json")

    def test_source_events_deduplicate_and_interleave_seasons(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "close.csv"
            _write_close(path)
            events = source_events(path)
        self.assertEqual(len(events), 2)
        ordered = stratified_event_order(events)
        self.assertEqual([event["id"] for event in ordered],
                         ["2023-a", "2024-a"])

    def test_eligible_cohort_requires_two_books_and_deduplicates_games(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quotes = root / "quotes.csv"
            results = root / "results.csv"
            quote_rows = [
                {"event_id": "a", "market": "totals_1st_1_innings",
                 "point": "0.5", "book_key": "one"},
                {"event_id": "a", "market": "totals_1st_1_innings",
                 "point": "0.5", "book_key": "two"},
                {"event_id": "b", "market": "totals_1st_1_innings",
                 "point": "0.5", "book_key": "one"},
                {"event_id": "b", "market": "totals_1st_1_innings",
                 "point": "0.5", "book_key": "two"},
                {"event_id": "b", "market": "totals_1st_1_innings",
                 "point": "0.5", "book_key": "three"},
                {"event_id": "thin", "market": "totals_1st_1_innings",
                 "point": "0.5", "book_key": "one"},
            ]
            with quotes.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=quote_rows[0])
                writer.writeheader()
                writer.writerows(quote_rows)
            result_rows = [
                {"event_id": "a", "game_pk": "10", "result_status": "final",
                 "game_type": "R"},
                {"event_id": "b", "game_pk": "10", "result_status": "final",
                 "game_type": "R"},
                {"event_id": "thin", "game_pk": "11",
                 "result_status": "final", "game_type": "R"},
            ]
            with results.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=result_rows[0])
                writer.writeheader()
                writer.writerows(result_rows)
            eligible = qualified_close_event_ids(quotes, results)
        self.assertEqual(eligible, {"b"})

    def test_pending_calls_keep_each_event_ladder_together_and_resume(self):
        events = [
            {"id": "a", "requested_date": "2023-06-01",
             "commence_time": "2023-06-01T20:00:00Z"},
            {"id": "b", "requested_date": "2024-06-01",
             "commence_time": "2024-06-01T20:00:00Z"},
        ]
        rows = pending_calls(events, (1440, 60), set())
        self.assertEqual([(row[0]["id"], row[1]) for row in rows],
                         [("a", 1440), ("a", 60), ("b", 1440), ("b", 60)])
        remaining = pending_calls(events, (1440, 60), {rows[0][3]})
        self.assertEqual([(row[0]["id"], row[1]) for row in remaining],
                         [("a", 60), ("b", 1440), ("b", 60)])

    def test_run_reuses_event_ids_without_discovery_and_resumes(self):
        calls = []

        def request(url):
            calls.append(url)
            path = urlparse(url).path
            self.assertIn("/events/", path)
            query = parse_qs(urlparse(url).query)
            event_id = path.split("/")[-2]
            stamp = query["date"][0]
            event = _priced_event(event_id)
            event["commence_time"] = ("2023-06-01T20:00:00Z" if event_id == "2023-a"
                                      else "2024-06-01T20:00:00Z")
            return {"timestamp": stamp, "data": event}, {
                "used": "10", "remaining": "999"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            close = root / "close.csv"
            manifest = root / "open.csv"
            quotes = root / "quotes.csv"
            _write_close(close)
            rows = run("key", max_calls=4, lead_minutes="1440,60",
                       close_manifest=close, manifest_path=manifest,
                       quotes_path=quotes,
                       eligible_event_ids={"2023-a", "2024-a"},
                       request=request)
            self.assertEqual(len(rows), 4)
            self.assertEqual(len(calls), 4)
            self.assertTrue(all("/events?" not in url for url in calls))
            again = run("key", max_calls=4, lead_minutes="1440,60",
                        close_manifest=close, manifest_path=manifest,
                        quotes_path=quotes,
                        eligible_event_ids={"2023-a", "2024-a"},
                        request=request)
            self.assertEqual(again, [])
            self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
