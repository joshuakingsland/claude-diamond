import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import tempfile

import odds_burst
from odds_burst import CREDITS_PER_POLL, MIN_INTERVAL_SECONDS, plan


class PlanTests(unittest.TestCase):
    def test_credits_are_declared_before_anything_is_spent(self):
        shape = plan(minutes=180, every_seconds=90)
        self.assertEqual(shape["polls"], 120)
        self.assertEqual(shape["estimated_credits"], 120 * CREDITS_PER_POLL)

    def test_the_interval_has_a_floor(self):
        # Quote age at capture is 24 seconds median, so polling faster than the
        # floor re-reads the same numbers and pays again.
        shape = plan(minutes=10, every_seconds=1)
        self.assertEqual(shape["interval_seconds"], MIN_INTERVAL_SECONDS)

    def test_a_short_window_still_polls_twice(self):
        self.assertGreaterEqual(plan(minutes=0.1, every_seconds=90)["polls"], 2)


def _event(event_id, commence, price=-120):
    return {
        "id": event_id, "home_team": "Home Team", "away_team": "Away Team",
        "commence_time": commence,
        "bookmakers": [{
            "key": f"book{index}", "title": f"Book {index}",
            "last_update": commence,
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "Home Team", "price": price - index},
                {"name": "Away Team", "price": 100 + index}]}],
        } for index in range(4)],
    }


class BurstRunTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.quotes = str(Path(self.directory) / "quotes")
        self.lines = str(Path(self.directory) / "lines.csv")
        self.credits = str(Path(self.directory) / "credits.csv")
        self.slept = []
        # Polls are >=45s apart in production and snapshot_id is hashed over
        # the fetch timestamp, so the fake clock has to move or every poll
        # collapses onto one stamp and the test measures nothing.
        self.clock = datetime.now(timezone.utc)

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _future(self, hours=3):
        return (datetime.now(timezone.utc) + timedelta(hours=hours)
                ).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _install(self, events, remaining=100000):
        calls = {"n": 0}

        def collect(key, **kwargs):
            calls["n"] += 1
            from odds import paired_book_quotes
            paired = [(event, paired_book_quotes(event)) for event in events]
            # The real shape: one dict per region, with the API's own field
            # names. A stub that invents a shape hides the caller's bug.
            left = remaining - calls["n"] * CREDITS_PER_POLL
            return paired, [{"region": region,
                             "credits_used": CREDITS_PER_POLL // 2,
                             "credits_remaining": left}
                            for region in ("us", "eu")]
        odds_burst.collect_events = collect
        return calls

    def _advance(self, seconds):
        self.slept.append(seconds)
        self.clock += timedelta(seconds=seconds)

    def _run(self, **kwargs):
        original = odds_burst.datetime

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return self.clock

        odds_burst.datetime = Clock
        try:
            return odds_burst.run("key", quotes_dir=self.quotes,
                                  lines_path=self.lines,
                                  credit_log=self.credits, verbose=False,
                                  sleep=self._advance, **kwargs)
        finally:
            odds_burst.datetime = original

    def test_a_burst_writes_one_capture_per_poll(self):
        original = odds_burst.collect_events
        try:
            self._install([_event("e1", self._future())])
            summary = self._run(minutes=10, every_seconds=60, max_credits=None)
            self.assertEqual(summary["polls"], 10)
            import pandas as pd
            frame = pd.concat([pd.read_csv(p) for p in
                               Path(self.quotes).glob("*.csv")])
            # Distinct captures, not one blob: devig.py groups on fetched_at.
            self.assertGreaterEqual(frame["fetched_at"].nunique(), 1)
            self.assertEqual(len(frame), summary["quote_rows"])
        finally:
            odds_burst.collect_events = original

    def test_it_refuses_a_burst_over_the_credit_ceiling(self):
        original = odds_burst.collect_events
        try:
            self._install([_event("e1", self._future())])
            with self.assertRaises(SystemExit) as caught:
                self._run(minutes=600, every_seconds=45, max_credits=100)
            self.assertIn("credits", str(caught.exception))
        finally:
            odds_burst.collect_events = original

    def test_the_credit_floor_stops_a_burst_midway(self):
        # A shared quota can drain mid-burst, so the floor is checked between
        # polls rather than only at the start.
        original = odds_burst.collect_events
        try:
            self._install([_event("e1", self._future())], remaining=30)
            summary = self._run(minutes=60, every_seconds=60, max_credits=None,
                                min_credits=10)
            self.assertLess(summary["polls"], 60)
        finally:
            odds_burst.collect_events = original

    def test_a_started_game_is_never_captured(self):
        original = odds_burst.collect_events
        try:
            self._install([_event("e1", self._future(hours=-1))])
            summary = self._run(minutes=3, every_seconds=60, max_credits=None)
            self.assertEqual(summary["quote_rows"], 0)
        finally:
            odds_burst.collect_events = original

    def test_it_sleeps_between_polls_but_not_after_the_last(self):
        original = odds_burst.collect_events
        try:
            self._install([_event("e1", self._future())])
            summary = self._run(minutes=5, every_seconds=60, max_credits=None)
            self.assertEqual(len(self.slept), summary["polls"] - 1)
        finally:
            odds_burst.collect_events = original


if __name__ == "__main__":
    unittest.main()
