"""What an alerting path has to get right before it is allowed to interrupt.

An alert that fires on a price nobody can take is worse than no alert, because
it trains its reader to ignore the next one. These cover the ways this one
could cry wolf: a panel that grows, a repeat of a price already sent, a quote
from outside the window, and a label that names the wrong side of a total.
"""

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import alerts
from config import MAX_LOCK_LEAD_MINUTES, MIN_LOCK_LEAD_MINUTES

COMMENCE = datetime(2026, 8, 5, 22, 0, tzinfo=timezone.utc)


def stamp(minutes_before):
    return (COMMENCE - timedelta(minutes=minutes_before)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def quote(book, home, away, lead=120, market="h2h", point=np.nan, event="g1"):
    return {"event_id": event, "market": market, "point": point,
            "book_key": book, "fetched_at": stamp(lead), "priced": 1,
            "commence_time": COMMENCE.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home_team": "Chicago Cubs", "away_team": "St. Louis Cardinals",
            "price_home": home, "price_away": away}


def board(outlier=(120.0, -160.0), lead=120, market="h2h", point=np.nan,
          books=11):
    """A panel in agreement except for one book well off the consensus."""
    rows = [quote(f"book{i}", -110.0, -110.0, lead, market, point)
            for i in range(books - 1)]
    rows.append(quote("outlier", *outlier, lead=lead, market=market,
                      point=point))
    return pd.DataFrame(rows)


class DetectionTests(unittest.TestCase):
    def test_the_outlier_is_found_and_named_by_its_book(self):
        found = alerts.current(board(), threshold=0.005)
        self.assertEqual(len(found), 1)
        row = found.iloc[0]
        self.assertEqual(row["book_key"], "outlier")
        self.assertEqual(float(row["price"]), 120.0)
        self.assertEqual(row["side"], "home")

    def test_an_agreeing_market_raises_nothing(self):
        rows = [quote(f"book{i}", -110.0, -110.0) for i in range(11)]
        self.assertEqual(len(alerts.current(pd.DataFrame(rows),
                                            threshold=0.005)), 0)

    def test_a_quote_outside_the_lock_window_is_not_alerted(self):
        # Both directions: too early to bet, and too late to reach the book.
        for lead in (MAX_LOCK_LEAD_MINUTES + 30, MIN_LOCK_LEAD_MINUTES - 5):
            found = alerts.current(board(lead=lead), threshold=0.005)
            self.assertEqual(len(found), 0, f"lead {lead} should not alert")

    def test_a_supplied_panel_is_not_widened_by_one_capture(self):
        # The bias this guards against: inside a single poll every book has
        # 100% coverage, so a panel derived here would admit a book that is
        # almost never up — and the best of N prices rises with N.
        frame = board()
        panel = [f"book{i}" for i in range(10)]
        self.assertEqual(len(alerts.current(frame, threshold=0.005,
                                            books=panel)), 0)
        self.assertEqual(len(alerts.current(frame, threshold=0.005,
                                            books=panel + ["outlier"])), 1)

    def test_a_thin_panel_refuses_to_alert(self):
        rows = [quote(f"book{i}", -110.0, -110.0) for i in range(3)]
        rows.append(quote("outlier", 120.0, -160.0))
        self.assertEqual(len(alerts.current(pd.DataFrame(rows),
                                            threshold=0.005)), 0)


class LabelTests(unittest.TestCase):
    """"home" on a total means Over. An alert must never print the raw side."""

    def test_a_total_is_named_over_and_under(self):
        over = alerts.current(board(market="totals", point=8.5),
                              threshold=0.005).iloc[0]
        self.assertEqual(over["selection"], "Over 8.5")
        under = alerts.current(board(outlier=(-160.0, 120.0), market="totals",
                                     point=8.5), threshold=0.005).iloc[0]
        self.assertEqual(under["selection"], "Under 8.5")

    def test_a_run_line_carries_the_point_from_the_backed_side(self):
        home = alerts.current(board(market="spreads", point=-1.5),
                              threshold=0.005).iloc[0]
        self.assertEqual(home["selection"], "Chicago Cubs -1.5")
        away = alerts.current(board(outlier=(-160.0, 120.0), market="spreads",
                                    point=-1.5), threshold=0.005).iloc[0]
        self.assertEqual(away["selection"], "St. Louis Cardinals +1.5")

    def test_a_moneyline_names_the_team(self):
        found = alerts.current(board(), threshold=0.005).iloc[0]
        self.assertEqual(found["selection"], "Chicago Cubs ML")


class LogTests(unittest.TestCase):
    """The log is append-only and is the forward test. It must not repeat."""

    def setUp(self):
        self.dir = TemporaryDirectory()
        self.path = str(Path(self.dir.name) / "shop_alerts.csv")
        self.addCleanup(self.dir.cleanup)

    def test_the_same_price_is_not_alerted_twice(self):
        found = alerts.current(board(), threshold=0.005)
        self.assertEqual(len(alerts.record(found, 0.005, path=self.path)), 1)
        self.assertEqual(len(alerts.record(found, 0.005, path=self.path)), 0)

    def test_a_book_drifting_further_raises_a_fresh_alert(self):
        alerts.record(alerts.current(board(), threshold=0.005), 0.005,
                      path=self.path)
        worse = alerts.current(board(outlier=(150.0, -190.0)), threshold=0.005)
        fresh = alerts.record(worse, 0.005, path=self.path)
        self.assertEqual(len(fresh), 1)
        self.assertEqual(fresh[0]["price"], "150")

    def test_the_log_is_appended_never_rewritten(self):
        alerts.record(alerts.current(board(), threshold=0.005), 0.005,
                      path=self.path)
        alerts.record(alerts.current(board(outlier=(150.0, -190.0)),
                                     threshold=0.005), 0.005, path=self.path)
        rows = pd.read_csv(self.path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(list(rows.columns), alerts.ALERT_FIELDS)


class DeliveryTests(unittest.TestCase):
    def test_absent_mail_settings_are_not_an_error(self):
        # Same contract as a missing odds key on a fork: detect, log, stay
        # green. A workflow going red for want of an SMTP host would stop the
        # capture, which is the part that cannot be bought back.
        sent, note = alerts.send([{"deviation_points": "1.0"}], 3.0, env={})
        self.assertFalse(sent)
        self.assertIn("no mail settings", note)

    def test_the_message_states_the_odds_of_the_price_surviving(self):
        found = alerts.current(board(), threshold=0.005)
        with TemporaryDirectory() as tmp:
            fresh = alerts.record(found, 0.005,
                                  path=str(Path(tmp) / "log.csv"))
        body = alerts.compose(fresh, 4.0)
        self.assertIn("Chicago Cubs ML", body)
        self.assertIn("outlier", body)
        self.assertIn("21%", body)
        self.assertIn("not a recommended stake", body)


class ScanTests(unittest.TestCase):
    """The burst hook. A capture must never die because alerting did."""

    def test_a_broken_row_does_not_stop_the_burst(self):
        found = alerts.scan([{"nonsense": 1}], ["book0", "book1"],
                            log="/nonexistent/dir/log.csv")
        self.assertEqual(found, [])

    def test_a_good_poll_is_detected_and_logged(self):
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "log.csv")
            panel = [f"book{i}" for i in range(10)] + ["outlier"]
            fresh = alerts.scan(board().to_dict("records"), panel,
                                threshold=0.005, log=path)
            self.assertEqual(len(fresh), 1)
            self.assertTrue(Path(path).exists())


class FreshnessTests(unittest.TestCase):
    def test_capture_age_is_measured_from_the_quote_not_the_run(self):
        old = (datetime.now(timezone.utc) - timedelta(minutes=42)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        frame = pd.DataFrame([{"fetched_at": old}])
        found, age = alerts.latest_capture(frame)
        self.assertEqual(found, old)
        self.assertAlmostEqual(age, 42.0, delta=1.0)

    def test_no_quotes_is_reported_rather_than_guessed(self):
        self.assertEqual(alerts.latest_capture(pd.DataFrame()), (None, None))


class PushTests(unittest.TestCase):
    """The channel that needs no account and no second factor."""

    def _fresh(self):
        return [{"deviation_points": "1.25", "selection": "Chicago Cubs ML",
                 "price": "120", "book_key": "betrivers", "books": 11,
                 "lead_minutes": "60"}]

    def test_no_topic_is_not_an_error(self):
        sent, note = alerts.push(self._fresh(), 2.0, env={})
        self.assertFalse(sent)
        self.assertIn("no ntfy topic", note)

    def test_a_topic_posts_the_alert_body(self):
        seen = {}

        class Response:
            def read(self): return b"{}"
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def opener(request, timeout=None):
            seen["url"] = request.full_url
            seen["body"] = request.data.decode("utf-8")
            seen["headers"] = request.headers
            return Response()

        sent, note = alerts.push(self._fresh(), 2.0,
                                 env={"NTFY_TOPIC": "secret-topic"},
                                 opener=opener)
        self.assertTrue(sent)
        self.assertEqual(seen["url"], "https://ntfy.sh/secret-topic")
        self.assertIn("Chicago Cubs ML", seen["body"])
        self.assertIn("1.25pt", seen["body"])
        # High priority, or a silenced phone swallows a 90-second opportunity.
        self.assertEqual(seen["headers"].get("Priority"), "high")

    def test_a_self_hosted_server_is_honoured(self):
        seen = {}

        class Response:
            def read(self): return b"{}"
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def opener(request, timeout=None):
            seen["url"] = request.full_url
            return Response()

        alerts.push(self._fresh(), 2.0,
                    env={"NTFY_TOPIC": "t", "NTFY_SERVER": "https://n.me/"},
                    opener=opener)
        self.assertEqual(seen["url"], "https://n.me/t")


class NotifyTests(unittest.TestCase):
    """One channel failing must never silence the other."""

    def _fresh(self):
        return [{"deviation_points": "1.25", "selection": "Cubs ML",
                 "price": "120", "book_key": "betrivers", "books": 11,
                 "lead_minutes": "60"}]

    def test_nothing_configured_reports_the_log_only(self):
        delivered, note = alerts.notify(self._fresh(), 2.0, env={})
        self.assertFalse(delivered)
        self.assertIn("written to the log only", note)

    def test_a_raising_channel_does_not_stop_the_other(self):
        # A mail server refusing a connection must not swallow the push, and
        # neither may end a capture that has already been paid for.
        def boom(*args, **kwargs):
            raise OSError("connection refused")
        original = alerts.send
        alerts.send = boom
        try:
            delivered, note = alerts.notify(
                self._fresh(), 2.0,
                env={"NTFY_TOPIC": "t", "NTFY_SERVER": "http://127.0.0.1:1"})
        finally:
            alerts.send = original
        # Both failed here, but neither raised out of notify.
        self.assertFalse(delivered)
        self.assertIn("failed", note)


class ResendTests(unittest.TestCase):
    """Email by API key, so no account needs a second factor."""

    def _fresh(self):
        return [{"deviation_points": "1.25", "selection": "Chicago Cubs ML",
                 "price": "120", "book_key": "betrivers", "books": 11,
                 "lead_minutes": "60"}]

    def _capture(self):
        seen = {}

        class Response:
            def read(self): return b'{"id":"abc"}'
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def opener(request, timeout=None):
            seen["url"] = request.full_url
            seen["headers"] = request.headers
            seen["payload"] = json.loads(request.data.decode("utf-8"))
            return Response()

        return seen, opener

    def test_no_key_is_not_an_error(self):
        sent, note = alerts.resend(self._fresh(), 2.0,
                                   env={"BET_EMAIL_TO": "a@b.c"})
        self.assertFalse(sent)
        self.assertIn("no resend key", note)

    def test_a_key_without_a_recipient_is_not_an_error(self):
        sent, note = alerts.resend(self._fresh(), 2.0,
                                   env={"RESEND_API_KEY": "re_x"})
        self.assertFalse(sent)
        self.assertIn("recipient", note)

    def test_the_request_carries_the_key_and_the_alert(self):
        seen, opener = self._capture()
        sent, _ = alerts.resend(
            self._fresh(), 2.0,
            env={"RESEND_API_KEY": "re_secret", "BET_EMAIL_TO": "me@x.com"},
            opener=opener)
        self.assertTrue(sent)
        self.assertEqual(seen["url"], "https://api.resend.com/emails")
        self.assertEqual(seen["headers"].get("Authorization"),
                         "Bearer re_secret")
        self.assertEqual(seen["payload"]["to"], ["me@x.com"])
        self.assertIn("Chicago Cubs ML", seen["payload"]["text"])
        self.assertIn("1.25pt", seen["payload"]["subject"])

    def test_the_sandbox_sender_is_the_default(self):
        # Resend only delivers from a verified domain; until there is one, its
        # sandbox sender reaches the account owner and nobody else.
        seen, opener = self._capture()
        alerts.resend(self._fresh(), 2.0,
                      env={"RESEND_API_KEY": "k", "BET_EMAIL_TO": "me@x.com"},
                      opener=opener)
        self.assertEqual(seen["payload"]["from"], "onboarding@resend.dev")

    def test_a_verified_sender_is_honoured(self):
        seen, opener = self._capture()
        alerts.resend(self._fresh(), 2.0,
                      env={"RESEND_API_KEY": "k", "BET_EMAIL_TO": "me@x.com",
                           "RESEND_FROM": "bets@mine.com"}, opener=opener)
        self.assertEqual(seen["payload"]["from"], "bets@mine.com")

    def test_several_recipients_are_split(self):
        seen, opener = self._capture()
        alerts.resend(self._fresh(), 2.0,
                      env={"RESEND_API_KEY": "k",
                           "BET_EMAIL_TO": "a@x.com, b@y.com"}, opener=opener)
        self.assertEqual(seen["payload"]["to"], ["a@x.com", "b@y.com"])

    def test_the_older_recipient_name_still_works(self):
        seen, opener = self._capture()
        alerts.resend(self._fresh(), 2.0,
                      env={"RESEND_API_KEY": "k",
                           "ALERT_EMAIL_TO": "old@x.com"}, opener=opener)
        self.assertEqual(seen["payload"]["to"], ["old@x.com"])


class SelfTestTests(unittest.TestCase):
    """A delivery check must be unmistakable and must not touch the log."""

    def test_a_test_message_says_so(self):
        body = alerts.compose([dict(alerts.SAMPLE_ALERT)], 0.0)
        self.assertIn("TEST MESSAGE", body)
        self.assertIn("no bet here", body)

    def test_nothing_configured_reports_failure(self):
        self.assertFalse(alerts.self_test(env={}))

    def test_the_sample_never_looks_like_a_real_price(self):
        self.assertTrue(alerts.SAMPLE_ALERT["selection"].startswith("TEST"))


if __name__ == "__main__":
    unittest.main()
