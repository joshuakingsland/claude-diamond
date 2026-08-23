"""Tell me when to bet: live line-shopping alerts, and an honest expiry.

`line_shopping.py` found the one thing in this repository that beats a closing
price — taking a book that sits well off the panel's own de-vigged consensus.
This turns that finding into a live signal. It reads the newest capture, applies
the same rule the study validated, and writes an alert per qualifying price.

**The number that governs the whole design.** An opportunity does not last.
Measured on the burst captures, where consecutive polls sit about 90 seconds
apart:

| Deviation | Gone by the next poll | Alive at 5 min | Alive at 15 min |
| --- | ---: | ---: | ---: |
| 0.25 pt | 75% | 19% | 15% |
| 0.50 pt | 67% | 21% | 15% |
| 1.00 pt | 70% | 10% | **0%** |

The bigger the mispricing the faster it dies, which is the opposite of
convenient. Three consequences follow, and they are design constraints rather
than caveats:

1. **An hourly alert is mostly a historical notice.** Three percent of
   opportunities survive an hour. Alerting off the hourly capture is worth
   doing because it costs nothing, but roughly four in five will already be
   gone when the mail is read. `EXPECTED_LIVE_AT_FIVE_MINUTES` is printed in
   every alert so that is never a surprise.
2. **A stale alert costs nothing but attention.** The price is either still on
   the screen or it is not. There is no losing bet here, only a wasted look —
   which is why the honest response is to send anyway and label the odds.
3. **Catching the other 80% needs a poller, not a mailbox.** That is
   `odds_burst.py` cadence, and it needs someone already at a terminal with the
   book open. This file cannot manufacture that.

**The alert log is the forward test.** Every alert appends to
`data/shop_alerts.csv` and is never revised. Later captures of the same
game-market let `line_shopping.py` score what was actually flagged, live and
without hindsight — the same discipline `signal_ledger.py` applies to the
movement probe. The study looked backwards over 13 dates; this is the record
that will eventually say whether it holds forwards.

**Nothing here places a bet or recommends a stake.** An alert is a price worth
looking at, on a rule with 82 historical observations. It is not a wager.

    python alerts.py
    python alerts.py --threshold 0.0025 --send
"""

import argparse
import csv
import json
import os
import smtplib
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import pandas as pd

from config import (MAX_LOCK_LEAD_MINUTES, MAX_ODDS_AGE_MINUTES,
                    MIN_LOCK_LEAD_MINUTES)
from line_shopping import (MIN_PANEL_BOOKS, captures, load_quotes, panel_books,
                           prepare)

# The study's middle arm: +0.49 points of closing line value against Pinnacle
# over 35 bets, at roughly three or four alerts a day. The 0.25pt arm carries
# more sample and more noise, the 1pt arm fires twice a week.
DEFAULT_THRESHOLD = 0.005
# Measured, not assumed. See the table above.
EXPECTED_LIVE_AT_FIVE_MINUTES = 0.21
ALERT_LOG = "data/shop_alerts.csv"
ALERT_FIELDS = ["alerted_at", "fetched_at", "event_id", "commence_time",
                "home_team", "away_team", "market", "point", "side",
                "selection", "book_key", "price", "deviation_points",
                "consensus_probability", "break_even", "books",
                "lead_minutes", "threshold"]

SIDE_LABELS = {
    ("h2h", "home"): "{home} ML",
    ("h2h", "away"): "{away} ML",
    ("spreads", "home"): "{home} {point:+g}",
    ("spreads", "away"): "{away} {minus_point:+g}",
    ("totals", "home"): "Over {point:g}",
    ("totals", "away"): "Under {point:g}",
}


def selection_label(row):
    """Name the bet the way a bettor would, not the way the model prices it.

    The model is home-oriented throughout, so on a total "home" means Over.
    Printing the raw side in an alert would put the word "home" next to a
    number about run scoring.
    """
    template = SIDE_LABELS.get((row["market"], row["side"]))
    if template is None:
        return f"{row['market']} {row['side']}"
    point = row.get("point")
    point = float(point) if pd.notna(point) else 0.0
    return template.format(home=row.get("home_team", "home"),
                           away=row.get("away_team", "away"),
                           point=point, minus_point=-point)


def latest_capture(quotes):
    """The newest capture in the log, and how old it is.

    Age matters more than usual here. A capture from an hour ago describes a
    market that has almost certainly moved on, and an alert built from it would
    read exactly like one built a second ago.
    """
    if not len(quotes):
        return None, None
    stamp = quotes["fetched_at"].max()
    captured = pd.to_datetime(stamp, utc=True, errors="coerce")
    age = (pd.Timestamp.now(tz="UTC") - captured).total_seconds() / 60.0
    return stamp, float(age)


def fixed_panel(pattern="data/market_quotes/*.csv"):
    """The book panel, measured once over the whole quote history.

    A caller polling every 90 seconds must pass this in rather than let
    `current` derive a panel from one capture. Coverage measured inside a
    single poll is 100% for every book present, which is precisely the
    growing-panel bias `line_shopping.py` exists to avoid: the best of N
    prices rises with N, so a panel that follows whichever books happen to be
    up inflates the deviation on a thin night and invents alerts.
    """
    return panel_books(load_quotes(pattern))


def current(quotes, threshold=DEFAULT_THRESHOLD, stamp=None, books=None):
    """Qualifying prices in the newest capture, on the study's rule exactly."""
    if not len(quotes):
        return pd.DataFrame()
    books = panel_books(quotes) if books is None else books
    if len(books) < MIN_PANEL_BOOKS:
        return pd.DataFrame()
    stamp = stamp if stamp is not None else quotes["fetched_at"].max()
    frame = prepare(quotes, books)
    frame = frame[frame["fetched_at"] == stamp]
    book = captures(frame, min_lead=MIN_LOCK_LEAD_MINUTES,
                    max_lead=MAX_LOCK_LEAD_MINUTES)
    if not len(book):
        return pd.DataFrame()

    rows = []
    for side in ("home", "away"):
        block = book[book[f"{side}_edge"] >= threshold].copy()
        block["side"] = side
        block["deviation"] = block[f"{side}_edge"]
        block["break_even"] = block[f"{side}_break_even"]
        block["price"] = block[f"best_{side}_price"]
        block["book_key"] = block[f"best_{side}_book"]
        block["consensus"] = (block["consensus_home"] if side == "home"
                              else 1.0 - block["consensus_home"])
        rows.append(block)
    found = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not len(found):
        return pd.DataFrame()

    # Team names live on the quote rows, not the aggregate.
    names = (quotes[["event_id", "home_team", "away_team"]]
             .drop_duplicates("event_id"))
    found = found.merge(names, on="event_id", how="left")
    found["selection"] = found.apply(selection_label, axis=1)
    return found.sort_values("deviation", ascending=False)


def scan(rows, books, threshold=DEFAULT_THRESHOLD, log=ALERT_LOG, send_mail=False):
    """Detect, log and optionally mail one poll's worth of fresh quotes.

    The hook `odds_burst.py` calls after every capture. This is the path that
    matters: three quarters of these prices are gone by the next poll, so an
    alert raised seconds after the quote is the only one with a real chance of
    still being on the screen when it is read.

    Never raises. A burst exists to capture data, and a mail server refusing a
    connection must not end it — the detection is already in the log.
    """
    try:
        quotes = pd.DataFrame(rows)
        found = current(quotes, threshold=threshold, books=books)
        if not len(found):
            return []
        fresh = record(found, threshold, path=log)
        if fresh and send_mail:
            notify(fresh, 0.0)
        return fresh
    except Exception as error:                       # noqa: BLE001
        print(f"  alerting failed ({error}); the capture continues")
        return []


def already_sent(path=ALERT_LOG):
    """Alerts previously written, keyed so a repeated capture cannot re-fire.

    The hourly workflow re-reads a market that may not have moved. Keying on
    the price as well as the side means a book drifting further off consensus
    raises a fresh alert, while an unchanged quote stays quiet.
    """
    file = Path(path)
    if not file.exists():
        return set()
    with file.open(newline="", encoding="utf-8") as handle:
        return {(row["event_id"], row["market"], row["point"], row["side"],
                 row["book_key"], row["price"])
                for row in csv.DictReader(handle)}


def record(found, threshold, path=ALERT_LOG, now=None):
    """Append new alerts to the permanent log. Never revises an existing row."""
    seen = already_sent(path)
    stamped = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    fresh = []
    for _, row in found.iterrows():
        point = "" if pd.isna(row.get("point")) else f"{float(row['point']):g}"
        key = (str(row["event_id"]), row["market"], point, row["side"],
               str(row["book_key"]), f"{float(row['price']):g}")
        if key in seen:
            continue
        seen.add(key)
        fresh.append({
            "alerted_at": stamped,
            "fetched_at": row["fetched_at"],
            "event_id": row["event_id"],
            "commence_time": row["commence"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),
            "market": row["market"],
            "point": point,
            "side": row["side"],
            "selection": row["selection"],
            "book_key": row["book_key"],
            "price": f"{float(row['price']):g}",
            "deviation_points": f"{100 * float(row['deviation']):.3f}",
            "consensus_probability": f"{float(row['consensus']):.6f}",
            "break_even": f"{float(row['break_even']):.6f}",
            "books": int(row["books"]),
            "lead_minutes": f"{float(row['lead_minutes']):.1f}",
            "threshold": f"{threshold:g}",
        })
    if not fresh:
        return []
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    exists = file.exists()
    with file.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALERT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(fresh)
    return fresh


def compose(fresh, age_minutes):
    """The message body. States the odds of the price still being there."""
    test = any(str(row.get("selection", "")).startswith("TEST") for row in fresh)
    lines = []
    if test:
        lines += ["TEST MESSAGE — delivery check only, no bet here.", ""]
    lines += [f"{len(fresh)} price{'s' if len(fresh) != 1 else ''} off the "
              f"market consensus.", ""]
    for row in fresh:
        lines.append(f"  {row['selection']}  {row['price']} at "
                     f"{row['book_key']}")
        lines.append(f"    {row['deviation_points']}pt better than the "
                     f"consensus of {row['books']} books, "
                     f"{row['lead_minutes']} min to first pitch")
        lines.append("")
    lines += [
        f"Quote is {age_minutes:.0f} minutes old.",
        f"About {100 * EXPECTED_LIVE_AT_FIVE_MINUTES:.0f}% of these are still "
        "on the screen five minutes after capture, so expect most to be gone.",
        "",
        "This is a price worth checking, not a recommended stake. The rule "
        "behind it has 82 historical observations over 14 dates and has never "
        "been tested forwards.",
    ]
    return "\n".join(lines)


def push(fresh, age_minutes, env=None, opener=None):
    """Push the alert to an ntfy topic, if one is configured.

    Preferred over mail for the reason the survival table gives: three
    quarters of these prices are gone inside ninety seconds, and a phone
    notification arrives while an inbox is still being checked. It also needs
    no account and no second factor anywhere — the topic name *is* the
    address.

    That convenience is also the caveat, and it is worth stating rather than
    burying: **anyone who knows the topic can read the alerts**. Use a long
    random topic, treat it as a password, or point `NTFY_SERVER` at your own
    instance.
    """
    env = os.environ if env is None else env
    topic = env.get("NTFY_TOPIC")
    if not topic:
        return False, "no ntfy topic set"
    server = env.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    best = max(float(row["deviation_points"]) for row in fresh)
    body = compose(fresh, age_minutes).encode("utf-8")
    request = urllib.request.Request(
        f"{server}/{topic}", data=body, method="POST",
        headers={
            "Title": (f"{len(fresh)} shop alert"
                      f"{'s' if len(fresh) != 1 else ''}, "
                      f"best {best:.2f}pt off consensus"),
            # High priority so it breaks through a silenced phone; these are
            # worth nothing if read an hour later.
            "Priority": "high",
            "Tags": "money_with_wings",
        })
    open_url = opener or urllib.request.urlopen
    with open_url(request, timeout=15) as response:
        response.read()
    return True, f"pushed {len(fresh)} alert(s) to {server}/{topic}"


def resend(fresh, age_minutes, env=None, opener=None):
    """Email the alert through Resend's HTTP API, if a key is configured.

    Resend rather than SMTP because it authenticates with a plain API key. A
    Gmail app password requires two-factor authentication on the Google
    account; nothing here does.

    **The sender address is the part that catches people.** Resend will only
    deliver from a domain you have verified. Until one is, the sandbox sender
    `onboarding@resend.dev` works but will *only* deliver to the address that
    owns the Resend account — which is exactly the case here, and why it is
    the default. Set `RESEND_FROM` once a domain is verified.
    """
    env = os.environ if env is None else env
    key = env.get("RESEND_API_KEY")
    to = env.get("BET_EMAIL_TO") or env.get("ALERT_EMAIL_TO")
    if not key or not to:
        missing = "key" if not key else "recipient"
        return False, f"no resend {missing}"
    best = max(float(row["deviation_points"]) for row in fresh)
    payload = json.dumps({
        "from": env.get("RESEND_FROM", "onboarding@resend.dev"),
        "to": [address.strip() for address in to.split(",") if address.strip()],
        "subject": (f"{len(fresh)} shop alert"
                    f"{'s' if len(fresh) != 1 else ''}, "
                    f"best {best:.2f}pt off consensus"),
        "text": compose(fresh, age_minutes),
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://api.resend.com/emails", data=payload, method="POST",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    open_url = opener or urllib.request.urlopen
    with open_url(request, timeout=20) as response:
        body = response.read().decode("utf-8", "replace")
    return True, f"emailed {len(fresh)} alert(s) to {to} via resend ({body[:80]})"


def notify(fresh, age_minutes, env=None):
    """Deliver by every configured channel; absent configuration is not an error.

    Each channel is attempted independently so one failing does not silence
    the other, and a channel raising cannot end a capture — the detection is
    already in the log by the time this runs.
    """
    env = os.environ if env is None else env
    notes, delivered = [], False
    for channel in (push, resend, send):
        try:
            sent, note = channel(fresh, age_minutes, env=env)
            delivered = delivered or sent
        except Exception as error:                   # noqa: BLE001
            note = f"{channel.__name__} failed ({error})"
        notes.append(note)
    if not delivered:
        notes.append("alert written to the log only")
    return delivered, "; ".join(notes)


def send(fresh, age_minutes, env=None):
    """Mail the alert, if and only if the mail settings exist.

    Absent configuration is not an error: the same pattern as the odds key, so
    a fork with no mail secrets runs the detector and writes the log without
    the workflow going red.

    Note that Gmail is not usable here without enabling two-factor
    authentication on the account, because an app password requires it. Any
    transactional sender with a plain API key works, and `push` avoids the
    question entirely.
    """
    env = os.environ if env is None else env
    host = env.get("SMTP_HOST")
    to = env.get("ALERT_EMAIL_TO")
    if not host or not to:
        return False, "no mail settings"
    message = EmailMessage()
    best = max(float(row["deviation_points"]) for row in fresh)
    message["Subject"] = (f"{len(fresh)} shop alert"
                          f"{'s' if len(fresh) != 1 else ''}, "
                          f"best {best:.2f}pt off consensus")
    message["From"] = env.get("SMTP_FROM", env.get("SMTP_USER", to))
    message["To"] = to
    message.set_content(compose(fresh, age_minutes))
    port = int(env.get("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls()
        user, password = env.get("SMTP_USER"), env.get("SMTP_PASSWORD")
        if user and password:
            server.login(user, password)
        server.send_message(message)
    return True, f"mailed {len(fresh)} alert(s) to {to}"


SAMPLE_ALERT = {
    "selection": "TEST — not a real price", "book_key": "example",
    "price": "+120", "deviation_points": "0.00", "books": 11,
    "lead_minutes": "60",
}


def self_test(env=None):
    """Send one clearly-labelled message through every configured channel.

    Delivery cannot be verified by waiting: alerts fire only when a book is
    genuinely off consensus, which may be hours away and cannot be summoned.
    Without this, the first test of a new channel is a real opportunity — and
    finding out then that the key was wrong wastes the one thing that cannot
    be re-bought.

    The message says TEST in the subject and in the body, because an alert
    that looks real and is not is worse than no alert.
    """
    env = os.environ if env is None else env
    fresh = [dict(SAMPLE_ALERT)]
    configured = [name for name, key in (("ntfy", "NTFY_TOPIC"),
                                         ("resend", "RESEND_API_KEY"),
                                         ("smtp", "SMTP_HOST"))
                  if env.get(key)]
    print(f"channels configured: {', '.join(configured) or 'none'}")
    if not configured:
        print("nothing to test; set NTFY_TOPIC or RESEND_API_KEY")
        return False
    delivered, note = notify(fresh, 0.0, env=env)
    print(note)
    print("delivered" if delivered
          else "NOT delivered — check the key and the recipient")
    return delivered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotes", default="data/market_quotes/*.csv")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--log", default=ALERT_LOG)
    parser.add_argument("--send", action="store_true",
                        help="deliver by every configured channel")
    parser.add_argument("--self-test", action="store_true",
                        help="send one labelled test message and exit")
    parser.add_argument("--max-age", type=float,
                        default=float(MAX_ODDS_AGE_MINUTES),
                        help="refuse to alert on a capture older than this")
    args = parser.parse_args()

    if args.self_test:
        # Never writes to the alert log: the log is the forward test and a
        # synthetic row in it would corrupt the one record that cannot be
        # re-derived.
        raise SystemExit(0 if self_test() else 1)

    quotes = load_quotes(args.quotes)
    stamp, age = latest_capture(quotes)
    if stamp is None:
        print("no quote logs found")
        return
    print(f"newest capture {stamp} ({age:.0f} min old)")
    if age > args.max_age:
        # A price from an hour ago is a fact about a market that has moved on.
        print(f"capture older than {args.max_age:g} min; not alerting")
        return

    found = current(quotes, threshold=args.threshold, stamp=stamp)
    if not len(found):
        print(f"no price is {100 * args.threshold:g}pt off consensus")
        return
    fresh = record(found, args.threshold, path=args.log)
    print(f"{len(found)} qualifying, {len(fresh)} new")
    for row in fresh:
        print(f"  {row['selection']:<28} {row['price']:>6} at "
              f"{row['book_key']:<16} {row['deviation_points']:>6}pt  "
              f"{row['lead_minutes']:>5} min out")
    if fresh and args.send:
        _, note = notify(fresh, age)
        print(note)


if __name__ == "__main__":
    main()
