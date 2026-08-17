"""High-frequency odds capture, for measuring how long a price survives.

`odds.py` samples the board about seventeen times a day, median gap 58 minutes.
That answers "where was the market" and cannot answer "how long was that number
available", which is the question standing between a measured dislocation and a
bet.

**What this is not for.** The first guess was that some book lags the market and
can be picked off, and higher frequency would expose it. The data already
refuses that: `book_updated_at` is populated on every quote and the median
quote age at capture is **24 seconds**, with the slowest book at 1.4 minutes.
Every book is fresh. So the 1.4-1.8 probability points by which the best book
beats the median are not staleness — they are books genuinely disagreeing about
the price, persistently.

**What it is for.** Duration. A book sitting 3 points off the market for four
minutes is invisible to hourly sampling — it would be caught about 7% of the
time — and whether such a number lasts four minutes or forty decides whether
anything could be taken at size. That is the untested assumption ranked first
of the ten in the README, and it is measurable without a broker.

Polling happens *inside one job* rather than through a tighter cron. GitHub's
scheduler has a five-minute floor, fires late under load, and drops runs; a
single job that sleeps between polls gets true minute resolution and one
predictable credit bill. A job may run six hours, which covers any card.

Cost is bounded and declared before anything is spent. Each poll costs
`regions x markets` credits exactly as a normal capture does — six today — so a
ninety-second cadence across a three-hour window is about 720 credits. Against
the four million on the account that is two years of daily bursts, but the
estimate is printed and `--max-credits` refuses to start a burst that would
exceed it rather than discovering the limit halfway through.

Rows land in the same quote log as `odds.py`, with the same schema and the same
de-vig, distinguished only by their `fetched_at`. Downstream code needs no
knowledge that a burst happened: `devig.py` groups by capture, and a burst is
simply many captures close together.

    python odds_burst.py --minutes 60 --every 90 --dry-run
    python odds_burst.py --minutes 180 --every 90 --require-key
"""

import argparse
import csv
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import MARKETS, ODDS_CONSENSUS_VERSION, ODDS_REGIONS
from odds import (LINE_FIELDS, QUOTE_FIELDS, _is_future, _quote_rows,
                  _write_atomic, collect_events, consensus_lines,
                  record_credits)

# Below this a "burst" is just a capture and the extra machinery is noise.
MIN_POLLS = 2
# The scheduler cannot be trusted below this and neither can the provider's own
# refresh: quote age at capture is 24 seconds median, so polling faster than
# this mostly re-reads the same numbers and pays for them again.
MIN_INTERVAL_SECONDS = 45
CREDITS_PER_POLL = len(ODDS_REGIONS) * len(MARKETS)


def plan(minutes, every_seconds):
    """Polls, credits and wall-clock, computed before anything is spent."""
    every_seconds = max(int(every_seconds), MIN_INTERVAL_SECONDS)
    polls = max(int((minutes * 60) // every_seconds), MIN_POLLS)
    return {
        "polls": polls,
        "interval_seconds": every_seconds,
        "minutes": round(polls * every_seconds / 60.0, 1),
        "credits_per_poll": CREDITS_PER_POLL,
        "estimated_credits": polls * CREDITS_PER_POLL,
    }


def append_quotes(path, rows):
    """Append rows to the monthly shard without reading it back.

    `odds.append_quote_log` reads the whole log, de-duplicates on snapshot_id
    and rewrites it atomically. That is right for seventeen captures a day and
    ruinous here: a burst adds tens of thousands of rows a day, so the monthly
    shard reaches a million rows and each of the 120 daily rewrites has to walk
    all of it. The framework would have been unusable inside a week.

    De-duplication is not needed because `snapshot_id` is hashed over the
    fetch timestamp, and the interval floor guarantees polls land in different
    seconds. Two bursts overlapping would each append their own rows, and
    `merge_data.py` already unions the log on snapshot_id when two runs commit
    together.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUOTE_FIELDS,
                                extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in QUOTE_FIELDS}
                         for row in rows)
    return len(rows)


def poll_once(key, now=None, verbose=True):
    """One capture, identical in shape to what `odds.py` writes."""
    now = now or datetime.now(timezone.utc)
    stamp = f"{now:%Y-%m-%dT%H:%M:%SZ}"
    lines, quotes, in_play = [], [], 0
    events, credits = collect_events(key)
    for event, paired in events:
        commence = event.get("commence_time", "")
        # Same fail-closed rule as the hourly path. A burst runs right up to
        # first pitch, which is exactly when in-play prices start appearing,
        # so this matters more here than there.
        if not _is_future(commence, now):
            in_play += 1
            continue
        for line in consensus_lines(event, paired):
            lines.append({
                "date": commence[:10],
                "commence_time": commence,
                "event_id": event.get("id", ""),
                "home_team": event.get("home_team", ""),
                "away_team": event.get("away_team", ""),
                **line,
                "odds_source": f"the-odds-api-{ODDS_CONSENSUS_VERSION}",
                "fetched_at": stamp,
            })
        quotes.extend(_quote_rows(event, paired, stamp))
    return {"stamp": stamp, "lines": lines, "quotes": quotes,
            "in_play": in_play, "credits": credits,
            "events": len(events)}


def run(key, minutes=180, every_seconds=90, quotes_dir="data/market_quotes",
        lines_path="data/lines_upcoming.csv",
        credit_log="data/credit_log.csv", max_credits=None,
        min_credits=0, verbose=True, sleep=time.sleep):
    """Poll on a fixed cadence, appending every poll to the quote log."""
    shape = plan(minutes, every_seconds)
    if verbose:
        print(f"burst plan: {shape['polls']} polls every "
              f"{shape['interval_seconds']}s over {shape['minutes']} min, "
              f"about {shape['estimated_credits']} credits")
    if max_credits is not None and shape["estimated_credits"] > max_credits:
        raise SystemExit(
            f"burst would cost about {shape['estimated_credits']} credits, "
            f"above the {max_credits} ceiling. Lower --minutes, raise "
            f"--every, or raise --max-credits deliberately.")

    deadline = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    polled, written, spent = 0, 0, 0
    last_lines = None
    remaining = None
    for index in range(shape["polls"]):
        now = datetime.now(timezone.utc)
        if index and now >= deadline:
            break
        result = poll_once(key, now=now, verbose=verbose)
        polled += 1
        spent += CREDITS_PER_POLL
        if result["quotes"]:
            # Monthly shards, the same convention `odds.py` uses. Passing the
            # directory straight to append_quote_log would write a file named
            # after the directory and split the log from the hourly path.
            shard = (Path(quotes_dir)
                     / f"quotes_{result['stamp'][:7]}.csv")
            append_quotes(shard, result["quotes"])
            written += len(result["quotes"])
        # The board file is a snapshot of one moment, so the last poll wins
        # rather than the first: a burst ends nearest first pitch, which is
        # the picture the card should be priced from.
        if result["lines"]:
            last_lines = result["lines"]
        # `collect_events` returns a LIST of per-region spend dicts, and
        # `record_credits` writes them and hands back the lowest remaining
        # balance across regions. Reading a `remaining` key off the list was
        # the first version's bug: the floor silently never engaged.
        if result["credits"]:
            balance = record_credits(credit_log, result["stamp"],
                                     result["credits"])
            if balance is not None:
                remaining = balance
        if verbose:
            note = ""
            if result["in_play"]:
                note += f" in-play skipped {result['in_play']}"
            if remaining is not None:
                note += f" credits left {remaining}"
            print(f"  poll {polled}/{shape['polls']} {result['stamp']} "
                  f"events {result['events']} "
                  f"quotes {len(result['quotes'])}{note}")
        # Checked between polls, not only at the start: a shared quota can be
        # drained by something else mid-burst, and the floor exists to stop
        # this run rather than to describe the balance when it began.
        if min_credits and remaining is not None and remaining < min_credits:
            print(f"credit floor reached ({remaining} < {min_credits}); "
                  f"stopping the burst with {polled} polls written")
            break
        if index + 1 < shape["polls"]:
            sleep(shape["interval_seconds"])

    if last_lines is not None:
        _write_atomic(lines_path, LINE_FIELDS, last_lines)
    summary = {"polls": polled, "quote_rows": written,
               "credits_spent": spent, "credits_remaining": remaining,
               "interval_seconds": shape["interval_seconds"]}
    if verbose:
        print(f"burst complete: {summary}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=180.0,
                        help="how long to keep polling")
    parser.add_argument("--every", type=float, default=90.0,
                        help=f"seconds between polls, floor "
                             f"{MIN_INTERVAL_SECONDS}")
    parser.add_argument("--max-credits", type=int, default=2000,
                        help="refuse to start a burst costing more than this")
    parser.add_argument("--min-credits", type=int, default=0,
                        help="stop the burst when fewer remain; 0 disables")
    parser.add_argument("--quotes-dir", default="data/market_quotes")
    parser.add_argument("--lines", default="data/lines_upcoming.csv")
    parser.add_argument("--credit-log", default="data/credit_log.csv")
    parser.add_argument("--require-key", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and its cost, fetch nothing")
    args = parser.parse_args(argv)

    shape = plan(args.minutes, args.every)
    if args.dry_run:
        print(f"dry run: {shape}")
        return shape

    key = os.environ.get("ODDS_API_KEY")
    if not key:
        if args.require_key:
            raise SystemExit(
                "ODDS_API_KEY is required. Add it as a GitHub Actions secret.")
        print("No ODDS_API_KEY set; nothing fetched.")
        print(f"would have run: {shape}")
        return shape

    return run(key, minutes=args.minutes, every_seconds=args.every,
               quotes_dir=args.quotes_dir, lines_path=args.lines,
               credit_log=args.credit_log, max_credits=args.max_credits,
               min_credits=args.min_credits)


if __name__ == "__main__":
    main()
