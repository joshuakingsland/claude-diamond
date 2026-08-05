"""Capped, resumable historical odds capture.

The historical endpoint is the only expensive thing this project touches. It
bills markets x regions x 10, so one snapshot of three markets in one region
costs 30 credits — measured, not assumed. Everything here exists so that cost
is bounded, visible, and never paid twice for the same snapshot:

- `--max-requests` is a hard ceiling checked before every call.
- Every completed snapshot is written to a manifest immediately, so an
  interrupted run resumes instead of restarting.
- The actual credit cost is read back from the response headers and recorded,
  so the manifest shows what was really spent rather than an estimate.

One daily snapshot does double duty. Taken shortly before the evening slate
starts, it is the closing proxy for those games and simultaneously the
roughly-24-hour entry price for the following evening's games. Capturing
entry and close separately would double the bill for the same information.

Afternoon games are the known gap: a 22:50 UTC snapshot is already in-play
for a game that started at 18:00, so those games get an entry price but no
usable close. Lead time is recorded per game and reported rather than
smoothed over.
"""

import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import MARKETS, PRICED_ODDS_REGIONS, SPORT_KEY
from odds import QUOTE_FIELDS, paired_book_quotes

API = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/odds"
MANIFEST = Path("data/historical_manifest.csv")
QUOTES = Path("data/historical_quotes.csv")

MANIFEST_FIELDS = [
    "requested_date", "snapshot_timestamp", "previous_timestamp",
    "events", "quotes", "credits_used", "credits_remaining", "fetched_at",
]


def _request(url, timeout=60, attempts=4):
    last = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
                headers = {
                    "used": response.headers.get("x-requests-last"),
                    "remaining": response.headers.get("x-requests-remaining"),
                }
                return payload, headers
        except Exception as error:  # noqa: BLE001 - retried, then re-raised
            last = error
            if attempt == attempts - 1:
                break
            time.sleep(2 ** attempt)
    raise last


def fetch_snapshot(key, moment, regions=None, markets=MARKETS):
    regions = regions or PRICED_ODDS_REGIONS
    query = urllib.parse.urlencode({
        "apiKey": key,
        "regions": ",".join(regions),
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "date": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    return _request(f"{API}?{query}")


def _load_manifest(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def snapshot_dates(start, end, hour=22, minute=50):
    """One snapshot per day at a fixed UTC time."""
    current = datetime.fromisoformat(start).replace(
        hour=hour, minute=minute, tzinfo=timezone.utc)
    final = datetime.fromisoformat(end).replace(
        hour=hour, minute=minute, tzinfo=timezone.utc)
    while current <= final:
        yield current
        current += timedelta(days=1)


def run(key, start, end, max_requests, manifest_path=MANIFEST,
        quotes_path=QUOTES, dry_run=False):
    done = {row["requested_date"] for row in _load_manifest(manifest_path)}
    spent_rows = _load_manifest(manifest_path)
    already_spent = sum(int(row["credits_used"] or 0) for row in spent_rows)
    pending = [moment for moment in snapshot_dates(start, end)
               if moment.strftime("%Y-%m-%d") not in done]
    print(f"{len(done)} snapshots already captured ({already_spent} credits); "
          f"{len(pending)} remaining in {start}..{end}")
    if dry_run:
        print(f"dry run: would issue up to {min(len(pending), max_requests)} "
              f"requests at ~30 credits each = "
              f"~{min(len(pending), max_requests) * 30} credits")
        return 0

    issued, spent = 0, 0
    for moment in pending:
        if issued >= max_requests:
            print(f"request cap of {max_requests} reached; rerun to continue")
            break
        label = moment.strftime("%Y-%m-%d")
        try:
            payload, headers = fetch_snapshot(key, moment)
        except Exception as error:  # noqa: BLE001 - recorded, run continues
            print(f"  {label} FAILED {error!r}"[:120])
            continue
        events = payload.get("data", []) or []
        stamp = payload.get("timestamp", "")
        rows = []
        for event in events:
            commence = event.get("commence_time", "")
            for region in PRICED_ODDS_REGIONS:
                for quote in paired_book_quotes(event, region):
                    rows.append({
                        "snapshot_id": f"{stamp}|{event.get('id','')}|"
                                       f"{quote['book_key']}|{quote['market']}|"
                                       f"{quote['point']}",
                        "fetched_at": stamp,
                        "event_id": event.get("id", ""),
                        "commence_time": commence,
                        "date": commence[:10],
                        "home_team": event.get("home_team", ""),
                        "away_team": event.get("away_team", ""),
                        **quote,
                    })
                break  # historical requests one region at a time
        _append(quotes_path, QUOTE_FIELDS, rows)
        used = int(headers.get("used") or 0)
        spent += used
        issued += 1
        _append(manifest_path, MANIFEST_FIELDS, [{
            "requested_date": label,
            "snapshot_timestamp": stamp,
            "previous_timestamp": payload.get("previous_timestamp", ""),
            "events": len(events),
            "quotes": len(rows),
            "credits_used": used,
            "credits_remaining": headers.get("remaining", ""),
            "fetched_at": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
        }])
        if issued % 10 == 0 or issued == 1:
            print(f"  {label}: {len(events):3d} events, {len(rows):5d} quotes, "
                  f"{spent} credits spent, {headers.get('remaining')} left")
    print(f"issued {issued} requests, spent {spent} credits")
    return issued


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-03-27")
    parser.add_argument("--end", default="2025-09-28")
    parser.add_argument("--max-requests", type=int, default=10,
                        help="hard ceiling; each costs about 30 credits")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--quotes", default=str(QUOTES))
    args = parser.parse_args()
    key = os.environ.get("ODDS_API_KEY")
    if not key and not args.dry_run:
        raise SystemExit("ODDS_API_KEY is required")
    run(key, args.start, args.end, args.max_requests,
        args.manifest, args.quotes, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
