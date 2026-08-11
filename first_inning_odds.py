"""Low-cost historical coverage audit for first-inning MLB totals.

This is deliberately a data-collection tool, not a YRFI/NRFI model.  A
first-inning total is a different contract from a full-game total: it needs
its own outcome labels, feature snapshots, evaluation, and close definition.
The first question is more basic -- does the selected region actually carry a
paired `totals_1st_1_innings` price near first pitch often enough to study?

The Odds API's historical event-odds endpoint is priced per event, market,
and region.  The default is therefore one event in one region at a fixed
pregame lead.  Every completed attempt, including a no-offer result, enters
an append-only audit manifest so a rerun neither hides missing coverage nor
repays for the same event snapshot.
"""

import argparse
import csv
import os
import urllib.parse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from config import FIRST_INNING_TOTALS_MARKET, SPORT_KEY
from historical_odds import _request
from odds import _quote_rows, append_quote_log, paired_book_quotes


EVENTS_API = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/events"
EVENT_ODDS_API = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/events"
DEFAULT_MANIFEST = Path("data/first_inning_audit.csv")
DEFAULT_QUOTES = Path("data/first_inning_quotes.csv")

AUDIT_FIELDS = [
    "audit_id", "requested_date", "event_id", "home_team", "away_team",
    "commence_time", "requested_snapshot", "returned_snapshot", "market",
    "region", "status", "quote_count", "book_count", "book_keys", "points",
    "odds_credits_used", "discovery_credits_used", "credits_remaining",
    "fetched_at", "error",
]


def _iso(value):
    """Parse a required UTC timestamp without silently accepting bad data."""
    moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def audit_snapshot(commence_time, lead_minutes):
    """Return the precise historical snapshot requested before first pitch."""
    if not 1 <= int(lead_minutes) <= 1440:
        raise ValueError("lead_minutes must be between 1 and 1,440")
    return _iso(commence_time) - timedelta(minutes=int(lead_minutes))


def day_lookup_time(day):
    """Use noon UTC, before every ordinary MLB game on the requested date."""
    return datetime.strptime(day, "%Y-%m-%d").replace(
        hour=12, tzinfo=timezone.utc)


def _url(base, key, **params):
    query = urllib.parse.urlencode({"apiKey": key, **params})
    return f"{base}?{query}"


def events_url(key, day):
    return _url(EVENTS_API, key, date=day_lookup_time(day).strftime(
        "%Y-%m-%dT%H:%M:%SZ"))


def event_odds_url(key, event_id, snapshot, region, market):
    return _url(
        f"{EVENT_ODDS_API}/{event_id}/odds", key,
        regions=region,
        markets=market,
        oddsFormat="american",
        date=snapshot.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def response_events(payload):
    """Normalise the documented list and event-odds response shapes."""
    data = (payload or {}).get("data", [])
    if isinstance(data, dict):
        return [data]
    return data if isinstance(data, list) else []


def events_on_day(payload, day):
    """Events belonging to the MLB calendar day, not merely its UTC date.

    A 10:10 p.m. Pacific first pitch has the following UTC date.  The
    historical event lookup is taken at noon UTC, so use the noon-to-noon
    window that contains the whole North American baseball slate instead of
    silently dropping those late games.
    """
    start = day_lookup_time(day)
    end = start + timedelta(days=1)
    events = []
    for event in response_events(payload):
        try:
            if start <= _iso(event.get("commence_time", "")) < end:
                events.append(event)
        except (TypeError, ValueError):
            continue
    return sorted(events, key=lambda event: event.get("commence_time", ""))


def _load_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append(path, fields, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if header:
            writer.writeheader()
        writer.writerows(rows)


def _credit(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _audit_id(event_id, snapshot, region, market):
    return "|".join((str(event_id), snapshot.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     region, market))


def stratified_days(start, end, per_season=10):
    """Precommit evenly spaced regular-season audit dates by calendar year.

    Period-market history begins 3 May 2023.  This deterministic calendar
    samples early, middle, and late season in every available year instead of
    buying a convenient contiguous block that could look unusually good by
    chance.  Dates are only candidates; the event-odds calls remain capped.
    """
    if not 1 <= int(per_season) <= 31:
        raise ValueError("days_per_season must be between 1 and 31")
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last:
        raise ValueError("start must not be after end")
    selected = []
    for year in range(first.year, last.year + 1):
        low = max(first, date(year, 5, 3))
        high = min(last, date(year, 9, 30))
        if low > high:
            continue
        slots = int(per_season)
        for position in range(slots):
            fraction = 0.5 if slots == 1 else position / (slots - 1)
            offset = round((high - low).days * fraction)
            selected.append((low + timedelta(days=offset)).isoformat())
    return sorted(set(selected))


def _round_robin(groups, limit):
    """Take candidate events evenly across sampled dates until the hard cap."""
    positions = [0] * len(groups)
    selected = []
    while len(selected) < limit:
        progressed = False
        for index, group in enumerate(groups):
            candidates = group["candidates"]
            if positions[index] >= len(candidates):
                continue
            event, snapshot, audit_id = candidates[positions[index]]
            positions[index] += 1
            discovery = group["credits"] if positions[index] == 1 else 0
            selected.append((event, snapshot, audit_id, discovery))
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def _audit_row(event, audit_id, snapshot, region, market, discovery_credits,
               payload=None, headers=None, quotes=None, status="offered", error=""):
    headers = headers or {}
    quotes = quotes or []
    points = sorted({str(quote.get("point", "")) for quote in quotes})
    books = sorted({quote.get("book_key", "") for quote in quotes
                    if quote.get("book_key")})
    return {
        "audit_id": audit_id,
        "requested_date": str(event.get("commence_time", ""))[:10],
        "event_id": event.get("id", ""),
        "home_team": event.get("home_team", ""),
        "away_team": event.get("away_team", ""),
        "commence_time": event.get("commence_time", ""),
        "requested_snapshot": snapshot.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "returned_snapshot": (payload or {}).get("timestamp", ""),
        "market": market,
        "region": region,
        "status": status,
        "quote_count": len(quotes),
        "book_count": len(books),
        "book_keys": ",".join(books),
        "points": ",".join(points),
        "odds_credits_used": _credit(headers.get("used")),
        "discovery_credits_used": discovery_credits,
        "credits_remaining": headers.get("remaining", ""),
        "fetched_at": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
        "error": error,
    }


def _capture_one(key, event, snapshot, audit_id, region, market,
                 discovery_credits, quotes_path, request):
    """Buy and immediately record one event-level historical snapshot."""
    try:
        payload, headers = request(event_odds_url(
            key, event.get("id", ""), snapshot, region, market))
        returned = response_events(payload)
        # Historical event odds normally returns one event.  Start from the
        # discovery response so identifiers survive an incomplete event-odds
        # payload, then replace it with the priced response.
        priced_event = dict(event)
        if returned:
            priced_event.update(returned[0])
        quotes = paired_book_quotes(
            priced_event, region=region, accepted_markets=(market,))
        quote_stamp = payload.get("timestamp") or snapshot.strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        append_quote_log(quotes_path, _quote_rows(priced_event, quotes,
                                                   quote_stamp))
        status = "offered" if quotes else "no_offer"
        return _audit_row(event, audit_id, snapshot, region, market,
                          discovery_credits, payload, headers, quotes, status)
    except Exception as error:  # record a paid/failed attempt for resume
        return _audit_row(event, audit_id, snapshot, region, market,
                          discovery_credits, status="failed", error=repr(error))


def run(key, day, max_events=1, lead_minutes=10, region="us",
        market=FIRST_INNING_TOTALS_MARKET, manifest_path=DEFAULT_MANIFEST,
        quotes_path=DEFAULT_QUOTES, dry_run=False, request=None):
    """Audit at most ``max_events`` first-inning total snapshots.

    ``request`` follows :func:`historical_odds._request` and is injectable so
    the accounting and resume behaviour are tested without spending credits.
    """
    # Fail before requesting anything for an invalid manual dispatch.
    day_lookup_time(day)
    if not 1 <= int(max_events):
        raise ValueError("max_events must be at least 1")
    if "," in region:
        raise ValueError("audit one region at a time so its cost is explicit")
    if dry_run:
        print(
            f"dry run: {day}, region {region}, market {market}; would make "
            f"one event-discovery call and at most {max_events} event-odds "
            f"calls (~10 credits each, plus discovery)."
        )
        return []
    if not key:
        raise ValueError("ODDS_API_KEY is required")
    request = _request if request is None else request

    discovery, discovery_headers = request(events_url(key, day))
    discovery_credits = _credit(discovery_headers.get("used"))
    done = {row.get("audit_id") for row in _load_rows(manifest_path)}
    candidates = []
    for event in events_on_day(discovery, day):
        snapshot = audit_snapshot(event["commence_time"], lead_minutes)
        audit_id = _audit_id(event.get("id", ""), snapshot, region, market)
        if audit_id not in done:
            candidates.append((event, snapshot, audit_id))
    selected = candidates[:int(max_events)]
    print(f"{len(events_on_day(discovery, day))} game(s) on {day}; "
          f"{len(done)} prior audit row(s); {len(selected)} new event(s) selected")
    if not selected:
        return []

    results = []
    for index, (event, snapshot, audit_id) in enumerate(selected):
        # The one discovery call is allocated to the first manifest row so
        # the resulting ledger reconciles exactly to the response headers.
        discovery_share = discovery_credits if index == 0 else 0
        row = _capture_one(key, event, snapshot, audit_id, region, market,
                           discovery_share, quotes_path, request)
        _append(manifest_path, AUDIT_FIELDS, [row])
        results.append(row)
        print(f"  {event.get('away_team')} @ {event.get('home_team')}: "
              f"{row['status']} — {row['book_count']} book(s), "
              f"{row['quote_count']} quote(s), "
              f"{row['odds_credits_used']} odds credits")

    total = sum(_credit(row["odds_credits_used"])
                + _credit(row["discovery_credits_used"]) for row in results)
    remaining = results[-1].get("credits_remaining", "unknown")
    print(f"recorded {len(results)} audit row(s); {total} measured credits; "
          f"{remaining} remaining")
    failures = [row for row in results if row["status"] == "failed"]
    if failures:
        raise RuntimeError(f"{len(failures)} first-inning audit request(s) failed")
    return results


def run_study(key, start, end, max_events=500, days_per_season=10,
              lead_minutes=10, region="us", market=FIRST_INNING_TOTALS_MARKET,
              manifest_path=DEFAULT_MANIFEST, quotes_path=DEFAULT_QUOTES,
              dry_run=False, request=None, fail_on_request_error=False):
    """Run a capped, evenly date-stratified historical period-market sample.

    The hard cap applies to expensive event-odds calls.  Historical event
    discovery is comparatively cheap but is still measured from the response
    headers and allocated to its first selected event in the audit manifest.
    """
    days = stratified_days(start, end, days_per_season)
    if not 1 <= int(max_events):
        raise ValueError("max_events must be at least 1")
    if "," in region:
        raise ValueError("study one region at a time so its cost is explicit")
    if dry_run:
        print(
            f"dry run: {len(days)} stratified date(s) in {start}..{end}; "
            f"at most {max_events} event-odds calls (~10 credits each) plus "
            f"up to {len(days)} discovery calls."
        )
        return []
    if not key:
        raise ValueError("ODDS_API_KEY is required")
    request = _request if request is None else request
    done = {row.get("audit_id") for row in _load_rows(manifest_path)}
    groups, discovery_total = [], 0
    for day in days:
        discovery, headers = request(events_url(key, day))
        credits = _credit(headers.get("used"))
        discovery_total += credits
        candidates = []
        for event in events_on_day(discovery, day):
            snapshot = audit_snapshot(event["commence_time"], lead_minutes)
            audit_id = _audit_id(event.get("id", ""), snapshot, region, market)
            if audit_id not in done:
                candidates.append((event, snapshot, audit_id))
        groups.append({"day": day, "credits": credits, "candidates": candidates})
    selected = _round_robin(groups, int(max_events))
    discovered = sum(len(group["candidates"]) for group in groups)
    print(f"{len(days)} stratified date(s), {discovered} uncaptured event(s); "
          f"{len(selected)} selected under the {max_events}-event cap")
    if not selected:
        return []

    results = []
    for index, (event, snapshot, audit_id, discovery_credits) in enumerate(selected, 1):
        row = _capture_one(key, event, snapshot, audit_id, region, market,
                           discovery_credits, quotes_path, request)
        _append(manifest_path, AUDIT_FIELDS, [row])
        results.append(row)
        if index == 1 or index % 25 == 0 or index == len(selected):
            offered = sum(item["status"] == "offered" for item in results)
            paid = sum(_credit(item["odds_credits_used"])
                       + _credit(item["discovery_credits_used"])
                       for item in results)
            print(f"  {index}/{len(selected)}: {offered} offered; "
                  f"{paid} recorded credits")

    # Normally every sampled day gets at least one selected event, so its
    # discovery credit lives in the manifest.  Keep the total explicit even
    # for a holiday/no-game day that had no row to carry its cost.
    event_credits = sum(_credit(row["odds_credits_used"]) for row in results)
    recorded_discovery = sum(_credit(row["discovery_credits_used"])
                             for row in results)
    total = event_credits + discovery_total
    offered = sum(row["status"] == "offered" for row in results)
    failed = sum(row["status"] == "failed" for row in results)
    remaining = results[-1].get("credits_remaining", "unknown")
    print(f"recorded {len(results)} event(s): {offered} offered, {failed} failed; "
          f"{total} measured credits ({event_credits} event + "
          f"{discovery_total} discovery; {recorded_discovery} allocated); "
          f"{remaining} remaining")
    if failed:
        failed_ids = ", ".join(row["event_id"] for row in results
                                 if row["status"] == "failed")
        message = (f"{failed} first-inning study request(s) failed "
                   f"({failed_ids}); successful and no-offer rows were recorded")
        if fail_on_request_error:
            raise RuntimeError(message)
        # A provider can have a missing historical event while every other
        # event is usable. Turning that into a process failure used to skip
        # the checkpoint step and discard hundreds of paid rows.
        print(f"WARNING: {message}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--date", help="one MLB game date to audit (YYYY-MM-DD)")
    target.add_argument("--start", help="first date of a stratified study")
    parser.add_argument("--end", help="last date of a stratified study; required with --start")
    parser.add_argument("--max-events", type=int, default=1,
                        help="hard event-odds call ceiling (default: 1)")
    parser.add_argument("--days-per-season", type=int, default=10,
                        help="evenly spaced dates per year for --start/--end")
    parser.add_argument("--lead-minutes", type=int, default=10,
                        help="snapshot this many minutes before first pitch")
    parser.add_argument("--region", default="us")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--quotes", default=str(DEFAULT_QUOTES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-request-error", action="store_true",
                        help="nonzero exit after preserving failed rows")
    args = parser.parse_args(argv)
    if args.date:
        run(os.environ.get("ODDS_API_KEY"), args.date, args.max_events,
            args.lead_minutes, args.region, manifest_path=args.manifest,
            quotes_path=args.quotes, dry_run=args.dry_run)
        return
    if not args.end:
        parser.error("--end is required with --start")
    run_study(os.environ.get("ODDS_API_KEY"), args.start, args.end,
              args.max_events, args.days_per_season, args.lead_minutes,
              args.region, manifest_path=args.manifest, quotes_path=args.quotes,
              dry_run=args.dry_run,
              fail_on_request_error=args.fail_on_request_error)


if __name__ == "__main__":
    main()
