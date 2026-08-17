"""Resumable historical opening-price ladder for YRFI/NRFI research.

The completed first-inning archive contains one ten-minute snapshot per event.
This collector reuses those exact provider event IDs and requests predeclared
earlier snapshots without paying for event discovery again.  It is a data
collector, not a betting strategy.
"""

import argparse
import json
import os
from pathlib import Path

from first_inning_odds import (
    AUDIT_FIELDS, _append, _audit_id, _capture_one, _credit, _load_rows,
    audit_snapshot,
)
from historical_odds import _request
from config import FIRST_INNING_TOTALS_MARKET


DEFAULT_CLOSE_MANIFEST = Path("data/first_inning_audit.csv")
DEFAULT_CLOSE_QUOTES = Path("data/first_inning_quotes.csv")
DEFAULT_RESULTS = Path("data/first_inning_results.csv")
DEFAULT_MANIFEST = Path("data/first_inning_open_audit.csv")
DEFAULT_QUOTES = Path("data/first_inning_open_quotes.csv")
DEFAULT_GATE_REPORT = Path("first_inning_open_evaluation.json")
DEFAULT_LEADS = (1440, 720, 360, 180, 60)
DEFAULT_SEASONS = (2023, 2024)
CONFIRMATION_YEAR = 2025
CONFIRMATION_UNLOCK_STATUS = "candidate_locked_2025_price_confirmation_sealed"


def parse_leads(value):
    """Return unique, earliest-to-latest lead minutes."""
    raw = value if isinstance(value, (list, tuple)) else str(value).split(",")
    try:
        leads = tuple(int(item) for item in raw)
    except (TypeError, ValueError) as error:
        raise ValueError("lead minutes must be comma-separated integers") from error
    if not leads or any(lead < 1 or lead > 1440 for lead in leads):
        raise ValueError("every lead minute must be between 1 and 1,440")
    if len(set(leads)) != len(leads):
        raise ValueError("lead minutes must be unique")
    return tuple(sorted(leads, reverse=True))


def parse_seasons(value):
    raw = value if isinstance(value, (list, tuple)) else str(value).split(",")
    try:
        seasons = tuple(int(item) for item in raw)
    except (TypeError, ValueError) as error:
        raise ValueError("seasons must be comma-separated years") from error
    if not seasons or any(season < 2023 or season > 2025 for season in seasons):
        raise ValueError("opening research seasons must be between 2023 and 2025")
    if len(set(seasons)) != len(seasons):
        raise ValueError("seasons must be unique")
    return tuple(sorted(seasons))


def qualified_close_event_ids(quotes_path=DEFAULT_CLOSE_QUOTES,
                              results_path=DEFAULT_RESULTS):
    """Return one settled, multi-book provider event per MLB game.

    The filter uses only close availability and settlement status, never the
    realised YRFI value. Events outside it cannot supply the study's required
    close benchmark, so buying their opening ladder would be pure waste.
    """
    books = {}
    for row in _load_rows(quotes_path):
        try:
            point = float(row.get("point", "nan"))
        except (TypeError, ValueError):
            continue
        if (row.get("market") != FIRST_INNING_TOTALS_MARKET
                or abs(point - 0.5) > 1e-9):
            continue
        event_id = str(row.get("event_id", "")).strip()
        book = str(row.get("book_key", "")).strip()
        if event_id and book:
            books.setdefault(event_id, set()).add(book)

    by_game = {}
    for row in _load_rows(results_path):
        if row.get("result_status") != "final":
            continue
        if (row.get("game_type") or "R") != "R":
            continue
        event_id = str(row.get("event_id", "")).strip()
        game_pk = str(row.get("game_pk", "")).strip()
        book_count = len(books.get(event_id, ()))
        if not event_id or not game_pk or book_count < 2:
            continue
        candidate = (-book_count, event_id)
        if game_pk not in by_game or candidate < by_game[game_pk]:
            by_game[game_pk] = candidate
    return {event_id for _, event_id in by_game.values()}


def source_events(path=DEFAULT_CLOSE_MANIFEST, seasons=None,
                  eligible_event_ids=None):
    """Load one immutable event description from each closing audit row."""
    allowed_seasons = set(seasons) if seasons is not None else None
    eligible = (set(eligible_event_ids)
                if eligible_event_ids is not None else None)
    events = {}
    for row in _load_rows(path):
        event_id = str(row.get("event_id", "")).strip()
        commence = str(row.get("commence_time", "")).strip()
        if not event_id or not commence:
            continue
        if eligible is not None and event_id not in eligible:
            continue
        try:
            season = int(str(row.get("requested_date", commence[:10]))[:4])
        except (TypeError, ValueError):
            continue
        if allowed_seasons is not None and season not in allowed_seasons:
            continue
        events.setdefault(event_id, {
            "id": event_id,
            "commence_time": commence,
            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),
            "requested_date": row.get("requested_date", commence[:10]),
        })
    return sorted(events.values(), key=lambda event: (
        event["requested_date"], event["commence_time"], event["id"]))


def stratified_event_order(events):
    """Interleave seasons so every checkpoint has temporal coverage."""
    groups = {}
    for event in events:
        try:
            season = int(str(event["requested_date"])[:4])
        except (TypeError, ValueError):
            season = 0
        groups.setdefault(season, []).append(event)
    positions = {season: 0 for season in groups}
    ordered = []
    while True:
        progressed = False
        for season in sorted(groups):
            position = positions[season]
            if position >= len(groups[season]):
                continue
            ordered.append(groups[season][position])
            positions[season] += 1
            progressed = True
        if not progressed:
            return ordered


def pending_calls(events, leads, done, region="us",
                  market=FIRST_INNING_TOTALS_MARKET):
    """Return complete event ladders in deterministic, resumable order."""
    pending = []
    for event in stratified_event_order(events):
        for lead in leads:
            snapshot = audit_snapshot(event["commence_time"], lead)
            audit_id = _audit_id(event["id"], snapshot, region, market)
            if audit_id not in done:
                pending.append((event, lead, snapshot, audit_id))
    return pending


def run(key, max_calls=1000, lead_minutes=DEFAULT_LEADS, region="us",
        market=FIRST_INNING_TOTALS_MARKET,
        close_manifest=DEFAULT_CLOSE_MANIFEST,
        close_quotes=DEFAULT_CLOSE_QUOTES, results_path=DEFAULT_RESULTS,
        manifest_path=DEFAULT_MANIFEST,
        quotes_path=DEFAULT_QUOTES, seasons=DEFAULT_SEASONS,
        gate_report=DEFAULT_GATE_REPORT, eligible_event_ids=None,
        dry_run=False, request=None):
    """Capture at most ``max_calls`` missing event/snapshot pairs."""
    if int(max_calls) < 1:
        raise ValueError("max_calls must be at least 1")
    if "," in region:
        raise ValueError("capture one region at a time so cost stays explicit")
    leads = parse_leads(lead_minutes)
    seasons = parse_seasons(seasons)
    if CONFIRMATION_YEAR in seasons and not dry_run:
        report_path = Path(gate_report)
        report = (json.loads(report_path.read_text(encoding="utf-8"))
                  if report_path.exists() else {})
        if report.get("status") != CONFIRMATION_UNLOCK_STATUS:
            raise RuntimeError(
                "2025 confirmation is sealed until the complete 2023/2024 "
                "archive produces a locked development candidate")
    if eligible_event_ids is None:
        eligible_event_ids = qualified_close_event_ids(
            close_quotes, results_path)
    if not eligible_event_ids:
        raise ValueError("no settled multi-book closing events are eligible")
    events = source_events(
        close_manifest, seasons=seasons,
        eligible_event_ids=eligible_event_ids)
    if not events:
        raise ValueError("closing manifest contains no reusable event IDs")
    done = {row.get("audit_id") for row in _load_rows(manifest_path)}
    candidates = pending_calls(events, leads, done, region, market)
    selected = candidates[:int(max_calls)]
    estimate = len(selected) * 10
    print(f"{len(events)} closing events in {','.join(map(str, seasons))}; "
          f"{len(leads)} frozen opening rungs; "
          f"{len(done)} prior opening attempts; {len(selected)} calls selected")
    if dry_run:
        print(f"dry run: at most {estimate:,} event-odds credits; "
              "zero event-discovery credits")
        return []
    if not key:
        raise ValueError("ODDS_API_KEY is required")
    request = _request if request is None else request

    results = []
    for index, (event, lead, snapshot, audit_id) in enumerate(selected, 1):
        row = _capture_one(
            key, event, snapshot, audit_id, region, market, 0,
            quotes_path, request)
        _append(manifest_path, AUDIT_FIELDS, [row])
        results.append(row)
        if index == 1 or index % 25 == 0 or index == len(selected):
            offered = sum(item["status"] == "offered" for item in results)
            paid = sum(_credit(item["odds_credits_used"]) for item in results)
            print(f"  {index}/{len(selected)}: {offered} offered; "
                  f"{paid} measured credits; last rung {lead}m")

    offered = sum(row["status"] == "offered" for row in results)
    failed = sum(row["status"] == "failed" for row in results)
    no_offer = sum(row["status"] == "no_offer" for row in results)
    paid = sum(_credit(row["odds_credits_used"]) for row in results)
    remaining = results[-1].get("credits_remaining", "unknown") if results else "unknown"
    print(f"recorded {len(results)} attempts: {offered} offered, "
          f"{no_offer} no-offer, {failed} failed; {paid} credits; "
          f"{remaining} remaining")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-calls", type=int, default=1000)
    parser.add_argument("--lead-minutes", default=",".join(
        str(value) for value in DEFAULT_LEADS))
    parser.add_argument("--seasons", default=",".join(
        str(value) for value in DEFAULT_SEASONS))
    parser.add_argument("--region", default="us")
    parser.add_argument("--close-manifest", default=str(DEFAULT_CLOSE_MANIFEST))
    parser.add_argument("--close-quotes", default=str(DEFAULT_CLOSE_QUOTES))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--quotes", default=str(DEFAULT_QUOTES))
    parser.add_argument("--gate-report", default=str(DEFAULT_GATE_REPORT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run(os.environ.get("ODDS_API_KEY"), args.max_calls, args.lead_minutes,
        args.region, close_manifest=args.close_manifest, seasons=args.seasons,
        close_quotes=args.close_quotes, results_path=args.results,
        manifest_path=args.manifest, quotes_path=args.quotes,
        gate_report=args.gate_report, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
