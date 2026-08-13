"""Resumable, per-game historical full-game odds capture.

The command is a dry run only when ``--dry-run`` is explicit. A completed
close manifest doubles as the event catalog for additional snapshot roles, so
an early backfill does not repeatedly buy and wait for date-discovery calls.
"""

import argparse
import os
from datetime import date, timedelta
from pathlib import Path

from config import MARKETS
from csv_collection import yearly_part
from first_inning_odds import (
    EVENT_ODDS_API, _append, _credit, _iso, _load_rows, _url,
    events_on_day, events_url, response_events,
)
from historical_odds import _request
from odds import _quote_rows, append_quote_log, paired_book_quotes


MANIFEST = Path("data/full_game_event_audit.csv")
QUOTES = Path("data/full_game_event_quotes")
FIELDS = [
    "audit_id", "event_id", "home_team", "away_team", "commence_time",
    "snapshot_role", "requested_snapshot", "returned_snapshot", "status",
    "quote_count", "odds_credits_used", "discovery_credits_used",
    "credits_remaining", "error",
]


def days(start, end):
    current = date.fromisoformat(start)
    final = date.fromisoformat(end)
    while current <= final:
        yield current.isoformat()
        current += timedelta(days=1)


def url(key, event, snapshot, region):
    return _url(
        f"{EVENT_ODDS_API}/{event}/odds", key, regions=region,
        markets=",".join(MARKETS), oddsFormat="american",
        date=snapshot.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def _audit_id(event_id, role, snapshot, region):
    return "|".join((str(event_id), role,
                     snapshot.strftime("%Y-%m-%dT%H:%M:%SZ"), region))


def _catalog_events(audit_rows, start, end):
    """Reuse completed close attempts as a zero-credit event catalog."""
    first = date.fromisoformat(start)
    final = date.fromisoformat(end)
    events = {}
    for row in audit_rows:
        if row.get("snapshot_role") != "close" or not row.get("event_id"):
            continue
        try:
            commence = _iso(row["commence_time"])
        except (KeyError, TypeError, ValueError):
            continue
        # Event discovery assigns a late West Coast game to the MLB slate
        # beginning at noon UTC, not to its following UTC calendar date.
        slate_day = (commence - timedelta(hours=12)).date()
        if not first <= slate_day <= final:
            continue
        events.setdefault(row["event_id"], {
            "id": row["event_id"],
            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),
            "commence_time": row["commence_time"],
        })
    return sorted(events.values(), key=lambda event: (
        event["commence_time"], event["id"]))


def _select_from_catalog(catalog, done, max_events, lead_minutes, role,
                         region):
    selected = []
    for event in catalog:
        snapshot = _iso(event["commence_time"]) - timedelta(
            minutes=lead_minutes)
        audit_id = _audit_id(event["id"], role, snapshot, region)
        if audit_id not in done:
            selected.append((event, snapshot, audit_id, 0))
            if len(selected) >= max_events:
                break
    return selected


def _discover_events(key, start, end, done, max_events, lead_minutes, role,
                     region):
    selected = []
    for day in days(start, end):
        payload, headers = _request(events_url(key, day))
        discovery = _credit(headers.get("used"))
        for event in events_on_day(payload, day):
            snapshot = _iso(event["commence_time"]) - timedelta(
                minutes=lead_minutes)
            audit_id = _audit_id(event["id"], role, snapshot, region)
            if audit_id in done or len(selected) >= max_events:
                continue
            selected.append((event, snapshot, audit_id, discovery))
            discovery = 0
        if len(selected) >= max_events:
            break
    return selected


def run(key, start, end, max_events, lead_minutes=20, role="close",
        region="us", manifest=MANIFEST, quotes=QUOTES, dry_run=True):
    if not 1 <= lead_minutes <= 1440 or max_events < 1:
        raise ValueError("invalid cap or lead")
    audit_rows = _load_rows(manifest)
    done = {row["audit_id"] for row in audit_rows}
    catalog = _catalog_events(audit_rows, start, end) if role != "close" else []
    catalog_selected = _select_from_catalog(
        catalog, done, max_events, lead_minutes, role, region) if catalog else []
    if dry_run:
        if catalog:
            calls = len(catalog_selected)
            print(f"dry run: {start}..{end}, {role}, {lead_minutes}m lead; "
                  f"{calls} event calls from the close catalog, "
                  f"~{calls * 30} credits; 0 discovery calls")
        else:
            estimate = min(max_events, len(list(days(start, end))) * 16) * 30
            print(f"dry run: {start}..{end}, {role}, {lead_minutes}m lead; "
                  f"at most {max_events} event calls, ~{estimate} credits "
                  "plus discovery")
        return []
    if not key:
        raise ValueError("ODDS_API_KEY is required")

    selected = (catalog_selected if catalog
                else _discover_events(key, start, end, done, max_events,
                                      lead_minutes, role, region))
    rows = []
    for event, snapshot, audit_id, discovery in selected:
        base = {
            "audit_id": audit_id,
            "event_id": event["id"],
            "home_team": event.get("home_team", ""),
            "away_team": event.get("away_team", ""),
            "commence_time": event["commence_time"],
            "snapshot_role": role,
            "requested_snapshot": snapshot.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "discovery_credits_used": discovery,
        }
        try:
            payload, headers = _request(url(
                key, event["id"], snapshot, region))
            priced = dict(event)
            returned = response_events(payload)
            if returned:
                priced.update(returned[0])
            paired = paired_book_quotes(priced, region, MARKETS)
            prefix = "quotes" if role == "close" else f"quotes_{role}"
            quote_path = yearly_part(
                quotes, event["commence_time"], prefix=prefix)
            append_quote_log(quote_path, _quote_rows(
                priced, paired,
                payload.get("timestamp") or base["requested_snapshot"]))
            row = {
                **base,
                "returned_snapshot": payload.get("timestamp", ""),
                "status": "offered" if paired else "no_offer",
                "quote_count": len(paired),
                "odds_credits_used": _credit(headers.get("used")),
                "credits_remaining": headers.get("remaining", ""),
                "error": "",
            }
        except Exception as error:  # Persist failures so reruns never repay.
            row = {
                **base, "returned_snapshot": "", "status": "failed",
                "quote_count": 0, "odds_credits_used": 0,
                "credits_remaining": "", "error": repr(error),
            }
        _append(manifest, FIELDS, [row])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--lead-minutes", type=int, default=20)
    parser.add_argument("--snapshot-role", choices=["early", "close"],
                        default="close")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    run(os.getenv("ODDS_API_KEY"), arguments.start, arguments.end,
        arguments.max_events, arguments.lead_minutes,
        arguments.snapshot_role, dry_run=arguments.dry_run)


if __name__ == "__main__":
    main()
