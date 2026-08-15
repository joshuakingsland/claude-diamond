"""Append-only forward probe for the small price-movement signal.

This is deliberately separate from ``paper_ledger.csv``.  Outcome evidence
does not support a wager, but expanding-date tests show a small ability to
predict a later main price.  The probe records one timestamped observation per
game risk bucket so future sharp-close CLV can test that claim without calling
it a bet or cherry-picking the best capture after the fact.
"""

import argparse
import csv
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import (MAX_BOOK_QUOTE_AGE_MINUTES, MAX_LOCK_LEAD_MINUTES,
                    MAX_ODDS_AGE_MINUTES, MIN_CLV_SIGNAL,
                    MIN_LOCK_LEAD_MINUTES, MIN_MARKET_BOOKS)
from ledger import risk_bucket, unset


SIGNAL_FIELDS = [
    "wager_id", "captured_at", "event_id", "game_pk", "official_date",
    "commence_time", "home_team", "away_team", "market", "point", "side",
    "predicted_clv", "price", "book", "book_updated_at", "market_books",
    "risk_bucket", "model_version", "market_offset_version",
    "movement_model_version", "movement_target",
    "execution_status",
]


def screen(card, existing=None, now=None):
    existing = set(existing or set())
    now = now or datetime.now(timezone.utc)
    eligible = []
    for row in card.to_dict("records"):
        signal = row.get("predicted_clv")
        if unset(signal) or abs(float(signal)) < MIN_CLV_SIGNAL:
            continue
        early_probe = bool(int(row.get("movement_model_eligible", 0) or 0)) \
            and row.get("movement_target") == "24h_entry_to_20m_close"
        if not early_probe and not int(row.get("lineups_confirmed", 0) or 0):
            continue
        if int(row.get("market_books", 0) or 0) < MIN_MARKET_BOOKS:
            continue
        lead = int(row.get("lead_minutes", 0) or 0)
        if (not early_probe
                and not MIN_LOCK_LEAD_MINUTES <= lead <= MAX_LOCK_LEAD_MINUTES):
            continue
        captured = pd.to_datetime(row.get("odds_fetched_at"), utc=True,
                                  errors="coerce")
        if pd.isna(captured) or (now - captured).total_seconds() / 60 \
                > MAX_ODDS_AGE_MINUTES:
            continue
        side = "home" if float(signal) > 0 else "away"
        updated_column = ("best_price_home_updated_at" if side == "home"
                          else "best_price_away_updated_at")
        updated = pd.to_datetime(row.get(updated_column), utc=True,
                                 errors="coerce")
        if pd.isna(updated) or (now - updated).total_seconds() / 60 \
                > MAX_BOOK_QUOTE_AGE_MINUTES:
            continue
        bucket = risk_bucket(row)
        target = (row.get("movement_target") if early_probe else "lock_window")
        # Keep the legacy identifier stable so deploying the new probe cannot
        # append duplicates of the lock-window observations already recorded.
        identifier = (f"clv|{target}|{row['game_pk']}|{bucket}" if early_probe
                      else f"clv|{row['game_pk']}|{bucket}")
        if identifier in existing:
            continue
        eligible.append({
            "wager_id": identifier,
            "captured_at": f"{now:%Y-%m-%dT%H:%M:%SZ}",
            "event_id": row.get("event_id", ""),
            "game_pk": row["game_pk"],
            "official_date": row["official_date"],
            "commence_time": row["commence_time"],
            "home_team": row["home_team"], "away_team": row["away_team"],
            "market": row["market"],
            "point": "" if unset(row.get("point")) else row["point"],
            "side": side,
            "predicted_clv": round(abs(float(signal)), 6),
            "price": (row["best_price_home"] if side == "home"
                      else row["best_price_away"]),
            "book": (row["best_book_home"] if side == "home"
                     else row["best_book_away"]),
            "book_updated_at": row.get(updated_column, ""),
            "market_books": row["market_books"],
            "risk_bucket": bucket,
            "model_version": row.get("model_version", ""),
            "market_offset_version": row.get("market_offset_version", ""),
            "movement_model_version": row.get("movement_model_version", ""),
            "movement_target": target,
            "execution_status": ("paper_clv_probe" if early_probe
                                 else "paper_quote"),
        })
    eligible.sort(key=lambda row: -row["predicted_clv"])
    selected, buckets = [], set()
    for row in eligible:
        key = (str(row["game_pk"]), row["risk_bucket"])
        if key in buckets:
            continue
        buckets.add(key)
        selected.append(row)
    return selected


def append(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fields = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_fields != SIGNAL_FIELDS:
            # Append-only logs still need an explicit schema migration when a
            # provenance column is added. Appending a wider row under the old
            # header silently shifts columns and corrupts every future read.
            with tempfile.NamedTemporaryFile(
                    "w", newline="", encoding="utf-8", delete=False,
                    dir=path.parent, prefix=f".{path.name}.") as handle:
                writer = csv.DictWriter(handle, fieldnames=SIGNAL_FIELDS,
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing_rows)
                writer.writerows(rows)
                temporary = handle.name
            os.replace(temporary, path)
            return
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIGNAL_FIELDS,
                                extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", default="data/predictions_upcoming.csv")
    parser.add_argument("--out", default="data/clv_signals.csv")
    args = parser.parse_args(argv)
    card = pd.read_csv(args.card) if Path(args.card).exists() else pd.DataFrame()
    existing = pd.read_csv(args.out) if Path(args.out).exists() else pd.DataFrame()
    ids = set(existing["wager_id"].astype(str)) if len(existing) else set()
    rows = screen(card, ids) if len(card) else []
    append(args.out, rows)
    print(f"recorded {len(rows)} new CLV probe signal(s)")


if __name__ == "__main__":
    main()
