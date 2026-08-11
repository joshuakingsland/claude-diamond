"""Append-only pregame snapshots of probable starters and game status.

``games.csv`` is a regenerated table and therefore tells us the latest answer,
not what the pricing process knew.  A late scratch can otherwise rewrite the
starter attached to an old card.  This log preserves each observed assignment
with its capture time; serving selects the latest snapshot available before it
prices the card.  Historical assignments that were never captured are not
reconstructed or labelled point-in-time.
"""

import argparse
import csv
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from results import fetch_schedule


SNAPSHOT_FIELDS = [
    "snapshot_id", "captured_at", "game_pk", "official_date",
    "game_date_utc", "status", "home_team_id", "away_team_id",
    "home_sp_id", "home_sp_name", "away_sp_id", "away_sp_name",
]


def snapshot_row(game, captured_at):
    teams = game.get("teams", {}) or {}

    def side(name):
        block = teams.get(name, {}) or {}
        team = block.get("team", {}) or {}
        pitcher = block.get("probablePitcher", {}) or {}
        return team, pitcher

    home, home_sp = side("home")
    away, away_sp = side("away")
    status = game.get("status", {}) or {}
    values = (
        captured_at, game.get("gamePk"), game.get("gameDate", ""),
        status.get("detailedState", status.get("abstractGameState", "")),
        home_sp.get("id", ""), away_sp.get("id", ""),
    )
    return {
        "snapshot_id": hashlib.sha256(
            "|".join(str(value) for value in values).encode()).hexdigest()[:20],
        "captured_at": captured_at,
        "game_pk": game.get("gamePk"),
        "official_date": game.get("officialDate", ""),
        "game_date_utc": game.get("gameDate", ""),
        "status": status.get("detailedState",
                             status.get("abstractGameState", "")),
        "home_team_id": home.get("id", ""),
        "away_team_id": away.get("id", ""),
        "home_sp_id": home_sp.get("id", ""),
        "home_sp_name": home_sp.get("fullName", ""),
        "away_sp_id": away_sp.get("id", ""),
        "away_sp_name": away_sp.get("fullName", ""),
    }


def append(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    known = {row.get("snapshot_id") for row in existing}
    combined = existing + [row for row in rows
                           if row.get("snapshot_id") not in known]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)
    temporary.replace(path)
    return len(combined) - len(existing)


def apply_probable_snapshots(games, snapshots, as_of=None):
    """Overlay latest captured pregame starters onto unfinished games."""
    if snapshots is None or not len(snapshots):
        return games.copy()
    required = {"captured_at", "game_pk", "home_sp_id", "away_sp_id"}
    if not required.issubset(snapshots.columns):
        return games.copy()
    frame = snapshots.copy()
    frame["_captured"] = pd.to_datetime(frame["captured_at"], utc=True,
                                         errors="coerce")
    cutoff = pd.Timestamp(as_of or datetime.now(timezone.utc))
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    frame = frame[frame["_captured"].notna()
                  & (frame["_captured"] <= cutoff)]
    if not len(frame):
        return games.copy()
    latest = (frame.sort_values("_captured")
              .drop_duplicates("game_pk", keep="last")
              .set_index("game_pk"))
    out = games.copy()
    for index, game in out.iterrows():
        if pd.notna(game.get("home_score")) or game["game_pk"] not in latest.index:
            continue
        snap = latest.loc[game["game_pk"]]
        for column in ("home_sp_id", "home_sp_name", "away_sp_id",
                       "away_sp_name"):
            if column in snap and pd.notna(snap[column]) \
                    and str(snap[column]).strip() != "":
                out.at[index, column] = snap[column]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/schedule_snapshots.csv")
    parser.add_argument("--days", type=int, default=2)
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc)
    last = now + timedelta(days=max(args.days, 0))
    captured = f"{now:%Y-%m-%dT%H:%M:%SZ}"
    games = fetch_schedule(now.date().isoformat(), last.date().isoformat())
    rows = [snapshot_row(game, captured) for game in games]
    added = append(args.out, rows)
    print(f"captured {len(rows)} schedule rows; appended {added}")


if __name__ == "__main__":
    main()
