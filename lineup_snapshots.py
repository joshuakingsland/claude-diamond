"""Append-only snapshots of confirmed batting orders before first pitch.

The runs model does not yet contain batter-level features, but the execution
policy must still distinguish a truly late-information card from one priced
before lineups existed.  StatsAPI exposes batting orders in the live game feed
once clubs submit them.  This module records the first observed version (and
any later change) without rewriting what an earlier decision knew.
"""

import argparse
import csv
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


FEED = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
LINEUP_FIELDS = [
    "snapshot_id", "captured_at", "game_pk", "game_date_utc", "status",
    "home_lineup_ids", "away_lineup_ids", "home_lineup_names",
    "away_lineup_names", "confirmed",
]


def _get(game_pk, timeout=30):
    request = urllib.request.Request(FEED.format(game_pk=game_pk),
                                     headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def parse(payload, captured_at):
    game = payload.get("gameData", {}) or {}
    live = payload.get("liveData", {}) or {}
    boxscore = live.get("boxscore", {}) or {}
    teams = boxscore.get("teams", {}) or {}
    players = game.get("players", {}) or {}

    def order(side):
        ids = (teams.get(side, {}) or {}).get("battingOrder", []) or []
        ids = [str(value) for value in ids]
        names = []
        for player_id in ids:
            person = players.get(f"ID{player_id}", {}) or {}
            names.append(person.get("fullName", ""))
        return ids, names

    home_ids, home_names = order("home")
    away_ids, away_names = order("away")
    if len(home_ids) < 9 or len(away_ids) < 9:
        return None
    game_pk = game.get("game", {}).get("pk") or payload.get("gamePk")
    date_time = game.get("datetime", {}) or {}
    status = game.get("status", {}) or {}
    identity = "|".join([str(game_pk), *home_ids, "away", *away_ids])
    return {
        "snapshot_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
        "captured_at": captured_at,
        "game_pk": game_pk,
        "game_date_utc": date_time.get("dateTime", ""),
        "status": status.get("detailedState",
                             status.get("abstractGameState", "")),
        "home_lineup_ids": "|".join(home_ids),
        "away_lineup_ids": "|".join(away_ids),
        "home_lineup_names": "|".join(home_names),
        "away_lineup_names": "|".join(away_names),
        "confirmed": 1,
    }


def append(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    known = {row.get("snapshot_id") for row in existing}
    new = [row for row in rows if row.get("snapshot_id") not in known]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LINEUP_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing + new)
    os.replace(temporary, path)
    return len(new)


def confirmed_games(snapshots, as_of=None):
    if snapshots is None or not len(snapshots) or "captured_at" not in snapshots:
        return set()
    frame = snapshots.copy()
    frame["_captured"] = pd.to_datetime(frame["captured_at"], utc=True,
                                         errors="coerce")
    cutoff = pd.Timestamp(as_of or datetime.now(timezone.utc))
    cutoff = (cutoff.tz_localize("UTC") if cutoff.tzinfo is None
              else cutoff.tz_convert("UTC"))
    frame = frame[frame["_captured"].notna()
                  & (frame["_captured"] <= cutoff)]
    if "confirmed" in frame:
        frame = frame[pd.to_numeric(frame["confirmed"], errors="coerce") == 1]
    return {str(value) for value in frame["game_pk"]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--out", default="data/lineup_snapshots.csv")
    parser.add_argument("--max-lead-hours", type=float, default=5.0)
    args = parser.parse_args(argv)
    games = pd.read_csv(args.games)
    now = datetime.now(timezone.utc)
    starts = pd.to_datetime(games["game_date_utc"], utc=True, errors="coerce")
    lead = (starts - pd.Timestamp(now)).dt.total_seconds() / 3600.0
    eligible = games[games["home_score"].isna()
                     & lead.between(-0.25, args.max_lead_hours)]
    captured = f"{now:%Y-%m-%dT%H:%M:%SZ}"
    rows, failures = [], 0
    for game_pk in eligible["game_pk"]:
        try:
            row = parse(_get(game_pk), captured)
        except Exception:  # transient and retried on the next capture
            failures += 1
            continue
        if row is not None:
            rows.append(row)
    added = append(args.out, rows)
    print(f"checked {len(eligible)} games; {len(rows)} confirmed lineups, "
          f"{added} new versions, {failures} fetch failures")


if __name__ == "__main__":
    main()
