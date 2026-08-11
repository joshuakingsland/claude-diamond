"""Settle first-inning historical audit rows from MLB's public linescores.

The Odds API identifies a market event; MLB StatsAPI identifies the played
game.  This module records the match and the two first-inning half scores
instead of inferring a YRFI/NRFI result from a full-game box score.  It does
not construct historical batting-order snapshots or a predictive model.
"""

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from results import fetch_schedule


RESULT_FIELDS = [
    "event_id", "game_pk", "official_date", "commence_time", "home_team",
    "away_team", "first_inning_home_runs", "first_inning_away_runs",
    "first_inning_total", "yrfi", "nrfi", "result_status", "result_source",
    "fetched_at", "error",
]


def _read(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS,
                                extrasaction="ignore")
        if header:
            writer.writeheader()
        writer.writerows(rows)


def _team_key(name):
    return "".join(char for char in str(name).lower() if char.isalnum())


def _iso(value):
    moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return (moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None
            else moment.astimezone(timezone.utc))


def candidate_dates(event):
    """Return the UTC date and its preceding local-MLB calendar date."""
    moment = _iso(event["commence_time"])
    return {(moment + timedelta(days=offset)).date().isoformat()
            for offset in (0, -1)}


def _side(game, side):
    return ((game.get("teams", {}) or {}).get(side, {}) or {})


def match_game(event, games):
    """Match sides exactly, then choose the closest scheduled first pitch."""
    home, away = _team_key(event.get("home_team")), _team_key(event.get("away_team"))
    matched = [game for game in games
               if _team_key(_side(game, "home").get("team", {}).get("name")) == home
               and _team_key(_side(game, "away").get("team", {}).get("name")) == away]
    if not matched:
        return None
    target = _iso(event["commence_time"])
    def gap(game):
        try:
            return abs((_iso(game.get("gameDate", "")) - target).total_seconds())
        except (TypeError, ValueError):
            return float("inf")
    candidate = min(matched, key=gap)
    # A day shift is useful for a late Pacific game; a team-name collision
    # days away is not.  Doubleheaders remain distinct by first-pitch time.
    return candidate if gap(candidate) <= 12 * 60 * 60 else None


def _integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def label_game(game):
    """Return the first-inning label or a reason it is not settled."""
    state = (game.get("status", {}) or {}).get("abstractGameState", "")
    if state != "Final":
        return None, "not_final"
    innings = (game.get("linescore", {}) or {}).get("innings", []) or []
    first = next((inning for inning in innings
                  if _integer(inning.get("num")) == 1), None)
    if first is None:
        return None, "missing_first_inning"
    home = _integer((first.get("home") or {}).get("runs"))
    away = _integer((first.get("away") or {}).get("runs"))
    if home is None or away is None:
        return None, "incomplete_first_inning"
    total = home + away
    return {
        "first_inning_home_runs": home,
        "first_inning_away_runs": away,
        "first_inning_total": total,
        "yrfi": int(total > 0),
        "nrfi": int(total == 0),
    }, "final"


def run(audit_path="data/first_inning_audit.csv",
        out_path="data/first_inning_results.csv", fetch=None):
    audits = [row for row in _read(audit_path) if row.get("status") == "offered"]
    existing = {row.get("event_id") for row in _read(out_path)}
    pending = [row for row in audits if row.get("event_id") not in existing]
    if not pending:
        print("no unlabelled offered first-inning events")
        return []
    fetch = fetch_schedule if fetch is None else fetch
    schedules = {}
    for day in sorted({day for event in pending for day in candidate_dates(event)}):
        schedules[day] = fetch(day, day)
    now = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
    rows = []
    for event in pending:
        games = [game for day in candidate_dates(event) for game in schedules[day]]
        game = match_game(event, games)
        base = {
            "event_id": event.get("event_id", ""),
            "commence_time": event.get("commence_time", ""),
            "home_team": event.get("home_team", ""),
            "away_team": event.get("away_team", ""),
            "result_source": "mlb-statsapi-linescore",
            "fetched_at": now,
        }
        if game is None:
            rows.append({**base, "result_status": "unmatched",
                         "error": "no same-sides StatsAPI game within 12h"})
            continue
        label, status = label_game(game)
        rows.append({
            **base,
            "game_pk": game.get("gamePk", ""),
            "official_date": game.get("officialDate", ""),
            "result_status": status,
            **(label or {}),
            "error": "" if label else status,
        })
    _append(out_path, rows)
    settled = sum(row["result_status"] == "final" for row in rows)
    print(f"wrote {len(rows)} first-inning result row(s); {settled} final labels")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="data/first_inning_audit.csv")
    parser.add_argument("--out", default="data/first_inning_results.csv")
    args = parser.parse_args(argv)
    run(args.audit, args.out)


if __name__ == "__main__":
    main()
