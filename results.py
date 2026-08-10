"""MLB schedule and result ingestion from the public StatsAPI.

One request returns a whole season, so history is cheap and there is no
scraping and no API key. The output is a flat game table keyed by `game_pk`,
the stable StatsAPI identifier. Team and player identity use numeric MLBAM
ids throughout; display names are never used as keys.

Only regular-season games are ingested by default. Spring training and the
postseason are different populations — different rosters, different usage
patterns, different market liquidity — and mixing them into a training set
quietly changes what the model is estimating.
"""

import argparse
import csv
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# The first season this project models. The last one is never written down,
# because a hardcoded end year is a bug with a delayed fuse: `2021-2025` was
# correct when it was typed and silently stopped including the current season
# the moment 2026 started.
FIRST_SEASON = 2021

SCHEDULE_API = "https://statsapi.mlb.com/api/v1/schedule"
HYDRATE = "probablePitcher,venue,linescore,weather"

GAME_FIELDS = [
    "game_pk", "official_date", "game_date_utc", "season", "game_type",
    "double_header", "game_number", "series_game_number",
    "day_night", "scheduled_innings", "status",
    "home_team_id", "home_team_name", "away_team_id", "away_team_name",
    "home_score", "away_score", "home_win", "total_runs", "run_diff",
    "home_sp_id", "home_sp_name", "away_sp_id", "away_sp_name",
    "venue_id", "venue_name",
    "reported_condition", "reported_temp_f", "reported_wind",
    "innings_played", "home_batted_ninth",
]


def _get(url, timeout=60):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def fetch_schedule(start_date, end_date, game_type="R", timeout=60):
    """Return raw StatsAPI game records for a date range."""
    query = urllib.parse.urlencode({
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
        "gameType": game_type,
        "hydrate": HYDRATE,
    })
    payload = _get(f"{SCHEDULE_API}?{query}", timeout=timeout)
    return [game for date in payload.get("dates", [])
            for game in date.get("games", [])]


def _optional_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _side(game, side):
    return game.get("teams", {}).get(side, {}) or {}


def _pitcher(game, side):
    return _side(game, side).get("probablePitcher") or {}


def parse_game(game):
    """Flatten one StatsAPI record into a row, or None if it is unusable.

    A game is only usable as a training or settlement row when it reached a
    final state with both scores present. Postponed, suspended, and cancelled
    games are dropped rather than defaulted, so a missing result can never be
    read as a nil-nil tie.
    """
    status = (game.get("status", {}) or {}).get("abstractGameState", "")
    home, away = _side(game, "home"), _side(game, "away")
    home_score = _optional_int(home.get("score"))
    away_score = _optional_int(away.get("score"))
    linescore = game.get("linescore", {}) or {}
    innings = linescore.get("innings", []) or []
    weather = game.get("weather", {}) or {}
    venue = game.get("venue", {}) or {}

    row = {
        "game_pk": game.get("gamePk"),
        "official_date": game.get("officialDate", ""),
        "game_date_utc": game.get("gameDate", ""),
        "season": game.get("season", ""),
        "game_type": game.get("gameType", ""),
        "double_header": game.get("doubleHeader", "N"),
        "game_number": game.get("gameNumber", 1),
        "series_game_number": game.get("seriesGameNumber", ""),
        "day_night": game.get("dayNight", ""),
        "scheduled_innings": game.get("scheduledInnings", 9),
        "status": status,
        "home_team_id": (home.get("team", {}) or {}).get("id"),
        "home_team_name": (home.get("team", {}) or {}).get("name", ""),
        "away_team_id": (away.get("team", {}) or {}).get("id"),
        "away_team_name": (away.get("team", {}) or {}).get("name", ""),
        "home_score": home_score,
        "away_score": away_score,
        "home_sp_id": _pitcher(game, "home").get("id"),
        "home_sp_name": _pitcher(game, "home").get("fullName", ""),
        "away_sp_id": _pitcher(game, "away").get("id"),
        "away_sp_name": _pitcher(game, "away").get("fullName", ""),
        "venue_id": venue.get("id"),
        "venue_name": venue.get("name", ""),
        "reported_condition": weather.get("condition", ""),
        "reported_temp_f": weather.get("temp", ""),
        "reported_wind": weather.get("wind", ""),
        "innings_played": len(innings) if innings else None,
    }
    if status != "Final" or home_score is None or away_score is None:
        # The raw scores go too, not just the derived columns. StatsAPI opens
        # a linescore at 0-0 as soon as a game is close to starting, so a
        # scheduled game arrives carrying a real-looking nil-nil. Everything
        # downstream tests `home_score.notna()` to mean "this game happened":
        # `models.fit` would train on it as a genuine shutout and
        # `predict_upcoming` would treat tonight's card as already played.
        row.update({"home_score": None, "away_score": None, "home_win": None,
                    "total_runs": None, "run_diff": None,
                    "home_batted_ninth": None})
        return row
    row["home_win"] = int(home_score > away_score)
    row["total_runs"] = home_score + away_score
    row["run_diff"] = home_score - away_score
    # The home team does not bat in the bottom of the final inning when it is
    # already ahead. That truncation is real and a run-scoring model that
    # ignores it will systematically under-predict home scoring.
    row["home_batted_ninth"] = _home_batted_last(innings)
    return row


def _home_batted_last(innings):
    """Did the home team bat in the bottom of the final inning?

    StatsAPI omits the `runs` key from the home half when it was never played,
    so this is directly observable rather than inferred. In 2024 the home team
    did not bat in the last inning of 1,057 of 2,430 games. Ignoring that
    truncation biases any run-scoring model downward for the home side, which
    then leaks straight into the moneyline and the total.
    """
    if not innings:
        return None
    return int("runs" in (innings[-1].get("home") or {}))


def build_table(games):
    rows = [parse_game(game) for game in games]
    return [row for row in rows if row["game_pk"] is not None]


def default_seasons(today=None):
    """Every season this project models, through the current one."""
    year = (today or datetime.now(timezone.utc)).year
    return f"{FIRST_SEASON}-{max(year, FIRST_SEASON)}"


def parse_seasons(text):
    if "-" in text:
        first, last = (int(part) for part in text.split("-", 1))
        return list(range(first, last + 1))
    return [int(text)]


def merge_table(rows, path):
    """Write the fetched seasons; leave every season not fetched alone.

    `write_table` replaces the file, which makes a narrow ``--seasons`` a
    delete rather than a no-op. That is how the entire 2026 season vanished:
    the weekly workflow passed a stale ``2021-2025``, the ingester did exactly
    as told, and the live card went empty because no game on the board existed
    in the table any more.

    Correcting the year would have fixed that instance. This makes the class
    of mistake impossible instead — a season can only be rewritten by a fetch
    that actually returned games for it.

    Fetched seasons are replaced wholesale rather than merged row by row, so a
    game cancelled upstream really does disappear. And the set comes from the
    rows in hand, not from what was requested, so a failed or empty fetch
    preserves what is already there instead of erasing it.
    """
    path = Path(path)
    fetched = {str(row.get("season")) for row in rows
               if row.get("season") not in (None, "")}
    kept = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            kept = [row for row in csv.DictReader(handle)
                    if str(row.get("season")) not in fetched]
    combined = kept + list(rows)
    combined.sort(key=lambda row: (str(row.get("season")),
                                   str(row.get("official_date")),
                                   str(row.get("game_pk"))))
    return write_table(combined, path), len(kept)


def write_table(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GAME_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return len(rows)


def season_bounds(season):
    """Generous bounds; StatsAPI only returns games that exist."""
    return f"{season}-02-15", f"{season}-11-15"


def deduplicate(rows):
    """One row per game_pk.

    A postponed game is returned twice by the schedule endpoint, under its
    original date and again under the date it was actually played, sharing a
    game_pk. Left alone this is not a cosmetic duplicate: `features.py` walks
    the table and folds each row's result into team state, so a duplicated
    game counts twice toward Elo, run rates, and park factors, and its weather
    is fetched for a date the game was not played on.

    The kept row is the one whose UTC start agrees with its official date,
    since that is the pair the rest of the pipeline joins on — `features.py`
    keys off `official_date` and `market.py` matches on `game_date_utc`, and a
    row where the two disagree would send them to different games. A played
    row beats an unplayed one, and a later start breaks any remaining tie.
    """
    def rank(row):
        agrees = str(row.get("game_date_utc", ""))[:10] == row.get("official_date")
        played = row.get("home_score") not in (None, "")
        return (agrees, played, str(row.get("game_date_utc", "")))

    best = {}
    for row in rows:
        key = row.get("game_pk")
        if key not in best or rank(row) > rank(best[key]):
            best[key] = row
    return list(best.values())


def fetch_seasons(seasons, game_type="R"):
    rows = []
    for season in seasons:
        start, end = season_bounds(season)
        games = fetch_schedule(start, end, game_type=game_type)
        parsed = deduplicate(build_table(games))
        final = [row for row in parsed if row["status"] == "Final"]
        print(f"  {season}: {len(parsed)} scheduled, {len(final)} final")
        rows.extend(parsed)
    return deduplicate(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default=None,
                        help="inclusive range, e.g. 2021-2026; defaults to "
                             "every season through the current one")
    parser.add_argument("--game-type", default="R")
    parser.add_argument("--out", default="data/games.csv")
    args = parser.parse_args()
    seasons = parse_seasons(args.seasons or default_seasons())
    print(f"fetching seasons {seasons}")
    rows = fetch_seasons(seasons, game_type=args.game_type)
    if not rows:
        raise SystemExit("StatsAPI returned no games; leaving the table alone")
    count, kept = merge_table(rows, args.out)
    print(f"wrote {count} games to {args.out} "
          f"({len(rows)} fetched, {kept} kept from seasons not fetched)")


if __name__ == "__main__":
    main()
