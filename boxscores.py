"""Per-start pitcher lines and bullpen usage, from StatsAPI boxscores.

This is the first genuinely new information added to the model rather than a
rearrangement of what it already had. Everything in `features.py` so far is
derived from runs scored and allowed, which is close to what the market has
already priced. A starter's own strikeout, walk and home-run rates, and the
state of the bullpen behind him, are different inputs.

Two rules carry over unchanged:

- Point-in-time. A pitcher's line for game N may only inform games after N.
  The accumulation lives in `features.py`; this module only records what
  happened, keyed by game_pk and date.
- Fail closed. A game whose boxscore cannot be parsed is skipped and counted,
  never defaulted to zeros, because a zeroed pitching line reads as a perfect
  start.

One request per game, roughly 2,400 a season, free and keyless. Resumable:
game_pks already recorded are skipped.
"""

import argparse
import csv
import json
import os
import time
import urllib.request
from pathlib import Path

BOX_API = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"

PITCHER_FIELDS = [
    "game_pk", "official_date", "team_id", "player_id", "is_starter",
    "outs", "batters_faced", "hits", "runs", "earned_runs", "home_runs",
    "walks", "hit_batsmen", "strike_outs", "pitches", "strikes",
]


def _get(url, timeout=30, attempts=3):
    last = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except Exception as error:  # noqa: BLE001 - retried then reported
            last = error
            if attempt == attempts - 1:
                break
            time.sleep(1.5 ** attempt)
    raise last


def parse_boxscore(game_pk, official_date, payload):
    """Return one row per pitcher who appeared."""
    rows = []
    for side in ("home", "away"):
        team = (payload.get("teams", {}) or {}).get(side, {}) or {}
        team_id = ((team.get("team", {}) or {}).get("id"))
        pitcher_ids = team.get("pitchers", []) or []
        players = team.get("players", {}) or {}
        for order, player_id in enumerate(pitcher_ids):
            entry = players.get(f"ID{player_id}", {}) or {}
            stats = ((entry.get("stats", {}) or {}).get("pitching", {}) or {})
            if not stats:
                continue
            rows.append({
                "game_pk": game_pk,
                "official_date": official_date,
                "team_id": team_id,
                "player_id": player_id,
                # StatsAPI lists pitchers in appearance order, so the first is
                # the starter. gamesStarted is not always populated.
                "is_starter": int(order == 0),
                "outs": stats.get("outs"),
                "batters_faced": stats.get("battersFaced"),
                "hits": stats.get("hits"),
                "runs": stats.get("runs"),
                "earned_runs": stats.get("earnedRuns"),
                "home_runs": stats.get("homeRuns"),
                "walks": stats.get("baseOnBalls"),
                "hit_batsmen": stats.get("hitByPitch"),
                "strike_outs": stats.get("strikeOuts"),
                "pitches": stats.get("numberOfPitches"),
                "strikes": stats.get("strikes"),
            })
    return rows


def deduplicate(out_path, verbose=True):
    """Drop repeated (game_pk, player_id) lines before appending more.

    A pitcher appears once per game, so a repeat is damage rather than data,
    and it is damage that does not announce itself: `features.py` folds every
    line it is given, so a duplicated start counts a pitcher's strikeouts and
    home runs twice and the resulting rate looks entirely plausible.

    This file is written in append mode over a long run, which makes it easy
    to corrupt from outside — anything that rewrites it while the ingester
    holds it open leaves the writer appending at a stale offset. That happened
    here: a `git stash` of this file mid-run cost 4,479 games and left 1,552
    duplicated rows. Repairing on resume costs one pass and means the next run
    heals the file instead of building on it.
    """
    out_path = Path(out_path)
    if not out_path.exists() or out_path.stat().st_size == 0:
        return 0
    with out_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seen, kept = set(), []
    for row in rows:
        marker = (row.get("game_pk"), row.get("player_id"))
        if marker in seen:
            continue
        seen.add(marker)
        kept.append(row)
    removed = len(rows) - len(kept)
    if not removed:
        return 0
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PITCHER_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    os.replace(temporary, out_path)
    if verbose:
        print(f"repaired {removed} duplicated line(s) before resuming")
    return removed


def run(games, out_path, limit=None, verbose=True):
    out_path = Path(out_path)
    deduplicate(out_path, verbose=verbose)
    done = set()
    if out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as handle:
            done = {row["game_pk"] for row in csv.DictReader(handle)}
    pending = [g for g in games if str(g["game_pk"]) not in done]
    if limit:
        pending = pending[:limit]
    if verbose:
        print(f"{len(done)} games already recorded; {len(pending)} pending")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    processed, failed = 0, 0
    with out_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PITCHER_FIELDS,
                                extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for game in pending:
            game_pk = game["game_pk"]
            try:
                payload = _get(BOX_API.format(game_pk=game_pk))
                rows = parse_boxscore(game_pk, game["official_date"], payload)
            except Exception:  # noqa: BLE001 - counted, never defaulted
                failed += 1
                continue
            if not rows:
                failed += 1
                continue
            writer.writerows(rows)
            processed += 1
            if verbose and processed % 250 == 0:
                handle.flush()
                print(f"  {processed} games, {failed} failed")
    print(f"recorded {processed} games, {failed} failed")
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--out", default="data/pitching.csv")
    parser.add_argument("--seasons", default="")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    with open(args.games, newline="", encoding="utf-8") as handle:
        games = [row for row in csv.DictReader(handle)
                 if row.get("status") == "Final"]
    if args.seasons:
        wanted = {s.strip() for s in args.seasons.split(",")}
        games = [g for g in games if str(g["season"]) in wanted]
    games.sort(key=lambda g: (g["official_date"], g["game_pk"]))
    run(games, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
