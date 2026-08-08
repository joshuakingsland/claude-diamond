"""Leverage-weighted bullpen workload, from play-by-play.

`features.py` already carries bullpen outs over three days, and that feature
earns a coefficient of +0.001 — the right sign and nearly nothing. The
hypothesis this module tests is that the blunt version is the problem: an out
recorded in a tied ninth costs a reliever more than an out recorded in a
six-run game, and the market adjusts for innings rather than for what those
innings were worth.

Two stages, because the weighting needs a table the fetch is producing:

- `fetch` walks each game's plays and records the state before every plate
  appearance, with the pitcher who faced it. Raw plays go to a scratch file,
  not the repository: they are an intermediate worth 8GB of downloads and
  nothing in version control.
- `build` fits an empirical win expectancy over those states, converts it to a
  leverage weight per plate appearance, and writes one row per game and
  pitcher.

Win expectancy is fitted over (inning, half, outs, score difference) and *not*
over the base state. Reconstructing runners from the movement arrays is
error-prone and the omission is second-order: leverage is dominated by how late
it is and how close the score is. The simplification is stated here rather than
hidden, because a leverage number that quietly ignores a bases-loaded jam is
worth knowing about.

One request per game, free and keyless, resumable.
"""

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PBP_API = "https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay"

PLAY_FIELDS = ["game_pk", "official_date", "inning", "is_top", "outs_before",
               "home_score_before", "away_score_before", "pitcher_id"]

WORK_FIELDS = ["game_pk", "official_date", "pitcher_id", "batters",
               "leverage_sum", "high_leverage_batters"]

# Above this a plate appearance counts as high leverage, the usual convention.
HIGH_LEVERAGE = 1.5
# Score differences past this are blowouts and share one bucket; the win
# expectancy is flat out there and splitting it only thins the table.
MAX_DIFF = 8


def _get(url, timeout=60, attempts=3):
    last = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except Exception as error:  # noqa: BLE001 - retried, then reported
            last = error
            if attempt == attempts - 1:
                break
            time.sleep(1.5 ** attempt)
    raise last


def plays_from(payload, game_pk, official_date):
    """State before each plate appearance, with the pitcher facing it."""
    rows = []
    home, away, outs, inning, half = 0, 0, 0, None, None
    for play in payload.get("allPlays", []) or []:
        about = play.get("about", {}) or {}
        result = play.get("result", {}) or {}
        pitcher = ((play.get("matchup", {}) or {}).get("pitcher", {}) or {})
        if about.get("inning") is None or not pitcher.get("id"):
            continue
        if (about.get("inning"), about.get("isTopInning")) != (inning, half):
            inning, half = about.get("inning"), about.get("isTopInning")
            outs = 0
        rows.append({
            "game_pk": game_pk, "official_date": official_date,
            "inning": int(about["inning"]),
            "is_top": int(bool(about.get("isTopInning"))),
            "outs_before": int(min(outs, 2)),
            "home_score_before": home, "away_score_before": away,
            "pitcher_id": pitcher["id"],
        })
        outs = int((play.get("count", {}) or {}).get("outs") or outs)
        home = int(result.get("homeScore", home) or 0)
        away = int(result.get("awayScore", away) or 0)
    return rows


def fetch(games, plays_path, limit=None, pause=0.0, verbose=True):
    plays_path = Path(plays_path)
    done = set()
    if plays_path.exists():
        with plays_path.open(newline="", encoding="utf-8") as handle:
            done = {row["game_pk"] for row in csv.DictReader(handle)}
    pending = [g for g in games if str(g["game_pk"]) not in done]
    if limit:
        pending = pending[:limit]
    if verbose:
        print(f"{len(done)} games already walked; {len(pending)} pending")

    plays_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not plays_path.exists() or plays_path.stat().st_size == 0
    recorded, failed = 0, 0
    with plays_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAY_FIELDS,
                                extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for game in pending:
            try:
                payload = _get(PBP_API.format(game_pk=game["game_pk"]))
                rows = plays_from(payload, game["game_pk"],
                                  game.get("official_date", ""))
            except Exception:  # noqa: BLE001 - counted, run continues
                rows = []
            if not rows:
                failed += 1
                continue
            writer.writerows(rows)
            recorded += 1
            if verbose and recorded % 250 == 0:
                handle.flush()
                print(f"  {recorded} games walked, {failed} failed")
            if pause:
                time.sleep(pause)
    if verbose:
        print(f"walked {recorded} games, {failed} failed")
    return recorded


def win_expectancy(plays, games):
    """Empirical P(home wins) by state, from the states themselves."""
    outcomes = games[["game_pk", "home_win"]].dropna()
    frame = plays.merge(outcomes, on="game_pk")
    frame["diff"] = np.clip(frame.home_score_before - frame.away_score_before,
                            -MAX_DIFF, MAX_DIFF)
    frame["late"] = np.minimum(frame.inning, 10)
    table = (frame.groupby(["late", "is_top", "outs_before", "diff"])
                  .home_win.agg(["mean", "size"]))
    # A state seen fewer than 30 times is noise; fall back to the league rate.
    overall = float(frame.home_win.mean())
    table["we"] = np.where(table["size"] >= 30, table["mean"], overall)
    return table["we"], frame, overall


def leverage_weights(plays, games):
    """Leverage per plate appearance: how much the state can still swing.

    Approximated by the spread of win expectancy reachable from the state —
    the difference between the run-scoring and out-making branches — scaled so
    the average plate appearance sits at 1.0.
    """
    table, frame, overall = win_expectancy(plays, games)
    keys = list(zip(frame.late, frame.is_top, frame.outs_before, frame["diff"]))
    here = np.array([table.get(k, overall) for k in keys])
    scored = np.array([table.get((k[0], k[1], k[2],
                                  int(np.clip(k[3] + (1 if k[1] == 0 else -1),
                                              -MAX_DIFF, MAX_DIFF))), overall)
                       for k in keys])
    retired = np.array([table.get((k[0], k[1], min(k[2] + 1, 2), k[3]), overall)
                        for k in keys])
    swing = np.abs(scored - here) + np.abs(retired - here)
    average = float(np.mean(swing)) or 1.0
    frame = frame.copy()
    frame["leverage"] = swing / average
    return frame


def build(plays, games):
    frame = leverage_weights(plays, games)
    frame["high"] = (frame.leverage >= HIGH_LEVERAGE).astype(int)
    work = (frame.groupby(["game_pk", "official_date", "pitcher_id"])
                 .agg(batters=("leverage", "size"),
                      leverage_sum=("leverage", "sum"),
                      high_leverage_batters=("high", "sum"))
                 .reset_index())
    work["leverage_sum"] = work.leverage_sum.round(4)
    return work


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="fetch", choices=["fetch", "build"])
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--plays", required=True,
                        help="scratch file of per-plate-appearance states")
    parser.add_argument("--out", default="data/bullpen_leverage.csv")
    parser.add_argument("--seasons", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pause", type=float, default=0.0)
    args = parser.parse_args()

    if args.stage == "fetch":
        with open(args.games, newline="", encoding="utf-8") as handle:
            games = [row for row in csv.DictReader(handle)
                     if row.get("status") == "Final"]
        if args.seasons:
            wanted = {s.strip() for s in args.seasons.split(",")}
            games = [g for g in games if str(g.get("season")) in wanted]
        fetch(games, args.plays, limit=args.limit, pause=args.pause)
        return

    plays = pd.read_csv(args.plays)
    games = pd.read_csv(args.games)
    work = build(plays, games)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(args.out, index=False)
    print(f"wrote {len(work)} game-pitcher rows to {args.out}")


if __name__ == "__main__":
    main()
