"""Statcast pitch-level data from Baseball Savant, aggregated on ingest.

Free and keyless, and the largest input this repository did not have. Two
things it brings that nothing already here can:

**Expected outcomes.** `estimated_woba_using_speedangle` scores a batted ball
by its exit velocity and launch angle rather than by whether it found grass.
That strips out the fielders, the park and the luck, which is why expected
statistics stabilise in a few dozen batted balls where actual results take a
season. It is the same property that made a starter's contact profile the only
non-scoreboard feature worth carrying: it says what happened rather than what
was recorded.

**Batters.** The model had none. Forty-seven features and not one about who was
holding a bat — team offence was a single shrunk run rate, which cannot know
that four of tonight's nine are resting. Aggregating by batter is the first
step to fixing that; `features.py` decides separately whether it helps.

Aggregated on the way in, deliberately. One day of pitches is 3MB and a season
is close to half a gigabyte, so six seasons of raw rows would be several
gigabytes committed to a repository whose whole game table is 3MB. Nothing
downstream wants a pitch — `features.py` wants "what did this pitcher do in
this game" — so the pitches are summed per game and per player and discarded.
The cost is that a new question about pitch sequencing means refetching, and
that is the right trade at this size.

Resumable by date. A date already present is skipped, so the ingest can be run
repeatedly and stopped at any point. Fail closed: a date that errors is counted
and skipped rather than written empty, because an empty day is
indistinguishable from a quiet one once it is on disk.

    python statcast.py --seasons 2021-2026     # everything, hours
    python statcast.py --seasons 2026 --limit 5 # a taste
"""

import argparse
import csv
import io
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SAVANT_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"

# Regular season only, matching `results.py`. Spring training and the
# postseason are different populations and mixing them changes what is being
# estimated.
GAME_TYPE = "R"

# A barrel is Statcast's own label for the exit-velocity/launch-angle
# combination that produces extra bases, encoded as 6 in launch_speed_angle.
BARREL_CODE = 6

PITCHER_FIELDS = [
    "game_pk", "game_date", "player_id", "role", "is_home",
    "pitches", "batters_faced", "swings", "whiffs", "called_strikes",
    "in_zone", "batted_balls", "barrels",
    "xwoba_sum", "xwoba_denom", "woba_sum", "woba_denom",
    "launch_speed_sum", "launch_speed_count",
    "release_speed_sum", "release_speed_count",
    "spin_rate_sum", "spin_rate_count", "delta_run_exp_sum",
]


def _get(url, timeout=180, attempts=4):
    last = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"Accept": "text/csv",
                              "User-Agent": "claude-diamond/research"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as error:  # noqa: BLE001 - retried, then reported
            last = error
            if attempt == attempts - 1:
                break
            time.sleep(2.0 * (attempt + 1))
    raise last


def fetch_day(day, game_type=GAME_TYPE):
    """Every pitch thrown on one date, as a frame."""
    query = urllib.parse.urlencode({
        "all": "true",
        "hfGT": f"{game_type}|",
        "game_date_gt": day,
        "game_date_lt": day,
        "type": "details",
    })
    text = _get(f"{SAVANT_CSV}?{query}")
    if not text.strip():
        return pd.DataFrame()
    frame = pd.read_csv(io.StringIO(text), low_memory=False)
    # Savant echoes the header even for an empty result, and occasionally
    # returns a stray row of nulls; both must not look like data.
    if not len(frame) or "game_pk" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["game_pk"].notna()]


def aggregate(pitches):
    """Per game, per player, once as pitcher and once as batter.

    Sums rather than rates. A rate needs a denominator that only makes sense
    once several games are pooled, and pooling is `features.py`'s job — it
    walks games in order and may only use what came before. Storing a rate
    here would decide the window for it.
    """
    if not len(pitches):
        return pd.DataFrame(columns=PITCHER_FIELDS)

    missing = [column for column in ("description", "inning_topbot")
               if column not in pitches.columns]
    if missing:
        # Fail closed and say which field. Without inning_topbot a batter row
        # cannot be attributed to a team, and silently defaulting it would
        # give every batter to the away side -- a wrong answer that looks like
        # data. If Savant ever renames the field the ingest should stop, not
        # quietly mislabel 400,000 rows.
        raise KeyError(
            f"statcast feed is missing {missing}; cannot attribute rows")

    frame = pitches.copy()
    description = frame["description"].astype(str)
    # Savant spells a swing several ways; the miss variants are the whiffs.
    swing_words = ("hit_into_play", "foul", "swinging_strike",
                   "swinging_strike_blocked", "foul_tip", "foul_bunt",
                   "missed_bunt", "bunt_foul_tip")
    whiff_words = ("swinging_strike", "swinging_strike_blocked", "foul_tip",
                   "missed_bunt")
    frame["is_swing"] = description.isin(swing_words).astype(float)
    frame["is_whiff"] = description.isin(whiff_words).astype(float)
    frame["is_called_strike"] = description.eq("called_strike").astype(float)
    # Zones 1-9 are inside the strike zone; 11-14 are the outside quadrants.
    zone = pd.to_numeric(frame.get("zone"), errors="coerce")
    frame["in_zone"] = ((zone >= 1) & (zone <= 9)).astype(float)
    frame["is_batted"] = frame["bb_type"].notna().astype(float)
    angle = pd.to_numeric(frame.get("launch_speed_angle"), errors="coerce")
    frame["is_barrel"] = angle.eq(BARREL_CODE).astype(float)
    # A plate appearance ends on a row carrying an event.
    frame["is_pa"] = frame["events"].notna().astype(float)
    # Which side the player is on, from the half-inning. The home team pitches
    # in the top and bats in the bottom. Without this a batter row cannot be
    # attributed to a team at all, which is what the first version of this
    # file got wrong -- it aggregated batters and then had no way to say whose
    # they were.
    top = frame["inning_topbot"].astype(str).str.lower().str.startswith("top")
    frame["pitcher_is_home"] = top.astype(float)
    frame["batter_is_home"] = (~top).astype(float)

    numeric = {}
    for column in ("estimated_woba_using_speedangle", "woba_value",
                   "woba_denom", "launch_speed", "release_speed",
                   "release_spin_rate", "delta_run_exp"):
        numeric[column] = pd.to_numeric(frame.get(column), errors="coerce")

    # xwOBA is only defined on a batted ball or a walk/strikeout, so its
    # denominator is counted from the rows that actually carry one rather than
    # assumed equal to plate appearances.
    frame["xwoba_sum"] = numeric["estimated_woba_using_speedangle"].fillna(0.0)
    frame["xwoba_denom"] = numeric[
        "estimated_woba_using_speedangle"].notna().astype(float)
    frame["woba_sum"] = numeric["woba_value"].fillna(0.0)
    frame["woba_denom_v"] = numeric["woba_denom"].fillna(0.0)
    frame["launch_speed_v"] = numeric["launch_speed"].fillna(0.0)
    frame["launch_speed_n"] = numeric["launch_speed"].notna().astype(float)
    frame["release_speed_v"] = numeric["release_speed"].fillna(0.0)
    frame["release_speed_n"] = numeric["release_speed"].notna().astype(float)
    frame["spin_v"] = numeric["release_spin_rate"].fillna(0.0)
    frame["spin_n"] = numeric["release_spin_rate"].notna().astype(float)
    frame["delta_run_exp_v"] = numeric["delta_run_exp"].fillna(0.0)

    rows = []
    for role, key in (("pitcher", "pitcher"), ("batter", "batter")):
        block = frame[frame[key].notna()]
        if not len(block):
            continue
        grouped = block.groupby(["game_pk", "game_date", key], dropna=False)
        summed = grouped.agg(
            is_home=(f"{role}_is_home", "max"),
            pitches=("is_swing", "size"),
            batters_faced=("is_pa", "sum"),
            swings=("is_swing", "sum"),
            whiffs=("is_whiff", "sum"),
            called_strikes=("is_called_strike", "sum"),
            in_zone=("in_zone", "sum"),
            batted_balls=("is_batted", "sum"),
            barrels=("is_barrel", "sum"),
            xwoba_sum=("xwoba_sum", "sum"),
            xwoba_denom=("xwoba_denom", "sum"),
            woba_sum=("woba_sum", "sum"),
            woba_denom=("woba_denom_v", "sum"),
            launch_speed_sum=("launch_speed_v", "sum"),
            launch_speed_count=("launch_speed_n", "sum"),
            release_speed_sum=("release_speed_v", "sum"),
            release_speed_count=("release_speed_n", "sum"),
            spin_rate_sum=("spin_v", "sum"),
            spin_rate_count=("spin_n", "sum"),
            delta_run_exp_sum=("delta_run_exp_v", "sum"),
        ).reset_index()
        summed = summed.rename(columns={key: "player_id"})
        summed["role"] = role
        rows.append(summed)
    if not rows:
        return pd.DataFrame(columns=PITCHER_FIELDS)
    out = pd.concat(rows, ignore_index=True)
    out["game_pk"] = out["game_pk"].astype("int64")
    out["player_id"] = out["player_id"].astype("int64")
    return out.reindex(columns=PITCHER_FIELDS)


def season_days(season, today=None):
    """Every date a regular-season game could fall on, generously bounded."""
    today = today or datetime.now(timezone.utc).date()
    first, last = date(season, 3, 1), date(season, 11, 15)
    last = min(last, today)
    day = first
    while day <= last:
        yield day.isoformat()
        day += timedelta(days=1)


def existing_days(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["game_date"] for row in csv.DictReader(handle)}


def run(seasons, out_path, limit=None, pause=1.0, verbose=True, today=None):
    out_path = Path(out_path)
    done = existing_days(out_path)
    pending = [day for season in seasons for day in season_days(season, today)
               if day not in done]
    if limit:
        pending = pending[:limit]
    if verbose:
        print(f"{len(done)} dates already aggregated; {len(pending)} pending")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    written, empty, failed = 0, 0, 0
    with out_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PITCHER_FIELDS,
                                extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for day in pending:
            try:
                pitches = fetch_day(day)
            except Exception:  # noqa: BLE001 - counted, run continues
                failed += 1
                continue
            if not len(pitches):
                # A real off day. Recorded as nothing rather than as a row, so
                # resume treats it as pending and retries -- cheap, and the
                # alternative is a silent hole that looks aggregated.
                empty += 1
                continue
            rows = aggregate(pitches)
            writer.writerows(rows.to_dict("records"))
            # Flushed every date, not every twentieth. A full ingest is hours
            # long and this container restarts; the flush costs nothing next
            # to a six-second fetch, and it caps what a restart loses at one
            # date instead of twenty.
            handle.flush()
            written += 1
            if verbose and written % 20 == 0:
                print(f"  {written} dates, {empty} with no games, "
                      f"{failed} failed")
            if pause:
                time.sleep(pause)
    if verbose:
        print(f"aggregated {written} dates, {empty} with no games, "
              f"{failed} failed")
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default=None,
                        help="inclusive range, e.g. 2021-2026; defaults to "
                             "every season through the current one")
    parser.add_argument("--out", default="data/statcast_games.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pause", type=float, default=1.0)
    args = parser.parse_args()

    from results import default_seasons, parse_seasons

    seasons = parse_seasons(args.seasons or default_seasons())
    print(f"seasons {seasons}")
    run(seasons, args.out, limit=args.limit, pause=args.pause)


if __name__ == "__main__":
    main()
