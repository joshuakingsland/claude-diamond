"""Do umpires move anything, and is the challenge system taking it away?

The precursor to any zone-bias model. A strike-zone edge has to show up first
as a plain umpire main effect, because an interaction with a pitcher's arsenal
is necessarily smaller than the effect it interacts with. If umpire identity
explains nothing, nothing built on top of it can.

Three tests, in order of how directly they see the zone:

**Strikeouts per game.** The most sensitive and the only one that needs no
model: a larger zone means more called strikes means more strikeouts. If
umpires differ at all, they differ here.

**Residual total runs.** The mechanism a bettor would care about on a total:
zone size moves scoring.

**Residual home win probability.** What the moneyline is priced on.

Each is a permutation test rather than a look at the extremes. Umpires work
different schedules, so the spread of per-umpire means is inflated by sample
size alone; shuffling assignments *within season* preserves that structure and
gives the spread the null actually predicts. Reporting the widest umpire
instead would be the same mistake `extremes.py` documents.

Everything is reported by season, and that is the point rather than a
breakdown. Teams now carry two challenges a game and keep them when correct,
which removes precisely the calls a bias model would exploit — the obvious
ones. A model fitted before challenges will overstate what remains, so the
pre-challenge seasons against the current one is the finding.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Umpires below this many games in a season are dropped: their mean is noise
# and including them inflates the observed spread for reasons that have nothing
# to do with the zone.
MIN_GAMES = 15
DRAWS = 2000


def spread(frame, value, group="hp_umpire_id"):
    """Sample-weighted standard deviation of per-umpire means."""
    grouped = frame.groupby(group)[value].agg(["mean", "size"])
    grouped = grouped[grouped["size"] >= MIN_GAMES]
    if len(grouped) < 5:
        return None, 0
    weights = grouped["size"] / grouped["size"].sum()
    centre = float((grouped["mean"] * weights).sum())
    variance = float((weights * (grouped["mean"] - centre) ** 2).sum())
    return float(np.sqrt(variance)), int(len(grouped))


def permutation_test(frame, value, draws=DRAWS, seed=5):
    """Observed spread against the spread from shuffled assignments.

    Shuffling happens inside a season, so each umpire keeps a realistic number
    of games and only the pairing with those games is broken.
    """
    observed, umpires = spread(frame, value)
    if observed is None:
        return None
    rng = np.random.default_rng(seed)
    shuffled = frame.copy()
    null = []
    for _ in range(draws):
        shuffled["hp_umpire_id"] = (
            frame.groupby("season")["hp_umpire_id"]
            .transform(lambda block: rng.permutation(block.to_numpy())))
        value_spread, _ = spread(shuffled, value)
        if value_spread is not None:
            null.append(value_spread)
    if len(null) < 100:
        return None
    null = np.array(null)
    return {
        "umpires": umpires,
        "games": int(len(frame)),
        "observed_spread": round(observed, 5),
        "null_spread_median": round(float(np.median(null)), 5),
        "p_value": round(float((null >= observed).mean()), 4),
        "excess": round(observed - float(np.median(null)), 5),
        # Observed spread contains sampling noise. What is left after removing
        # the null's own spread is the between-umpire standard deviation the
        # data implies -- the number that decides whether this is worth
        # building on, separately from whether it is distinguishable.
        "implied_true_sd": round(float(np.sqrt(max(
            observed ** 2 - float(np.median(null)) ** 2, 0.0))), 5),
    }


def load(games_path, umpires_path, predictions_path, pitching_path):
    games = pd.read_csv(games_path)
    umpires = pd.read_csv(umpires_path)
    frame = games[games["home_score"].notna()].merge(
        umpires[["game_pk", "hp_umpire_id", "hp_umpire_name"]], on="game_pk")

    predictions = pd.read_csv(predictions_path)
    keep = ["game_pk", "p_home_ml", "expected_home_runs", "expected_away_runs"]
    frame = frame.merge(predictions[keep], on="game_pk", how="left")
    frame["resid_win"] = frame["home_win"] - frame["p_home_ml"]
    frame["resid_total"] = frame["total_runs"] - (
        frame["expected_home_runs"] + frame["expected_away_runs"])

    if Path(pitching_path).exists():
        pitching = pd.read_csv(pitching_path)
        strikeouts = (pitching.groupby("game_pk")["strike_outs"].sum()
                      .rename("game_strikeouts").reset_index())
        frame = frame.merge(strikeouts, on="game_pk", how="left")
    return frame


def report(frame):
    out = {"games": int(len(frame)),
           "umpires": int(frame["hp_umpire_id"].nunique())}
    for label, column in (("strikeouts_per_game", "game_strikeouts"),
                          ("residual_total_runs", "resid_total"),
                          ("residual_home_win", "resid_win")):
        if column not in frame:
            continue
        block = frame[frame[column].notna()]
        if not len(block):
            continue
        out[label] = {"pooled": permutation_test(block, column)}
        by_season = {}
        for season, season_block in block.groupby("season"):
            result = permutation_test(season_block, column, draws=DRAWS // 2)
            if result:
                by_season[int(season)] = result
        out[label]["by_season"] = by_season
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--umpires", default="data/umpires.csv")
    parser.add_argument("--predictions", default="data/predictions_glm.csv")
    parser.add_argument("--pitching", default="data/pitching.csv")
    parser.add_argument("--report", default="umpire_effect.json")
    args = parser.parse_args()

    frame = load(args.games, args.umpires, args.predictions, args.pitching)
    result = report(frame)
    Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"{result['games']:,} games, {result['umpires']} umpires\n")
    for label in ("strikeouts_per_game", "residual_total_runs",
                  "residual_home_win"):
        block = result.get(label)
        if not block or not block.get("pooled"):
            continue
        pooled = block["pooled"]
        print(f"{label}")
        print(f"  pooled: spread {pooled['observed_spread']} vs null "
              f"{pooled['null_spread_median']}  p={pooled['p_value']}  "
              f"({pooled['umpires']} umpires)")
        for season, season_block in sorted(block["by_season"].items()):
            print(f"    {season}: spread {season_block['observed_spread']:.4f} "
                  f"vs null {season_block['null_spread_median']:.4f}  "
                  f"p={season_block['p_value']}")
        print()


if __name__ == "__main__":
    main()
