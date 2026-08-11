"""Is 2022 baseball the same game as 2026 baseball? The model assumes so.

`walk_forward` trains on every prior season with equal weight and no era term.
That is honest about lookahead and silent about regime: inside this window the
pitch clock and the shift ban arrived in 2023, bases got bigger the same year,
and the challenge system arrived in 2026. Each moved the run environment, and a
2021 game currently counts for exactly as much as a 2025 one.

Stationarity is a wrong assumption sitting in plain sight, and wrong
assumptions are the category that has paid in this repository — twice, against
six failures for adding information. So it is worth the test rather than the
argument.

Three ways to stop pretending the seasons are interchangeable, all applied to
training only and all judged by the same walk-forward:

**Recency weighting.** Every training game gets a weight that decays with how
many seasons back it is. One parameter, the half-life, and the model is
otherwise untouched. A half-life of one season means the previous year counts
double the one before it.

**Recent seasons only.** The blunt version: drop everything older than a
window. Loses data, which is the trade being measured.

**Season-mean recentring.** Neither of the above touches the *level*. If 2026
simply scores more than 2023, the estimator can be handed each season's runs
relative to that season's own league average, which removes the level shift
without discarding any rows.

The first two are re-weightings the estimator supports directly. The third is
a change to the target, so it is measured separately and reported as such.

    python stationarity.py                 # the diagnostic
    python stationarity.py --walk-forward  # refit and reprice, several minutes
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS
from models import RunsModel, _design, mirror
from runs import GRID

# Half-lives tried, in seasons. A half-life of one means last season counts
# twice the one before it; four is close to no decay across this window.
HALF_LIVES = (0.5, 1.0, 2.0, 4.0)
# Trailing windows tried, in seasons.
WINDOWS = (1, 2, 3)
DRAWS = 2000
MIN_TRAIN_GAMES = 1500


def league_by_season(games):
    """Runs per team per game, by season. The regime, as one number."""
    played = games[games["home_score"].notna()]
    return (played.groupby("season")["total_runs"].mean() / 2.0).round(4)


def drift(games):
    """How much does the run environment actually move season to season?"""
    rates = league_by_season(games)
    out = {"runs_per_team_per_game": {int(k): float(v) for k, v in rates.items()}}
    values = rates.to_numpy()
    out["range"] = round(float(values.max() - values.min()), 4)
    out["season_to_season_sd"] = round(float(np.std(np.diff(values), ddof=0)), 4)
    # Against the spread of a single game, so the drift can be judged rather
    # than admired: a regime shift far below game noise cannot be worth much.
    played = games[games["home_score"].notna()]
    game_sd = float(np.concatenate([played["home_score"],
                                    played["away_score"]]).std())
    out["single_game_sd"] = round(game_sd, 4)
    out["drift_as_share_of_game_noise"] = round(
        float((values.max() - values.min()) / game_sd), 4)
    return out


def _weights(seasons, target, half_life):
    """Exponential decay in seasons before the one being predicted."""
    age = (target - np.asarray(seasons, dtype=float)) - 1.0
    return 0.5 ** (np.clip(age, 0.0, None) / half_life)


def _fit(features, games, weight=None, kind="glm", reference=None):
    """A RunsModel fitted with optional per-game sample weights.

    Weights are duplicated across the two dugout views, because `fit` trains
    on every game twice and a weight belongs to the game rather than to the
    view of it.
    """
    model = RunsModel(kind=kind)
    if weight is None:
        return model.fit(features, games)

    merged = features.merge(
        games[["game_pk", "home_score", "away_score", "home_win",
               "innings_played", "scheduled_innings", "home_batted_ninth"]],
        on="game_pk", how="inner")
    merged = merged[merged["home_score"].notna()]
    order = merged["game_pk"].map(dict(zip(features["game_pk"], weight)))
    paired = np.concatenate([order.to_numpy(dtype=float)] * 2)

    from sklearn.linear_model import PoissonRegressor
    from sklearn.preprocessing import StandardScaler

    design = pd.concat([_design(merged), _design(mirror(merged))], axis=0)
    target = np.concatenate([merged["home_score"].to_numpy(dtype=float),
                             merged["away_score"].to_numpy(dtype=float)])
    model.scaler = StandardScaler().fit(design)
    model.estimator = PoissonRegressor(alpha=1e-3, max_iter=2000)
    model.estimator.fit(model.scaler.transform(design), target,
                        sample_weight=paired)
    # Width is measured the same way regardless of how the mean was fitted;
    # reusing `fit`'s own holdout would mean refitting unweighted, so the
    # weighted estimator is paired with the unweighted width deliberately and
    # both arms of the comparison carry it. Passed in when the caller already
    # has it -- every decay scheme in a season shares one training block, and
    # fitting the width once per scheme was most of the runtime.
    if reference is None:
        reference = RunsModel(kind=kind).fit(features, games)
    model.dispersion = reference.dispersion
    model.shape = reference.shape
    model.extra_home_edge = reference.extra_home_edge
    model.extra_margins = reference.extra_margins
    model.walk_off_margins = reference.walk_off_margins
    return model


def _losses(model, features, games, season):
    test = features[features["season"] == season]
    lengths = (test[["game_pk"]]
               .merge(games[["game_pk", "scheduled_innings"]],
                      on="game_pk", how="left")["scheduled_innings"])
    priced = model.price(test, innings=lengths.to_numpy())
    truth = test[["game_pk", "official_date"]].merge(
        games[["game_pk", "home_score", "away_score", "home_win",
               "total_runs"]], on="game_pk", how="left")
    keep = truth["home_score"].notna().to_numpy()
    out = {"date": truth.loc[keep, "official_date"].to_numpy()}
    pairs = (
        ("moneyline", priced["p_home_ml"].to_numpy(),
         truth["home_win"].to_numpy()),
        ("runline", priced["p_home_rl_-1.5"].to_numpy(),
         ((truth["home_score"] - truth["away_score"]) > 1.5).to_numpy()),
        ("total", priced["p_over_8.5"].to_numpy(),
         (truth["total_runs"] > 8.5).to_numpy()),
    )
    for name, probability, outcome in pairs:
        probability = np.clip(probability[keep], 1e-9, 1 - 1e-9)
        actual = outcome[keep].astype(float)
        out[name] = -(actual * np.log(probability)
                      + (1 - actual) * np.log(1 - probability))
    return out


def walk_forward(features, games, kind="glm", verbose=True):
    """Every scheme, judged on the same seasons and the same games."""
    schemes = {"equal": None}
    schemes.update({f"half_life_{h:g}": ("decay", h) for h in HALF_LIVES})
    schemes.update({f"last_{w}": ("window", w) for w in WINDOWS})

    collected = {name: [] for name in schemes}
    dates = []
    for season in sorted(features["season"].unique()):
        train = features[features["season"] < season]
        test = features[features["season"] == season]
        if len(train) < MIN_TRAIN_GAMES or not len(test):
            continue
        if verbose:
            print(f"  {season}: training on {len(train)}, predicting {len(test)}")
        # One unweighted fit on the full training block, shared as the width
        # for every decay scheme and reused directly as the baseline.
        reference = RunsModel(kind=kind).fit(train, games)
        first = True
        for name, scheme in schemes.items():
            if scheme is None:
                weight, block, model = None, train, reference
            elif scheme[0] == "decay":
                block = train
                weight = _weights(train["season"], season, scheme[1])
                model = _fit(block, games, weight, kind=kind,
                             reference=reference)
            else:
                block = train[train["season"] >= season - scheme[1]]
                weight = None
                if len(block) < MIN_TRAIN_GAMES:
                    block = train
                model = _fit(block, games, None, kind=kind)
            result = _losses(model, features, games, season)
            collected[name].append(result)
            if first:
                dates.append(result["date"])
                first = False
    return collected, np.concatenate(dates)


def compare(collected, dates, baseline="equal", draws=DRAWS, seed=0):
    unique = np.unique(dates)
    index = {date: np.flatnonzero(dates == date) for date in unique}
    rng = np.random.default_rng(seed)
    out = {}
    for market in ("moneyline", "runline", "total"):
        base = np.concatenate([block[market] for block in collected[baseline]])
        entry = {"baseline": round(float(base.mean()), 5), "schemes": {}}
        for name, blocks in collected.items():
            if name == baseline:
                continue
            other = np.concatenate([block[market] for block in blocks])
            sample = []
            for _ in range(draws):
                pick = rng.choice(unique, len(unique), replace=True)
                take = np.concatenate([index[date] for date in pick])
                sample.append(other[take].mean() - base[take].mean())
            low, high = np.percentile(sample, [5, 95])
            entry["schemes"][name] = {
                "log_loss": round(float(other.mean()), 5),
                "delta": round(float(other.mean() - base.mean()), 5),
                "ci90_date_clustered": [round(float(low), 5),
                                        round(float(high), 5)],
                "helps": bool(high < 0),
            }
        out[market] = entry

    # Look-elsewhere. Seven schemes across three markets is 21 comparisons at
    # 90% intervals, so about 2.1 should exclude zero under a pure null. A
    # count near that is not evidence, whatever the individual cells say --
    # the same correction `extremes.py` exists to apply.
    helped = sum(1 for market in out.values()
                 for entry in market["schemes"].values() if entry["helps"])
    hurt = sum(1 for market in out.values()
               for entry in market["schemes"].values()
               if entry["ci90_date_clustered"][0] > 0)
    tests = sum(len(market["schemes"]) for market in out.values())
    out["look_elsewhere"] = {
        "comparisons": tests,
        "expected_by_chance_each_side": round(0.05 * tests, 2),
        "excluded_zero_helping": helped,
        "excluded_zero_hurting": hurt,
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features.csv")
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--kind", default="glm", choices=["glm", "gbm"])
    parser.add_argument("--report", default="stationarity.json")
    parser.add_argument("--walk-forward", action="store_true",
                        help="refit and reprice under each scheme; minutes")
    args = parser.parse_args()

    games = pd.read_csv(args.games)
    features = pd.read_csv(args.features)
    result = {"drift": drift(games)}

    print("run environment by season (runs per team per game)")
    for season, rate in result["drift"]["runs_per_team_per_game"].items():
        print(f"  {season}: {rate:.3f}")
    print(f"\n  range across seasons      {result['drift']['range']:.3f} runs")
    print(f"  season-to-season sd       {result['drift']['season_to_season_sd']:.3f}")
    print(f"  one game's own sd         {result['drift']['single_game_sd']:.3f}")
    print(f"  drift as a share of that  "
          f"{result['drift']['drift_as_share_of_game_noise']:.3f}")

    if args.walk_forward:
        print("\nrefitting under each weighting scheme")
        collected, dates = walk_forward(features, games, kind=args.kind)
        result["walk_forward"] = compare(collected, dates)
        for market, block in result["walk_forward"].items():
            if market == "look_elsewhere":
                continue
            print(f"\n{market}  (equal weighting = {block['baseline']:.5f})")
            print(f"  {'scheme':<16}{'log loss':>10}{'delta':>10}"
                  f"   90% interval")
            for name, entry in block["schemes"].items():
                flag = "  <-- helps" if entry["helps"] else ""
                print(f"  {name:<16}{entry['log_loss']:>10.5f}"
                      f"{entry['delta']:>+10.5f}   "
                      f"[{entry['ci90_date_clustered'][0]:+.5f}, "
                      f"{entry['ci90_date_clustered'][1]:+.5f}]{flag}")

        look = result["walk_forward"]["look_elsewhere"]
        print(f"\nlook-elsewhere: {look['comparisons']} comparisons at 90%, "
              f"so ~{look['expected_by_chance_each_side']} should exclude zero "
              f"each way by chance")
        print(f"  schemes that helped: {look['excluded_zero_helping']}   "
              f"schemes that hurt: {look['excluded_zero_hurting']}")

    Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
