"""The predicted means are over-spread. Correcting them makes pricing worse.

This is a negative result about a defect that is unambiguously real, which is
why it gets its own file rather than a footnote.

**The defect.** Regress observed runs on predicted runs across the walk-forward
and the slope is 0.712, not 1. Split the predictions into quintiles and the
middle three are within 0.10 of the truth with per-season gaps that change
sign, while the extremes are off in the same direction every season: the
quietest fifth of games score 0.209 more than predicted (5 seasons out of 5)
and the loudest score 0.339 fewer (0 out of 5). That is not sampling noise and
it is not a level shift. It is regression to the mean, the ordinary
consequence of conditioning on a noisy estimate.

**The correction, twice.** Refitting the run-scoring mean is standard: a linear
recalibration is the textbook answer and isotonic is the safer one, monotone so
it cannot reorder two games and free to leave the well-behaved middle alone.
Both are fitted out of sample, and the dispersion and inning shape are refitted
on the corrected predictions so the comparison does not reward the old width
for the wrong reason.

Both make every market worse, with intervals excluding zero.

**Why.** Because the run-scoring mean and the price are not the same quantity.
The between-game spread of predicted means is 0.743 runs against a residual
spread of 3.143 — the signal is under a quarter of the per-game noise — and the
dispersion is fitted from residuals around the *unshrunk* mean, so the priced
distribution has already absorbed the attenuation. Correcting the mean on top
of that charges for the same uncertainty twice. It shows up exactly where the
mechanism predicts: the moneyline calibration slope goes from 0.876, slightly
over-confident, through 1.0 and out the other side to 1.10 under isotonic and
1.29 under linear, while the spread of predicted probabilities collapses by 22%
and 33%.

Isotonic is the instructive one. It leaves calibration *error* essentially
where it found it, 0.01354 against 0.01341, and still loses on all three
markets — because the loss is not in calibration at all, it is in
discrimination: the spread of predicted probabilities falls from 0.0837 to
0.0652. Calibration is necessary and not sufficient — a model that answers 0.5
to everything is perfectly calibrated and worth nothing — and this is what
trading discrimination for nothing looks like on a real model.

The conclusion is not that the means are fine. They are measurably not fine,
and an estimator that fixed the attenuation *at source*, by being less noisy,
would be a real improvement. What does not work is repairing the symptom
downstream of a width that already accounts for it.

    python mean_calibration.py             # the diagnostic only, seconds
    python mean_calibration.py --corrections   # refits and prices, minutes
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from runs import (GRID, calibrate_extra_innings, calibrate_walk_off,
                  fit_dispersion, fit_inning_shape, joint_distribution,
                  uncensor_home_mean)

QUINTILES = 5
CALIBRATION_BINS = 10
DRAWS = 2000
MIN_TRAIN_GAMES = 2000


def load(games_path, predictions_path):
    """Completed nine-inning games joined to their walk-forward predictions."""
    games = pd.read_csv(games_path)
    games = games[(games["status"] == "Final")
                  & (games["scheduled_innings"] == 9)]
    predictions = pd.read_csv(predictions_path)
    frame = games.merge(predictions, on="game_pk", suffixes=("", "_predicted"))
    frame = frame.dropna(subset=["home_score", "away_score",
                                 "expected_home_runs"])
    return frame.sort_values("official_date").reset_index(drop=True)


def stacked(frame):
    """Both dugouts as one column of observations. Run scoring is run scoring."""
    observed = np.concatenate([frame["home_score"], frame["away_score"]])
    predicted = np.concatenate([frame["expected_home_runs"],
                                frame["expected_away_runs"]])
    season = np.concatenate([frame["season"], frame["season"]])
    return observed.astype(float), predicted.astype(float), season


def attenuation(frame):
    """Regression slope of observed on predicted, per side and pooled.

    A slope below one means the predictions are spread wider than the truth.
    """
    out = {}
    for side in ("home", "away"):
        observed = frame[f"{side}_score"].to_numpy(float)
        predicted = frame[f"expected_{side}_runs"].to_numpy(float)
        slope, intercept = np.polyfit(predicted, observed, 1)
        out[side] = {"slope": round(float(slope), 4),
                     "intercept": round(float(intercept), 4),
                     "sd_predicted": round(float(predicted.std()), 4)}
    observed, predicted, _ = stacked(frame)
    slope, _ = np.polyfit(predicted, observed, 1)
    out["pooled"] = {
        "slope": round(float(slope), 4),
        "sd_predicted": round(float(predicted.std()), 4),
        "sd_residual": round(float(np.std(observed - predicted)), 4),
        # How much of a game's variation the model can even see. Below a
        # quarter, which is why the priced width dominates the priced mean.
        "signal_to_noise": round(float(predicted.std()
                                       / np.std(observed - predicted)), 4),
    }
    return out


def quintile_gaps(frame):
    """Observed minus predicted by predicted-runs quintile, and per season.

    Cut points come from the pooled distribution so every season is scored on
    the same buckets rather than on its own quantiles, which would hide a
    common shift by construction.
    """
    observed, predicted, season = stacked(frame)
    edges = np.quantile(predicted, np.linspace(0, 1, QUINTILES + 1))
    buckets = []
    for low, high in zip(edges[:-1], edges[1:]):
        top = high >= edges[-1]
        inside = (predicted >= low) & (predicted <= high if top
                                       else predicted < high)
        per_season = []
        for value in sorted(set(season)):
            block = inside & (season == value)
            if block.sum() > 50:
                per_season.append(float(observed[block].mean()
                                        - predicted[block].mean()))
        per_season = np.array(per_season)
        buckets.append({
            "predicted_from": round(float(low), 2),
            "predicted_to": round(float(high), 2),
            "games": int(inside.sum()),
            "gap": round(float(observed[inside].mean()
                               - predicted[inside].mean()), 4),
            "seasons_positive": int((per_season > 0).sum()),
            "seasons": len(per_season),
            "season_gap_mean": round(float(per_season.mean()), 4),
            "season_gap_sd": round(float(per_season.std(ddof=1)), 4),
        })
    return buckets


def calibration(probability, outcome, bins=CALIBRATION_BINS):
    """Slope and mean absolute gap of the reliability curve.

    The slope is the number that matters here. One is right; below one is
    over-confident and above one is under-confident, and a correction that
    overshoots shows up as a slope that crosses from one side to the other.
    """
    # Deduplicated, because a concentrated forecast collapses the quantile
    # edges onto each other and every bin then selects the same games. That
    # returns a slope of zero from ten identical points, which reads as a
    # catastrophically bad forecast rather than as an unanswerable question.
    edges = np.unique(np.quantile(probability, np.linspace(0, 1, bins + 1)))
    if len(edges) < 4:
        return None
    predicted, actual, weights = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        top = high >= edges[-1]
        inside = (probability >= low) & (probability <= high if top
                                         else probability < high)
        if inside.sum() > 30:
            predicted.append(float(probability[inside].mean()))
            actual.append(float(outcome[inside].mean()))
            weights.append(int(inside.sum()))
    # Three distinct points, not three bins: a slope through repeated x is
    # not a slope.
    if len(set(predicted)) < 3:
        return None
    slope = float(np.polyfit(predicted, actual, 1)[0])
    error = float(np.average(np.abs(np.array(predicted) - np.array(actual)),
                             weights=weights))
    return {"slope": round(slope, 4), "error": round(error, 5)}


def _fits(kind, predicted, observed):
    """A recalibration map from predicted runs to observed runs."""
    if kind == "linear":
        slope, intercept = np.polyfit(predicted, observed, 1)
        return lambda values: intercept + slope * values
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(out_of_bounds="clip").fit(predicted, observed)
    return lambda values: model.predict(values)


def _price(mean_home, mean_away, dispersion, shape, walk_off, edge, margins):
    home = uncensor_home_mean(mean_home, mean_away, dispersion,
                              walk_off_margins=walk_off,
                              extra_inning_home_edge=edge,
                              extra_inning_margins=margins, shape=shape)
    joint = joint_distribution(home, mean_away, dispersion,
                               walk_off_margins=walk_off,
                               extra_inning_home_edge=edge,
                               extra_inning_margins=margins, shape=shape)
    home_axis, away_axis = GRID[:, None], GRID[None, :]
    return {
        "moneyline": (joint * (home_axis > away_axis)).sum(axis=(-2, -1)),
        "runline": (joint * ((home_axis - away_axis) > 1.5)).sum(axis=(-2, -1)),
        "total": (joint * ((home_axis + away_axis) > 8.5)).sum(axis=(-2, -1)),
    }


def _loss(probability, outcome):
    probability = np.clip(np.asarray(probability, float), 1e-9, 1 - 1e-9)
    outcome = np.asarray(outcome, float)
    return -(outcome * np.log(probability)
             + (1 - outcome) * np.log(1 - probability))


def corrections(frame, verbose=True):
    """Walk forward pricing three ways: uncorrected, linear, isotonic.

    Everything is fitted on prior seasons only, the corrections included. The
    width is refitted on the corrected predictions rather than carried over,
    because a mean and a width that disagree about which predictions they
    describe would make the comparison meaningless.
    """
    kinds = ("plain", "linear", "isotonic")
    blocks, probabilities = [], {kind: {"p": [], "y": []} for kind in kinds}
    for season in sorted(frame["season"].unique()):
        train = frame[frame["season"] < season]
        test = frame[frame["season"] == season].reset_index(drop=True)
        if len(train) < MIN_TRAIN_GAMES or not len(test):
            continue
        walk_off = calibrate_walk_off(train)
        edge, margins = calibrate_extra_innings(train)
        train_mean = {side: train[f"expected_{side}_runs"].to_numpy(float)
                      for side in ("home", "away")}
        train_obs = {side: train[f"{side}_score"].to_numpy(float)
                     for side in ("home", "away")}
        test_mean = {side: test[f"expected_{side}_runs"].to_numpy(float)
                     for side in ("home", "away")}
        outcomes = {
            "moneyline": test["home_win"].to_numpy(float),
            "runline": ((test["home_score"] - test["away_score"]) > 1.5)
            .to_numpy(float),
            "total": (test["total_runs"] > 8.5).to_numpy(float),
        }
        block = {"season": int(season),
                 "date": test["official_date"].to_numpy()}
        for kind in kinds:
            if kind == "plain":
                mean_home, mean_away = test_mean["home"], test_mean["away"]
                shape = (float(test["inning_scoreless"].iloc[0]),
                         float(test["inning_tail"].iloc[0]))
                dispersion = (float(test["dispersion_home"].iloc[0]),
                              float(test["dispersion_away"].iloc[0]))
            else:
                maps = {side: _fits(kind, train_mean[side], train_obs[side])
                        for side in ("home", "away")}
                mean_home = np.clip(maps["home"](test_mean["home"]), 0.2, 20.0)
                mean_away = np.clip(maps["away"](test_mean["away"]), 0.2, 20.0)
                fitted = {side: np.clip(maps[side](train_mean[side]), 0.2, 20.0)
                          for side in ("home", "away")}
                dispersion = (fit_dispersion(train_obs["home"], fitted["home"]),
                              fit_dispersion(train_obs["away"], fitted["away"]))
                shape = fit_inning_shape(train_obs["away"], fitted["away"])
            priced = _price(mean_home, mean_away, dispersion, shape,
                            walk_off, edge, margins)
            for market, probability in priced.items():
                block[f"{kind}_{market}"] = _loss(probability,
                                                  outcomes[market])
            probabilities[kind]["p"].append(priced["moneyline"])
            probabilities[kind]["y"].append(outcomes["moneyline"])
        blocks.append(pd.DataFrame(block))
        if verbose:
            print(f"  {season}: priced {len(test)} games three ways")

    losses = pd.concat(blocks, ignore_index=True)
    rng = np.random.default_rng(0)
    dates = losses["date"].to_numpy()
    unique = np.unique(dates)
    by_date = {date: np.flatnonzero(dates == date) for date in unique}

    out = {"games": int(len(losses)), "markets": {}, "probability": {}}
    for kind in kinds:
        stack = np.concatenate(probabilities[kind]["p"])
        out["probability"][kind] = calibration(
            stack, np.concatenate(probabilities[kind]["y"]))
        out["probability"][kind]["sd_predicted"] = round(float(stack.std()), 4)
    for market in ("moneyline", "runline", "total"):
        base = losses[f"plain_{market}"].to_numpy()
        entry = {"plain": round(float(base.mean()), 5)}
        for kind in ("linear", "isotonic"):
            other = losses[f"{kind}_{market}"].to_numpy()
            draws = []
            for _ in range(DRAWS):
                pick = rng.choice(unique, len(unique), replace=True)
                index = np.concatenate([by_date[date] for date in pick])
                draws.append(other[index].mean() - base[index].mean())
            low, high = np.percentile(draws, [5, 95])
            entry[kind] = {
                "log_loss": round(float(other.mean()), 5),
                "delta": round(float(other.mean() - base.mean()), 5),
                "delta_ci90_date_clustered": [round(float(low), 5),
                                              round(float(high), 5)],
                "helps": bool(high < 0),
            }
        out["markets"][market] = entry
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--predictions", default="data/predictions_glm.csv")
    parser.add_argument("--report", default="mean_calibration.json")
    parser.add_argument("--corrections", action="store_true",
                        help="also refit and reprice; takes minutes")
    args = parser.parse_args()

    frame = load(args.games, args.predictions)
    result = {"games": int(len(frame)),
              "attenuation": attenuation(frame),
              "quintiles": quintile_gaps(frame)}

    print(f"{len(frame):,} walk-forward games\n")
    pooled = result["attenuation"]["pooled"]
    print(f"regression of observed runs on predicted: slope "
          f"{pooled['slope']} (1.0 would be calibrated)")
    print(f"  home {result['attenuation']['home']['slope']}, "
          f"away {result['attenuation']['away']['slope']}")
    print(f"  sd of predicted mean {pooled['sd_predicted']} against residual "
          f"{pooled['sd_residual']}: signal is {pooled['signal_to_noise']} "
          f"of the noise\n")
    print("predicted runs      games      gap   seasons with the same sign")
    for bucket in result["quintiles"]:
        print(f"{bucket['predicted_from']:>6.2f}-{bucket['predicted_to']:<6.2f}"
              f"{bucket['games']:>10}  {bucket['gap']:>+7.3f}   "
              f"{bucket['seasons_positive']}/{bucket['seasons']} positive")

    if args.corrections:
        print("\nrefitting and repricing")
        result["corrections"] = corrections(frame)
        print("\nmarket       plain    linear      delta   isotonic      delta")
        for market, entry in result["corrections"]["markets"].items():
            print(f"{market:<10} {entry['plain']:>7.5f} "
                  f"{entry['linear']['log_loss']:>9.5f} "
                  f"{entry['linear']['delta']:>+10.5f} "
                  f"{entry['isotonic']['log_loss']:>10.5f} "
                  f"{entry['isotonic']['delta']:>+10.5f}")
        print("\nmoneyline probability calibration "
              "(1.0 right, >1 under-confident)")
        for kind, entry in result["corrections"]["probability"].items():
            print(f"  {kind:<9} slope {entry['slope']:>6.3f}   "
                  f"error {entry['error']:.5f}   "
                  f"sd(prob) {entry['sd_predicted']:.4f}")

    Path(args.report).write_text(json.dumps(result, indent=2),
                                 encoding="utf-8")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
