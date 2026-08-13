"""Frozen research comparison of walk-forward models versus MLB closes.

This script does not tune a model, select a betting threshold, or write to a
ledger.  It applies the evaluation plan frozen before the completed archive
was inspected: 2022-23 are diagnostic development seasons and 2024 is the
confirmation season.  The ensemble is fixed at a 50/50 probability average.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from csv_collection import read_csv_collection
from market import build_priced_games, delta_interval, log_loss, verdict
from models import reprice_requests
from provenance import repository_revision
from validate import brier, calibration_error


DEVELOPMENT_SEASONS = (2022, 2023)
CONFIRMATION_SEASONS = (2024,)
MODEL_COLUMNS = {
    "gbm": "model_prob_gbm",
    "glm": "model_prob_glm",
    "equal_weight_ensemble": "model_prob_ensemble",
}


def _outcome(frame, market):
    if market == "h2h":
        return frame["home_win"].astype(float), np.zeros(len(frame), dtype=bool)
    point = frame["close_point"].astype(float)
    if market == "spreads":
        return ((frame["run_diff"] > -point).astype(float),
                np.isclose(frame["run_diff"], -point))
    return ((frame["total_runs"] > point).astype(float),
            np.isclose(frame["total_runs"], point))


def build_evaluation_rows(quotes, games, predictions):
    """One no-push game-market row with close and all candidate forecasts."""
    priced, unmatched = build_priced_games(quotes, games)
    outcomes = games[["game_pk", "season", "home_win", "total_runs",
                      "run_diff"]]
    base = priced.merge(outcomes, on="game_pk", how="inner")
    base = base[(base["close_books"] >= 3) & base["close_prob"].notna()]
    base = base[base["home_win"].notna() & base["total_runs"].notna()
                & base["run_diff"].notna()]

    rows = []
    for market in ("h2h", "spreads", "totals"):
        block = base[base["market"] == market].copy().reset_index(drop=True)
        if market != "h2h":
            block = block[block["close_point"].notna()].reset_index(drop=True)
        outcome, pushed = _outcome(block, market)
        block["outcome"] = np.asarray(outcome, dtype=float)
        block = block.loc[~np.asarray(pushed)].reset_index(drop=True)
        requests = block[["game_pk", "official_date", "market",
                          "close_point"]].rename(columns={"close_point": "point"})
        for name, prediction_frame in predictions.items():
            repriced = reprice_requests(requests, prediction_frame)
            block[f"model_prob_{name}"] = repriced["model_prob_home"].to_numpy()
        block["model_prob_ensemble"] = block[
            ["model_prob_gbm", "model_prob_glm"]].mean(axis=1, skipna=False)
        keep = ["event_id", "game_pk", "official_date", "season", "market",
                "close_point", "close_prob", "close_books", "outcome",
                "model_prob_gbm", "model_prob_glm", "model_prob_ensemble"]
        rows.append(block[keep])
    frame = pd.concat(rows, ignore_index=True)
    return frame, unmatched, priced


def _metric_block(frame, model_column):
    sample = frame.dropna(subset=[model_column, "close_prob", "outcome"]).copy()
    if len(sample) < 50:
        return {"games": int(len(sample)), "status": "insufficient_sample"}
    model = sample[model_column].to_numpy(float)
    market = sample["close_prob"].to_numpy(float)
    outcome = sample["outcome"].to_numpy(float)
    interval = delta_interval(sample, model_column, "close_prob", outcome)
    return {
        "games": int(len(sample)),
        "model_log_loss": round(log_loss(model, outcome), 6),
        "market_log_loss": round(log_loss(market, outcome), 6),
        "log_loss_delta_model_minus_market": round(
            log_loss(model, outcome) - log_loss(market, outcome), 6),
        "delta_ci90_date_clustered": interval,
        "model_brier": round(brier(model, outcome), 6),
        "market_brier": round(brier(market, outcome), 6),
        "model_calibration_error": calibration_error(model, outcome),
        "market_calibration_error": calibration_error(market, outcome),
        "mean_abs_disagreement_points": round(
            float(np.mean(np.abs(model - market)) * 100), 3),
        "verdict": verdict(interval),
    }


def select_market_anchor_weight(frame, model_column):
    """Development-only convex shrinkage from the close toward the model."""
    sample = frame.dropna(subset=[model_column, "close_prob", "outcome"])
    if len(sample) < 50:
        return None
    market = sample["close_prob"].to_numpy(float)
    model = sample[model_column].to_numpy(float)
    outcome = sample["outcome"].to_numpy(float)
    best_weight, best_loss = 0.0, log_loss(market, outcome)
    # Ties stay closer to the market because the burden is on the model to
    # demonstrate incremental information, not merely equal performance.
    for weight in np.linspace(0.01, 1.0, 100):
        probability = market + weight * (model - market)
        loss = log_loss(probability, outcome)
        if loss < best_loss - 1e-12:
            best_weight, best_loss = float(weight), loss
    return round(best_weight, 2)


def market_anchored_test(rows):
    """Select shrinkage on 2022-23 and apply it unchanged to 2024."""
    development = rows[rows["season"].isin(DEVELOPMENT_SEASONS)]
    confirmation = rows[rows["season"].isin(CONFIRMATION_SEASONS)]
    result = {
        "status": "exploratory_second_stage_no_promotion",
        "formula": "close + weight * (model - close)",
        "weight_selection": "0.00..1.00 by 0.01; minimum 2022-23 log loss",
        "markets": {},
    }
    for market in ("h2h", "spreads", "totals"):
        result["markets"][market] = {}
        dev_market = development[development["market"] == market]
        confirm_market = confirmation[confirmation["market"] == market]
        for model, column in MODEL_COLUMNS.items():
            weight = select_market_anchor_weight(dev_market, column)
            if weight is None:
                result["markets"][market][model] = {
                    "status": "insufficient_sample"}
                continue
            dev_scored = dev_market.copy()
            confirm_scored = confirm_market.copy()
            for frame in (dev_scored, confirm_scored):
                frame["market_anchored_prob"] = (
                    frame["close_prob"]
                    + weight * (frame[column] - frame["close_prob"]))
            confirmation_metric = _metric_block(
                confirm_scored, "market_anchored_prob")
            interval = confirmation_metric.get("delta_ci90_date_clustered")
            result["markets"][market][model] = {
                "selected_weight": weight,
                "development": _metric_block(dev_scored, "market_anchored_prob"),
                "confirmation": confirmation_metric,
                "confirmation_incremental_signal": bool(
                    interval is not None and interval[1] < 0),
            }
    return result


def evaluate(rows, unmatched, priced, predictions, raw_quote_events):
    trained_through = {
        name: sorted(set(frame["trained_through"].dropna().astype(str)))
        for name, frame in predictions.items()
    }
    leakage_violations = {}
    for name, frame in predictions.items():
        probe = rows[["game_pk", "official_date"]].drop_duplicates().merge(
            frame[["game_pk", "trained_through"]], on="game_pk", how="inner")
        leakage_violations[name] = int((
            pd.to_datetime(probe["trained_through"], errors="coerce")
            >= pd.to_datetime(probe["official_date"], errors="coerce")
        ).sum())

    unmatched_by_year = {}
    for day, _, _ in unmatched:
        unmatched_by_year[day[:4]] = unmatched_by_year.get(day[:4], 0) + 1
    eligible = rows.dropna(subset=["model_prob_gbm", "model_prob_glm"])
    games_by_season = {
        str(int(season)): int(count)
        for season, count in eligible.groupby("season")["game_pk"].nunique().items()
    }
    report = {
        "status": "research_only_no_promotion",
        "repository_revision": repository_revision(),
        "protocol": {
            "development_seasons": list(DEVELOPMENT_SEASONS),
            "confirmation_seasons": list(CONFIRMATION_SEASONS),
            "ensemble": "arithmetic mean of GBM and GLM probabilities",
            "primary_metric": "log loss versus devigged closing consensus",
            "selection_rule": "no ROI-only selection or betting promotion",
        },
        "integrity": {
            "raw_quote_events": int(raw_quote_events),
            "matched_to_schedule_events": int(raw_quote_events - len(unmatched)),
            "matched_with_qualified_market_events": (
                int(priced["event_id"].nunique()) if len(priced) else 0),
            "unmatched_odds_events": int(len(unmatched)),
            "unmatched_odds_events_by_year": unmatched_by_year,
            "evaluation_rows": int(len(rows)),
            "evaluation_games": int(rows["game_pk"].nunique()),
            "candidate_games_2022_2024": int(eligible["game_pk"].nunique()),
            "candidate_games_by_season": games_by_season,
            "prediction_trained_through_values": trained_through,
            "trained_through_on_or_after_game_violations": leakage_violations,
        },
        "splits": {},
        "note": (
            "2024 is a confirmation window for this run, not a pristine trial: "
            "the repository was developed around historical MLB data. No result "
            "here alone authorizes a wager."
        ),
    }
    split_definitions = {
        "development_2022_2023": DEVELOPMENT_SEASONS,
        "confirmation_2024": CONFIRMATION_SEASONS,
        "all_2022_2024_diagnostic": DEVELOPMENT_SEASONS + CONFIRMATION_SEASONS,
    }
    for split_name, seasons in split_definitions.items():
        split = rows[rows["season"].isin(seasons)]
        report["splits"][split_name] = {}
        for market in ("h2h", "spreads", "totals"):
            market_rows = split[split["market"] == market]
            report["splits"][split_name][market] = {
                model: _metric_block(market_rows, column)
                for model, column in MODEL_COLUMNS.items()
            }
    report["market_anchored_incremental_test"] = market_anchored_test(rows)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", default="data/full_game_event_quotes")
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--gbm", default="data/predictions_gbm.csv")
    parser.add_argument("--glm", default="data/predictions_glm.csv")
    parser.add_argument("--rows", default="data/research/full_game_close_rows.csv")
    parser.add_argument("--report", default="full_game_close_evaluation.json")
    args = parser.parse_args(argv)

    quotes = read_csv_collection(args.quotes)
    games = pd.read_csv(args.games)
    predictions = {"gbm": pd.read_csv(args.gbm), "glm": pd.read_csv(args.glm)}
    rows, unmatched, priced = build_evaluation_rows(quotes, games, predictions)
    row_path = Path(args.rows)
    row_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(row_path, index=False)
    report = evaluate(rows, unmatched, priced, predictions,
                      quotes["event_id"].nunique())
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
