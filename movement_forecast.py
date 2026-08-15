"""Serve the confirmed 24-hour-to-close movement study as a paper probe.

This is not an outcome model and it does not create a betting path.  It turns
the candidate selected on 2022-23 into a frozen linear artifact, applies it
only near the historical 24-hour entry horizon, and lets ``signal_ledger``
record the prediction for later sharp-close evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from csv_collection import read_csv_collection
from full_game_movement_evaluation import (
    CONFIRMATION_YEAR, MOVE_CLIP, SELECTION_YEAR, TRAIN_YEAR,
    _candidate, _logit, build_movement_rows,
)
from market import build_priced_games
from provenance import repository_revision


VERSION = "full-game-24h-close-probe-v1"
FIT_YEARS = (TRAIN_YEAR, SELECTION_YEAR)
# The historical feature is concentrated at 24.01-24.10 hours. Hourly live
# capture needs a little tolerance, but applying this model in the 4-hour lock
# window would be unsupported extrapolation.
ENTRY_WINDOW_MINUTES = (23 * 60, 25 * 60)
MIN_BOOKS = 3


def _sigmoid(value):
    value = np.clip(np.asarray(value, float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def fit(rows, evaluation):
    """Fit only the candidate configurations locked before confirmation."""
    models = {}
    for market, block in evaluation.get("markets", {}).items():
        selection = block.get("candidate_selection") or {}
        features = tuple(selection.get("selected_features") or ())
        alpha = selection.get("selected_alpha")
        if not features or alpha is None or not block.get("confirmation_signal"):
            continue
        train = rows[(rows["market"] == market)
                     & rows["season"].isin(FIT_YEARS)].copy()
        if not len(train):
            continue
        pipeline = _candidate(features, alpha)
        pipeline.fit(train[list(features)].to_numpy(float),
                     train["move_logit"].to_numpy(float))
        scaler = pipeline.named_steps["standardscaler"]
        ridge = pipeline.named_steps["ridge"]
        models[market] = {
            "features": list(features),
            "alpha": float(alpha),
            "training_rows": int(len(train)),
            "training_years": list(FIT_YEARS),
            "scaler_mean": [round(float(value), 12)
                            for value in scaler.mean_],
            "scaler_scale": [round(float(value), 12)
                             for value in scaler.scale_],
            "coefficients": [round(float(value), 12)
                             for value in ridge.coef_],
            "intercept": round(float(ridge.intercept_), 12),
            "confirmation_2024": block.get("confirmation_2024"),
        }
    dates = pd.to_datetime(
        rows.loc[rows["season"].isin(FIT_YEARS), "official_date"],
        errors="coerce")
    return {
        "version": VERSION,
        "repository_revision": repository_revision(),
        "source_protocol": evaluation.get("protocol", {}).get("version"),
        "source_protocol_hash": evaluation.get(
            "protocol", {}).get("protocol_hash"),
        "target": "24h_entry_to_20m_close",
        "entry_window_minutes": list(ENTRY_WINDOW_MINUTES),
        "minimum_books": MIN_BOOKS,
        "fit_years": list(FIT_YEARS),
        "confirmation_year": CONFIRMATION_YEAR,
        "fitted_through": (str(dates.max().date())
                           if len(dates) and dates.notna().any() else None),
        "promotion_status": "paper_clv_probe_only",
        "bets_placed": 0,
        "markets": models,
    }


def load(path="movement_model.json"):
    path = Path(path)
    if not path.exists():
        return {"version": VERSION, "promotion_status": "unavailable",
                "markets": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value, default=0.0):
    try:
        number = float(value)
        return number if np.isfinite(number) else float(default)
    except (TypeError, ValueError):
        return float(default)


def live_features(model_probability, market_probability, leader_probability,
                  follower_probability, market_spread, market_books,
                  lead_minutes, point, official_date):
    entry_logit = float(_logit([market_probability])[0])
    model_logit = float(_logit([model_probability])[0])
    leader = _number(leader_probability, np.nan)
    follower = _number(follower_probability, np.nan)
    leader_available = float(np.isfinite(leader))
    follower_available = float(np.isfinite(follower))
    date = pd.to_datetime(official_date, errors="coerce")
    if pd.isna(date):
        raise ValueError("official_date is required for movement features")
    month = float(date.month)
    return {
        "entry_logit": entry_logit,
        "abs_entry_logit": abs(entry_logit),
        "leader_gap_logit": (float(_logit([leader])[0]) - entry_logit
                             if leader_available else 0.0),
        "follower_gap_logit": (float(_logit([follower])[0]) - entry_logit
                               if follower_available else 0.0),
        "leader_available": leader_available,
        "follower_available": follower_available,
        "entry_market_spread": _number(market_spread),
        "entry_books": _number(market_books),
        "entry_lead_hours": _number(lead_minutes) / 60.0,
        "point": _number(point),
        "month_sin": float(np.sin(2 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * month / 12.0)),
        "model_gap_logit": model_logit - entry_logit,
    }


def apply(model_probability, market_probability, market, artifact,
          leader_probability=None, follower_probability=None,
          market_spread=0.0, market_books=0, lead_minutes=0, point=0.0,
          official_date=None):
    """Predict the later close only inside the artifact's supported horizon."""
    version = artifact.get("version", VERSION)
    target = artifact.get("target", "24h_entry_to_20m_close")
    model = artifact.get("markets", {}).get(market)
    window = artifact.get("entry_window_minutes", ENTRY_WINDOW_MINUTES)
    eligible = bool(
        model
        and len(window) == 2
        and float(window[0]) <= _number(lead_minutes) <= float(window[1])
        and _number(market_books) >= float(artifact.get(
            "minimum_books", MIN_BOOKS))
    )
    if not eligible:
        return {
            "predicted_close_prob_home": float(market_probability),
            "predicted_clv": 0.0,
            "eligible": False,
            "version": version,
            "target": target,
        }
    try:
        values = live_features(
            model_probability, market_probability, leader_probability,
            follower_probability, market_spread, market_books, lead_minutes,
            point, official_date)
        vector = np.asarray([values[name] for name in model["features"]], float)
        mean = np.asarray(model["scaler_mean"], float)
        scale = np.asarray(model["scaler_scale"], float)
        coefficients = np.asarray(model["coefficients"], float)
        if not (len(vector) == len(mean) == len(scale) == len(coefficients)):
            raise ValueError("movement artifact feature lengths differ")
        standard = (vector - mean) / np.where(scale == 0, 1.0, scale)
        movement = float(model["intercept"] + np.dot(coefficients, standard))
        movement = float(np.clip(movement, -MOVE_CLIP, MOVE_CLIP))
        entry_logit = float(_logit([market_probability])[0])
        predicted = float(_sigmoid([entry_logit + movement])[0])
    except (KeyError, TypeError, ValueError):
        return {
            "predicted_close_prob_home": float(market_probability),
            "predicted_clv": 0.0,
            "eligible": False,
            "version": version,
            "target": target,
        }
    return {
        "predicted_close_prob_home": predicted,
        "predicted_clv": predicted - float(market_probability),
        "predicted_move_logit": movement,
        "eligible": True,
        "version": version,
        "target": target,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", default="data/full_game_event_quotes")
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--predictions", default="data/predictions_glm.csv")
    parser.add_argument("--evaluation",
                        default="full_game_movement_evaluation.json")
    parser.add_argument("--out", default="movement_model.json")
    args = parser.parse_args(argv)

    quotes = read_csv_collection(args.quotes)
    priced, _ = build_priced_games(quotes, pd.read_csv(args.games))
    rows, integrity = build_movement_rows(
        priced, pd.read_csv(args.predictions))
    if integrity.get("trained_through_violations"):
        raise SystemExit("refusing to fit movement artifact with leakage")
    evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    artifact = fit(rows, evaluation)
    Path(args.out).write_text(json.dumps(artifact, indent=2) + "\n",
                              encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
