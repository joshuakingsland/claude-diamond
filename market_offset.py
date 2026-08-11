"""A conservative market-offset model for live fair probabilities.

The standalone runs model is useful baseball research but the 2025 comparison
shows that it is a worse probability prior than the market.  The live decision
therefore starts from the de-vigged market logit and lets historical evidence
choose how much of the model-market residual to retain.  A weight of zero is
the market; one is the old standalone model.  Weights are constrained to
``[0, 1]`` so a thin sample cannot turn disagreement into an anti-model trade.

A second weight asks whether the same residual predicts movement from entry to
the later main-line snapshot.  That is an explicit closing-line-value target,
kept separate from outcome prediction.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from market import build_priced_games, log_loss
from models import reprice_requests
from provenance import repository_revision

OFFSET_VERSION = "market-logit-offset-v1"
MIN_FIT_ROWS = 250
RIDGE = 0.002


def _logit(probability):
    probability = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1.0 - probability))


def _sigmoid(value):
    value = np.clip(np.asarray(value, float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def blend_probability(market_probability, model_probability, weight):
    market_logit = _logit(market_probability)
    residual = _logit(model_probability) - market_logit
    return _sigmoid(market_logit + float(weight) * residual)


def _date_cluster_interval(dates, values, draws=2000, seed=31):
    dates = np.asarray(dates)
    values = np.asarray(values, float)
    unique = np.unique(dates)
    if len(unique) < 10 or not len(values):
        return None
    positions = {date: np.flatnonzero(dates == date) for date in unique}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        selected = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([positions[date] for date in selected])
        samples.append(float(values[index].mean()))
    return [round(float(np.percentile(samples, 2.5)), 8),
            round(float(np.percentile(samples, 97.5)), 8)]


def _outcomes(frame):
    if not len(frame):
        return np.array([]), np.array([], dtype=bool)
    market = frame["market"].iloc[0]
    if market == "h2h":
        return frame["home_win"].to_numpy(float), np.zeros(len(frame), bool)
    point = frame["point"].to_numpy(float)
    if market == "spreads":
        result = frame["run_diff"].to_numpy(float)
        return (result > -point).astype(float), np.isclose(result, -point)
    result = frame["total_runs"].to_numpy(float)
    return (result > point).astype(float), np.isclose(result, point)


def training_rows(priced, predictions, games, label="close"):
    """Outcome rows at the exact main point carried by ``label``."""
    probability_column = f"{label}_prob"
    point_column = f"{label}_point"
    outcomes = games[["game_pk", "home_win", "run_diff", "total_runs"]]
    merged = (priced.merge(predictions, on="game_pk", how="inner")
                    .merge(outcomes, on="game_pk", how="inner"))
    merged = merged[merged[probability_column].notna()
                    & merged["home_win"].notna()].copy()
    rows = []
    for market, block in merged.groupby("market"):
        block = block.copy()
        if market != "h2h":
            block = block[block[point_column].notna()]
        requests = block[["game_pk", "official_date", "market",
                          point_column]].rename(columns={point_column: "point"})
        repriced = reprice_requests(requests, block)
        block = block.reset_index(drop=True)
        block["point"] = requests.reset_index(drop=True)["point"]
        block["model_prob"] = repriced["model_prob_home"]
        block["market_prob"] = block[probability_column].to_numpy(float)
        outcome, pushed = _outcomes(block)
        block["outcome"] = outcome
        block = block.loc[~pushed & block["model_prob"].notna()]
        rows.append(block[["game_pk", "official_date", "market", "point",
                           "model_prob", "market_prob", "outcome"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _walk_forward_outcome(frame):
    if "official_date" not in frame or len(frame) < MIN_FIT_ROWS * 2:
        return {"rows": 0, "status": "insufficient forward sample"}
    frame = frame.sort_values("official_date").reset_index(drop=True)
    dates = np.array(sorted(frame["official_date"].astype(str).unique()))
    if len(dates) < 20:
        return {"rows": 0, "status": "insufficient forward sample"}
    cuts = [int(len(dates) * fraction) for fraction in (0.5, 0.67, 0.84)]
    market_values, offset_values, outcomes, test_dates, weights = [], [], [], [], []
    for start, stop in zip(cuts, cuts[1:] + [len(dates)]):
        train = frame[frame["official_date"].astype(str).isin(set(dates[:start]))]
        test = frame[frame["official_date"].astype(str).isin(
            set(dates[start:stop]))]
        if len(train) < MIN_FIT_ROWS or not len(test):
            continue
        weight = fit_outcome_weight(train, include_forward=False)["weight"]
        market = test["market_prob"].to_numpy(float)
        model = test["model_prob"].to_numpy(float)
        market_values.extend(market)
        offset_values.extend(blend_probability(market, model, weight))
        outcomes.extend(test["outcome"].to_numpy(float))
        test_dates.extend(test["official_date"].astype(str))
        weights.append(weight)
    if not outcomes:
        return {"rows": 0, "status": "insufficient forward sample"}
    market_values = np.asarray(market_values, float)
    offset_values = np.asarray(offset_values, float)
    outcomes = np.asarray(outcomes, float)
    market_row_loss = -(outcomes * np.log(np.clip(market_values, 1e-9, 1))
                        + (1 - outcomes) * np.log(np.clip(
                            1 - market_values, 1e-9, 1)))
    offset_row_loss = -(outcomes * np.log(np.clip(offset_values, 1e-9, 1))
                        + (1 - outcomes) * np.log(np.clip(
                            1 - offset_values, 1e-9, 1)))
    improvement = market_row_loss - offset_row_loss
    return {
        "rows": int(len(outcomes)),
        "mean_weight": round(float(np.mean(weights)), 6),
        "log_loss_market": round(log_loss(market_values, outcomes), 6),
        "log_loss_offset": round(log_loss(offset_values, outcomes), 6),
        "log_loss_improvement": round(float(improvement.mean()), 8),
        "improvement_ci95_date_clustered": _date_cluster_interval(
            test_dates, improvement),
        "status": "forward by date; research only",
    }


def fit_outcome_weight(frame, include_forward=True):
    if len(frame) < MIN_FIT_ROWS:
        return {"rows": int(len(frame)), "weight": 0.0,
                "status": "insufficient sample; market-only"}
    market = frame["market_prob"].to_numpy(float)
    model = frame["model_prob"].to_numpy(float)
    outcome = frame["outcome"].to_numpy(float)

    def objective(weight):
        return log_loss(blend_probability(market, model, weight), outcome) \
            + RIDGE * float(weight) ** 2

    fitted = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    research_weight = float(fitted.x)
    blended = blend_probability(market, model, research_weight)
    report = {
        "rows": int(len(frame)),
        "weight": round(research_weight, 6),
        "log_loss_market": round(log_loss(market, outcome), 6),
        "log_loss_standalone": round(log_loss(model, outcome), 6),
        "log_loss_research_offset": round(log_loss(blended, outcome), 6),
        "log_loss_offset": round(log_loss(blended, outcome), 6),
        "status": "research_only; selected on historical outcomes",
    }
    if include_forward:
        forward = _walk_forward_outcome(frame)
        interval = forward.get("improvement_ci95_date_clustered")
        supported = bool(interval and interval[0] > 0)
        report["research_weight"] = report["weight"]
        report["weight"] = (round(float(forward.get("mean_weight", 0.0)), 6)
                            if supported else 0.0)
        report["walk_forward"] = forward
        report["deployment_status"] = (
            "forward-supported" if supported else "market-only")
    return report


def _movement_arrays(block):
    entry = _logit(block["entry_prob"])
    close = _logit(block["close_prob"])
    model_residual = _logit(block["model_prob"]) - entry
    leader_probability = block["entry_leader_prob"].fillna(
        block["entry_prob"]).to_numpy(float)
    leader_residual = _logit(leader_probability) - entry
    return entry, close, model_residual, leader_residual


def _fit_movement_coefficients(block):
    entry, close, model_residual, leader_residual = _movement_arrays(block)

    def objective(weights):
        predicted = (entry + float(weights[0]) * model_residual
                      + float(weights[1]) * leader_residual)
        return float(np.mean((close - predicted) ** 2)
                     + RIDGE * np.sum(np.asarray(weights) ** 2))

    fitted = minimize(objective, x0=np.array([0.0, 0.0]),
                      bounds=((0.0, 1.0), (0.0, 1.0)), method="L-BFGS-B")
    return np.asarray(fitted.x, float)


def _walk_forward_movement(block):
    """Expanding-date evaluation; every target comes after its fitted rows."""
    block = block.sort_values("official_date").reset_index(drop=True)
    dates = np.array(sorted(block["official_date"].astype(str).unique()))
    if len(block) < MIN_FIT_ROWS * 2 or len(dates) < 20:
        return {"rows": 0, "status": "insufficient forward sample"}
    cut_points = [int(len(dates) * fraction) for fraction in (0.5, 0.67, 0.84)]
    predicted, baseline, target, evaluated_dates = [], [], [], []
    weights = []
    for start, stop in zip(cut_points, cut_points[1:] + [len(dates)]):
        train_dates = set(dates[:start])
        fold_dates = set(dates[start:stop])
        train = block[block["official_date"].astype(str).isin(train_dates)]
        test = block[block["official_date"].astype(str).isin(fold_dates)]
        if len(train) < MIN_FIT_ROWS or not len(test):
            continue
        fitted = _fit_movement_coefficients(train)
        entry, close, model_residual, leader_residual = _movement_arrays(test)
        predicted.extend(entry + fitted[0] * model_residual
                         + fitted[1] * leader_residual)
        baseline.extend(entry)
        target.extend(close)
        evaluated_dates.extend(test["official_date"].astype(str))
        weights.append(fitted)
    if not predicted:
        return {"rows": 0, "status": "insufficient forward sample"}
    predicted = np.asarray(predicted)
    baseline = np.asarray(baseline)
    target = np.asarray(target)
    improvement = ((target - baseline) ** 2
                   - (target - predicted) ** 2)
    return {
        "rows": int(len(target)),
        "rmse_entry_logit": round(float(np.sqrt(np.mean(
            (target - baseline) ** 2))), 6),
        "rmse_offset_logit": round(float(np.sqrt(np.mean(
            (target - predicted) ** 2))), 6),
        "mse_improvement": round(float(improvement.mean()), 8),
        "improvement_ci95_date_clustered": _date_cluster_interval(
            evaluated_dates, improvement),
        "mean_model_weight": round(float(np.mean([w[0] for w in weights])), 6),
        "mean_leader_weight": round(float(np.mean([w[1] for w in weights])), 6),
        "status": "forward by date; research only",
    }


def fit_movement_weight(priced, predictions):
    """Model and market-leader offsets that predict the later main line."""
    both = priced[priced["entry_prob"].notna()
                  & priced["close_prob"].notna()].copy()
    same = ((both["market"] == "h2h")
            | np.isclose(both["entry_point"].astype(float),
                         both["close_point"].astype(float), equal_nan=False))
    both = both[same]
    if not len(both):
        return {}
    requests = both[["game_pk", "official_date", "market",
                     "entry_point"]].rename(columns={"entry_point": "point"})
    repriced = reprice_requests(requests, predictions)
    both = both.reset_index(drop=True)
    both["model_prob"] = repriced["model_prob_home"]
    result = {}
    for market, block in both.groupby("market"):
        block = block[block["model_prob"].notna()]
        if len(block) < MIN_FIT_ROWS:
            result[market] = {"rows": int(len(block)), "weight": 0.0,
                              "status": "insufficient sample"}
            continue
        fitted = _fit_movement_coefficients(block)
        entry, close, model_residual, leader_residual = _movement_arrays(block)
        offset = entry + fitted[0] * model_residual + fitted[1] * leader_residual
        forward = _walk_forward_movement(block)
        interval = forward.get("improvement_ci95_date_clustered")
        supported = bool(interval and interval[0] > 0)
        deployed = fitted if supported else np.array([0.0, 0.0])
        result[market] = {
            "rows": int(len(block)),
            "leader_rows": int(block["entry_leader_prob"].notna().sum()),
            "research_model_weight": round(float(fitted[0]), 6),
            "research_leader_weight": round(float(fitted[1]), 6),
            "model_weight": round(float(deployed[0]), 6),
            "leader_weight": round(float(deployed[1]), 6),
            # Backward-compatible field consumed by older cards.
            "weight": round(float(deployed[0]), 6),
            "rmse_entry_logit": round(float(np.sqrt(np.mean(
                (close - entry) ** 2))), 6),
            "rmse_offset_logit": round(float(np.sqrt(np.mean(
                (close - offset) ** 2))), 6),
            "walk_forward": forward,
            "deployment_status": (
                "forward-supported" if supported else "market-only"),
            "status": "research_only; predicts a later main price",
        }
    return result


def fit(priced, predictions, games):
    rows = training_rows(priced, predictions, games, "close")
    outcome = {market: fit_outcome_weight(block)
               for market, block in rows.groupby("market")}
    movement = fit_movement_weight(priced, predictions)
    dates = pd.to_datetime(rows.get("official_date"), errors="coerce")
    fitted_through = (str(dates.max().date()) if len(dates) and dates.notna().any()
                      else None)
    return {
        "version": OFFSET_VERSION,
        "repository_revision": repository_revision(),
        "fitted_through": fitted_through,
        "promotion_status": "research_only",
        "outcome": outcome,
        "movement": movement,
    }


def load(path="market_offset.json"):
    path = Path(path)
    if not path.exists():
        return {"version": OFFSET_VERSION, "promotion_status": "unavailable",
                "outcome": {}, "movement": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def apply(model_probability, market_probability, market, artifact,
          leader_probability=None):
    outcome = artifact.get("outcome", {}).get(market, {})
    movement = artifact.get("movement", {}).get(market, {})
    outcome_weight = float(outcome.get("weight", 0.0))
    movement_weight = float(movement.get("model_weight",
                                         movement.get("weight", 0.0)))
    leader_weight = float(movement.get("leader_weight", 0.0))
    try:
        leader_probability = float(leader_probability)
        if not np.isfinite(leader_probability):
            raise ValueError
    except (TypeError, ValueError):
        leader_probability = float(market_probability)
    predicted_logit = (_logit(market_probability)
                       + movement_weight * (_logit(model_probability)
                                            - _logit(market_probability))
                       + leader_weight * (_logit(leader_probability)
                                          - _logit(market_probability)))
    return {
        "fair_prob_home": float(blend_probability(
            market_probability, model_probability, outcome_weight)),
        "predicted_close_prob_home": float(_sigmoid(predicted_logit)),
        "outcome_weight": outcome_weight,
        "movement_weight": movement_weight,
        "leader_weight": leader_weight,
        "offset_version": artifact.get("version", OFFSET_VERSION),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotes", default="data/historical_quotes.csv")
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--predictions", default="data/predictions_glm.csv")
    parser.add_argument("--out", default="market_offset.json")
    args = parser.parse_args(argv)
    quotes = pd.read_csv(args.quotes)
    games = pd.read_csv(args.games)
    predictions = pd.read_csv(args.predictions)
    priced, _ = build_priced_games(quotes, games)
    report = fit(priced, predictions, games)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
