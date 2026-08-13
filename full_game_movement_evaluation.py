"""Frozen, time-safe evaluation of MLB entry-to-close price movement.

The closing market beat every standalone outcome model in the completed
2022--24 comparison. This study asks the narrower question the archive can
answer honestly: does information visible at a 24-hour snapshot predict the
20-minute closing consensus better than assuming no movement?

The protocol is deliberately fixed before the early archive is collected:
2022 fits candidates, 2023 selects one candidate, and 2024 is not opened until
the early-snapshot audit for that season is complete. Nothing here places or
authorises a wager.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from csv_collection import read_csv_collection
from market import build_priced_games
from models import reprice_requests
from provenance import repository_revision


PROTOCOL_VERSION = "full-game-entry-close-v1"
TRAIN_YEAR = 2022
SELECTION_YEAR = 2023
CONFIRMATION_YEAR = 2024
MIN_ROWS = 500
MOVE_CLIP = 0.50
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)

MICRO_FEATURES = (
    "entry_logit", "abs_entry_logit", "leader_gap_logit",
    "follower_gap_logit", "leader_available", "follower_available",
    "entry_market_spread", "entry_books", "entry_lead_hours", "point",
    "month_sin", "month_cos",
)
HYBRID_FEATURES = MICRO_FEATURES + ("model_gap_logit",)
CANDIDATE_FEATURES = {
    "microstructure": MICRO_FEATURES,
    "microstructure_plus_model": HYBRID_FEATURES,
}
ROW_COLUMNS = list(dict.fromkeys([
    "event_id", "game_pk", "official_date", "season", "market", "point",
    "entry_prob", "close_prob", "entry_books", "close_books",
    "entry_lead_hours", "close_lead_hours", "trained_through",
    "model_prob", "entry_logit", "close_logit", "move_logit",
    *dict.fromkeys(feature for features in CANDIDATE_FEATURES.values()
                   for feature in features),
]))


def _logit(values):
    values = np.clip(np.asarray(values, float), 1e-6, 1 - 1e-6)
    return np.log(values / (1.0 - values))


def _date_cluster_interval(dates, values, draws=3000, seed=73):
    dates = np.asarray(dates, str)
    values = np.asarray(values, float)
    unique = np.unique(dates)
    if len(unique) < 10 or not len(values):
        return None
    positions = {date: np.flatnonzero(dates == date) for date in unique}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([positions[date] for date in sampled])
        estimates.append(float(values[index].mean()))
    return [round(float(np.percentile(estimates, 2.5)), 8),
            round(float(np.percentile(estimates, 97.5)), 8)]


def audit_coverage(audit):
    """Attempt coverage by season; failed calls still count as attempted."""
    if not len(audit):
        return {str(year): {"close_attempted": 0, "early_attempted": 0,
                            "early_offered": 0, "complete": False}
                for year in (TRAIN_YEAR, SELECTION_YEAR, CONFIRMATION_YEAR)}
    frame = audit.copy()
    frame["year"] = pd.to_datetime(frame["commence_time"], utc=True,
                                   errors="coerce").dt.year
    result = {}
    for year in (TRAIN_YEAR, SELECTION_YEAR, CONFIRMATION_YEAR):
        season = frame[frame["year"] == year]
        close = season[season["snapshot_role"] == "close"]
        early = season[season["snapshot_role"] == "early"]
        target = int(len(close))
        attempted = int(len(early))
        result[str(year)] = {
            "close_attempted": target,
            "close_offered": int((close["status"] == "offered").sum()),
            "early_attempted": attempted,
            "early_offered": int((early["status"] == "offered").sum()),
            "early_failed": int((early["status"] == "failed").sum()),
            "complete": bool(target > 0 and attempted >= target),
        }
    return result


def build_movement_rows(priced, predictions):
    """Create entry-only features and a later close target.

    Run lines and totals are comparable only when entry and close carry the
    same point. Moneyline has no point and is the primary research market.
    """
    required = (priced["entry_prob"].notna() & priced["close_prob"].notna()
                & (priced["entry_books"] >= 3) & (priced["close_books"] >= 3))
    both = priced[required].copy()
    entry_point = pd.to_numeric(both["entry_point"], errors="coerce")
    close_point = pd.to_numeric(both["close_point"], errors="coerce")
    same_point = ((both["market"] == "h2h")
                  | np.isclose(entry_point, close_point, equal_nan=False))
    both = both[same_point].reset_index(drop=True)
    if not len(both):
        return pd.DataFrame(columns=ROW_COLUMNS), {
            "trained_through_violations": 0}

    requests = both[["game_pk", "official_date", "market",
                     "entry_point"]].rename(columns={"entry_point": "point"})
    repriced = reprice_requests(requests, predictions)
    both["model_prob"] = repriced["model_prob_home"].to_numpy(float)
    provenance = predictions[["game_pk", "trained_through"]].drop_duplicates(
        "game_pk", keep="last")
    both = both.merge(provenance, on="game_pk", how="left")
    game_date = pd.to_datetime(both["official_date"], errors="coerce")
    trained = pd.to_datetime(both["trained_through"], errors="coerce")
    safe = trained.notna() & game_date.notna() & (trained < game_date)
    violations = int((trained.notna() & game_date.notna() & ~safe).sum())
    both = both[safe & both["model_prob"].notna()].copy()
    if not len(both):
        return pd.DataFrame(columns=ROW_COLUMNS), {
            "trained_through_violations": violations}

    both["season"] = pd.to_datetime(
        both["official_date"], errors="coerce").dt.year
    both["entry_logit"] = _logit(both["entry_prob"])
    both["close_logit"] = _logit(both["close_prob"])
    both["move_logit"] = both["close_logit"] - both["entry_logit"]
    both["model_gap_logit"] = _logit(both["model_prob"]) - both["entry_logit"]

    for label in ("leader", "follower"):
        probability = pd.to_numeric(both[f"entry_{label}_prob"], errors="coerce")
        available = probability.notna()
        both[f"{label}_available"] = available.astype(float)
        gap = np.zeros(len(both), float)
        gap[available.to_numpy()] = (_logit(probability[available])
                                     - both.loc[available, "entry_logit"].to_numpy())
        both[f"{label}_gap_logit"] = gap

    both["entry_market_spread"] = pd.to_numeric(
        both.get("entry_market_spread", 0.0), errors="coerce").fillna(0.0)
    both["entry_books"] = pd.to_numeric(both["entry_books"], errors="coerce")
    both["entry_lead_hours"] = pd.to_numeric(
        both["entry_lead_hours"], errors="coerce")
    both["point"] = pd.to_numeric(both["entry_point"], errors="coerce").fillna(0.0)
    both["abs_entry_logit"] = both["entry_logit"].abs()
    month = pd.to_datetime(both["official_date"]).dt.month.to_numpy(float)
    both["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    both["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    rows = both[ROW_COLUMNS].sort_values(
        ["official_date", "game_pk", "market"]).reset_index(drop=True)
    return rows, {"trained_through_violations": violations}


def _candidate(feature_family, alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))


def _predict(train, test, features, alpha):
    model = _candidate(features, alpha)
    model.fit(train[list(features)].to_numpy(float),
              train["move_logit"].to_numpy(float))
    predicted = model.predict(test[list(features)].to_numpy(float))
    return np.clip(predicted, -MOVE_CLIP, MOVE_CLIP)


def movement_metrics(frame, predicted, draws=3000):
    actual = frame["move_logit"].to_numpy(float)
    predicted = np.asarray(predicted, float)
    baseline_error = actual ** 2
    candidate_error = (actual - predicted) ** 2
    improvement = baseline_error - candidate_error
    meaningful = np.abs(actual) >= 0.01
    direction = (float(np.mean(np.sign(actual[meaningful])
                               == np.sign(predicted[meaningful])))
                 if meaningful.any() else None)
    correlation = (float(np.corrcoef(actual, predicted)[0, 1])
                   if np.std(actual) > 0 and np.std(predicted) > 0 else None)
    return {
        "rows": int(len(frame)),
        "rmse_no_move_logit": round(float(np.sqrt(baseline_error.mean())), 8),
        "rmse_candidate_logit": round(float(np.sqrt(candidate_error.mean())), 8),
        "mse_improvement": round(float(improvement.mean()), 10),
        "relative_mse_reduction": round(float(
            improvement.mean() / baseline_error.mean()), 8),
        "improvement_ci95_date_clustered": _date_cluster_interval(
            frame["official_date"], improvement, draws=draws),
        "direction_accuracy_nontrivial_moves": (
            round(direction, 6) if direction is not None else None),
        "movement_correlation": (
            round(correlation, 6) if correlation is not None else None),
        "mean_actual_move_logit": round(float(actual.mean()), 8),
        "mean_predicted_move_logit": round(float(predicted.mean()), 8),
    }


def evaluate_market(rows, coverage, market, draws=3000):
    block = rows[rows["market"] == market].copy()
    counts = {str(year): int((block["season"] == year).sum())
              for year in (TRAIN_YEAR, SELECTION_YEAR, CONFIRMATION_YEAR)}
    result = {"rows_by_season": counts, "primary_baseline": "no movement"}
    development_complete = (coverage[str(TRAIN_YEAR)]["complete"]
                            and coverage[str(SELECTION_YEAR)]["complete"])
    if not development_complete:
        result["status"] = "awaiting complete 2022-23 early archive"
        return result
    train = block[block["season"] == TRAIN_YEAR]
    selection = block[block["season"] == SELECTION_YEAR]
    if len(train) < MIN_ROWS or len(selection) < MIN_ROWS:
        result["status"] = "insufficient development rows"
        return result

    candidates = []
    for family, features in CANDIDATE_FEATURES.items():
        for alpha in RIDGE_ALPHAS:
            predicted = _predict(train, selection, features, alpha)
            metrics = movement_metrics(selection, predicted, draws=draws)
            candidates.append({
                "candidate": f"{family}_ridge_alpha_{alpha:g}",
                "feature_family": family,
                "alpha": alpha,
                "features": list(features),
                "selection_2023": metrics,
            })
    chosen = min(candidates,
                 key=lambda item: item["selection_2023"]["rmse_candidate_logit"])
    result["candidate_selection"] = {
        "fit_year": TRAIN_YEAR,
        "selection_year": SELECTION_YEAR,
        "candidates": candidates,
        "selected": chosen["candidate"],
        "selected_features": chosen["features"],
        "selected_alpha": chosen["alpha"],
    }
    development_signal = chosen["selection_2023"]["mse_improvement"] > 0
    result["development_signal"] = bool(development_signal)
    if not development_signal:
        result["status"] = "rejected in 2023 selection; confirmation not opened"
        result["confirmation_signal"] = False
        return result
    if not coverage[str(CONFIRMATION_YEAR)]["complete"]:
        result["status"] = "candidate locked; 2024 confirmation remains sealed"
        return result

    confirmation = block[block["season"] == CONFIRMATION_YEAR]
    if len(confirmation) < MIN_ROWS:
        result["status"] = "insufficient confirmation rows"
        result["confirmation_signal"] = False
        return result
    refit = block[block["season"].isin([TRAIN_YEAR, SELECTION_YEAR])]
    predicted = _predict(refit, confirmation, chosen["features"], chosen["alpha"])
    metrics = movement_metrics(confirmation, predicted, draws=draws)
    interval = metrics["improvement_ci95_date_clustered"]
    confirmed = bool(interval and interval[0] > 0
                     and metrics["mse_improvement"] > 0)
    result["confirmation_2024"] = metrics
    result["confirmation_signal"] = confirmed
    result["status"] = ("confirmed research signal; paper CLV only"
                        if confirmed else "not confirmed; no promotion")
    return result


def evaluate(rows, audit, integrity, unmatched_events, raw_quote_events,
             draws=3000):
    coverage = audit_coverage(audit)
    markets = {market: evaluate_market(rows, coverage, market, draws=draws)
               for market in ("h2h", "spreads", "totals")}
    confirmed = [market for market, result in markets.items()
                 if result.get("confirmation_signal")]
    protocol = {
        "version": PROTOCOL_VERSION,
        "entry_snapshot": "24 hours before scheduled first pitch",
        "close_snapshot": "20 minutes before scheduled first pitch",
        "training_year": TRAIN_YEAR,
        "candidate_selection_year": SELECTION_YEAR,
        "sealed_confirmation_year": CONFIRMATION_YEAR,
        "primary_market": "h2h",
        "secondary_market_rule": "same point at entry and close",
        "primary_metric": "close-logit MSE improvement versus no movement",
        "confirmation_gate": "date-clustered 95% interval lower bound > 0",
        "minimum_rows_per_stage": MIN_ROWS,
        "candidate_features": {name: list(features)
                               for name, features in CANDIDATE_FEATURES.items()},
        "ridge_alphas": list(RIDGE_ALPHAS),
        "confirmation_is_not_used_for_selection": True,
    }
    protocol_hash = hashlib.sha256(json.dumps(
        protocol, sort_keys=True).encode()).hexdigest()[:16]
    return {
        "study": "MLB 24-hour entry to 20-minute close movement",
        "repository_revision": repository_revision(),
        "protocol": {**protocol, "protocol_hash": protocol_hash},
        "coverage": coverage,
        "integrity": {
            **integrity,
            "raw_quote_events": int(raw_quote_events),
            "unmatched_schedule_events": int(len(unmatched_events)),
            "qualified_movement_rows": int(len(rows)),
            "confirmation_sealed": not coverage[str(CONFIRMATION_YEAR)]["complete"],
        },
        "markets": markets,
        "confirmed_markets": confirmed,
        "promotion_status": "research_only; no wager path",
        "bets_placed": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotes", default="data/full_game_event_quotes")
    parser.add_argument("--audit", default="data/full_game_event_audit.csv")
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--predictions", default="data/predictions_glm.csv")
    parser.add_argument("--rows", default="data/research/full_game_movement_rows.csv")
    parser.add_argument("--report", default="full_game_movement_evaluation.json")
    args = parser.parse_args(argv)

    quotes = read_csv_collection(args.quotes)
    audit = pd.read_csv(args.audit)
    games = pd.read_csv(args.games)
    predictions = pd.read_csv(args.predictions)
    priced, unmatched = build_priced_games(quotes, games)
    rows, integrity = build_movement_rows(priced, predictions)
    report = evaluate(rows, audit, integrity, unmatched,
                      quotes["event_id"].nunique() if len(quotes) else 0)

    rows_path = Path(args.rows)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(rows_path, index=False)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
