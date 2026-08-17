"""Frozen evaluation of first-inning opening prices versus the close.

The study asks whether information available at the first broadly quoted
YRFI/NRFI price predicts the ten-minute consensus close.  Price movement and
executable closing-line value are primary; historical outcome ROI is secondary
and cannot place or authorize a wager.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FIRST_INNING_TOTALS_MARKET
from first_inning_model_evaluation import build_evaluation_rows
from first_inning_open_odds import DEFAULT_LEADS
from odds import american_to_prob
from provenance import repository_revision


PROTOCOL_VERSION = "yrfi-open-close-v1"
TRAIN_YEAR = 2023
SELECTION_YEAR = 2024
CONFIRMATION_YEAR = 2025
EXCLUDED_YEAR = 2026
PROSPECTIVE_FORWARD_START = "2026-08-15"
MIN_BOOKS = 2
MIN_ROWS = 400
MIN_STRATEGY_BETS = 150
MIN_BOOK_ROBUSTNESS_BETS = 30
MIN_POSITIVE_BOOKS = 2
MOVE_CLIP = 0.50
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
EV_THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.05)
BOOK_KEYS = (
    "betmgm", "fanduel", "williamhill_us", "betrivers", "betonlineag",
    "barstool", "superbook", "unibet_us", "bovada",
)

MICRO_FEATURES = (
    "open_logit", "abs_open_logit", "open_market_spread",
    "open_books", "open_lead_hours", "open_median_vig",
    "open_prob_std", "open_mean_minus_median",
    "open_median_staleness_minutes", "open_max_staleness_minutes",
    "month_sin", "month_cos",
)
BOOK_FEATURES = tuple(
    f"book_{book}_{suffix}"
    for book in BOOK_KEYS
    for suffix in ("deviation", "present", "staleness_minutes")
)
OPEN_SAFE_CONTEXT = (
    "home_off", "home_def", "away_off", "away_def",
    "home_recent_off", "home_recent_def",
    "away_recent_off", "away_recent_def",
    "home_rest", "away_rest", "rest_diff",
    "home_games_played", "away_games_played",
    "home_bp_rate", "away_bp_rate",
    "home_bp_workload", "away_bp_workload",
    "park_factor", "elevation_km",
    "expected_home_runs_prior", "expected_away_runs_prior",
)
CANDIDATE_FEATURES = {
    "opening_microstructure": MICRO_FEATURES,
    "opening_microstructure_plus_context": MICRO_FEATURES + OPEN_SAFE_CONTEXT,
    "opening_book_leaders": MICRO_FEATURES + BOOK_FEATURES,
    "opening_book_leaders_plus_context": (
        MICRO_FEATURES + BOOK_FEATURES + OPEN_SAFE_CONTEXT),
}
ROW_COLUMNS = list(dict.fromkeys([
    "event_id", "game_pk", "official_date", "season", "yrfi",
    "open_prob_yrfi", "close_prob_yrfi", "open_books",
    "open_market_spread", "open_median_vig", "open_requested_lead_minutes",
    "open_returned_lead_minutes", "best_price_yrfi", "best_price_nrfi",
    "best_book_yrfi", "best_book_nrfi",
    "close_logit", "move_logit", *MICRO_FEATURES, *BOOK_FEATURES,
    *OPEN_SAFE_CONTEXT,
]))


def _logit(values):
    values = np.clip(np.asarray(values, float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def _date_cluster_interval(dates, values, draws=3000, seed=97):
    dates = np.asarray(dates, str)
    values = np.asarray(values, float)
    unique = np.unique(dates)
    if len(unique) < 10 or not len(values):
        return None
    positions = {date: np.flatnonzero(dates == date) for date in unique}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(draws)):
        selected = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([positions[date] for date in selected])
        estimates.append(float(values[index].mean()))
    return [round(float(np.percentile(estimates, 2.5)), 8),
            round(float(np.percentile(estimates, 97.5)), 8)]


def _american_decimal(value):
    value = float(value)
    return 1.0 + (value / 100.0 if value > 0 else 100.0 / -value)


def _snapshot_aggregate(quotes):
    columns = [
        "event_id", "snapshot_time", "open_prob_yrfi", "open_books",
        "open_market_spread", "open_median_vig", "best_price_yrfi",
        "best_price_nrfi", "best_book_yrfi", "best_book_nrfi",
        "open_prob_std", "open_mean_minus_median",
        "open_median_staleness_minutes", "open_max_staleness_minutes",
        *BOOK_FEATURES,
    ]
    if quotes is None or not len(quotes):
        return pd.DataFrame(columns=columns)
    frame = quotes.copy()
    frame["point"] = pd.to_numeric(frame["point"], errors="coerce")
    frame["devig_prob_home"] = pd.to_numeric(
        frame["devig_prob_home"], errors="coerce")
    frame["price_home"] = pd.to_numeric(frame["price_home"], errors="coerce")
    frame["price_away"] = pd.to_numeric(frame["price_away"], errors="coerce")
    frame["snapshot_time"] = pd.to_datetime(
        frame["fetched_at"], utc=True, errors="coerce")
    if "book_updated_at" in frame:
        frame["book_updated_at"] = pd.to_datetime(
            frame["book_updated_at"], utc=True, errors="coerce")
    else:
        frame["book_updated_at"] = pd.Series(
            pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    frame = frame[
        (frame["market"] == FIRST_INNING_TOTALS_MARKET)
        & np.isclose(frame["point"], 0.5)
        & frame["snapshot_time"].notna()
        & frame["devig_prob_home"].notna()
        & frame["price_home"].notna()
        & frame["price_away"].notna()
    ].copy()
    if not len(frame):
        return pd.DataFrame(columns=columns)
    home_implied = frame["price_home"].map(american_to_prob)
    away_implied = frame["price_away"].map(american_to_prob)
    frame["vig"] = home_implied + away_implied - 1.0
    frame["book_staleness_minutes"] = np.maximum(
        (frame["snapshot_time"] - frame["book_updated_at"])
        .dt.total_seconds() / 60.0,
        0.0,
    )
    keys = ["event_id", "snapshot_time"]
    best_yrfi = (frame.sort_values(
        keys + ["price_home", "book_key"],
        ascending=[True, True, False, True])
        .drop_duplicates(keys)
        [keys + ["book_key"]]
        .rename(columns={"book_key": "best_book_yrfi"}))
    best_nrfi = (frame.sort_values(
        keys + ["price_away", "book_key"],
        ascending=[True, True, False, True])
        .drop_duplicates(keys)
        [keys + ["book_key"]]
        .rename(columns={"book_key": "best_book_nrfi"}))
    grouped = frame.groupby(keys, as_index=False).agg(
        open_prob_yrfi=("devig_prob_home", "median"),
        open_books=("book_key", "nunique"),
        open_min_prob=("devig_prob_home", "min"),
        open_max_prob=("devig_prob_home", "max"),
        open_mean_prob=("devig_prob_home", "mean"),
        open_prob_std=("devig_prob_home", "std"),
        open_median_vig=("vig", "median"),
        open_median_staleness_minutes=("book_staleness_minutes", "median"),
        open_max_staleness_minutes=("book_staleness_minutes", "max"),
        best_price_yrfi=("price_home", "max"),
        best_price_nrfi=("price_away", "max"),
    )
    grouped["open_market_spread"] = (
        grouped.pop("open_max_prob") - grouped.pop("open_min_prob"))
    grouped["open_mean_minus_median"] = (
        grouped.pop("open_mean_prob") - grouped["open_prob_yrfi"])
    grouped["open_prob_std"] = grouped["open_prob_std"].fillna(0.0)
    grouped = grouped.merge(best_yrfi, on=keys, how="left")
    grouped = grouped.merge(best_nrfi, on=keys, how="left")
    for book in BOOK_KEYS:
        prefix = f"book_{book}"
        values = (frame[frame["book_key"] == book]
                  .groupby(keys, as_index=False)
                  .agg(**{
                      f"{prefix}_probability": ("devig_prob_home", "median"),
                      f"{prefix}_staleness_minutes": (
                          "book_staleness_minutes", "median"),
                  }))
        grouped = grouped.merge(values, on=keys, how="left")
        probability = f"{prefix}_probability"
        grouped[f"{prefix}_present"] = grouped[probability].notna().astype(int)
        grouped[f"{prefix}_deviation"] = (
            grouped[probability] - grouped["open_prob_yrfi"])
        grouped = grouped.drop(columns=probability)
    return grouped[columns]


def build_open_snapshots(quotes, audit, min_books=MIN_BOOKS,
                         leads=DEFAULT_LEADS):
    """Choose the earliest predeclared multi-book 0.5 snapshot per event."""
    aggregate = _snapshot_aggregate(quotes)
    integrity = {
        "raw_open_quote_events": int(quotes["event_id"].nunique())
        if quotes is not None and len(quotes) and "event_id" in quotes else 0,
        "offered_open_audit_rows": 0,
        "qualified_open_snapshots": 0,
        "qualified_open_events": 0,
        "post_start_open_snapshot_violations": 0,
        "unexpected_open_lead_rows": 0,
    }
    if audit is None or not len(audit) or not len(aggregate):
        return aggregate.iloc[0:0].copy(), integrity
    offered = audit[audit["status"] == "offered"].copy()
    integrity["offered_open_audit_rows"] = int(len(offered))
    for column in ("commence_time", "requested_snapshot", "returned_snapshot"):
        offered[column] = pd.to_datetime(
            offered[column], utc=True, errors="coerce")
    offered["requested_lead"] = (
        offered["commence_time"] - offered["requested_snapshot"]
    ).dt.total_seconds() / 60.0
    offered["returned_lead"] = (
        offered["commence_time"] - offered["returned_snapshot"]
    ).dt.total_seconds() / 60.0
    allowed = {int(value) for value in leads}
    rounded = offered["requested_lead"].round().astype("Int64")
    integrity["unexpected_open_lead_rows"] = int(
        (~rounded.isin(allowed)).sum())
    integrity["post_start_open_snapshot_violations"] = int(
        ((offered["requested_snapshot"] >= offered["commence_time"])
         | (offered["returned_snapshot"] >= offered["commence_time"])).sum())
    joined = offered[[
        "event_id", "commence_time", "requested_snapshot",
        "returned_snapshot", "requested_lead", "returned_lead",
    ]].merge(
        aggregate, left_on=["event_id", "returned_snapshot"],
        right_on=["event_id", "snapshot_time"], how="inner")
    joined = joined[
        (joined["open_books"] >= int(min_books))
        & joined["requested_lead"].round().isin(allowed)
    ].copy()
    integrity["qualified_open_snapshots"] = int(len(joined))
    if not len(joined):
        return joined, integrity
    # Larger lead means earlier.  Book count is only a deterministic tie-break.
    joined = joined.sort_values(
        ["event_id", "requested_lead", "open_books", "snapshot_time"],
        ascending=[True, False, False, True])
    opening = joined.drop_duplicates("event_id", keep="first").copy()
    opening = opening.rename(columns={
        "requested_lead": "open_requested_lead_minutes",
        "returned_lead": "open_returned_lead_minutes",
    })
    integrity["qualified_open_events"] = int(len(opening))
    return opening.reset_index(drop=True), integrity


def archive_coverage(open_audit, close_audit, leads=DEFAULT_LEADS,
                     eligible_event_ids=None):
    """Attempt coverage by season and rung; no-offers count as attempted."""
    result = {}
    close = close_audit.copy() if close_audit is not None else pd.DataFrame()
    opened = open_audit.copy() if open_audit is not None else pd.DataFrame()
    if eligible_event_ids is not None:
        eligible = {str(value) for value in eligible_event_ids}
        if len(close):
            close = close[close["event_id"].astype(str).isin(eligible)].copy()
        if len(opened):
            opened = opened[
                opened["event_id"].astype(str).isin(eligible)].copy()
    if len(close):
        close["season"] = pd.to_datetime(
            close["commence_time"], utc=True, errors="coerce").dt.year
    if len(opened):
        opened["season"] = pd.to_datetime(
            opened["commence_time"], utc=True, errors="coerce").dt.year
        requested = pd.to_datetime(
            opened["requested_snapshot"], utc=True, errors="coerce")
        commence = pd.to_datetime(
            opened["commence_time"], utc=True, errors="coerce")
        opened["lead"] = ((commence - requested).dt.total_seconds() / 60).round()
    for year in (TRAIN_YEAR, SELECTION_YEAR, CONFIRMATION_YEAR, EXCLUDED_YEAR):
        close_year = close[close.get("season", pd.Series(dtype=float)) == year]
        open_year = opened[opened.get("season", pd.Series(dtype=float)) == year]
        event_ids = set(close_year.get("event_id", pd.Series(dtype=str)).astype(str))
        pairs = set(zip(
            open_year.get("event_id", pd.Series(dtype=str)).astype(str),
            pd.to_numeric(open_year.get("lead", pd.Series(dtype=float)),
                          errors="coerce").fillna(-1).astype(int),
        ))
        missing = sum((event_id, int(lead)) not in pairs
                      for event_id in event_ids for lead in leads)
        result[str(year)] = {
            "closing_events": int(len(event_ids)),
            "expected_open_attempts": int(len(event_ids) * len(leads)),
            "open_attempts": int(len(open_year)),
            "open_offered": int((open_year.get("status") == "offered").sum())
            if len(open_year) else 0,
            "open_no_offer": int((open_year.get("status") == "no_offer").sum())
            if len(open_year) else 0,
            "open_failed": int((open_year.get("status") == "failed").sum())
            if len(open_year) else 0,
            "missing_event_rungs": int(missing),
            "complete": bool(event_ids and missing == 0),
        }
    return result


def build_open_rows(open_quotes, open_audit, close_rows,
                    min_books=MIN_BOOKS, leads=DEFAULT_LEADS):
    opening, integrity = build_open_snapshots(
        open_quotes, open_audit, min_books=min_books, leads=leads)
    if not len(opening) or close_rows is None or not len(close_rows):
        return pd.DataFrame(columns=ROW_COLUMNS), {
            **integrity, "open_close_joined_events": 0}
    frame = close_rows.merge(opening, on="event_id", how="inner")
    frame = frame.rename(columns={"market_prob_yrfi": "close_prob_yrfi"})
    frame["open_prob_yrfi"] = np.clip(pd.to_numeric(
        frame["open_prob_yrfi"], errors="coerce"), 1e-6, 1 - 1e-6)
    frame["close_prob_yrfi"] = np.clip(pd.to_numeric(
        frame["close_prob_yrfi"], errors="coerce"), 1e-6, 1 - 1e-6)
    frame = frame.dropna(subset=[
        "open_prob_yrfi", "close_prob_yrfi", "best_price_yrfi",
        "best_price_nrfi", "best_book_yrfi", "best_book_nrfi", "yrfi",
        "official_date",
    ]).copy()
    frame["season"] = pd.to_datetime(
        frame["official_date"], errors="coerce").dt.year
    frame["open_logit"] = _logit(frame["open_prob_yrfi"])
    frame["close_logit"] = _logit(frame["close_prob_yrfi"])
    frame["move_logit"] = frame["close_logit"] - frame["open_logit"]
    frame["abs_open_logit"] = frame["open_logit"].abs()
    frame["open_lead_hours"] = pd.to_numeric(
        frame["open_requested_lead_minutes"], errors="coerce") / 60.0
    month = pd.to_datetime(frame["official_date"]).dt.month.to_numpy(float)
    frame["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    frame["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    numeric = set(ROW_COLUMNS) - {
        "event_id", "official_date", "best_book_yrfi", "best_book_nrfi",
    }
    for column in numeric:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows = frame[ROW_COLUMNS].sort_values(
        ["official_date", "game_pk"]).reset_index(drop=True)
    return rows, {**integrity, "open_close_joined_events": int(len(rows))}


def _candidate(alpha):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])


def _predict(train, test, features, alpha):
    model = _candidate(alpha)
    model.fit(train[list(features)], train["move_logit"].to_numpy(float))
    predicted = model.predict(test[list(features)])
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
    return {
        "rows": int(len(frame)),
        "rmse_no_move_logit": round(float(np.sqrt(baseline_error.mean())), 8),
        "rmse_candidate_logit": round(float(np.sqrt(candidate_error.mean())), 8),
        "mse_improvement": round(float(improvement.mean()), 10),
        "relative_mse_reduction": round(float(
            improvement.mean() / baseline_error.mean()), 8)
        if baseline_error.mean() else 0.0,
        "improvement_ci95_date_clustered": _date_cluster_interval(
            frame["official_date"], improvement, draws=draws),
        "direction_accuracy_nontrivial_moves": (
            round(direction, 6) if direction is not None else None),
    }


def strategy_metrics(frame, predicted_move, threshold, draws=3000):
    predicted_close = 1.0 / (1.0 + np.exp(-np.clip(
        frame["open_logit"].to_numpy(float) + predicted_move, -30, 30)))
    decimal_yrfi = frame["best_price_yrfi"].map(_american_decimal).to_numpy(float)
    decimal_nrfi = frame["best_price_nrfi"].map(_american_decimal).to_numpy(float)
    predicted_ev_yrfi = predicted_close * decimal_yrfi - 1.0
    predicted_ev_nrfi = (1.0 - predicted_close) * decimal_nrfi - 1.0
    choose_yrfi = predicted_ev_yrfi >= predicted_ev_nrfi
    predicted_ev = np.where(choose_yrfi, predicted_ev_yrfi, predicted_ev_nrfi)
    selected = predicted_ev >= float(threshold)
    if not selected.any():
        return {
            "threshold": float(threshold), "bets": 0,
            "mean_predicted_ev": None, "mean_close_expected_value": None,
            "close_ev_ci95_date_clustered": None, "roi": None,
            "roi_ci95_date_clustered": None, "book_breakdown": [],
            "positive_books_with_minimum_sample": 0,
        }
    close = frame["close_prob_yrfi"].to_numpy(float)
    outcome = frame["yrfi"].to_numpy(float)
    decimal = np.where(choose_yrfi, decimal_yrfi, decimal_nrfi)
    close_side = np.where(choose_yrfi, close, 1.0 - close)
    won = np.where(choose_yrfi, outcome, 1.0 - outcome)
    close_ev = close_side * decimal - 1.0
    profit = np.where(won == 1.0, decimal - 1.0, -1.0)
    dates = frame["official_date"].astype(str).to_numpy()
    book = np.where(
        choose_yrfi,
        frame["best_book_yrfi"].astype(str).to_numpy(),
        frame["best_book_nrfi"].astype(str).to_numpy(),
    )
    book_breakdown = []
    for name in sorted(set(book[selected])):
        mask = selected & (book == name)
        book_breakdown.append({
            "book": name,
            "bets": int(mask.sum()),
            "share": round(float(mask.sum() / selected.sum()), 6),
            "mean_close_expected_value": round(float(close_ev[mask].mean()), 8),
            "roi": round(float(profit[mask].mean()), 8),
        })
    positive_books = sum(
        item["bets"] >= MIN_BOOK_ROBUSTNESS_BETS
        and item["mean_close_expected_value"] > 0
        for item in book_breakdown)
    return {
        "threshold": float(threshold),
        "bets": int(selected.sum()),
        "bet_rate": round(float(selected.mean()), 6),
        "yrfi_bets": int((selected & choose_yrfi).sum()),
        "nrfi_bets": int((selected & ~choose_yrfi).sum()),
        "mean_predicted_ev": round(float(predicted_ev[selected].mean()), 8),
        "mean_close_expected_value": round(float(close_ev[selected].mean()), 8),
        "close_ev_ci95_date_clustered": _date_cluster_interval(
            dates[selected], close_ev[selected], draws=draws),
        "roi": round(float(profit[selected].mean()), 8),
        "roi_ci95_date_clustered": _date_cluster_interval(
            dates[selected], profit[selected], draws=draws, seed=101),
        "win_rate": round(float(won[selected].mean()), 6),
        "book_breakdown": book_breakdown,
        "positive_books_with_minimum_sample": int(positive_books),
    }


def _protocol(leads=DEFAULT_LEADS):
    protocol = {
        "version": PROTOCOL_VERSION,
        "market": FIRST_INNING_TOTALS_MARKET,
        "point": 0.5,
        "opening_ladder_minutes": list(leads),
        "opening_definition": "earliest rung with at least two paired books",
        "close_definition": "existing ten-minute multi-book consensus",
        "training_year": TRAIN_YEAR,
        "candidate_selection_year": SELECTION_YEAR,
        "historical_price_confirmation_year": CONFIRMATION_YEAR,
        "excluded_outcome_year": EXCLUDED_YEAR,
        "prospective_forward_start": PROSPECTIVE_FORWARD_START,
        "primary_metric": "close-logit MSE improvement versus no movement",
        "confirmation_gate": (
            "movement improvement and best-price close EV date-clustered "
            "95% lower bounds above zero; positive point-estimate ROI"
        ),
        "minimum_books": MIN_BOOKS,
        "minimum_rows_per_stage": MIN_ROWS,
        "minimum_strategy_bets": MIN_STRATEGY_BETS,
        "minimum_book_robustness_bets": MIN_BOOK_ROBUSTNESS_BETS,
        "minimum_positive_books": MIN_POSITIVE_BOOKS,
        "candidate_features": {name: list(features)
                               for name, features in CANDIDATE_FEATURES.items()},
        "predeclared_book_keys": list(BOOK_KEYS),
        "ridge_alphas": list(RIDGE_ALPHAS),
        "expected_value_thresholds": list(EV_THRESHOLDS),
        "pitcher_weather_umpire_lineup_features_excluded": True,
        "historical_outcome_confirmation_is_pristine": False,
        "execution_assumption": "best captured US price; book mix reported",
    }
    digest = hashlib.sha256(json.dumps(
        protocol, sort_keys=True).encode()).hexdigest()[:16]
    return {**protocol, "protocol_hash": digest}


def evaluate(rows, coverage, integrity, draws=3000):
    report = {
        "study": "MLB first-inning opening price to close",
        "status": "research_only_no_promotion",
        "repository_revision": repository_revision(),
        "protocol": _protocol(),
        "coverage": coverage,
        "integrity": integrity,
        "bets_placed": 0,
        "excluded_2026": {
            "status": "excluded_from_historical_outcome_evaluation",
            "eligible_rows": int((rows.get("season", pd.Series(dtype=float))
                                  == EXCLUDED_YEAR).sum()),
            "outcomes_evaluated": False,
            "prospective_forward_start": PROSPECTIVE_FORWARD_START,
        },
    }
    development_complete = (
        coverage.get(str(TRAIN_YEAR), {}).get("complete", False)
        and coverage.get(str(SELECTION_YEAR), {}).get("complete", False))
    if not development_complete:
        report["status"] = "awaiting_complete_2023_2024_opening_archive"
        return report
    train = rows[rows["season"] == TRAIN_YEAR].copy()
    selection = rows[rows["season"] == SELECTION_YEAR].copy()
    if min(len(train), len(selection)) < MIN_ROWS:
        report["status"] = "insufficient_development_rows"
        return report

    candidates = []
    for family, features in CANDIDATE_FEATURES.items():
        for alpha in RIDGE_ALPHAS:
            predicted = _predict(train, selection, features, alpha)
            candidates.append({
                "candidate": f"{family}_ridge_alpha_{alpha:g}",
                "feature_family": family,
                "features": list(features),
                "alpha": float(alpha),
                "selection_movement": movement_metrics(
                    selection, predicted, draws=draws),
            })
    chosen = min(candidates, key=lambda item: (
        item["selection_movement"]["rmse_candidate_logit"],
        len(item["features"]), item["alpha"]))
    selected_move = _predict(
        train, selection, tuple(chosen["features"]), chosen["alpha"])
    thresholds = [strategy_metrics(
        selection, selected_move, threshold, draws=draws)
        for threshold in EV_THRESHOLDS]
    eligible = [item for item in thresholds
                if item["bets"] >= MIN_STRATEGY_BETS
                and item["mean_close_expected_value"] is not None]
    selected_strategy = max(
        eligible,
        key=lambda item: (item["mean_close_expected_value"], item["threshold"]),
        default=None,
    )
    report["selection_2024"] = {
        "rows": int(len(selection)),
        "candidates": candidates,
        "selected_candidate": chosen["candidate"],
        "selected_features": chosen["features"],
        "selected_alpha": chosen["alpha"],
        "threshold_candidates": thresholds,
        "selected_strategy": selected_strategy,
    }
    development_signal = bool(
        chosen["selection_movement"]["mse_improvement"] > 0
        and selected_strategy
        and selected_strategy["mean_close_expected_value"] > 0)
    report["development_signal"] = development_signal
    if not development_signal:
        report["status"] = "rejected_in_2024_selection_no_confirmation_opened"
        return report
    if not coverage.get(str(CONFIRMATION_YEAR), {}).get("complete", False):
        report["status"] = "candidate_locked_2025_price_confirmation_sealed"
        return report
    confirmation = rows[rows["season"] == CONFIRMATION_YEAR].copy()
    if len(confirmation) < MIN_ROWS:
        report["status"] = "insufficient_confirmation_rows"
        return report
    development = pd.concat([train, selection], ignore_index=True)
    predicted = _predict(
        development, confirmation, tuple(chosen["features"]), chosen["alpha"])
    movement = movement_metrics(confirmation, predicted, draws=draws)
    strategy = strategy_metrics(
        confirmation, predicted, selected_strategy["threshold"], draws=draws)
    move_interval = movement["improvement_ci95_date_clustered"]
    clv_interval = strategy["close_ev_ci95_date_clustered"]
    confirmed = bool(
        strategy["bets"] >= MIN_STRATEGY_BETS
        and move_interval and move_interval[0] > 0
        and clv_interval and clv_interval[0] > 0
        and strategy["positive_books_with_minimum_sample"] >= MIN_POSITIVE_BOOKS
        and strategy["roi"] is not None and strategy["roi"] > 0)
    report["confirmation_2025"] = {
        "rows": int(len(confirmation)),
        "movement": movement,
        "strategy": strategy,
        "historical_outcome_roi_is_secondary_non_pristine": True,
        "confirmed_opening_price_signal": confirmed,
    }
    report["status"] = (
        "confirmed_opening_price_signal_paper_only" if confirmed
        else "opening_price_signal_not_confirmed_no_promotion")
    report["promotion_status"] = (
        "paper_only; requires prospective executable fills and positive CLV")
    return report


def _read(path):
    path = Path(path)
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-quotes", default="data/first_inning_open_quotes.csv")
    parser.add_argument("--open-audit", default="data/first_inning_open_audit.csv")
    parser.add_argument("--close-quotes", default="data/first_inning_quotes.csv")
    parser.add_argument("--close-audit", default="data/first_inning_audit.csv")
    parser.add_argument("--results", default="data/first_inning_results.csv")
    parser.add_argument("--features", default="data/features.csv")
    parser.add_argument("--rows", default="data/research/yrfi_open_rows.csv")
    parser.add_argument("--report", default="first_inning_open_evaluation.json")
    parser.add_argument("--draws", type=int, default=3000)
    args = parser.parse_args(argv)

    open_quotes, open_audit = _read(args.open_quotes), _read(args.open_audit)
    close_quotes, close_audit = _read(args.close_quotes), _read(args.close_audit)
    results, features = _read(args.results), _read(args.features)
    close_rows, close_integrity = build_evaluation_rows(
        close_quotes, results, features, close_audit)
    rows, open_integrity = build_open_rows(
        open_quotes, open_audit, close_rows)
    coverage = archive_coverage(
        open_audit, close_audit,
        eligible_event_ids=(set(close_rows["event_id"].astype(str))
                            if len(close_rows) else set()))
    report = evaluate(rows, coverage, {
        **open_integrity,
        "close_feature_integrity": close_integrity,
        "qualified_rows_by_season": {
            str(int(year)): int(count) for year, count in
            rows.groupby("season")["game_pk"].count().items()
        } if len(rows) else {},
    }, draws=args.draws)
    row_path = Path(args.rows)
    row_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(row_path, index=False)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
