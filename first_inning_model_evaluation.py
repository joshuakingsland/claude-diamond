"""Frozen, market-anchored YRFI model evaluation.

The archive supplies a devigged market probability roughly ten minutes before
first pitch.  This study asks two deliberately separate questions:

1. Does a simple, temporally fitted recalibration improve that consensus?
2. Do point-in-time baseball features add information after recalibration?

Candidate selection uses 2024 only.  The selected candidates are refit on
2023-24 and scored once on 2025.  No 2026 outcome is evaluated or written to
the research rows. Those already collected rows are excluded rather than
misrepresented as pristine; prospective evidence starts after this lock.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FIRST_INNING_TOTALS_MARKET
from provenance import repository_revision
from validate import brier, calibration_error, calibration_table, log_loss


PROTOCOL_VERSION = "yrfi-market-anchor-v1"
TRAIN_YEAR = 2023
SELECTION_YEAR = 2024
CONFIRMATION_YEAR = 2025
EXCLUDED_YEAR = 2026
PROSPECTIVE_FORWARD_START = "2026-08-15"
MIN_BOOKS = 2
MIN_ROWS = 400
C_VALUES = (0.01, 0.05, 0.2, 1.0)

MARKET_FEATURES = ("market_logit",)
RUN_CONTEXT = MARKET_FEATURES + (
    "expected_home_runs_prior", "expected_away_runs_prior",
    "home_sp_rate", "away_sp_rate", "home_sp_starts", "away_sp_starts",
    "park_factor",
)
PITCHER_CONTACT = RUN_CONTEXT + (
    "home_sp_k_rate", "away_sp_k_rate",
    "home_sp_bb_rate", "away_sp_bb_rate",
    "home_sp_hr_rate", "away_sp_hr_rate",
)
OFFENSE_AND_PITCHER = PITCHER_CONTACT + (
    "home_off", "away_off", "home_def", "away_def",
    "home_recent_off", "away_recent_off",
    "home_recent_def", "away_recent_def",
)
ENVIRONMENT = PITCHER_CONTACT + (
    "elevation_km", "temp_c", "air_density_index",
    "wind_out_to_center_ms", "precip_mm", "roof_retractable", "roof_dome",
)
FULL_PREGAME = tuple(dict.fromkeys(OFFENSE_AND_PITCHER + ENVIRONMENT))

FEATURE_FAMILIES = {
    "run_context": RUN_CONTEXT,
    "pitcher_contact": PITCHER_CONTACT,
    "offense_and_pitcher": OFFENSE_AND_PITCHER,
    "environment": ENVIRONMENT,
    "full_pregame": FULL_PREGAME,
}


def _logit(probability):
    probability = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def _protocol(feature_families=None, c_values=None):
    families = feature_families or FEATURE_FAMILIES
    values = tuple(c_values or C_VALUES)
    protocol = {
        "version": PROTOCOL_VERSION,
        "market": FIRST_INNING_TOTALS_MARKET,
        "point": 0.5,
        "minimum_books": MIN_BOOKS,
        "training_year": TRAIN_YEAR,
        "candidate_selection_year": SELECTION_YEAR,
        "confirmation_year": CONFIRMATION_YEAR,
        "excluded_year": EXCLUDED_YEAR,
        "prospective_forward_start": PROSPECTIVE_FORWARD_START,
        "market_recalibration_features": list(MARKET_FEATURES),
        "baseball_feature_families": {
            name: list(columns) for name, columns in families.items()
        },
        "logistic_c_values": list(values),
        "primary_metric": "log loss versus devigged market consensus",
        "confirmation_gate": (
            "date-clustered 95% upper bounds below zero versus both raw "
            "market and market-only recalibration"
        ),
        "selection_uses_confirmation": False,
        "excluded_2026_outcomes_evaluated": False,
    }
    payload = json.dumps(protocol, sort_keys=True).encode()
    return {**protocol,
            "protocol_hash": hashlib.sha256(payload).hexdigest()[:16]}


def build_evaluation_rows(quotes, results, features, audit=None,
                          min_books=MIN_BOOKS):
    """Return one qualified, point-in-time row per MLB game."""
    quote_columns = {"event_id", "market", "point", "book_key",
                     "devig_prob_home"}
    result_columns = {"event_id", "game_pk", "official_date", "yrfi",
                      "result_status"}
    if not quote_columns.issubset(quotes) or not result_columns.issubset(results):
        raise ValueError("first-inning quote or result schema is incomplete")
    if "game_pk" not in features:
        raise ValueError("feature schema has no game_pk")

    quotes = quotes.copy()
    quotes["point"] = pd.to_numeric(quotes["point"], errors="coerce")
    quotes["devig_prob_home"] = pd.to_numeric(
        quotes["devig_prob_home"], errors="coerce")
    quotes = quotes[
        (quotes["market"] == FIRST_INNING_TOTALS_MARKET)
        & np.isclose(quotes["point"], 0.5)
        & quotes["devig_prob_home"].notna()
    ]
    consensus = quotes.groupby("event_id", as_index=False).agg(
        market_prob_yrfi=("devig_prob_home", "median"),
        market_books=("book_key", "nunique"),
        quote_rows=("book_key", "size"),
    )
    consensus = consensus[consensus["market_books"] >= int(min_books)]

    settled = results.copy()
    settled = settled[settled["result_status"] == "final"]
    if "game_type" in settled:
        settled = settled[settled["game_type"].fillna("R") == "R"]
    settled["game_pk"] = pd.to_numeric(settled["game_pk"], errors="coerce")
    settled["yrfi"] = pd.to_numeric(settled["yrfi"], errors="coerce")
    settled = settled.dropna(subset=["game_pk", "official_date", "yrfi"])
    settled["game_pk"] = settled["game_pk"].astype(int)

    joined = consensus.merge(settled, on="event_id", how="inner")
    event_rows = len(joined)
    # A small number of provider event IDs map to the same MLB game.  Collapse
    # them before modeling so one outcome cannot receive extra weight.
    joined = joined.sort_values(
        ["game_pk", "market_books", "event_id"], ascending=[True, False, True])
    games = joined.groupby("game_pk", as_index=False).agg(
        event_id=("event_id", "first"),
        official_date=("official_date", "first"),
        commence_time=("commence_time", "first"),
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        yrfi=("yrfi", "first"),
        market_prob_yrfi=("market_prob_yrfi", "median"),
        market_books=("market_books", "max"),
        provider_events=("event_id", "nunique"),
    )

    feature_frame = features.copy()
    feature_frame["game_pk"] = pd.to_numeric(
        feature_frame["game_pk"], errors="coerce")
    feature_frame = feature_frame.dropna(subset=["game_pk"])
    feature_frame["game_pk"] = feature_frame["game_pk"].astype(int)
    feature_dates = feature_frame[["game_pk", "official_date"]].rename(
        columns={"official_date": "feature_date"})
    feature_frame = feature_frame.drop(columns=["official_date"],
                                       errors="ignore")
    rows = games.merge(feature_frame, on="game_pk", how="left", indicator=True)
    feature_misses = int((rows["_merge"] != "both").sum())
    rows = rows[rows["_merge"] == "both"].drop(columns="_merge")
    rows = rows.merge(feature_dates, on="game_pk", how="left")
    rows["official_date"] = pd.to_datetime(
        rows["official_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    rows["feature_date"] = pd.to_datetime(
        rows["feature_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    date_mismatches = int((rows["official_date"] != rows["feature_date"]).sum())
    rows = rows[rows["official_date"] == rows["feature_date"]].copy()
    rows["season"] = pd.to_datetime(rows["official_date"]).dt.year
    rows["market_prob_yrfi"] = np.clip(
        pd.to_numeric(rows["market_prob_yrfi"], errors="coerce"), 1e-6, 1-1e-6)
    rows["market_logit"] = _logit(rows["market_prob_yrfi"])
    rows = rows.sort_values(["official_date", "game_pk"]).reset_index(drop=True)

    snapshot_violations = 0
    offered_events = 0
    if audit is not None and len(audit):
        offered = audit[audit["status"] == "offered"].copy()
        offered_events = int(offered["event_id"].nunique())
        commence = pd.to_datetime(offered["commence_time"], utc=True,
                                  errors="coerce")
        returned = pd.to_datetime(offered["returned_snapshot"], utc=True,
                                  errors="coerce")
        requested = pd.to_datetime(offered["requested_snapshot"], utc=True,
                                   errors="coerce")
        snapshot_violations = int(
            ((returned.notna() & commence.notna() & (returned >= commence))
             | (requested.notna() & commence.notna() & (requested >= commence))).sum()
        )

    rows_by_season = {
        str(int(year)): int(count)
        for year, count in rows.groupby("season")["game_pk"].count().items()
    }
    integrity = {
        "raw_quote_events": int(quotes["event_id"].nunique()),
        "multi_book_consensus_events": int(len(consensus)),
        "settled_regular_events": int(len(settled)),
        "qualified_event_rows_before_game_deduplication": int(event_rows),
        "duplicate_provider_event_rows_collapsed": int(event_rows - len(games)),
        "qualified_unique_games": int(len(rows)),
        "feature_join_misses": feature_misses,
        "feature_date_mismatches": date_mismatches,
        "offered_audit_events": offered_events,
        "post_start_snapshot_violations": snapshot_violations,
        "rows_by_season": rows_by_season,
    }
    return rows, integrity


def _pipeline(c_value):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=float(c_value), max_iter=3000,
                                      solver="lbfgs", random_state=0)),
    ])


def _fit_predict(train, test, features, c_value):
    model = _pipeline(c_value)
    model.fit(train[list(features)], train["yrfi"].to_numpy(int))
    probability = model.predict_proba(test[list(features)])[:, 1]
    return model, np.clip(probability, 1e-6, 1 - 1e-6)


def _date_cluster_delta_interval(frame, left, right, outcome, draws=3000,
                                 seed=17):
    """95% interval for left-minus-right log loss, resampling game dates."""
    dates = frame["official_date"].astype(str).to_numpy()
    unique = np.unique(dates)
    if len(unique) < 10:
        return None
    positions = {date: np.flatnonzero(dates == date) for date in unique}
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    outcome = np.asarray(outcome, float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(draws)):
        selected = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([positions[date] for date in selected])
        values.append(log_loss(left[index], outcome[index])
                      - log_loss(right[index], outcome[index]))
    return [round(float(np.percentile(values, 2.5)), 6),
            round(float(np.percentile(values, 97.5)), 6)]


def _probability_metrics(probability, outcome):
    return {
        "log_loss": round(log_loss(probability, outcome), 6),
        "brier": round(brier(probability, outcome), 6),
        "calibration_error": calibration_error(probability, outcome),
        "mean_probability": round(float(np.mean(probability)), 6),
        "observed_yrfi_rate": round(float(np.mean(outcome)), 6),
        "calibration": calibration_table(probability, outcome),
    }


def _comparison(frame, candidate, benchmark, benchmark_name, draws=3000):
    outcome = frame["yrfi"].to_numpy(float)
    delta = log_loss(candidate, outcome) - log_loss(benchmark, outcome)
    interval = _date_cluster_delta_interval(
        frame, candidate, benchmark, outcome, draws=draws)
    return {
        "benchmark": benchmark_name,
        "log_loss_delta_candidate_minus_benchmark": round(delta, 6),
        "delta_ci95_date_clustered": interval,
        "candidate_better_interval_excludes_zero": bool(
            interval is not None and interval[1] < 0),
    }


def _candidate_record(train, test, name, features, c_value):
    _, probability = _fit_predict(train, test, features, c_value)
    outcome = test["yrfi"].to_numpy(float)
    market = test["market_prob_yrfi"].to_numpy(float)
    return {
        "candidate": name,
        "c": float(c_value),
        "features": list(features),
        "rows": int(len(test)),
        **_probability_metrics(probability, outcome),
        "log_loss_delta_vs_market": round(
            log_loss(probability, outcome) - log_loss(market, outcome), 6),
    }


def _selected_coefficients(model, features):
    coefficients = model.named_steps["model"].coef_[0]
    rows = [{"feature": feature, "standardized_logit_coefficient": round(
        float(value), 6)} for feature, value in zip(features, coefficients)]
    return sorted(rows, key=lambda row: abs(
        row["standardized_logit_coefficient"]), reverse=True)


def evaluate(rows, integrity, draws=3000, feature_families=None,
             c_values=None):
    """Select on 2024 and evaluate the locked candidates once on 2025."""
    families = feature_families or FEATURE_FAMILIES
    values = tuple(c_values or C_VALUES)
    protocol = _protocol(families, values)
    report = {
        "study": "MLB first-inning YRFI market-anchored model",
        "status": "research_only_no_promotion",
        "repository_revision": repository_revision(),
        "protocol": protocol,
        "integrity": integrity,
        "bets_placed": 0,
    }
    train = rows[rows["season"] == TRAIN_YEAR].copy()
    selection = rows[rows["season"] == SELECTION_YEAR].copy()
    confirmation = rows[rows["season"] == CONFIRMATION_YEAR].copy()
    excluded = rows[rows["season"] == EXCLUDED_YEAR]
    report["excluded_2026"] = {
        "status": "excluded_from_v1_not_a_pristine_holdout",
        "eligible_rows": int(len(excluded)),
        "outcomes_evaluated": False,
        "prospective_forward_start": PROSPECTIVE_FORWARD_START,
    }
    if min(len(train), len(selection), len(confirmation)) < MIN_ROWS:
        report["status"] = "insufficient_temporal_sample"
        return report

    market_selection = selection["market_prob_yrfi"].to_numpy(float)
    selection_outcome = selection["yrfi"].to_numpy(float)
    recalibration_candidates = []
    for c_value in values:
        recalibration_candidates.append(_candidate_record(
            train, selection, "market_recalibration", MARKET_FEATURES, c_value))
    selected_recalibration = min(
        recalibration_candidates,
        key=lambda item: (item["log_loss"], item["c"]),
    )

    baseball_candidates = []
    for family, columns in families.items():
        for c_value in values:
            baseball_candidates.append(_candidate_record(
                train, selection, family, columns, c_value))
    selected_baseball = min(
        baseball_candidates,
        key=lambda item: (item["log_loss"], len(item["features"]), item["c"]),
    )
    report["selection_2024"] = {
        "rows": int(len(selection)),
        "raw_market": _probability_metrics(market_selection, selection_outcome),
        "market_recalibration_candidates": recalibration_candidates,
        "baseball_candidates": baseball_candidates,
        "selected_market_recalibration": selected_recalibration,
        "selected_baseball_candidate": selected_baseball,
    }

    development = pd.concat([train, selection], ignore_index=True)
    recal_features = tuple(selected_recalibration["features"])
    recal_model, recal_probability = _fit_predict(
        development, confirmation, recal_features,
        selected_recalibration["c"])
    baseball_features = tuple(selected_baseball["features"])
    baseball_model, baseball_probability = _fit_predict(
        development, confirmation, baseball_features,
        selected_baseball["c"])
    market = confirmation["market_prob_yrfi"].to_numpy(float)
    outcome = confirmation["yrfi"].to_numpy(float)

    recal_vs_market = _comparison(
        confirmation, recal_probability, market, "raw_market", draws=draws)
    baseball_vs_market = _comparison(
        confirmation, baseball_probability, market, "raw_market", draws=draws)
    baseball_vs_recal = _comparison(
        confirmation, baseball_probability, recal_probability,
        "market_recalibration", draws=draws)
    confirmed_recalibration = recal_vs_market[
        "candidate_better_interval_excludes_zero"]
    confirmed_baseball = bool(
        baseball_vs_market["candidate_better_interval_excludes_zero"]
        and baseball_vs_recal["candidate_better_interval_excludes_zero"])

    report["confirmation_2025"] = {
        "rows": int(len(confirmation)),
        "raw_market": _probability_metrics(market, outcome),
        "market_recalibration": {
            **_probability_metrics(recal_probability, outcome),
            "comparison": recal_vs_market,
            "selected_c": selected_recalibration["c"],
            "coefficients": _selected_coefficients(
                recal_model, recal_features),
        },
        "baseball_candidate": {
            "family": selected_baseball["candidate"],
            "selected_c": selected_baseball["c"],
            "features": list(baseball_features),
            **_probability_metrics(baseball_probability, outcome),
            "comparison_vs_market": baseball_vs_market,
            "comparison_vs_market_recalibration": baseball_vs_recal,
            "coefficients": _selected_coefficients(
                baseball_model, baseball_features),
        },
        "confirmed_market_recalibration": bool(confirmed_recalibration),
        "confirmed_incremental_baseball_signal": bool(confirmed_baseball),
    }
    if confirmed_baseball:
        report["status"] = (
            "confirmed_incremental_baseball_signal_research_only")
    elif confirmed_recalibration:
        report["status"] = "confirmed_market_recalibration_only_research_only"
    else:
        report["status"] = "no_confirmed_yrfi_edge"
    report["promotion_status"] = (
        "paper_only; requires post-lock forward confirmation, executable "
        "price testing, and positive CLV before any review")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", default="data/first_inning_quotes.csv")
    parser.add_argument("--results", default="data/first_inning_results.csv")
    parser.add_argument("--features", default="data/features.csv")
    parser.add_argument("--audit", default="data/first_inning_audit.csv")
    parser.add_argument("--rows", default="data/research/yrfi_model_rows.csv")
    parser.add_argument("--report", default="first_inning_model_evaluation.json")
    parser.add_argument("--draws", type=int, default=3000)
    args = parser.parse_args(argv)

    rows, integrity = build_evaluation_rows(
        pd.read_csv(args.quotes), pd.read_csv(args.results),
        pd.read_csv(args.features), pd.read_csv(args.audit))
    # The persisted research rows end with the confirmation year.  Existing
    # 2026 rows are excluded from v1 rather than represented as a pristine
    # holdout; keeping them out of this artifact prevents accidental reuse.
    visible = rows[rows["season"] <= CONFIRMATION_YEAR].copy()
    row_path = Path(args.rows)
    row_path.parent.mkdir(parents=True, exist_ok=True)
    visible.to_csv(row_path, index=False)
    report = evaluate(rows, integrity, draws=args.draws)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
