"""Walk-forward validation for all three markets.

What this can and cannot establish is worth stating plainly, because the
distinction is the whole point of the exercise:

- It CAN establish whether the model predicts baseball. Calibration, log
  loss, and Brier score against realised outcomes need no odds at all.
- It CANNOT establish whether the model beats a price. Edge, ROI, and
  closing-line value require historical odds, so that question belongs to
  `market.py` and is imported here rather than inferred from the numbers
  above.

A model that is well calibrated and still loses money is the normal case in
a liquid market, so the second question is the one that decides whether any
of this is worth running. The report carries whatever `market.py` last
concluded, and `unavailable` when it has not run, rather than quietly
reporting accuracy as though it were edge.

Intervals are bootstrapped over whole seasons, not games. Games within a
season share teams, parks, a run environment, and a rule set, so treating
12,400 of them as independent would produce intervals several times too
narrow.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS
from provenance import feature_schema, model_version, repository_revision


def log_loss(probability, outcome):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    outcome = np.asarray(outcome, dtype=float)
    return float(-np.mean(outcome * np.log(probability)
                          + (1 - outcome) * np.log(1 - probability)))


def brier(probability, outcome):
    return float(np.mean((np.asarray(probability, dtype=float)
                          - np.asarray(outcome, dtype=float)) ** 2))


def calibration_table(probability, outcome, bins=10):
    """Predicted versus realised rate, in equal-width probability bins."""
    probability = np.asarray(probability, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (probability < high)
        if mask.sum() < 20:
            continue
        rows.append({
            "bin": f"{low:.1f}-{high:.1f}",
            "n": int(mask.sum()),
            "predicted": round(float(probability[mask].mean()), 4),
            "realised": round(float(outcome[mask].mean()), 4),
            "error": round(float(outcome[mask].mean() - probability[mask].mean()), 4),
        })
    return rows


def calibration_error(probability, outcome, bins=10):
    """Sample-weighted mean absolute gap between predicted and realised."""
    table = calibration_table(probability, outcome, bins)
    if not table:
        return None
    total = sum(row["n"] for row in table)
    return round(sum(abs(row["error"]) * row["n"] for row in table) / total, 5)


def season_bootstrap(frame, statistic, draws=4000, seed=11):
    """Resample whole seasons, since games inside one are far from independent."""
    rng = np.random.default_rng(seed)
    seasons = frame["season"].unique()
    if len(seasons) < 2:
        return None
    values = []
    for _ in range(draws):
        picked = rng.choice(seasons, len(seasons), replace=True)
        sample = pd.concat([frame[frame["season"] == season] for season in picked])
        try:
            value = statistic(sample)
        except Exception:  # noqa: BLE001 - a degenerate resample is skipped
            continue
        if value is not None and np.isfinite(value):
            values.append(value)
    if len(values) < 100:
        return None
    return [round(float(np.percentile(values, 5)), 5),
            round(float(np.percentile(values, 95)), 5)]


def evaluate(predictions, games, runline_point=-1.5, total_point=8.5,
             comparison_path="market_comparison.json",
             forward_path="forward_evidence.json"):
    """Accuracy of every market the model prices, against realised outcomes."""
    merged = predictions.merge(
        games[["game_pk", "home_score", "away_score", "home_win", "total_runs",
               "run_diff"]],
        on="game_pk", how="inner",
    )
    merged = merged[merged["home_win"].notna()].copy()
    report = {"games": int(len(merged)),
              "seasons": sorted(int(s) for s in merged["season"].unique())}

    # ---------------------------------------------------------- moneyline
    home_win = merged["home_win"].to_numpy(dtype=float)
    model_probability = merged["p_home_ml"].to_numpy(dtype=float)
    base_rate = float(home_win.mean())
    report["moneyline"] = {
        "home_win_rate": round(base_rate, 5),
        "log_loss": round(log_loss(model_probability, home_win), 5),
        "log_loss_home_field_baseline": round(
            log_loss(np.full_like(model_probability, base_rate), home_win), 5),
        "brier": round(brier(model_probability, home_win), 5),
        "calibration_error": calibration_error(model_probability, home_win),
        "calibration": calibration_table(model_probability, home_win),
    }
    report["moneyline"]["log_loss_ci90_season_clustered"] = season_bootstrap(
        merged, lambda f: log_loss(f["p_home_ml"], f["home_win"]))

    # ---------------------------------------------------------- run line
    column = f"p_home_rl_{runline_point}"
    if column in merged:
        covered = (merged["run_diff"].to_numpy(dtype=float)
                   > -float(runline_point)).astype(float)
        probability = merged[column].to_numpy(dtype=float)
        report["runline"] = {
            "point": runline_point,
            "cover_rate": round(float(covered.mean()), 5),
            "log_loss": round(log_loss(probability, covered), 5),
            "brier": round(brier(probability, covered), 5),
            "calibration_error": calibration_error(probability, covered),
            "calibration": calibration_table(probability, covered),
        }

    # ---------------------------------------------------------- total
    column = f"p_over_{total_point}"
    if column in merged:
        over = (merged["total_runs"].to_numpy(dtype=float)
                > float(total_point)).astype(float)
        probability = merged[column].to_numpy(dtype=float)
        report["total"] = {
            "point": total_point,
            "over_rate": round(float(over.mean()), 5),
            "log_loss": round(log_loss(probability, over), 5),
            "brier": round(brier(probability, over), 5),
            "calibration_error": calibration_error(probability, over),
            "calibration": calibration_table(probability, over),
        }

    # ------------------------------------------------- expected run accuracy
    for side in ("home", "away"):
        predicted = merged[f"expected_{side}_runs"].to_numpy(dtype=float)
        actual = merged[f"{side}_score"].to_numpy(dtype=float)
        report.setdefault("expected_runs", {})[side] = {
            "mean_predicted": round(float(predicted.mean()), 4),
            "mean_actual": round(float(actual.mean()), 4),
            "mae": round(float(np.mean(np.abs(predicted - actual))), 4),
            "bias": round(float(np.mean(predicted - actual)), 4),
        }

    report["market_comparison"] = market_comparison(comparison_path)
    report["forward_evidence"] = forward_evidence(forward_path)
    report["live_gate"] = live_gate(report["market_comparison"],
                                    report["forward_evidence"])
    return report


def market_comparison(path):
    """Read the market comparison if `market.py` has produced one.

    This block used to be a hardcoded `unavailable`, which was true when it
    was written and quietly false afterwards: historical odds arrived, the
    comparison ran, and the validation report went on announcing that the
    question could not be asked. A report that cannot notice its own evidence
    is worse than one that has none.
    """
    path = Path(path)
    if not path.exists():
        return {
            "status": "unavailable",
            "reason": (
                "No market comparison found. Predictive accuracy above says "
                "nothing about whether the model beats a price; edge, ROI, and "
                "closing-line value all require a priced market to compare "
                "against. Run `python historical_odds.py` then `python "
                "market.py` to produce one."
            ),
        }
    comparison = json.loads(path.read_text(encoding="utf-8"))
    close = comparison.get("close_prob", {})
    keep = ("games", "delta", "delta_ci90_date_clustered", "verdict")
    return {
        "status": "available",
        "source": str(path),
        "coverage": comparison.get("coverage"),
        "close": {market: {key: block[key] for key in keep if key in block}
                  for market, block in close.items()},
        "markets_model_beats_close": sorted(
            market for market, block in close.items()
            if (block.get("delta_ci90_date_clustered") or [0, 0])[1] < 0),
    }


def forward_evidence(path):
    path = Path(path)
    if not path.exists():
        return {"status": "unavailable",
                "reason": "No forward-evidence report has been produced."}
    evidence = json.loads(path.read_text(encoding="utf-8"))
    return {"status": "available", "source": str(path), **evidence}


def live_gate(comparison, evidence=None):
    """No path to `live` exists in this code; this records why, not whether."""
    if comparison.get("status") != "available":
        return {
            "status": "research_only",
            "reason": (
                "A market comparison with a positive clustered interval is "
                "required before any staking discussion, and none has been "
                "produced. This repository does not place wagers."
            ),
        }
    beaten = comparison.get("markets_model_beats_close") or []
    evidence = evidence or {"status": "unavailable"}
    forward_status = evidence.get("promotion_status", "unavailable")
    return {
        "status": "research_only",
        "reason": (
            f"The model beats the closing price on {len(beaten)} of "
            f"{len(comparison.get('close') or {})} markets with an interval "
            "excluding zero. Beating a closing price on log loss would in any "
            "case be a necessary condition for staking, not a sufficient one. "
            f"Forward evidence is {forward_status}; it requires independent "
            "games, accepted fills, and positive sharp-close CLV before any "
            "promotion review. This repository does not place wagers."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features.csv")
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--kind", default="glm", choices=["gbm", "glm"])
    parser.add_argument("--report", default="validation.json")
    parser.add_argument("--predictions", default="data/predictions.csv")
    parser.add_argument(
        "--reuse-predictions", action="store_true",
        help="evaluate an already generated prediction file after market and "
             "forward reports have been refreshed",
    )
    parser.add_argument("--market-comparison", default="market_comparison.json",
                        help="report from market.py; absent is reported as "
                             "unavailable rather than assumed")
    parser.add_argument("--forward-evidence", default="forward_evidence.json")
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    games = pd.read_csv(args.games)
    if args.reuse_predictions:
        predictions = pd.read_csv(args.predictions)
        print(f"re-evaluating {len(predictions)} stored {args.kind} predictions")
    else:
        from models import walk_forward

        seasons = sorted(int(season) for season in features["season"].unique())
        print(f"walk-forward over seasons {seasons} using {args.kind}")
        predictions = walk_forward(features, games, seasons, kind=args.kind)
        if not len(predictions):
            raise SystemExit("no predictions produced")
        predictions.to_csv(args.predictions, index=False)
    report = evaluate(predictions, games,
                      comparison_path=args.market_comparison,
                      forward_path=args.forward_evidence)
    report["model_kind"] = args.kind
    revision = repository_revision()
    report["model_revision"] = revision
    report["feature_schema"] = feature_schema(FEATURE_COLUMNS)
    report["model_version"] = model_version(
        args.kind, FEATURE_COLUMNS, revision=revision)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {key: report[key] for key in ("games", "seasons", "model_kind")}
    print(json.dumps(summary, indent=2))
    for market in ("moneyline", "runline", "total"):
        if market in report:
            block = {k: v for k, v in report[market].items() if k != "calibration"}
            print(f"{market}: {json.dumps(block)}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
