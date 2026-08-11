"""Market-only integrity report for the separately captured first-inning data."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import FIRST_INNING_TOTALS_MARKET


def _load(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_report(quotes, results):
    needed_quotes = {"event_id", "market", "point", "devig_prob_home"}
    needed_results = {"event_id", "yrfi", "result_status"}
    if not needed_quotes.issubset(quotes.columns) or not needed_results.issubset(results.columns):
        return {"status": "waiting_for_data", "events": 0}
    quotes = quotes.copy()
    quotes["point"] = pd.to_numeric(quotes["point"], errors="coerce")
    quotes["devig_prob_home"] = pd.to_numeric(quotes["devig_prob_home"],
                                                errors="coerce")
    quotes = quotes[(quotes["market"] == FIRST_INNING_TOTALS_MARKET)
                    & np.isclose(quotes["point"], 0.5)
                    & quotes["devig_prob_home"].notna()]
    consensus = quotes.groupby("event_id", as_index=False).agg(
        market_prob_yrfi=("devig_prob_home", "median"),
        market_books=("book_key", "nunique"),
    )
    settled = results[results["result_status"] == "final"].copy()
    settled["yrfi"] = pd.to_numeric(settled["yrfi"], errors="coerce")
    joined = consensus.merge(settled[["event_id", "yrfi"]], on="event_id",
                             how="inner").dropna()
    if not len(joined):
        return {"status": "waiting_for_final_labels", "events": 0,
                "quoted_events": int(len(consensus))}
    probability = np.clip(joined["market_prob_yrfi"].to_numpy(float), 1e-6, 1 - 1e-6)
    outcome = joined["yrfi"].to_numpy(float)
    return {
        "status": "market_baseline_only",
        "events": int(len(joined)),
        "quoted_events": int(len(consensus)),
        "mean_books": round(float(joined["market_books"].mean()), 3),
        "yrfi_rate": round(float(outcome.mean()), 5),
        "market_mean_yrfi_probability": round(float(probability.mean()), 5),
        "market_brier": round(float(np.mean((probability - outcome) ** 2)), 6),
        "market_log_loss": round(float(-np.mean(outcome * np.log(probability)
                                                  + (1 - outcome) * np.log(1 - probability))), 6),
        "note": "Data-quality baseline only; no first-inning predictive model or bet rule exists.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", default="data/first_inning_quotes.csv")
    parser.add_argument("--results", default="data/first_inning_results.csv")
    parser.add_argument("--out", default="first_inning_evidence.json")
    args = parser.parse_args(argv)
    report = build_report(_load(args.quotes), _load(args.results))
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
