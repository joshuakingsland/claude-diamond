"""Forward-test evidence, measured by independent games and sharp close.

Paper ROI is secondary.  The primary diagnostic is whether the accepted price
beats a later market-setting consensus on the backed side.  Multiple wagers on
one game remain one independent game for promotion counts.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (LEADER_BOOK_KEYS, MIN_FORWARD_ACCEPTED_FILL_RATE,
                    MIN_FORWARD_INDEPENDENT_GAMES, STAKING_POLICY_VERSION)
from odds import american_to_prob
from provenance import repository_revision


def sharp_closes(ledger, quotes):
    required = {"event_id", "market", "point", "side", "wager_id",
                "game_pk", "official_date", "price"}
    quote_required = {"fetched_at", "commence_time", "book_key", "event_id",
                      "market", "point", "devig_prob_home"}
    if not required.issubset(ledger.columns) or not quote_required.issubset(
            quotes.columns):
        return pd.DataFrame()
    rows = []
    quotes = quotes.copy()
    quotes["taken"] = pd.to_datetime(quotes["fetched_at"], utc=True,
                                      errors="coerce")
    quotes["commence"] = pd.to_datetime(quotes["commence_time"], utc=True,
                                         errors="coerce")
    quotes = quotes[(quotes["taken"] < quotes["commence"])
                    & quotes["book_key"].isin(LEADER_BOOK_KEYS)]
    for wager in ledger.to_dict("records"):
        event_id = wager.get("event_id")
        if not event_id:
            continue
        subset = quotes[(quotes["event_id"] == event_id)
                        & (quotes["market"] == wager["market"])]
        point = wager.get("point")
        if wager["market"] != "h2h":
            try:
                subset = subset[np.isclose(subset["point"].astype(float),
                                            float(point))]
            except (TypeError, ValueError):
                continue
        if not len(subset):
            continue
        latest = subset["taken"].max()
        close = subset[subset["taken"] == latest]
        home = float(close["devig_prob_home"].median())
        sharp_side = home if wager["side"] == "home" else 1.0 - home
        break_even = american_to_prob(wager["price"])
        rows.append({
            "wager_id": wager["wager_id"],
            "game_pk": wager["game_pk"],
            "official_date": wager["official_date"],
            "sharp_close_side": sharp_side,
            "execution_break_even": break_even,
            "clv_probability": sharp_side - break_even,
        })
    return pd.DataFrame(rows)


def date_interval(frame, column, draws=4000, seed=19):
    if not len(frame) or frame["official_date"].nunique() < 10:
        return None
    dates = frame["official_date"].unique()
    positions = {date: np.flatnonzero(frame["official_date"].to_numpy() == date)
                 for date in dates}
    values = frame[column].to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        selected = rng.choice(dates, len(dates), replace=True)
        index = np.concatenate([positions[date] for date in selected])
        samples.append(float(values[index].mean()))
    return [round(float(np.percentile(samples, 2.5)), 6),
            round(float(np.percentile(samples, 97.5)), 6)]


def evaluate(ledger, closes):
    independent_games = int(ledger["game_pk"].nunique()) if len(ledger) else 0
    statuses = ledger["execution_status"] if "execution_status" in ledger \
        else pd.Series("paper", index=ledger.index, dtype=object)
    accepted = statuses.eq("accepted")
    accepted_rate = float(accepted.mean()) if len(ledger) else 0.0
    report = {
        "repository_revision": repository_revision(),
        "staking_policy_version": STAKING_POLICY_VERSION,
        "wagers": int(len(ledger)),
        "independent_games": independent_games,
        "accepted_fill_rate": round(accepted_rate, 5),
        "sharp_close_matches": int(len(closes)),
    }
    if len(closes):
        report["mean_clv_probability"] = round(
            float(closes["clv_probability"].mean()), 6)
        report["clv_ci95_date_clustered"] = date_interval(
            closes, "clv_probability")
    settled = ledger[ledger.get("profit", pd.Series(index=ledger.index,
                                                      dtype=float)).notna()]
    if len(settled):
        stake = settled["stake"].astype(float).sum()
        report["settled_roi"] = round(
            float(settled["profit"].astype(float).sum() / stake), 6)

    failures = []
    if independent_games < MIN_FORWARD_INDEPENDENT_GAMES:
        failures.append(
            f"{independent_games} independent games < "
            f"{MIN_FORWARD_INDEPENDENT_GAMES}")
    if accepted_rate < MIN_FORWARD_ACCEPTED_FILL_RATE:
        failures.append(
            f"accepted fill rate {accepted_rate:.1%} < "
            f"{MIN_FORWARD_ACCEPTED_FILL_RATE:.1%}")
    interval = report.get("clv_ci95_date_clustered")
    if interval is None or interval[0] <= 0:
        failures.append("positive sharp-close CLV is not established")
    report["promotion_status"] = "research_only" if failures else "eligible_for_review"
    report["promotion_failures"] = failures
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="data/paper_ledger.csv")
    parser.add_argument("--quotes", default="data/market_quotes/*.csv")
    parser.add_argument("--signals", default="data/clv_signals.csv")
    parser.add_argument("--out", default="forward_evidence.json")
    args = parser.parse_args(argv)
    ledger_path = Path(args.ledger)
    ledger = (pd.read_csv(ledger_path) if ledger_path.exists()
              else pd.DataFrame())
    signal_path = Path(args.signals)
    signals = (pd.read_csv(signal_path) if signal_path.exists()
               else pd.DataFrame())
    quote_paths = sorted(glob.glob(args.quotes))
    quotes = (pd.concat([pd.read_csv(path) for path in quote_paths],
                        ignore_index=True) if quote_paths else pd.DataFrame())
    evidence_rows = signals if len(signals) else ledger
    closes = (sharp_closes(evidence_rows, quotes)
              if len(evidence_rows) and len(quotes) else pd.DataFrame())
    report = evaluate(evidence_rows, closes)
    report["source"] = "clv_signals" if len(signals) else "paper_ledger"
    if len(signals):
        report["signals"] = report.pop("wagers")
    if len(ledger):
        settled = ledger[ledger.get("profit", pd.Series(
            index=ledger.index, dtype=float)).notna()]
        if len(settled):
            stake = settled["stake"].astype(float).sum()
            report["paper_ledger_settled_roi"] = round(float(
                settled["profit"].astype(float).sum() / stake), 6)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
