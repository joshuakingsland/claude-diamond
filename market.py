"""Join historical prices to model predictions and ask the only question.

Everything else in this repository measures whether the model predicts
baseball. This measures whether it beats a price, which is a different and
much harder claim.

The join is the delicate part. Odds carry sportsbook team names and a
commence time; games carry MLBAM ids and an official date. They are matched
on (date, home team, away team) after normalising names, and a game that does
not match exactly is dropped rather than guessed at. A mismatched join would
attach one game's price to another game's outcome and manufacture edge out of
nothing.

Each game gets two prices where available:

- entry: the earliest snapshot at least 20 hours before first pitch.
- close: the latest snapshot strictly before first pitch.

A snapshot taken after first pitch is never eligible for either. Afternoon
games routinely have no usable close, because the daily snapshot lands while
they are already under way; those games keep their entry price and are
reported as lacking close coverage rather than silently filled.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MIN_ENTRY_LEAD_HOURS = 20


def normalise(name):
    """Team name to a comparable key. Books and StatsAPI mostly agree."""
    name = re.sub(r"[^a-z ]", "", str(name).lower()).strip()
    return re.sub(r"\s+", " ", name)


def _devig_consensus(quotes, market, point=None):
    """Median de-vigged home probability across books for one market."""
    subset = quotes[quotes["market"] == market]
    if point is not None:
        subset = subset[np.isclose(subset["point"].astype(float), float(point))]
    if not len(subset):
        return None, 0
    return float(subset["devig_prob_home"].median()), int(subset["book_key"].nunique())


def build_priced_games(quotes, games, min_books=3):
    """One row per game per market, with entry and close consensus."""
    quotes = quotes.copy()
    quotes["commence"] = pd.to_datetime(quotes["commence_time"], utc=True,
                                        errors="coerce")
    quotes["taken"] = pd.to_datetime(quotes["fetched_at"], utc=True,
                                     errors="coerce")
    quotes = quotes.dropna(subset=["commence", "taken"])
    quotes = quotes[quotes["taken"] < quotes["commence"]]
    quotes["lead_hours"] = ((quotes["commence"] - quotes["taken"])
                            .dt.total_seconds() / 3600.0)
    quotes["home_key"] = quotes["home_team"].map(normalise)
    quotes["away_key"] = quotes["away_team"].map(normalise)
    quotes["date"] = quotes["commence"].dt.strftime("%Y-%m-%d")

    games = games.copy()
    games["home_key"] = games["home_team_name"].map(normalise)
    games["away_key"] = games["away_team_name"].map(normalise)
    lookup = {}
    for row in games.to_dict("records"):
        lookup.setdefault(
            (row["official_date"], row["home_key"], row["away_key"]), row)

    rows, unmatched = [], set()
    grouped = quotes.groupby(["date", "home_key", "away_key"])
    for (date, home_key, away_key), group in grouped:
        game = lookup.get((date, home_key, away_key))
        if game is None:
            unmatched.add((date, home_key, away_key))
            continue
        entry_pool = group[group["lead_hours"] >= MIN_ENTRY_LEAD_HOURS]
        close_pool = group
        for market, point in (("h2h", None), ("spreads", -1.5), ("totals", 8.5)):
            record = {"game_pk": game["game_pk"], "official_date": date,
                      "market": market, "point": point}
            for label, pool in (("entry", entry_pool), ("close", close_pool)):
                if not len(pool):
                    record[f"{label}_prob"], record[f"{label}_books"] = None, 0
                    continue
                target = (pool["taken"].min() if label == "entry"
                          else pool["taken"].max())
                snap = pool[pool["taken"] == target]
                probability, books = _devig_consensus(snap, market, point)
                record[f"{label}_prob"] = probability
                record[f"{label}_books"] = books
                record[f"{label}_lead_hours"] = round(
                    float(snap["lead_hours"].iloc[0]), 2) if len(snap) else None
            rows.append(record)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame[(frame["entry_books"] >= min_books)
                      | (frame["close_books"] >= min_books)]
    return frame, sorted(unmatched)[:20]


def log_loss(probability, outcome):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    outcome = np.asarray(outcome, dtype=float)
    return float(-np.mean(outcome * np.log(probability)
                          + (1 - outcome) * np.log(1 - probability)))


def compare(priced, predictions, games, price_column="close_prob"):
    """Model versus market on the same games, per market."""
    outcomes = games[["game_pk", "home_win", "total_runs", "run_diff"]]
    merged = (priced.merge(predictions, on="game_pk", how="inner")
                    .merge(outcomes, on="game_pk", how="inner"))
    merged = merged[merged[price_column].notna()]
    # A postponed game has a price but no outcome. Dropping it explicitly
    # matters more than it looks: a comparison on `run_diff > 1.5` evaluates
    # NaN to False, so an unplayed game would be silently scored as a loss for
    # both the model and the market on the run line and the total.
    merged = merged[merged["home_win"].notna() & merged["total_runs"].notna()
                    & merged["run_diff"].notna()]
    report = {}
    specs = {
        "h2h": ("p_home_ml", lambda f: f["home_win"].astype(float)),
        "spreads": ("p_home_rl_-1.5",
                    lambda f: (f["run_diff"] > 1.5).astype(float)),
        "totals": ("p_over_8.5", lambda f: (f["total_runs"] > 8.5).astype(float)),
    }
    for market, (column, outcome_fn) in specs.items():
        subset = merged[(merged["market"] == market) & merged[column].notna()]
        if len(subset) < 50:
            report[market] = {"games": int(len(subset)),
                              "status": "insufficient sample"}
            continue
        outcome = outcome_fn(subset).to_numpy()
        model = subset[column].to_numpy(dtype=float)
        market_probability = subset[price_column].to_numpy(dtype=float)
        report[market] = {
            "games": int(len(subset)),
            "price": price_column,
            "log_loss_model": round(log_loss(model, outcome), 5),
            "log_loss_market": round(log_loss(market_probability, outcome), 5),
            "delta": round(log_loss(model, outcome)
                           - log_loss(market_probability, outcome), 5),
            "mean_abs_disagreement_pts": round(
                float(np.mean(np.abs(model - market_probability)) * 100), 3),
            "model_beats_market": bool(
                log_loss(model, outcome) < log_loss(market_probability, outcome)),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotes", default="data/historical_quotes.csv")
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--predictions", default="data/predictions_glm.csv")
    parser.add_argument("--report", default="market_comparison.json")
    args = parser.parse_args()

    quotes = pd.read_csv(args.quotes)
    games = pd.read_csv(args.games)
    predictions = pd.read_csv(args.predictions)
    priced, unmatched = build_priced_games(quotes, games)
    print(f"{len(quotes)} quotes -> {len(priced)} priced game-markets")
    if unmatched:
        print(f"unmatched keys (first {len(unmatched)}): {unmatched[:5]}")
    coverage = {
        "priced_game_markets": int(len(priced)),
        "with_entry": int((priced["entry_books"] >= 3).sum()) if len(priced) else 0,
        "with_close": int((priced["close_books"] >= 3).sum()) if len(priced) else 0,
        "median_entry_lead_hours": (
            round(float(priced["entry_lead_hours"].median()), 2)
            if len(priced) and priced["entry_lead_hours"].notna().any() else None),
        "median_close_lead_hours": (
            round(float(priced["close_lead_hours"].median()), 2)
            if len(priced) and priced["close_lead_hours"].notna().any() else None),
    }
    report = {"coverage": coverage}
    for price_column in ("close_prob", "entry_prob"):
        report[price_column] = compare(priced, predictions, games, price_column)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
