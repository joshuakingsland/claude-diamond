"""Join historical prices to model predictions and ask the only question.

Everything else in this repository measures whether the model predicts
baseball. This measures whether it beats a price, which is a different and
much harder claim.

The join is the delicate part. Odds carry sportsbook team names and a
commence time; games carry MLBAM ids and a scheduled UTC start. They are
matched on the team pair plus the closest scheduled start, and an odds event
with no game within `MAX_START_DRIFT_HOURS` is dropped rather than guessed at.
A mismatched join would attach one game's price to another game's outcome and
manufacture edge out of nothing.

Matching on the start time rather than on a calendar date is not fussiness.
An 19:10 Pacific first pitch is 02:10 UTC the following day, so a date-keyed
join loses every late West Coast game — and it loses them non-randomly, which
is worse than losing them at random. Start-time matching also separates the
two halves of a doubleheader, which share a date and a team pair and would
otherwise collapse onto whichever game was seen first.

Unmatched events are counted in full and reported. A join that quietly drops
part of the card produces a comparison over a sample nobody chose.

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

# How far an odds event's advertised start may sit from a scheduled game
# before the two are considered different games. Books and StatsAPI agree on
# first pitch to within minutes; a gap of hours means a postponement, and the
# price was struck on a game that did not happen then.
MAX_START_DRIFT_HOURS = 12

# Books and StatsAPI disagree on two franchises, and neither disagreement is
# cosmetic. StatsAPI dropped the city from the Athletics' name for 2025 while
# the books carried "Oakland Athletics" all season, so an unmapped name takes
# every A's game out of the sample. Cleveland renamed inside the training
# span, so the same key has to cover both spellings.
TEAM_ALIASES = {
    "oakland athletics": "athletics",
    "sacramento athletics": "athletics",
    "las vegas athletics": "athletics",
    "cleveland indians": "cleveland guardians",
}


def normalise(name):
    """Team name to a comparable key. Books and StatsAPI mostly agree."""
    name = re.sub(r"[^a-z ]", "", str(name).lower()).strip()
    name = re.sub(r"\s+", " ", name)
    return TEAM_ALIASES.get(name, name)


def _devig_consensus(quotes, market, point=None):
    """Median de-vigged home probability across books for one market."""
    subset = quotes[quotes["market"] == market]
    if point is not None:
        subset = subset[np.isclose(subset["point"].astype(float), float(point))]
    if not len(subset):
        return None, 0
    return float(subset["devig_prob_home"].median()), int(subset["book_key"].nunique())


def match_events_to_games(events, games):
    """Map each odds event to a game_pk by team pair and scheduled start.

    Each game is claimed at most once. Two events for the same team pair on
    the same day are the two halves of a doubleheader, and letting both claim
    the earlier game would score one set of prices against the wrong result.
    Events are considered in start order so the earlier price meets the
    earlier game.
    """
    schedule = {}
    for game in games.to_dict("records"):
        schedule.setdefault((game["home_key"], game["away_key"]), []).append(game)

    matched, unmatched, claimed = {}, [], set()
    for event in events.sort_values("commence").to_dict("records"):
        best, best_gap = None, None
        for game in schedule.get((event["home_key"], event["away_key"]), []):
            if game["game_pk"] in claimed or pd.isna(game["start"]):
                continue
            gap = abs((event["commence"] - game["start"]).total_seconds()) / 3600.0
            if gap > MAX_START_DRIFT_HOURS:
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = game, gap
        if best is None:
            unmatched.append((event["commence"].strftime("%Y-%m-%d"),
                              event["home_key"], event["away_key"]))
            continue
        claimed.add(best["game_pk"])
        matched[event["event_id"]] = (best["game_pk"], best["official_date"])
    return matched, unmatched


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

    games = games.copy()
    games["home_key"] = games["home_team_name"].map(normalise)
    games["away_key"] = games["away_team_name"].map(normalise)
    games["start"] = pd.to_datetime(games["game_date_utc"], utc=True,
                                    errors="coerce")

    events = quotes.drop_duplicates("event_id")[
        ["event_id", "home_key", "away_key", "commence"]]
    matched, unmatched = match_events_to_games(events, games)

    rows = []
    for event_id, group in quotes.groupby("event_id"):
        target = matched.get(event_id)
        if target is None:
            continue
        game_pk, official_date = target
        entry_pool = group[group["lead_hours"] >= MIN_ENTRY_LEAD_HOURS]
        close_pool = group
        for market, point in (("h2h", None), ("spreads", -1.5), ("totals", 8.5)):
            record = {"game_pk": game_pk, "official_date": official_date,
                      "market": market, "point": point}
            for label, pool in (("entry", entry_pool), ("close", close_pool)):
                if not len(pool):
                    record[f"{label}_prob"], record[f"{label}_books"] = None, 0
                    continue
                target_time = (pool["taken"].min() if label == "entry"
                               else pool["taken"].max())
                snap = pool[pool["taken"] == target_time]
                probability, books = _devig_consensus(snap, market, point)
                # A consensus thinner than the gate is not a market price.
                # Gating per column rather than per row matters because a game
                # can carry a well-covered entry and a one-book close, and the
                # comparison must not read that single book as the close.
                record[f"{label}_prob"] = probability if books >= min_books else None
                record[f"{label}_books"] = books
                record[f"{label}_lead_hours"] = round(
                    float(snap["lead_hours"].iloc[0]), 2) if len(snap) else None
            rows.append(record)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame[(frame["entry_books"] >= min_books)
                      | (frame["close_books"] >= min_books)]
    return frame, unmatched


def log_loss(probability, outcome):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    outcome = np.asarray(outcome, dtype=float)
    return float(-np.mean(outcome * np.log(probability)
                          + (1 - outcome) * np.log(1 - probability)))


def delta_interval(frame, model_column, price_column, outcome, draws=2000,
                   seed=11):
    """90% interval on the model-minus-market log loss gap, clustered by date.

    A bare point estimate of the gap is the trap this whole repository is
    built to avoid. On one season of prices the totals gap came to -0.00007,
    which flips `model_beats_market` to true and means nothing whatsoever.
    Resampling is by slate rather than by game because games on the same day
    share a run environment, a set of starters, and a market state.
    """
    dates = frame["official_date"].to_numpy()
    unique = np.unique(dates)
    if len(unique) < 10:
        return None
    positions = {date: np.flatnonzero(dates == date) for date in unique}
    model = frame[model_column].to_numpy(dtype=float)
    market = frame[price_column].to_numpy(dtype=float)
    outcome = np.asarray(outcome, dtype=float)

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        picked = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([positions[date] for date in picked])
        drawn = outcome[index]
        if drawn.min() == drawn.max():
            continue
        values.append(log_loss(model[index], drawn)
                      - log_loss(market[index], drawn))
    if len(values) < 100:
        return None
    return [round(float(np.percentile(values, 5)), 5),
            round(float(np.percentile(values, 95)), 5)]


def verdict(interval):
    """Read the interval, not the sign of the point estimate."""
    if interval is None:
        return "insufficient sample for an interval"
    low, high = interval
    if low > 0:
        return "market better; interval excludes zero"
    if high < 0:
        return "model better; interval excludes zero"
    return "undecided; interval spans zero"


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
        interval = delta_interval(subset, column, price_column, outcome)
        report[market] = {
            "games": int(len(subset)),
            "price": price_column,
            "log_loss_model": round(log_loss(model, outcome), 5),
            "log_loss_market": round(log_loss(market_probability, outcome), 5),
            "delta": round(log_loss(model, outcome)
                           - log_loss(market_probability, outcome), 5),
            "delta_ci90_date_clustered": interval,
            "mean_abs_disagreement_pts": round(
                float(np.mean(np.abs(model - market_probability)) * 100), 3),
            "model_beats_market": bool(
                log_loss(model, outcome) < log_loss(market_probability, outcome)),
            "verdict": verdict(interval),
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
    events = quotes["event_id"].nunique()
    print(f"{len(quotes)} quotes -> {len(priced)} priced game-markets")
    if unmatched:
        print(f"unmatched odds events: {len(unmatched)} of {events} "
              f"({100.0 * len(unmatched) / max(events, 1):.1f}%); "
              f"first: {unmatched[:5]}")
    coverage = {
        "odds_events": int(events),
        "unmatched_events": len(unmatched),
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
