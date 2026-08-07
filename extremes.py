"""Does a large disagreement pay? Testing the extreme bucket properly.

The sibling UFC project could not answer this: at roughly two wagers a month,
the tail of its disagreement distribution never fills. Baseball fills it in one
season, which is the whole reason this repository exists.

The question is natural and the naive answer is a trap. Bucket the priced
history by how far the model sits from the market, and the widest bucket looks
spectacular — a no-vig ROI above 40% with a bootstrap interval excluding zero.
Three checks kill it:

**The shape.** Cumulative cuts hide it; disjoint bands show it. The 13-14 point
band loses 37.8% over 44 games while the adjacent 14-15 band wins 35.1% over
36. No mechanism makes a 13.5-point disagreement lose badly and a 14.5-point
one win big. The apparent edge at 14+ exists because the cut excludes the bad
band.

**The search.** Thresholds from 8 to 20 were tried. Under a null where the
market price is simply correct, the best-looking cut still averages about +33%
and clears +30% in two seasons out of five. Finding a spectacular bucket is
what searching produces, not evidence.

**The direction.** In every bucket the market's number is the accurate one. At
10-15 points of disagreement the market implied 41.1% and 41.9% came in; the
model said 53.0%. The model is not finding value where it disagrees most, it is
being most wrong there.

Everything here is computed before vig. The measured overround across the
captured quotes is 4.55%, so roughly 2.3 points per side come off any of these
numbers before a real price is paid.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from market import build_priced_games

# Disjoint, because cumulative cuts hide exactly the discontinuity that gives
# the game away.
BANDS = [(0, 3), (3, 5), (5, 8), (8, 10), (10, 12), (12, 13), (13, 14),
         (14, 15), (15, 17), (17, 100)]

# The thresholds a searcher would plausibly try. Recorded so the correction
# below prices the search that was actually performed.
SEARCH_CUTS = list(range(8, 21))

SPECS = {
    "h2h": ("p_home_ml", lambda f: f["home_win"].astype(float)),
    "spreads": ("p_home_rl_-1.5", lambda f: (f["run_diff"] > 1.5).astype(float)),
    "totals": ("p_over_8.5", lambda f: (f["total_runs"] > 8.5).astype(float)),
}


def build_frame(priced, predictions, games, price_column="close_prob"):
    """One row per priced game-market, oriented to the side the model backs."""
    outcomes = games[["game_pk", "home_win", "total_runs", "run_diff",
                      "official_date"]]
    merged = (priced.merge(predictions, on="game_pk")
                    .merge(outcomes, on="game_pk", suffixes=("", "_game")))
    merged = merged[merged["home_win"].notna() & merged["total_runs"].notna()
                    & merged["run_diff"].notna()]
    frames = []
    for market, (column, outcome_fn) in SPECS.items():
        subset = merged[(merged["market"] == market) & merged[column].notna()
                        & merged[price_column].notna()].copy()
        if not len(subset):
            continue
        outcome = outcome_fn(subset)
        model = subset[column].astype(float)
        market_probability = subset[price_column].astype(float)
        gap = model - market_probability
        backing_home = gap > 0
        subset["market_key"] = market
        subset["gap_pts"] = (gap.abs() * 100)
        # The bettor's side: what the model prefers, what the market charges
        # for it, and whether it came in.
        subset["model_prob"] = np.where(backing_home, model, 1 - model)
        subset["market_prob"] = np.where(backing_home, market_probability,
                                         1 - market_probability)
        subset["won"] = np.where(backing_home, outcome, 1 - outcome)
        frames.append(subset[["market_key", "official_date", "gap_pts",
                              "model_prob", "market_prob", "won",
                              f"{price_column.split('_')[0]}_books"]]
                      .rename(columns={f"{price_column.split('_')[0]}_books":
                                       "books"}))
    return pd.concat(frames, ignore_index=True)


def roi(won, market_prob):
    """Return per unit staked at the market's own fair (de-vigged) price."""
    return float((np.asarray(won, dtype=float)
                  / np.asarray(market_prob, dtype=float)).mean() - 1)


def band_table(frame):
    rows = []
    for low, high in BANDS:
        block = frame[(frame["gap_pts"] >= low) & (frame["gap_pts"] < high)]
        if not len(block):
            continue
        rows.append({
            "band": f"{low}-{high}" if high < 100 else f"{low}+",
            "games": int(len(block)),
            "model_says": round(float(block["model_prob"].mean()) * 100, 1),
            "market_says": round(float(block["market_prob"].mean()) * 100, 1),
            "actually_won": round(float(block["won"].mean()) * 100, 1),
            "no_vig_roi_pct": round(roi(block["won"], block["market_prob"]) * 100, 1),
        })
    return rows


def search_correction(frame, draws=5000, seed=3):
    """Price the threshold search under a null where the market is correct.

    Without this the widest cut looks significant. With it, the best cut a
    searcher can find on pure noise is about as good as the one found here.
    """
    gap = frame["gap_pts"].to_numpy(dtype=float)
    market = frame["market_prob"].to_numpy(dtype=float)
    won = frame["won"].to_numpy(dtype=float)
    usable = [cut for cut in SEARCH_CUTS if (gap >= cut).sum() >= 5]
    if not usable:
        return None
    observed = max(roi(won[gap >= cut], market[gap >= cut]) for cut in usable)
    rng = np.random.default_rng(seed)
    best = np.empty(draws)
    for draw in range(draws):
        simulated = (rng.random(len(market)) < market).astype(float)
        best[draw] = max(roi(simulated[gap >= cut], market[gap >= cut])
                         for cut in usable)
    return {
        "cuts_searched": usable,
        "best_observed_roi_pct": round(observed * 100, 1),
        "p_value_after_search": round(float((best >= observed).mean()), 4),
        "null_best_cut_mean_roi_pct": round(float(best.mean()) * 100, 1),
        "null_share_above_30pct": round(float((best > 0.30).mean()), 3),
    }


def overround(quotes):
    """Vig actually charged, measured rather than assumed."""
    frame = quotes.dropna(subset=["price_home", "price_away"])
    if not len(frame):
        return None

    def implied(prices):
        prices = prices.astype(float)
        return np.where(prices < 0, -prices / (-prices + 100),
                        100 / (prices + 100))

    total = implied(frame["price_home"]) + implied(frame["price_away"]) - 1
    return round(float(np.median(total)) * 100, 2)


def report(frame, quotes=None):
    bands = band_table(frame)
    correction = search_correction(frame)
    wide = frame[frame["gap_pts"] >= 10]
    return {
        "games": int(len(frame)),
        "bands": bands,
        "wide_disagreements": {
            "threshold_pts": 10,
            "games": int(len(wide)),
            "model_says": round(float(wide["model_prob"].mean()) * 100, 1),
            "market_says": round(float(wide["market_prob"].mean()) * 100, 1),
            "actually_won": round(float(wide["won"].mean()) * 100, 1),
        },
        "threshold_search": correction,
        "measured_overround_pct": overround(quotes) if quotes is not None else None,
        "verdict": (
            "No support for large disagreements being profitable. Adjacent "
            "bands of similar size swing from heavily negative to heavily "
            "positive, the apparent edge at the widest cut does not survive "
            "pricing the threshold search, and in every band the market's "
            "probability is the accurate one while the model's is not. All "
            "figures are before vig."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotes", default="data/historical_quotes.csv")
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--predictions", default="data/predictions_glm.csv")
    parser.add_argument("--price", default="close_prob",
                        choices=["close_prob", "entry_prob"])
    parser.add_argument("--report", default="extremes.json")
    args = parser.parse_args(argv)

    quotes = pd.read_csv(args.quotes)
    games = pd.read_csv(args.games)
    predictions = pd.read_csv(args.predictions)
    priced, _ = build_priced_games(quotes, games)
    frame = build_frame(priced, predictions, games, args.price)
    result = report(frame, quotes)
    Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"{result['games']:,} priced game-markets with a close and a result\n")
    print(f"{'band':>8} {'n':>5} {'model':>7} {'market':>7} {'won':>7} {'no-vig ROI':>11}")
    for row in result["bands"]:
        print(f"{row['band']:>8} {row['games']:5d} {row['model_says']:6.1f}% "
              f"{row['market_says']:6.1f}% {row['actually_won']:6.1f}% "
              f"{row['no_vig_roi_pct']:+10.1f}%")
    search = result["threshold_search"]
    if search:
        print(f"\nbest cut found: {search['best_observed_roi_pct']:+.1f}%  "
              f"p after crediting the search = {search['p_value_after_search']}")
        print(f"under the null, the best cut averages "
              f"{search['null_best_cut_mean_roi_pct']:+.1f}% and clears +30% in "
              f"{search['null_share_above_30pct'] * 100:.0f}% of seasons")
    print(f"\nmeasured overround: {result['measured_overround_pct']}% "
          f"(~{result['measured_overround_pct'] / 2:.2f} pts per side, not deducted above)")
    print(f"\n{result['verdict']}")


if __name__ == "__main__":
    main()
