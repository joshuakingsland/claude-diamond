"""Which de-vig is right? The question is about the scoreboard, not the model.

`odds.py` removes the bookmaker's margin proportionally: each side's implied
probability divided by the pair's sum. That is the convenient choice rather
than the right one, and it is not a detail — the de-vigged consensus *is* the
number this project is scored against. `market.py` reports that the closing
price beats the model on the moneyline with an interval excluding zero. If the
de-vig is biased, part of that verdict is arithmetic rather than evidence.

Four ways to strip a two-way margin, all reducing to the same thing when the
two sides are equally priced and differing exactly where they are not:

**Proportional** (current). Divide by the overround. Assumes the margin is
loaded on each side in proportion to its probability, which implies the
bookmaker takes the same percentage cut on a heavy favourite as on a longshot.

**Additive.** Subtract the overround in equal shares. The opposite assumption:
the same number of percentage points off each side, so the longshot loses a
far larger fraction of its price.

**Power.** Find the exponent that makes the raised probabilities sum to one.
Sits between the two, and is the usual answer when favourite-longshot bias is
believed to be multiplicative in the odds.

**Shin.** Solves for the share of money coming from insiders, which is the one
of the four with a story about *why* the margin is distributed unevenly rather
than a functional form chosen for convenience.

The test is direct: a de-vig is a claim about the true probability, so score
each against what actually happened. Lower log loss is a better estimate. The
methods have no fitted parameters — the power exponent and the Shin insider
share are solved per quote — so there is nothing to hold out, but the choice
among four is itself a selection, and the intervals are date-clustered and
reported so it can be judged rather than asserted.

    python devig.py
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from csv_collection import read_quote_shards

from market import normalise

# Below this the pair is not a real two-way price and the margin cannot be
# read off it; above it the quote is stale or mis-mapped rather than wide.
MIN_OVERROUND = 1.0001
MAX_OVERROUND = 1.25
DRAWS = 2000


def american_to_prob(price):
    """Implied probability from an American price, both branches guarded.

    np.where evaluates both sides whatever the condition, so the unused branch
    has to be safe too: at -100 the positive formula divides by zero and the
    warning is real even though the result is discarded.
    """
    price = np.asarray(price, dtype=float)
    negative = -price / np.maximum(-price + 100.0, 1e-9)
    positive = 100.0 / np.maximum(price + 100.0, 1e-9)
    return np.where(price < 0, negative, positive)


def proportional(home, away):
    """Divide by the overround. What `odds.py` does today."""
    return home / (home + away)


def additive(home, away):
    """Take the margin off both sides in equal points."""
    excess = (home + away - 1.0) / 2.0
    return np.clip(home - excess, 1e-6, 1 - 1e-6)


def power(home, away, rounds=60):
    """Exponent k with home**k + away**k = 1, solved by bisection.

    Monotone in k, so bisection is exact to machine precision in a fixed
    number of rounds and cannot fail to converge the way a Newton step can on
    a near-degenerate pair.
    """
    home = np.asarray(home, dtype=float)
    away = np.asarray(away, dtype=float)
    low = np.full(home.shape, 0.5)
    high = np.full(home.shape, 3.0)
    for _ in range(rounds):
        mid = (low + high) / 2.0
        total = home ** mid + away ** mid
        # Sum falls as the exponent rises, so overshoot means k is too small.
        low = np.where(total > 1.0, mid, low)
        high = np.where(total > 1.0, high, mid)
    k = (low + high) / 2.0
    return home ** k


def shin(home, away, rounds=60):
    """Shin's insider-trading de-vig: solve for the informed share z.

    The only one of the four with a mechanism rather than a shape. The
    bookmaker widens against the side insiders would take, so the margin sits
    unevenly and by an amount the model can name.
    """
    home = np.asarray(home, dtype=float)
    away = np.asarray(away, dtype=float)
    total = home + away

    def implied(z):
        # Shin's inversion, per outcome, given the informed share z.
        def side(quoted):
            root = np.sqrt(np.maximum(z ** 2 + 4.0 * (1.0 - z)
                                      * quoted ** 2 / total, 0.0))
            return (root - z) / (2.0 * (1.0 - z))
        return side(home), side(away)

    low = np.zeros(home.shape)
    high = np.full(home.shape, 0.4)
    for _ in range(rounds):
        mid = (low + high) / 2.0
        h, a = implied(mid)
        # The sum falls as z rises; too high a sum means z is still too small.
        low = np.where(h + a > 1.0, mid, low)
        high = np.where(h + a > 1.0, high, mid)
    z = (low + high) / 2.0
    h, a = implied(z)
    return np.clip(h / np.maximum(h + a, 1e-9), 1e-6, 1 - 1e-6)


METHODS = {"proportional": proportional, "additive": additive,
           "power": power, "shin": shin}


def load_quotes(pattern="data/market_quotes/*.csv"):
    frames = [pd.read_csv(path) for path in sorted(glob.glob(pattern))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def attach_outcomes(quotes, games):
    """Join each quote to what the game did, by team pair and date."""
    games = games[games["home_score"].notna()].copy()
    games["home_key"] = games["home_team_name"].map(normalise)
    games["away_key"] = games["away_team_name"].map(normalise)
    quotes = quotes.copy()
    quotes["home_key"] = quotes["home_team"].map(normalise)
    quotes["away_key"] = quotes["away_team"].map(normalise)
    quotes["date"] = pd.to_datetime(quotes["commence_time"], utc=True,
                                    errors="coerce").dt.strftime("%Y-%m-%d")
    keep = ["home_key", "away_key", "official_date", "home_score", "away_score",
            "home_win", "total_runs", "game_pk"]
    merged = quotes.merge(games[keep], left_on=["home_key", "away_key", "date"],
                          right_on=["home_key", "away_key", "official_date"],
                          how="inner")
    return merged


def outcome_for(frame):
    """Did the home side of this quote win? NaN where it cannot be settled."""
    result = pd.Series(np.nan, index=frame.index)
    h2h = frame["market"] == "h2h"
    result[h2h] = frame.loc[h2h, "home_win"]
    spreads = frame["market"] == "spreads"
    margin = frame["home_score"] - frame["away_score"]
    result[spreads] = (margin.loc[spreads]
                       > -frame.loc[spreads, "point"]).astype(float)
    push = spreads & (margin == -frame["point"])
    result[push] = np.nan
    totals = frame["market"] == "totals"
    result[totals] = (frame.loc[totals, "total_runs"]
                      > frame.loc[totals, "point"]).astype(float)
    push = totals & (frame["total_runs"] == frame["point"])
    result[push] = np.nan
    return result


def log_loss(probability, outcome):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    outcome = np.asarray(outcome, dtype=float)
    return -(outcome * np.log(probability)
             + (1 - outcome) * np.log(1 - probability))


# One capture is identified by `fetched_at`, not by `snapshot_id` -- the
# latter is a per-quote id, so grouping on it returns the quotes back
# unchanged and silently turns a consensus test into a single-book one.
CAPTURE_KEYS = ["game_pk", "market", "point", "fetched_at"]


def consensus(frame, column):
    """Median across books of one quote-level probability, per capture."""
    return frame.groupby(CAPTURE_KEYS, dropna=False)[column].median()


def evaluate(frame, per_book=True, priced_only=True):
    """Log loss of each de-vig, at quote level and at consensus level."""
    # Carried as columns rather than as loose arrays, so every filter below
    # keeps the prices aligned with the row they came from. Doing this with
    # parallel numpy arrays is how the first version silently misaligned.
    frame = frame.copy()
    frame["implied_home"] = american_to_prob(frame["price_home"].to_numpy())
    frame["implied_away"] = american_to_prob(frame["price_away"].to_numpy())
    overround = frame["implied_home"] + frame["implied_away"]
    frame = frame[(overround > MIN_OVERROUND) & (overround < MAX_OVERROUND)]
    frame = frame.assign(outcome=outcome_for(frame))
    frame = frame[frame["outcome"].notna()].reset_index(drop=True)

    home = frame["implied_home"].to_numpy()
    away = frame["implied_away"].to_numpy()
    for name, method in METHODS.items():
        frame[name] = method(home, away)

    out = {"quotes": int(len(frame)),
           "median_overround": round(float(np.median(home + away)), 5)}
    if per_book:
        out["per_quote"] = {
            name: round(float(log_loss(frame[name], frame["outcome"]).mean()), 5)
            for name in METHODS}

    # Consensus is what the model is actually compared against, and it is
    # built from the priced regions only, exactly as `market.py` does -- `eu`
    # is captured so Pinnacle accumulates history, not so it can price.
    if "priced" in frame.columns and priced_only:
        frame = frame[frame["priced"] == 1]
    grouped = frame.groupby(CAPTURE_KEYS, dropna=False)
    rows = grouped.agg({**{name: "median" for name in METHODS},
                        "outcome": "first", "date": "first"}).reset_index()
    out["consensus_rows"] = int(len(rows))
    out["books_per_consensus"] = round(float(
        grouped.size().mean()), 2)
    out["per_consensus"] = {
        name: round(float(log_loss(rows[name], rows["outcome"]).mean()), 5)
        for name in METHODS}
    return out, rows


def intervals(rows, baseline="proportional", draws=DRAWS, seed=0):
    """Date-clustered interval on each method's gap to the current one."""
    rng = np.random.default_rng(seed)
    dates = rows["date"].to_numpy()
    unique = np.unique(dates)
    index = {date: np.flatnonzero(dates == date) for date in unique}
    base = log_loss(rows[baseline], rows["outcome"])
    out = {}
    for name in METHODS:
        if name == baseline:
            continue
        other = log_loss(rows[name], rows["outcome"])
        sample = []
        for _ in range(draws):
            pick = rng.choice(unique, len(unique), replace=True)
            take = np.concatenate([index[date] for date in pick])
            sample.append(other[take].mean() - base[take].mean())
        low, high = np.percentile(sample, [5, 95])
        out[name] = {"delta": round(float(other.mean() - base.mean()), 5),
                     "ci90_date_clustered": [round(float(low), 5),
                                             round(float(high), 5)],
                     "better": bool(high < 0)}
    return out


# The sharp reference for baseball. Captured under `eu` and deliberately not
# priced -- `config.PRICED_ODDS_REGIONS` is `us` only, and promoting a region
# is a model change rather than a config flip.
REFERENCE_BOOK = "pinnacle"
# Below this many distinct dates the date-clustered interval is resampling a
# handful of clusters and means very little, however many rows sit inside them.
MIN_CLUSTERS = 20


def benchmark(frame, book=REFERENCE_BOOK, draws=DRAWS, seed=0):
    """Is the sharp book a better price than the median of the priced books?

    The benchmark is the other half of the scoreboard. `market.py` scores the
    model against the median of US books; if a captured-but-unpriced book is
    systematically closer to the truth, then the comparison is against a
    softer number than the market actually offers -- and the model's deficit
    is understated rather than overstated.
    """
    frame = frame.copy()
    frame["implied_home"] = american_to_prob(frame["price_home"].to_numpy())
    frame["implied_away"] = american_to_prob(frame["price_away"].to_numpy())
    overround = frame["implied_home"] + frame["implied_away"]
    frame = frame[(overround > MIN_OVERROUND) & (overround < MAX_OVERROUND)]
    frame = frame.assign(outcome=outcome_for(frame))
    frame = frame[frame["outcome"].notna()]
    frame["prob"] = proportional(frame["implied_home"].to_numpy(),
                                 frame["implied_away"].to_numpy())

    priced = frame[frame["priced"] == 1]
    if not len(priced) or book not in set(frame["book_key"]):
        return {"status": f"no overlap between {book} and the priced books"}
    house = priced.groupby(CAPTURE_KEYS, dropna=False).agg(
        prob=("prob", "median"), outcome=("outcome", "first"))
    sharp = frame[frame["book_key"] == book].groupby(
        CAPTURE_KEYS, dropna=False).agg(prob=("prob", "median"),
                                        outcome=("outcome", "first"))
    both = house.join(sharp, how="inner", rsuffix="_sharp").dropna()
    both = both.reset_index()
    if len(both) < 200:
        return {"status": "too few overlapping rows", "rows": int(len(both))}

    house_loss = log_loss(both["prob"], both["outcome"])
    sharp_loss = log_loss(both["prob_sharp"], both["outcome_sharp"])
    dates = both["fetched_at"].str[:10].to_numpy()
    unique = np.unique(dates)
    index = {date: np.flatnonzero(dates == date) for date in unique}
    rng = np.random.default_rng(seed)
    sample = []
    for _ in range(draws):
        pick = rng.choice(unique, len(unique), replace=True)
        take = np.concatenate([index[date] for date in pick])
        sample.append(sharp_loss[take].mean() - house_loss[take].mean())
    low, high = np.percentile(sample, [5, 95])
    out = {
        "book": book,
        "rows": int(len(both)),
        "dates": int(len(unique)),
        "priced_median_log_loss": round(float(house_loss.mean()), 5),
        "reference_log_loss": round(float(sharp_loss.mean()), 5),
        "delta": round(float(sharp_loss.mean() - house_loss.mean()), 5),
        "ci90_date_clustered": [round(float(low), 5), round(float(high), 5)],
        "mean_abs_gap_points": round(float(
            100 * (both["prob"] - both["prob_sharp"]).abs().mean()), 3),
        "by_market": {},
    }
    for market, block in both.groupby("market"):
        out["by_market"][market] = {
            "rows": int(len(block)),
            "priced_median": round(float(
                log_loss(block["prob"], block["outcome"]).mean()), 5),
            "reference": round(float(
                log_loss(block["prob_sharp"], block["outcome_sharp"]).mean()), 5),
        }
    # Stated rather than left for the reader to notice: rows are plentiful and
    # clusters are not, and it is the clusters the interval rests on.
    out["interval_trustworthy"] = bool(len(unique) >= MIN_CLUSTERS)
    out["status"] = ("reference book is better"
                     if high < 0 else "indistinguishable"
                     if low < 0 < high else "priced median is better")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotes", default="data/market_quotes/*.csv")
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--report", default="devig.json")
    args = parser.parse_args()

    quotes = load_quotes(args.quotes)
    if not len(quotes):
        raise SystemExit("no quote logs found")
    games = pd.read_csv(args.games)
    frame = attach_outcomes(quotes, games)
    if not len(frame):
        raise SystemExit("no quotes could be joined to a settled game")

    result, rows = evaluate(frame)
    result["by_market"] = {}
    for market, block in rows.groupby("market"):
        result["by_market"][market] = {
            name: round(float(log_loss(block[name], block["outcome"]).mean()), 5)
            for name in METHODS}
    result["vs_proportional"] = intervals(rows)
    result["benchmark"] = benchmark(frame)

    Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{result['quotes']:,} settled quotes -> "
          f"{result['consensus_rows']:,} consensus rows")
    print(f"median overround {result['median_overround']}\n")
    print(f"{'method':<15}{'per quote':>12}{'consensus':>12}"
          f"{'delta':>10}   90% interval")
    for name in METHODS:
        delta = result["vs_proportional"].get(name)
        tail = ""
        if delta:
            flag = "  <-- better" if delta["better"] else ""
            tail = (f"{delta['delta']:>+10.5f}   "
                    f"[{delta['ci90_date_clustered'][0]:+.5f}, "
                    f"{delta['ci90_date_clustered'][1]:+.5f}]{flag}")
        else:
            tail = f"{'(baseline)':>10}"
        print(f"{name:<15}{result['per_quote'][name]:>12.5f}"
              f"{result['per_consensus'][name]:>12.5f}{tail}")
    print("\nby market (consensus):")
    print(f"{'market':<10}" + "".join(f"{n:>14}" for n in METHODS))
    for market, block in result["by_market"].items():
        print(f"{market:<10}" + "".join(f"{block[n]:>14.5f}" for n in METHODS))
    bench = result["benchmark"]
    if bench.get("rows"):
        print(f"\nbenchmark: {bench['book']} against the median of the priced "
              f"books, {bench['rows']:,} rows over {bench['dates']} dates")
        print(f"  priced median {bench['priced_median_log_loss']}   "
              f"{bench['book']} {bench['reference_log_loss']}   "
              f"delta {bench['delta']:+.5f} "
              f"[{bench['ci90_date_clustered'][0]:+.5f}, "
              f"{bench['ci90_date_clustered'][1]:+.5f}]")
        print(f"  {bench['status']}; mean gap "
              f"{bench['mean_abs_gap_points']} probability points")
        if not bench["interval_trustworthy"]:
            print(f"  NOTE: only {bench['dates']} distinct dates, so the "
                  f"clustered interval rests on very few clusters and should "
                  f"not be read as settled.")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
