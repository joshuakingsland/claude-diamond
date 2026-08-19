"""Does taking the best available price beat the sharp close?

The largest quantity measured anywhere in this repository, and until now the
only one never tested. Everything else here asks whether the *model* beats the
price. This asks whether the *market's own disagreement* does, with no model
involved at all.

**Three overrounds, all correct, easily confused.** The repository quotes 2.3
points of vig a side in places and 1.8 in others, and both are right because
they measure different objects. Measured on the same captures:

    one book's own two-way quote      1.0451   2.26 points a side
    panel median, each side separately 1.0367   1.84 points a side
    panel best, each side separately   1.0185   0.92 points a side

Taking the median of each side *independently* already beats any single book,
because the books disagree about where the line is and the median of the two
sides is not the median book. Shopping to the best price is worth a further
**0.91 points a side** — roughly twice the whole 24-hour movement signal, and
the reason this study was worth running at all.

It is decisive in both directions, which is why it is worth doing before
anything else. If shopping produces positive closing-line value then the edge
was never in the run distribution and the project should be pointed at
execution. If it does not, then even a perfect shopper loses to the close and
the honest answer to "does anything here beat the price" is settled.

**The rule, decidable at bet time.** For each game-market at each capture in
the lock window, take the best raw price on each side across a fixed panel of
books, and compare it against the contemporaneous de-vigged consensus of that
same panel. Deviation is `consensus_probability - break_even`, so a positive
number means one book is offering a price whose implied probability is *below*
what the market collectively thinks — value against the market's own opinion,
not against a forecast. The first capture in the window where a side clears the
threshold is the bet. Nothing here looks forward.

**The measure.** `clv = closing_probability - break_even`, the same convention
`forward_evidence.py` uses: a de-vigged closing probability for the side taken,
minus the raw break-even of the price actually taken. Raw on purpose — the vig
you paid belongs in the number, and a study that de-vigs the entry price would
be marking its own homework by removing the very thing being shopped for.

**A fixed panel is not a detail.** The best of N prices rises with N, so a
panel that grows as the provider adds books manufactures an improving edge out
of nothing. Only books present in at least `MIN_BOOK_COVERAGE` of captures are
eligible, and the panel is reported with the result.

**Why a threshold arm alone would prove nothing.** Selecting captures where the
best price beats the contemporaneous consensus by a full point, and then scoring
against a *later* consensus, is close to a tautology: if the consensus is a
martingale then closing-line value equals the entry deviation in expectation,
and the study reports back the condition it selected on. The content is not the
level but the **decay** — how much of the claimed deviation the market takes
back by the close. Nothing eaten means the outlier book was simply wrong; all of
it eaten means the outlier was early and the rest of the panel followed.

So the threshold arm is run beside an unconditional one that selects nothing:
every side of every game-market at the first capture in the lock window, best
price on the panel, scored the same way. That arm carries the full sample and
answers the blunt question — is the best-price market beatable at all, or does
the vig survive shopping?

**What this cannot see.** Whether the book would have accepted the bet, and at
what size. The books that post the best number are the ones that limit fastest,
and no amount of quote history reveals that. A positive result here is a
necessary condition for an edge, never a sufficient one.

    python line_shopping.py
    python line_shopping.py --threshold 0.02 --report line_shopping.json
"""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import MAX_LOCK_LEAD_MINUTES, MIN_LOCK_LEAD_MINUTES
from csv_collection import read_quote_shards
from devig import (MAX_OVERROUND, MIN_OVERROUND, american_to_prob,
                   proportional)

# A book must appear in this share of captures to join the panel. The best of
# N prices rises with N, so a panel that grows over time invents an edge.
MIN_BOOK_COVERAGE = 0.5
# Minimum books behind a consensus for it to be a market price at all.
MIN_PANEL_BOOKS = 5
# Deviation from the contemporaneous consensus that counts as an opportunity.
DEFAULT_THRESHOLD = 0.01
# Scored against this book as well as the panel median: it sits outside the
# shopping panel, so it is not a reference built from the quotes shopped among.
SHARP_BOOK = "pinnacle"
# Reported beside the headline so one selection rule cannot carry the verdict.
THRESHOLD_SWEEP = (0.0025, 0.005, 0.01, 0.02)
# Below this a date-clustered bootstrap is not an interval: resampling one or
# two dates returns nearly the same mean every draw, so the bounds collapse
# onto the point estimate and any positivity test on them passes for free.
MIN_INTERVAL_DATES = 3
DRAWS = 2000


def load_quotes(pattern="data/market_quotes/*.csv"):
    """Every quote shard as one frame, each quote counted once.

    De-duplicated because shards can overlap: a resharding once left the same
    quotes in both a monthly and a daily file, and every count downstream
    silently doubled without anything failing.
    """
    return read_quote_shards(pattern)


def panel_books(quotes, min_coverage=MIN_BOOK_COVERAGE, priced_only=True):
    """Books steady enough across captures to shop among.

    Priced books only by default, matching `config.PRICED_ODDS_REGIONS`: `eu`
    is captured so a sharp reference accumulates, not so it can be bet.
    """
    if priced_only and "priced" in quotes.columns:
        quotes = quotes[quotes["priced"] == 1]
    captures = quotes["fetched_at"].nunique()
    if not captures:
        return []
    coverage = quotes.groupby("book_key")["fetched_at"].nunique() / captures
    return sorted(coverage[coverage >= min_coverage].index)


def prepare(quotes, books):
    """Two-way quotes from the panel, with both sides' break-even prices."""
    frame = quotes[quotes["book_key"].isin(books)].copy()
    frame["home_break_even"] = american_to_prob(
        frame["price_home"].to_numpy())
    frame["away_break_even"] = american_to_prob(
        frame["price_away"].to_numpy())
    overround = frame["home_break_even"] + frame["away_break_even"]
    frame = frame[(overround > MIN_OVERROUND) & (overround < MAX_OVERROUND)]
    frame["fair_home"] = proportional(frame["home_break_even"].to_numpy(),
                                      frame["away_break_even"].to_numpy())
    frame["commence"] = pd.to_datetime(frame["commence_time"], utc=True,
                                       errors="coerce")
    frame["captured"] = pd.to_datetime(frame["fetched_at"], utc=True,
                                       errors="coerce")
    frame["lead_minutes"] = (
        (frame["commence"] - frame["captured"]).dt.total_seconds() / 60.0)
    return frame.dropna(subset=["commence", "captured"])


def closes(frame, books=None):
    """The de-vigged consensus nearest first pitch, per game-market.

    The reference every bet is scored against. Taken from the last capture
    still before first pitch, because the odds feed keeps returning a game
    afterwards and those prices reflect the score. Pass ``books`` to score
    against a chosen reference — Pinnacle alone rather than the shopping
    panel's own median — so the close is not built from the same quotes the
    bet deviated from.
    """
    pregame = frame[frame["lead_minutes"] > 0]
    if books is not None:
        pregame = pregame[pregame["book_key"].isin(books)]
    keys = ["event_id", "market", "point"]
    last = (pregame.groupby(keys, dropna=False)["captured"].max()
            .rename("close_captured").reset_index())
    merged = pregame.merge(last, on=keys, how="inner")
    merged = merged[merged["captured"] == merged["close_captured"]]
    grouped = merged.groupby(keys, dropna=False)
    return grouped.agg(close_fair_home=("fair_home", "median"),
                       close_books=("book_key", "nunique"),
                       close_captured=("close_captured", "first"),
                       close_lead=("lead_minutes", "first")).reset_index()


def captures(frame, min_lead=MIN_LOCK_LEAD_MINUTES,
             max_lead=MAX_LOCK_LEAD_MINUTES):
    """One row per game-market per capture: the panel's best two-way price.

    The best of a set of American odds is their numeric maximum — the ordering
    holds within the negatives, within the positives, and across the sign — so
    ``max`` is the shopper's price and no sorting rule is needed.
    """
    window = frame[(frame["lead_minutes"] >= min_lead)
                   & (frame["lead_minutes"] <= max_lead)]
    window = window.dropna(subset=["price_home", "price_away"])
    keys = ["event_id", "market", "point", "fetched_at"]
    if not len(window):
        return pd.DataFrame()
    grouped = window.groupby(keys, dropna=False)
    book = grouped.agg(
        consensus_home=("fair_home", "median"),
        books=("book_key", "nunique"),
        best_home_price=("price_home", "max"),
        best_away_price=("price_away", "max"),
        median_home_price=("price_home", "median"),
        median_away_price=("price_away", "median"),
        lead_minutes=("lead_minutes", "first"),
        captured=("captured", "first"),
        commence=("commence", "first"),
    ).reset_index()
    # Which book is posting the outlier matters operationally: it is the one
    # that will limit or move first, and it is not recoverable after the fact.
    for side in ("home", "away"):
        rows = window.loc[grouped[f"price_{side}"].idxmax(),
                          keys + ["book_key"]]
        book = book.merge(rows.rename(columns={"book_key": f"best_{side}_book"}),
                          on=keys, how="left")

    book = book[book["books"] >= MIN_PANEL_BOOKS]
    if not len(book):
        return pd.DataFrame()

    book["home_break_even"] = american_to_prob(
        book["best_home_price"].to_numpy())
    book["away_break_even"] = american_to_prob(
        book["best_away_price"].to_numpy())
    # The overround a shopper actually faces. Below 1.0 is a literal arbitrage;
    # the gap above 1.0 is what any shopping edge has to clear.
    book["best_overround"] = book["home_break_even"] + book["away_break_even"]
    book["median_overround"] = (
        american_to_prob(book["median_home_price"].to_numpy())
        + american_to_prob(book["median_away_price"].to_numpy()))
    # Value against the market's own opinion: the best price implies a lower
    # probability than the panel's median thinks the side deserves.
    book["home_edge"] = book["consensus_home"] - book["home_break_even"]
    book["away_edge"] = ((1.0 - book["consensus_home"])
                         - book["away_break_even"])
    return book


def _side_rows(book, side):
    """Recast a capture table as one row per side, ready to settle."""
    out = book.copy()
    out["side"] = side
    out["edge"] = out[f"{side}_edge"]
    out["break_even"] = out[f"{side}_break_even"]
    out["price"] = out[f"best_{side}_price"]
    out["book_key"] = out[f"best_{side}_book"]
    return out


def opportunities(book, threshold=DEFAULT_THRESHOLD):
    """The first qualifying shop per game-market, decided without hindsight.

    At each capture the panel's own median is the market's opinion and the
    best price on a side is what one book will actually pay. Where the second
    is better than the first by more than ``threshold``, that is the bet.
    """
    if not len(book):
        return pd.DataFrame()
    home_better = book["home_edge"] >= book["away_edge"]
    taken = pd.concat([_side_rows(book[home_better], "home"),
                       _side_rows(book[~home_better], "away")])
    qualifying = taken[taken["edge"] >= threshold]
    if not len(qualifying):
        return pd.DataFrame()
    # First qualifying capture only. Taking the best across the window would
    # be choosing with hindsight; a shopper watching live takes the first one
    # that clears the bar.
    qualifying = qualifying.sort_values("lead_minutes", ascending=False)
    return qualifying.groupby(["event_id", "market", "point"], dropna=False,
                              as_index=False).first()


def routine(book):
    """Both sides of every game-market at the first in-window capture.

    Selects nothing, so it cannot select its own answer — but it is also not
    an experiment. Sum the two sides of one game and the consensus cancels:
    ``clv_home + clv_away`` is ``1 - best_overround`` whatever the close does,
    so this arm measures the overround a shopper faces and nothing else. It is
    reported because that number is the bar every other arm has to clear, not
    because the market was tested here.
    """
    if not len(book):
        return pd.DataFrame()
    first = (book.sort_values("lead_minutes", ascending=False)
             .groupby(["event_id", "market", "point"], dropna=False,
                      as_index=False).first())
    return pd.concat([_side_rows(first, "home"), _side_rows(first, "away")],
                     ignore_index=True)


def settle(picks, close_table, min_close_books=MIN_PANEL_BOOKS):
    """Attach the closing price and compute closing-line value.

    A bet taken at the last capture before first pitch has no later close to
    be scored against; comparing it to the consensus at its own capture would
    make its closing line value identical to the deviation it was selected on,
    by construction. Those are dropped rather than counted.
    """
    if not len(picks) or not len(close_table):
        return pd.DataFrame()
    merged = picks.merge(close_table, on=["event_id", "market", "point"],
                         how="inner")
    merged = merged[merged["close_books"] >= min_close_books]
    merged = merged[merged["captured"] < merged["close_captured"]]
    close_side = np.where(merged["side"] == "home",
                          merged["close_fair_home"],
                          1.0 - merged["close_fair_home"])
    merged["close_probability"] = close_side
    # Same convention as forward_evidence.py: a de-vigged closing probability
    # for the side taken, minus the raw break-even of the price taken. The vig
    # paid stays in the number on purpose.
    merged["clv"] = merged["close_probability"] - merged["break_even"]
    # Did the bet still look good at the close, or did the price come back?
    merged["beat_close"] = (merged["clv"] > 0).astype(float)
    return merged


def summarise(settled, draws=DRAWS, seed=0):
    if not len(settled):
        return {"picks": 0, "status": "no qualifying opportunities"}
    dates = settled["commence"].dt.strftime("%Y-%m-%d").to_numpy()
    unique = np.unique(dates)
    index = {date: np.flatnonzero(dates == date) for date in unique}
    values = settled["clv"].to_numpy(float)
    bounds = None
    if len(unique) >= MIN_INTERVAL_DATES:
        rng = np.random.default_rng(seed)
        sample = []
        for _ in range(draws):
            pick = rng.choice(unique, len(unique), replace=True)
            take = np.concatenate([index[date] for date in pick])
            sample.append(float(values[take].mean()))
        low, high = np.percentile(sample, [5, 95])
        bounds = [round(100 * float(low), 4), round(100 * float(high), 4)]
    out = {
        "picks": int(len(settled)),
        "dates": int(len(unique)),
        "mean_clv_probability_points": round(100 * float(values.mean()), 4),
        "ci90_date_clustered_points": bounds,
        "interval_note": (None if bounds else
                          f"fewer than {MIN_INTERVAL_DATES} dates; resampling "
                          f"them returns the same mean every draw, so no "
                          f"interval is computable"),
        "share_beating_close": round(float(settled["beat_close"].mean()), 4),
        "mean_edge_at_entry_points": round(
            100 * float(settled["edge"].mean()), 4),
        "median_lead_minutes": round(float(settled["lead_minutes"].median()), 1),
        # No interval, no verdict. The 2.00pt arm previously reported a "90%
        # interval" of [2.2845, 2.2845] off a single date and was marked
        # profitable on it, which is a bootstrap collapsing onto its own point
        # estimate rather than evidence of anything.
        "profitable": bool(bounds and bounds[0] > 0),
    }
    # The part that is not a restatement of the entry rule: how much of the
    # claimed deviation the market takes back before the close.
    edge = float(settled["edge"].mean())
    out["decay_points"] = round(100 * (edge - float(values.mean())), 4)
    out["share_of_edge_retained"] = (round(float(values.mean()) / edge, 4)
                                     if abs(edge) > 1e-9 else None)
    out["mean_best_overround"] = round(
        float(settled["best_overround"].mean()), 4)
    out["by_market"] = {}
    for market, block in settled.groupby("market"):
        out["by_market"][market] = {
            "picks": int(len(block)),
            "mean_clv_points": round(100 * float(block["clv"].mean()), 4),
            "share_beating_close": round(float(block["beat_close"].mean()), 4),
        }
    out["by_book"] = {book: int(count) for book, count
                      in settled["book_key"].value_counts().items()}
    return out


def leave_one_out(quotes, books, threshold):
    """Rebuild the whole study eleven times, each without one book.

    Outlier prices concentrate: one book supplies nearly half the bets. If the
    result is that book rather than the market, dropping it collapses the
    number, and this is the only way to find out. The panel, consensus and
    close are all rebuilt each time — dropping a book only from the selection
    would leave it in the reference it deviated from.
    """
    out = {}
    for drop in books:
        keep = [book for book in books if book != drop]
        frame = prepare(quotes, keep)
        settled = settle(opportunities(captures(frame), threshold=threshold),
                         closes(frame))
        out[drop] = {
            "bets": int(len(settled)),
            "mean_clv_points": (round(100 * float(settled["clv"].mean()), 4)
                                if len(settled) else None),
        }
    return out


README_ROW = re.compile(
    r"\| \d[\d.]* pt \| \d+ \| \d+ \| \*\*[+-][\d.]+\*\* \[[^\]]+\] \| "
    r"\*\*[+-][\d.]+\*\* \[[^\]]+\] \|")


def readme_table(report, levels=("0.0025", "0.005", "0.01")):
    """The README's results table, as a pure function of the report.

    The table was transcribed by hand once and was wrong within two days: the
    study picked up new captures, the headline moved from +0.312 to +0.250, and
    every sentence around it still said +0.312. Generating it means the digits
    cannot drift while the argument stands still. The prose is deliberately
    *not* generated — a changed conclusion needs a person to write it, and
    `tests/test_documented_numbers.py` fails when the prose disagrees.
    """
    rows = []
    for key in levels:
        panel = report["threshold_sweep"].get(key) or {}
        sharp = report["threshold_sweep_vs_sharp"].get(key) or {}
        if not panel.get("picks") or not panel.get("ci90_date_clustered_points"):
            continue
        low, high = panel["ci90_date_clustered_points"]
        sharp_low, sharp_high = sharp["ci90_date_clustered_points"]
        rows.append(
            f"| {float(key) * 100:.2f} pt | {panel['picks']} | "
            f"{panel['dates']} | "
            f"**{panel['mean_clv_probability_points']:+.3f}** "
            f"[{low:+.3f}, {high:+.3f}] | "
            f"**{sharp['mean_clv_probability_points']:+.3f}** "
            f"[{sharp_low:+.3f}, {sharp_high:+.3f}] |")
    return "\n".join(rows)


def sync_readme(report, path="README.md"):
    """Rewrite the README's table rows in place. Returns True if it changed."""
    table = readme_table(report)
    file = Path(path)
    if not table or not file.exists():
        return False
    text = file.read_text(encoding="utf-8")
    matches = list(README_ROW.finditer(text))
    if not matches:
        return False
    # Splicing from the first match to the last would delete anything sitting
    # between them, so this refuses unless the rows are one contiguous block.
    # The README already holds a second table whose first column reads
    # "0.25 pt" — the survival curve — and a looser pattern one day would turn
    # a tidy-up into silent data loss.
    for before, after in zip(matches, matches[1:]):
        if text[before.end():after.start()].strip():
            raise SystemExit(
                "README results rows are not contiguous; refusing to splice. "
                "Check whether another table now matches README_ROW.")
    updated = (text[:matches[0].start()] + table + text[matches[-1].end():])
    if updated == text:
        return False
    file.write_text(updated, encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotes", default="data/market_quotes/*.csv")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--report", default="line_shopping.json")
    parser.add_argument("--sync-readme", action="store_true",
                        help="rewrite the README's results table from this run")
    args = parser.parse_args()

    quotes = load_quotes(args.quotes)
    if not len(quotes):
        raise SystemExit("no quote logs found")
    books = panel_books(quotes)
    frame = prepare(quotes, books)
    close_table = closes(frame)
    book = captures(frame)

    # An independent close. The panel median is built from the same quotes the
    # bet deviated from; Pinnacle is a sharp book outside the shopping panel,
    # so agreement between the two references is worth more than either alone.
    sharp = prepare(quotes, [SHARP_BOOK])
    sharp_close = closes(sharp) if len(sharp) else pd.DataFrame()

    def arm(picks, close=None, min_books=MIN_PANEL_BOOKS):
        close = close_table if close is None else close
        if not len(picks) or not len(close):
            return summarise(pd.DataFrame())
        return summarise(settle(picks, close, min_close_books=min_books))

    shopped = arm(opportunities(book, threshold=args.threshold))
    everything = arm(routine(book))
    sweep = {f"{level:g}": arm(opportunities(book, threshold=level))
             for level in THRESHOLD_SWEEP}
    against_sharp = {
        f"{level:g}": arm(opportunities(book, threshold=level),
                          close=sharp_close, min_books=1)
        for level in THRESHOLD_SWEEP}

    result = {
        "threshold": args.threshold,
        "panel_books": books,
        "panel_size": len(books),
        "captures": int(quotes["fetched_at"].nunique()),
        "quote_rows": int(len(quotes)),
        "closes_available": int(len(close_table)),
        "best_price_overround": round(float(book["best_overround"].mean()), 4),
        "median_price_overround": round(
            float(book["median_overround"].mean()), 4),
        "unconditional": everything,
        "threshold_sweep": sweep,
        "threshold_sweep_vs_sharp": against_sharp,
        "leave_one_book_out": leave_one_out(quotes, books, THRESHOLD_SWEEP[0]),
        "leave_one_out_threshold": THRESHOLD_SWEEP[0],
        **shopped,
    }
    Path(args.report).write_text(json.dumps(result, indent=2),
                                 encoding="utf-8")

    print(f"panel of {result['panel_size']} books over "
          f"{result['captures']} captures, {result['quote_rows']:,} quotes")
    print(f"closes available: {result['closes_available']:,}")
    print(f"overround at the best price {result['best_price_overround']:.4f}, "
          f"at the median {result['median_price_overround']:.4f}\n")

    def show(label, block, decay=True):
        if not block.get("picks"):
            print(f"{label}: {block.get('status', 'nothing qualified')}")
            return
        print(f"{label}: {block['picks']} bets over {block['dates']} dates")
        print(f"  edge claimed at entry  "
              f"{block['mean_edge_at_entry_points']:+.3f} points")
        bounds = block["ci90_date_clustered_points"]
        span = (f"90% CI [{bounds[0]:+.3f}, {bounds[1]:+.3f}]" if bounds
                else f"no interval (<{MIN_INTERVAL_DATES} dates)")
        print(f"  CLV at the close       "
              f"{block['mean_clv_probability_points']:+.3f} points   {span}")
        if decay:
            print(f"  market took back        {block['decay_points']:+.3f} "
                  f"points ({block['share_of_edge_retained']:.0%} of the claim "
                  f"retained)")
        print(f"  share beating the close {100*block['share_beating_close']:.1f}%")

    show("every side, no selection", everything, decay=False)
    print("  (both sides of one game sum to 1 - best_overround whatever the "
          "close does,\n   so this arm prices the vig a shopper faces; it does "
          "not test the market)")
    print()
    show(f"best price {100*args.threshold:g}pt over consensus", result)
    if result.get("picks"):
        print("\n  by market:")
        for market, block in result["by_market"].items():
            print(f"    {market:<9} {block['picks']:>5} shops  "
                  f"CLV {block['mean_clv_points']:+.3f}p  "
                  f"beat close {100*block['share_beating_close']:.0f}%")
        print("  by book:", ", ".join(f"{k} {v}"
                                      for k, v in result["by_book"].items()))
    for label, table in (("panel median", sweep),
                         (f"{SHARP_BOOK} close", against_sharp)):
        print(f"\n  threshold sweep, scored against the {label}:")
        for level, block in table.items():
            if block.get("picks"):
                bounds = block["ci90_date_clustered_points"]
                span = (f"90% CI [{bounds[0]:+.3f}, {bounds[1]:+.3f}]" if bounds
                        else f"no interval ({block['dates']} date"
                             f"{'s' if block['dates'] != 1 else ''})")
                print(f"    {float(level)*100:>5.2f}pt  {block['picks']:>4} bets  "
                      f"CLV {block['mean_clv_probability_points']:+.3f}p  {span}")
            else:
                print(f"    {float(level)*100:>5.2f}pt     0 bets")
    print(f"\n  leave one book out at "
          f"{100*result['leave_one_out_threshold']:g}pt:")
    for drop, block in result["leave_one_book_out"].items():
        clv = block["mean_clv_points"]
        shown = f"CLV {clv:+.3f}p" if clv is not None else "no bets"
        print(f"    without {drop:<16} {block['bets']:>3} bets  {shown}")
    if args.sync_readme:
        print("README table rewritten" if sync_readme(result)
              else "README table already current")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
