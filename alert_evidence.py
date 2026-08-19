"""Did the alerts that actually fired beat the close?

`line_shopping.py` looked backwards: it swept 261,890 stored quotes, chose a
rule, and reported what that rule would have earned. Every study in this
repository that ever looked good looked good that way, and most of them stopped
looking good the moment something independent was asked of them. This file asks
the independent question. It scores `data/shop_alerts.csv` — the prices that
were flagged live, by the running detector, before anyone knew what they would
do.

**What makes this different from the study.** The alerts cannot be re-selected.
The rule, the threshold and the panel were fixed when each row was written, so
there is no version of this report where a better rule is tried on the same
alerts. A disappointing number here cannot be tuned away; it can only be
reported. That is the entire point of an append-only log.

**Scored the same way as the study**, so the two numbers are comparable:
`clv = close_probability - break_even`, a de-vigged closing probability for the
side taken minus the raw break-even of the price that was flagged. Against the
panel median and against Pinnacle, which sits outside the shopping panel and is
the reference to believe when the two disagree.

**Only closed games count.** A game still in its lock window has no close, and
the newest quote in the log is not a closing quote merely because nothing newer
exists yet — the same leak `forward_evidence.sharp_closes` was written to
avoid. An alert whose game has not started is carried, not scored.

**The gate is deliberately hard to pass.** `MIN_ALERTS` and `MIN_ALERT_DATES`
exist because 70 backward-looking observations over 13 dates is already thin,
and a forward record needs to clear that bar independently before it means
anything. Nothing here promotes an alert to a wager; the status is a statement
about evidence, not a green light.

    python alert_evidence.py
    python alert_evidence.py --report alert_evidence.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from line_shopping import (DRAWS, MIN_PANEL_BOOKS, SHARP_BOOK, closes,
                           load_quotes, panel_books, prepare)
from provenance import repository_revision

# A forward record has to stand on its own before it says anything. The study
# it is testing had 70 observations over 13 dates; matching that is the floor,
# not the target.
MIN_ALERTS = 70
MIN_ALERT_DATES = 13
# Below this a date-clustered bootstrap is not an interval. Resampling one date
# returns the same mean every draw, so the "90% interval" collapses onto the
# point estimate and a positivity test on it passes automatically — which is
# exactly the kind of number that reads as evidence and is not.
MIN_INTERVAL_DATES = 3


def load_alerts(path="data/shop_alerts.csv"):
    file = Path(path)
    if not file.exists():
        return pd.DataFrame()
    frame = pd.read_csv(file)
    if not len(frame):
        return frame
    for column in ("commence_time", "fetched_at", "alerted_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True,
                                       errors="coerce")
    return frame.dropna(subset=["commence_time", "fetched_at"])


def score(alerts, close_table, now=None):
    """Attach closes to alerts, keeping only games that have finished starting.

    ``now`` is a parameter so tests and replays are deterministic; production
    passes nothing and gets the current UTC time.
    """
    if not len(alerts) or not len(close_table):
        return pd.DataFrame()
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    started = alerts[alerts["commence_time"] <= now].copy()
    if not len(started):
        return pd.DataFrame()
    # The log writes an absent point as an empty cell; the quote log carries
    # NaN. Merging the two spellings silently drops every moneyline row.
    started["point"] = pd.to_numeric(started["point"], errors="coerce")
    merged = started.merge(close_table, on=["event_id", "market", "point"],
                           how="inner")
    # An alert raised at the last capture before first pitch has no later
    # market to be scored against, exactly as in the study.
    merged = merged[merged["fetched_at"] < merged["close_captured"]]
    if not len(merged):
        return pd.DataFrame()
    close_side = np.where(merged["side"] == "home",
                          merged["close_fair_home"],
                          1.0 - merged["close_fair_home"])
    merged["close_probability"] = close_side
    merged["clv"] = merged["close_probability"] - merged["break_even"]
    merged["beat_close"] = (merged["clv"] > 0).astype(float)
    return merged


def interval(settled, column="clv", draws=DRAWS, seed=0):
    """A 90% interval resampled over dates, never over alerts.

    Several alerts can fire on one slate off one book's bad afternoon, so
    treating them as independent draws would narrow the interval by pretending
    to a sample size the data does not have.
    """
    dates = settled["commence_time"].dt.strftime("%Y-%m-%d").to_numpy()
    unique = np.unique(dates)
    if len(unique) < MIN_INTERVAL_DATES:
        return None
    index = {date: np.flatnonzero(dates == date) for date in unique}
    values = settled[column].to_numpy(float)
    rng = np.random.default_rng(seed)
    sample = [float(values[np.concatenate(
        [index[date] for date in rng.choice(unique, len(unique),
                                            replace=True)])].mean())
              for _ in range(draws)]
    low, high = np.percentile(sample, [5, 95])
    return [round(100 * float(low), 4), round(100 * float(high), 4)]


def summarise(settled, label):
    if not len(settled):
        return {"alerts": 0, "status": f"nothing scored against the {label}"}
    values = settled["clv"].to_numpy(float)
    return {
        "alerts": int(len(settled)),
        "dates": int(settled["commence_time"].dt.strftime("%Y-%m-%d")
                     .nunique()),
        "mean_clv_probability_points": round(100 * float(values.mean()), 4),
        "ci90_date_clustered_points": interval(settled),
        "interval_note": (None if settled["commence_time"].dt.strftime(
            "%Y-%m-%d").nunique() >= MIN_INTERVAL_DATES else
            f"fewer than {MIN_INTERVAL_DATES} dates; no interval is "
            f"computable from this"),
        "share_beating_close": round(float(settled["beat_close"].mean()), 4),
        "mean_deviation_at_alert_points": round(
            float(settled["deviation_points"].astype(float).mean()), 4),
        "by_book": {book: int(count) for book, count
                    in settled["book_key"].value_counts().items()},
    }


def evaluate(alerts, panel_scored, sharp_scored):
    report = {
        "repository_revision": repository_revision(),
        "alerts_logged": int(len(alerts)),
        "alerts_awaiting_first_pitch": int(
            (alerts["commence_time"] > pd.Timestamp.now(tz="UTC")).sum())
        if len(alerts) else 0,
        "against_panel_median": summarise(panel_scored, "panel median"),
        "against_sharp_close": summarise(sharp_scored, f"{SHARP_BOOK} close"),
    }

    # The sharp close decides. The panel median is built from the same quotes
    # the alert deviated from, so it is the flattering reference, and a gate
    # that could be passed on it alone would be marking its own homework.
    block = report["against_sharp_close"]
    failures = []
    if block.get("alerts", 0) < MIN_ALERTS:
        failures.append(f"{block.get('alerts', 0)} scored alerts < {MIN_ALERTS}")
    if block.get("dates", 0) < MIN_ALERT_DATES:
        failures.append(f"{block.get('dates', 0)} dates < {MIN_ALERT_DATES}")
    bounds = block.get("ci90_date_clustered_points")
    if bounds is None or bounds[0] <= 0:
        failures.append(
            "positive CLV against the sharp close is not established forwards")
    report["forward_status"] = ("research_only" if failures
                               else "consistent_with_the_study")
    report["forward_failures"] = failures
    # Never a recommendation. The rule has never been tested against whether a
    # book would accept the bet, which no quote log can show.
    report["stake_recommendation"] = "none"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--alerts", default="data/shop_alerts.csv")
    parser.add_argument("--quotes", default="data/market_quotes/*.csv")
    parser.add_argument("--report", default="alert_evidence.json")
    args = parser.parse_args(argv)

    alerts = load_alerts(args.alerts)
    quotes = load_quotes(args.quotes)
    if not len(alerts):
        report = {"alerts_logged": 0,
                  "forward_status": "research_only",
                  "forward_failures": ["no alerts have fired yet"],
                  "stake_recommendation": "none"}
        Path(args.report).write_text(json.dumps(report, indent=2),
                                     encoding="utf-8")
        print("no alerts logged yet; nothing to score")
        return report

    books = panel_books(quotes)
    panel_frame = prepare(quotes, books)
    sharp_frame = prepare(quotes, [SHARP_BOOK])
    panel_close = closes(panel_frame)
    sharp_close = closes(sharp_frame) if len(sharp_frame) else pd.DataFrame()
    panel_close = panel_close[panel_close["close_books"] >= MIN_PANEL_BOOKS]

    report = evaluate(alerts, score(alerts, panel_close),
                      score(alerts, sharp_close))
    Path(args.report).write_text(json.dumps(report, indent=2),
                                 encoding="utf-8")

    print(f"{report['alerts_logged']} alerts logged, "
          f"{report['alerts_awaiting_first_pitch']} still awaiting first pitch")
    for name, key in (("panel median", "against_panel_median"),
                      (f"{SHARP_BOOK} close", "against_sharp_close")):
        block = report[key]
        if not block.get("alerts"):
            print(f"  vs {name:<16} {block['status']}")
            continue
        bounds = block["ci90_date_clustered_points"]
        shown = (f"90% CI [{bounds[0]:+.3f}, {bounds[1]:+.3f}]" if bounds
                 else f"no interval (<{MIN_INTERVAL_DATES} dates)")
        print(f"  vs {name:<16} {block['alerts']:>3} alerts over "
              f"{block['dates']} dates  "
              f"CLV {block['mean_clv_probability_points']:+.3f}p  {shown}  "
              f"beat close {100*block['share_beating_close']:.0f}%")
    print(f"\nforward status: {report['forward_status']}")
    for failure in report["forward_failures"]:
        print(f"  - {failure}")
    print(f"wrote {args.report}")
    return report


if __name__ == "__main__":
    main()
