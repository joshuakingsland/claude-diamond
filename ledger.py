"""The paper forward test, and the only place the staking policy is real.

Until now `config.py` declared an edge rule, a stake size, a day cap, lock
timing, and execution limits, and not one of them was referenced anywhere in
the codebase. That is worse than having no policy: a constant that is written
down and never applied reads, to anyone auditing later, as a control that was
in force. This module applies them, and every rejection is written down with
the gate that caused it.

**No money moves here.** Wagers are recorded at a price that was on the board,
settled against real results, and totalled. That is a forward test, not a
staking system, and `market.py` has already established that the closing price
beats this model on two of three markets. The expected outcome of this ledger
is a loss, and it is worth running precisely because it is the measurement
that would show that honestly rather than an argument about whether it might.

Two decisions carry the design.

**Append-only, one lock per game-market.** A wager is written once and never
revised. Re-pricing an open position as the line moves is how a paper ledger
quietly becomes a record of the best moment to have bet rather than a record
of decisions actually taken.

**The consensus decides, the best price executes.** The edge test runs against
the paired-book de-vigged consensus; the stake is recorded at the best quote
available. These are deliberately different numbers — a better sportsbook
quote must never move the model's input — and a best price that has run too
far from the consensus is rejected as a broken quote rather than banked as
edge.
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import (EDGE_RULE, GAME_DAY_STAKE_CAP, MARKET_DISAGREEMENT_WARNING,
                    MAX_EXECUTION_DEVIATION, MAX_ODDS_AGE_MINUTES,
                    MAX_LOCK_LEAD_MINUTES, MAX_STAKE, MIN_LOCK_LEAD_MINUTES,
                    MIN_MARKET_BOOKS, MODEL_VERSION, STAKING_POLICY_VERSION)
from odds import american_to_prob

LEDGER_FIELDS = [
    "wager_id", "placed_at", "game_pk", "official_date", "commence_time",
    "home_team", "away_team", "market", "point", "side",
    "model_prob", "market_prob", "disagreement",
    "price", "book", "stake", "market_books", "market_spread",
    "lead_minutes", "wide_market", "model_version", "model_kind",
    "staking_policy_version",
    "settled_at", "outcome", "profit",
]

REJECTION_FIELDS = ["screened_at", "game_pk", "market", "point", "side",
                    "disagreement", "gate", "detail"]


def unset(value):
    """True when a cell carries no value.

    An empty CSV column read back through pandas is NaN, not the empty string,
    and NaN is truthy. A naive emptiness test therefore reads every open wager
    as already settled: settlement skips them forever and the summary reports
    them as settled with no result. The ledger would have shown three settled
    wagers and a 0-0-0 record indefinitely.
    """
    return value is None or pd.isna(value) or str(value).strip() == ""


def payout(price, stake=1.0):
    """Profit on a won wager at American odds, excluding the returned stake."""
    price = float(price)
    return stake * (price / 100.0 if price > 0 else 100.0 / -price)


def _wager_id(row, side):
    point = "" if row.get("point") in (None, "") else row["point"]
    return f"{row['game_pk']}|{row['market']}|{point}|{side}"


def screen(card, open_ids=None, now=None):
    """Apply the staking policy to a priced card.

    Returns ``(wagers, rejections)``. Every row that does not become a wager
    leaves a rejection carrying the gate that stopped it, so the ledger can be
    audited for what it declined as well as what it took.
    """
    open_ids = open_ids or set()
    now = now or datetime.now(timezone.utc)
    stamp = f"{now:%Y-%m-%dT%H:%M:%SZ}"

    candidates, rejections = [], []
    for row in card.to_dict("records"):
        disagreement = float(row["disagreement"])
        side = "home" if disagreement > 0 else "away"
        identifier = _wager_id(row, side)

        def reject(gate, detail=""):
            rejections.append({
                "screened_at": stamp, "game_pk": row["game_pk"],
                "market": row["market"], "point": row["point"], "side": side,
                "disagreement": round(disagreement, 6),
                "gate": gate, "detail": str(detail),
            })

        if identifier in open_ids:
            reject("already_locked")
            continue
        if abs(disagreement) < EDGE_RULE:
            reject("below_edge_rule", f"{abs(disagreement):.4f} < {EDGE_RULE}")
            continue
        if int(row["market_books"]) < MIN_MARKET_BOOKS:
            reject("too_few_books", row["market_books"])
            continue

        lead = int(row["lead_minutes"])
        if lead < MIN_LOCK_LEAD_MINUTES or lead > MAX_LOCK_LEAD_MINUTES:
            # Baseball's decisive information arrives late: lineups post about
            # three hours out and the bullpen picture settles only once the
            # previous day's games are final. Too early is as disqualifying as
            # too late.
            reject("outside_lock_window", f"{lead} min")
            continue

        age = (now - pd.to_datetime(row["odds_fetched_at"], utc=True))
        age_minutes = age.total_seconds() / 60.0
        if age_minutes > MAX_ODDS_AGE_MINUTES:
            reject("stale_quote", f"{age_minutes:.0f} min old")
            continue

        price = row["best_price_home"] if side == "home" else row["best_price_away"]
        consensus = (row["consensus_price_home"] if side == "home"
                     else row["consensus_price_away"])
        if pd.isna(price) or pd.isna(consensus):
            reject("no_executable_price")
            continue
        # Both sides of this comparison carry vig, so it measures line shopping
        # rather than the de-vig. A point or two is a real edge in a liquid
        # market; a gap this wide is a stale or mis-mapped quote.
        gap = american_to_prob(consensus) - american_to_prob(price)
        if gap > MAX_EXECUTION_DEVIATION:
            reject("execution_deviation", f"{gap:.4f} > {MAX_EXECUTION_DEVIATION}")
            continue

        market_probability = (float(row["market_prob_home"]) if side == "home"
                              else 1.0 - float(row["market_prob_home"]))
        model_probability = (float(row["model_prob_home"]) if side == "home"
                             else 1.0 - float(row["model_prob_home"]))
        candidates.append({
            "wager_id": identifier,
            "placed_at": stamp,
            "game_pk": row["game_pk"],
            "official_date": row["official_date"],
            "commence_time": row["commence_time"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "market": row["market"],
            "point": row["point"],
            "side": side,
            "model_prob": round(model_probability, 6),
            "market_prob": round(market_probability, 6),
            "disagreement": round(abs(disagreement), 6),
            "price": price,
            "book": (row["best_book_home"] if side == "home"
                     else row["best_book_away"]),
            "stake": float(MAX_STAKE),
            "market_books": row["market_books"],
            "market_spread": row["market_spread"],
            "lead_minutes": lead,
            # Books disagreeing this much among themselves usually means a
            # mis-mapped line rather than a market. Recorded, not rejected:
            # config calls it a warning, and a forward test that silently drops
            # its own awkward rows is not measuring anything.
            "wide_market": int(float(row["market_spread"])
                               > MARKET_DISAGREEMENT_WARNING),
            "model_version": row.get("model_version", MODEL_VERSION),
            "model_kind": row.get("model_kind", ""),
            "staking_policy_version": STAKING_POLICY_VERSION,
            "settled_at": "", "outcome": "", "profit": "",
        })

    # The day cap is applied last, to the widest disagreements first. Applying
    # it while scanning would hand the cap to whichever games happened to sort
    # earliest, which is a property of the file rather than of the card.
    candidates.sort(key=lambda item: -item["disagreement"])
    taken, per_day = [], {}
    for candidate in candidates:
        day = candidate["official_date"]
        if per_day.get(day, 0.0) + candidate["stake"] > GAME_DAY_STAKE_CAP:
            rejections.append({
                "screened_at": stamp, "game_pk": candidate["game_pk"],
                "market": candidate["market"], "point": candidate["point"],
                "side": candidate["side"],
                "disagreement": candidate["disagreement"],
                "gate": "day_cap", "detail": f"{day} at {GAME_DAY_STAKE_CAP}u",
            })
            continue
        per_day[day] = per_day.get(day, 0.0) + candidate["stake"]
        taken.append(candidate)
    return taken, rejections


def settle(ledger, games):
    """Score open wagers against final results.

    A postponed or suspended game has no result and stays open rather than
    being scored as a loss.
    """
    results = games.set_index("game_pk")
    settled = 0
    stamp = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
    # A ledger whose wagers are all open reads back with these columns empty,
    # which pandas types as float64. Writing "win" into a float column raises,
    # so the very first settlement would crash on a freshly created ledger.
    for column in ("outcome", "settled_at"):
        if column in ledger:
            ledger[column] = ledger[column].astype(object)
    for index, wager in ledger.iterrows():
        if not unset(wager.get("outcome")):
            continue
        game_pk = wager["game_pk"]
        if game_pk not in results.index:
            continue
        game = results.loc[game_pk]
        if pd.isna(game.get("home_score")) or pd.isna(game.get("away_score")):
            continue
        outcome = _outcome(wager, game)
        if outcome is None:
            continue
        ledger.at[index, "outcome"] = outcome
        ledger.at[index, "settled_at"] = stamp
        stake = float(wager["stake"])
        ledger.at[index, "profit"] = round(
            {"win": payout(wager["price"], stake),
             "loss": -stake, "push": 0.0}[outcome], 4)
        settled += 1
    return ledger, settled


def _outcome(wager, game):
    """Win, loss or push for one wager, from the home side's perspective."""
    home_score = float(game["home_score"])
    away_score = float(game["away_score"])
    market, side = wager["market"], wager["side"]
    point = wager["point"]

    if market == "h2h":
        home_result = "win" if home_score > away_score else "loss"
    elif market == "spreads":
        margin = home_score - away_score
        threshold = -float(point)
        if margin == threshold:
            return "push"
        home_result = "win" if margin > threshold else "loss"
    elif market == "totals":
        total = home_score + away_score
        if total == float(point):
            return "push"
        # `home` is the Over on a total, following the odds capture's
        # convention that Over and Under are the two sides of one line.
        home_result = "win" if total > float(point) else "loss"
    else:
        return None
    if side == "home":
        return home_result
    return "loss" if home_result == "win" else "win"


def _load(path, fields):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=fields)
    frame = pd.read_csv(path)
    return frame.reindex(columns=fields) if len(frame) else pd.DataFrame(
        columns=fields)


def _append(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def summarise(ledger):
    """Totals for the forward test. Settled wagers only."""
    if not len(ledger):
        return {"wagers": 0, "status": "no wagers recorded"}
    done = ledger[~ledger["outcome"].map(unset)]
    report = {
        "wagers": int(len(ledger)),
        "open": int(len(ledger) - len(done)),
        "settled": int(len(done)),
        "staking_policy_version": STAKING_POLICY_VERSION,
    }
    if not len(done):
        report["status"] = "nothing settled yet"
        return report
    staked = float(done["stake"].astype(float).sum())
    profit = float(done["profit"].astype(float).sum())
    report.update({
        "units_staked": round(staked, 2),
        "units_profit": round(profit, 4),
        "roi": round(profit / staked, 5) if staked else None,
        "record": {
            outcome: int((done["outcome"] == outcome).sum())
            for outcome in ("win", "loss", "push")
        },
        "by_market": {
            market: {
                "n": int(len(block)),
                "profit": round(float(block["profit"].astype(float).sum()), 4),
            }
            for market, block in done.groupby("market")
        },
    })
    # One season of a flat-stake ledger is a very small sample. Say so in the
    # report rather than leaving an ROI to be read as a result.
    report["interpretation"] = (
        "A flat-stake paper ledger over a few hundred wagers cannot separate "
        "a real edge from variance. Read the market comparison in "
        "market_comparison.json first; this records what the policy would "
        "have done, not whether it works."
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", default="data/predictions_upcoming.csv")
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--ledger", default="data/paper_ledger.csv")
    parser.add_argument("--rejections", default="data/paper_rejections.csv")
    parser.add_argument("--settle-only", action="store_true",
                        help="score open wagers without screening a new card")
    args = parser.parse_args(argv)

    ledger = _load(args.ledger, LEDGER_FIELDS)

    if not args.settle_only:
        card_path = Path(args.card)
        if not card_path.exists():
            raise SystemExit(f"{args.card} not found; run predict_upcoming.py first")
        card = pd.read_csv(card_path)
        if len(card):
            open_ids = set(ledger["wager_id"].astype(str)) if len(ledger) else set()
            wagers, rejections = screen(card, open_ids)
            _append(args.ledger, LEDGER_FIELDS, wagers)
            _append(args.rejections, REJECTION_FIELDS, rejections)
            counts = {}
            for rejection in rejections:
                counts[rejection["gate"]] = counts.get(rejection["gate"], 0) + 1
            print(f"screened {len(card)} card rows -> {len(wagers)} wagers")
            if counts:
                print(f"  rejected: {counts}")
            for wager in wagers:
                point = "" if wager["point"] == "" else f" {float(wager['point']):+g}"
                print(f"  {wager['away_team']} @ {wager['home_team']} "
                      f"{wager['market']}{point} {wager['side']} "
                      f"at {wager['price']:+g} ({wager['book']}), "
                      f"disagreement {wager['disagreement']:.3f}")
            ledger = _load(args.ledger, LEDGER_FIELDS)
        else:
            print("card is empty; nothing to screen")

    if len(ledger):
        games = pd.read_csv(args.games)
        ledger, settled = settle(ledger, games)
        ledger.to_csv(args.ledger, index=False)
        print(f"settled {settled} wager(s)")

    import json
    print(json.dumps(summarise(ledger), indent=2))


if __name__ == "__main__":
    main()
