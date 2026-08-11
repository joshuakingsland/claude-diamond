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

from config import (EDGE_RULE, GAME_DAY_STAKE_CAP, GAME_RISK_BUCKET_STAKE_CAP,
                    MARKET_DISAGREEMENT_WARNING, MAX_BOOK_QUOTE_AGE_MINUTES,
                    MAX_EXECUTION_DEVIATION, MAX_ODDS_AGE_MINUTES,
                    MAX_LOCK_LEAD_MINUTES, MAX_STAKE, MIN_EXPECTED_VALUE,
                    MIN_LOCK_LEAD_MINUTES, MIN_MARKET_BOOKS,
                    REQUIRE_CONFIRMED_LINEUPS, STAKING_POLICY_VERSION)
from odds import american_to_prob

LEDGER_FIELDS = [
    "wager_id", "placed_at", "event_id", "game_pk", "official_date",
    "commence_time",
    "home_team", "away_team", "market", "point", "side",
    "standalone_model_prob", "model_prob", "market_prob", "disagreement",
    "predicted_clv", "expected_value", "risk_bucket",
    "price", "book", "stake", "market_books", "market_spread",
    "book_updated_at", "quote_age_minutes", "lead_minutes", "wide_market",
    "model_version", "model_revision", "feature_schema", "model_kind",
    "distribution_family",
    "trained_through", "market_offset_version", "staking_policy_version",
    "leader_weight",
    "execution_status",
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
    """Stable identity for one wager. This is the key that stops a re-lock.

    The moneyline has no point, and a card read back through pandas delivers
    that absence as NaN rather than "". Formatting NaN straight into the key
    produced ids like `824082|h2h|nan|away`, which work only for as long as
    the column keeps typing as float: a card whose points all parse, or a
    different reader, yields "" instead, every moneyline id changes, and every
    open moneyline wager is locked a second time.
    """
    point = "" if unset(row.get("point")) else row["point"]
    return f"{row['game_pk']}|{row['market']}|{point}|{side}"


def staked_by_day(ledger):
    """Units already committed per game day, from the ledger on disk."""
    totals = {}
    if not len(ledger):
        return totals
    for row in ledger.to_dict("records"):
        day = row.get("official_date")
        if unset(day):
            continue
        try:
            totals[day] = totals.get(day, 0.0) + float(row.get("stake") or 0)
        except (TypeError, ValueError):
            continue
    return totals


def risk_bucket(row):
    """Correlated positions that compete for one unit of game exposure."""
    return "total" if row.get("market") == "totals" else "side"


def locked_risk_buckets(ledger):
    """Risk buckets already used by the append-only ledger across runs."""
    if not len(ledger):
        return set()
    locked = set()
    for row in ledger.to_dict("records"):
        game_pk = row.get("game_pk")
        if unset(game_pk):
            continue
        bucket = row.get("risk_bucket")
        if unset(bucket):
            bucket = risk_bucket(row)
        locked.add((str(game_pk), str(bucket)))
    return locked


def screen(card, open_ids=None, prior_stakes=None, prior_risk_buckets=None,
           now=None):
    """Apply the staking policy to a priced card.

    Returns ``(wagers, rejections)``. Every row that does not become a wager
    leaves a rejection carrying the gate that stopped it, so the ledger can be
    audited for what it declined as well as what it took.

    ``prior_stakes`` carries units already committed per day. Without it the
    day cap only holds inside a single call, which is worthless: the capture
    workflow screens the same card every hour, and each run would grant a
    fresh cap. Thirteen runs against a three-unit cap is thirty-nine units on
    a day the policy limits to three.
    """
    open_ids = open_ids or set()
    prior_stakes = dict(prior_stakes or {})
    prior_risk_buckets = set(prior_risk_buckets or set())
    now = now or datetime.now(timezone.utc)
    stamp = f"{now:%Y-%m-%dT%H:%M:%SZ}"

    candidates, rejections = [], []
    for row in card.to_dict("records"):
        disagreement = float(row.get("fair_disagreement",
                                     row["disagreement"]))
        side = "home" if disagreement > 0 else "away"
        identifier = _wager_id(row, side)
        bucket = risk_bucket(row)
        bucket_key = (str(row["game_pk"]), bucket)

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
        if bucket_key in prior_risk_buckets:
            reject("risk_bucket_already_locked", bucket)
            continue
        if abs(disagreement) < EDGE_RULE:
            reject("below_edge_rule", f"{abs(disagreement):.4f} < {EDGE_RULE}")
            continue
        if int(row["market_books"]) < MIN_MARKET_BOOKS:
            reject("too_few_books", row["market_books"])
            continue
        if REQUIRE_CONFIRMED_LINEUPS and not int(
                row.get("lineups_confirmed", 0) or 0):
            reject("lineups_unconfirmed")
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
        update_column = ("best_price_home_updated_at" if side == "home"
                         else "best_price_away_updated_at")
        book_updated_at = row.get(update_column)
        if unset(book_updated_at):
            # Backward-compatible for old captured cards. New captures always
            # carry the individual book timestamp.
            book_updated_at = row["odds_fetched_at"]
        book_age = (now - pd.to_datetime(book_updated_at, utc=True))
        book_age_minutes = book_age.total_seconds() / 60.0
        if book_age_minutes > MAX_BOOK_QUOTE_AGE_MINUTES:
            reject("stale_book_quote",
                   f"{book_age_minutes:.1f} min > {MAX_BOOK_QUOTE_AGE_MINUTES}")
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
        fair_home = float(row.get("fair_prob_home", row["model_prob_home"]))
        model_probability = (fair_home if side == "home" else 1.0 - fair_home)
        standalone_home = float(row["model_prob_home"])
        standalone_probability = (standalone_home if side == "home"
                                  else 1.0 - standalone_home)
        expected_value = (model_probability * payout(price)
                          - (1.0 - model_probability))
        if expected_value < MIN_EXPECTED_VALUE:
            reject("below_expected_value",
                   f"{expected_value:.4f} < {MIN_EXPECTED_VALUE}")
            continue
        raw_predicted_clv = row.get("predicted_clv", 0.0)
        predicted_clv = (0.0 if unset(raw_predicted_clv)
                         else float(raw_predicted_clv))
        if side == "away":
            predicted_clv = -predicted_clv
        candidates.append({
            "wager_id": identifier,
            "placed_at": stamp,
            "event_id": row.get("event_id", ""),
            "game_pk": row["game_pk"],
            "official_date": row["official_date"],
            "commence_time": row["commence_time"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "market": row["market"],
            "point": "" if unset(row["point"]) else row["point"],
            "side": side,
            "standalone_model_prob": round(standalone_probability, 6),
            "model_prob": round(model_probability, 6),
            "market_prob": round(market_probability, 6),
            "disagreement": round(abs(disagreement), 6),
            "predicted_clv": round(predicted_clv, 6),
            "expected_value": round(expected_value, 6),
            "risk_bucket": bucket,
            "price": price,
            "book": (row["best_book_home"] if side == "home"
                     else row["best_book_away"]),
            "stake": float(MAX_STAKE),
            "market_books": row["market_books"],
            "market_spread": row["market_spread"],
            "book_updated_at": book_updated_at,
            "quote_age_minutes": round(book_age_minutes, 3),
            "lead_minutes": lead,
            # Books disagreeing this much among themselves usually means a
            # mis-mapped line rather than a market. Recorded, not rejected:
            # config calls it a warning, and a forward test that silently drops
            # its own awkward rows is not measuring anything.
            "wide_market": int(float(row["market_spread"])
                               > MARKET_DISAGREEMENT_WARNING),
            "model_version": row.get("model_version", "unknown"),
            "model_revision": row.get("model_revision", "unknown"),
            "feature_schema": row.get("feature_schema", "unknown"),
            "model_kind": row.get("model_kind", ""),
            "distribution_family": row.get("distribution_family", ""),
            "trained_through": row.get("trained_through", ""),
            "market_offset_version": row.get("market_offset_version", ""),
            "leader_weight": row.get("leader_weight", ""),
            "staking_policy_version": STAKING_POLICY_VERSION,
            "execution_status": "paper",
            "settled_at": "", "outcome": "", "profit": "",
        })

    # One non-dominated position per correlated game bucket. H2H and spreads
    # compete for the side bucket; all total points compete for the total
    # bucket. Ranking is expected profit at the executable quote, then
    # predicted closing-line value, not raw probability disagreement.
    candidates.sort(key=lambda item: (-item["expected_value"],
                                      -item["predicted_clv"],
                                      -item["disagreement"]))
    unique, used = [], set(prior_risk_buckets)
    for candidate in candidates:
        key = (str(candidate["game_pk"]), candidate["risk_bucket"])
        if key in used:
            rejections.append({
                "screened_at": stamp, "game_pk": candidate["game_pk"],
                "market": candidate["market"], "point": candidate["point"],
                "side": candidate["side"],
                "disagreement": candidate["disagreement"],
                "gate": "risk_bucket_dominated",
                "detail": candidate["risk_bucket"],
            })
            continue
        used.add(key)
        unique.append(candidate)
    candidates = unique

    # The day cap is applied last, to the highest expected values first.
    taken, per_day = [], prior_stakes
    for candidate in candidates:
        day = candidate["official_date"]
        bucket_stake = candidate["stake"]
        if bucket_stake > GAME_RISK_BUCKET_STAKE_CAP:
            bucket_stake = float(GAME_RISK_BUCKET_STAKE_CAP)
            candidate["stake"] = bucket_stake
        if per_day.get(day, 0.0) + bucket_stake > GAME_DAY_STAKE_CAP:
            rejections.append({
                "screened_at": stamp, "game_pk": candidate["game_pk"],
                "market": candidate["market"], "point": candidate["point"],
                "side": candidate["side"],
                "disagreement": candidate["disagreement"],
                "gate": "day_cap", "detail": f"{day} at {GAME_DAY_STAKE_CAP}u",
            })
            continue
        per_day[day] = per_day.get(day, 0.0) + bucket_stake
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
            wagers, rejections = screen(card, open_ids,
                                        prior_stakes=staked_by_day(ledger),
                                        prior_risk_buckets=locked_risk_buckets(
                                            ledger))
            _append(args.ledger, LEDGER_FIELDS, wagers)
            _append(args.rejections, REJECTION_FIELDS, rejections)
            counts = {}
            for rejection in rejections:
                counts[rejection["gate"]] = counts.get(rejection["gate"], 0) + 1
            print(f"screened {len(card)} card rows -> {len(wagers)} wagers")
            if counts:
                print(f"  rejected: {counts}")
            for wager in wagers:
                point = ("" if unset(wager["point"])
                         else f" {float(wager['point']):+g}")
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
