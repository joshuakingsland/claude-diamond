"""Buy the winner of a finished tennis match before Novig closes it.

When a match ends, the live moneyline stays open until Novig's own feed
marks the event ended. In that window a resting bid on the loser is
liquidity for the winner at ``1 - bid``. This takes it, but only inside a
narrow band (0.990-0.999 by default):

- An ask *below* the band means the book disagrees with our feed: a stale
  or wrong result, a match still in play, a review. The trade is skipped,
  never chased.
- The cap at 0.999 keeps every fill worth something after the taker fee.

A winning contract pays 1 cent, so profit per contract at price ``p`` is
``(1 - p) * (1 - c * p)`` cents with fee coefficient ``c``. At 0.995 and
c = 0.06 that is 0.47% of the stake: real money only at size, and wiped out
by one wrong result. The guards below exist for that one wrong result.

Usage (paper, dry run -- logs what it would do, places nothing):

    python -m novig.sniper --feed results.jsonl

Add ``--execute`` to place orders, and ``--env live`` for real money.
"""

import argparse
import json
import sys
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

from novig.feeds import COMPLETED, JsonlResultFeed

CONTRACT_VALUE = Decimal("0.01")
TICK = Decimal("0.001")
DEFAULT_FEE_COEFFICIENT = Decimal("0.06")


@dataclass
class Limits:
    min_price: Decimal = Decimal("0.990")
    max_price: Decimal = Decimal("0.999")
    max_cost_per_match: Decimal = Decimal("50")
    max_cost_per_day: Decimal = Decimal("500")
    # Below this expected profit an order is not worth a place token.
    min_expected_profit: Decimal = Decimal("0.05")
    # A result older than this has either been traded already or the market
    # has closed; acting on it can only be wrong.
    max_result_age_s: float = 120.0


@dataclass
class Plan:
    market_id: str
    winner_outcome_id: str
    limit_price: Decimal
    qty: int
    cost: Decimal
    fee: Decimal
    expected_profit: Decimal
    levels: list = field(default_factory=list)


# ------------------------------------------------------------------ names
def name_tokens(name):
    """Lowercase ASCII tokens: accents stripped, punctuation split.

    ``"Auger-Aliassime F."`` -> ``["auger", "aliassime", "f"]``
    ``"Alcaraz, Carlos"``    -> ``["carlos", "alcaraz"]``
    """
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    if "," in text:
        last, _, first = text.partition(",")
        text = f"{first} {last}"
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return cleaned.split()


def _surname_and_initials(name):
    tokens = name_tokens(name)
    words = [t for t in tokens if len(t) > 1]
    initials = {t for t in tokens if len(t) == 1}
    return words, initials


def names_match(feed_name, outcome_name):
    """True when two spellings plausibly name the same player.

    The outcome name is taken as authoritative ("First [Middle] Last"). The
    feed name must contain its surname, and any initial the feed carries
    must match a given name's first letter. Doubles never match.
    """
    if "/" in feed_name or "/" in outcome_name:
        return False
    outcome_words, _ = _surname_and_initials(outcome_name)
    feed_words, feed_initials = _surname_and_initials(feed_name)
    if not outcome_words or not feed_words:
        return False
    surname = outcome_words[-1]
    if surname not in feed_words:
        return False
    given = outcome_words[:-1]
    if feed_initials and given:
        if not feed_initials & {g[0] for g in given}:
            return False
    feed_given = [w for w in feed_words if w not in outcome_words]
    # A full first name in the feed that the outcome lacks is a different
    # player who shares the surname (e.g. two Zverevs).
    return not feed_given


def find_market(result, markets):
    """``(market, winner_outcome, loser_outcome)`` or ``(None, reason)``.

    Exactly one open two-outcome market may name both players; anything
    else is ambiguous and skipped.
    """
    matches = []
    for market in markets:
        outcomes = market.get("outcomes") or []
        if market.get("status") != "OPEN" or len(outcomes) != 2:
            continue
        for winner, loser in (outcomes, outcomes[::-1]):
            if (names_match(result.winner, winner["name"])
                    and names_match(result.loser, loser["name"])):
                matches.append((market, winner, loser))
    if not matches:
        return None, "no open market names both players"
    if len(matches) > 1:
        return None, f"{len(matches)} markets match; ambiguous"
    return matches[0], None


# ------------------------------------------------------------------ money
def fee_per_contract(price, coefficient):
    return coefficient * price * (1 - price) * CONTRACT_VALUE


def plan_order(market, winner, loser, book, limits, budget):
    """Size an IOC buy of ``winner`` from the loser's resting bids.

    ``budget`` is the most this order may cost, already reduced by the
    per-day cap. Returns ``(Plan, None)`` or ``(None, reason)``.
    """
    coefficient = Decimal(str((market.get("fee") or {}).get(
        "coefficient", DEFAULT_FEE_COEFFICIENT)))
    bids = (book.get("orders") or {}).get(loser["outcomeId"]) or []
    if not bids:
        return None, "no resting bid on the loser"
    asks = sorted((1 - Decimal(str(b["price"])), int(b["qty"])) for b in bids)
    best = asks[0][0]
    if best < limits.min_price:
        return None, (f"best ask {best} below {limits.min_price}: "
                      "the book disagrees with the result")
    qty = 0
    cost = fee = profit = Decimal(0)
    limit_price = None
    levels = []
    for price, available in asks:
        if price > limits.max_price:
            break
        unit_cost = price * CONTRACT_VALUE + fee_per_contract(price,
                                                              coefficient)
        affordable = int(((budget - cost) / unit_cost).to_integral_value(
            ROUND_DOWN))
        take = min(available, affordable)
        if take <= 0:
            break
        unit_fee = fee_per_contract(price, coefficient)
        qty += take
        cost += take * price * CONTRACT_VALUE
        fee += take * unit_fee
        profit += take * ((1 - price) * CONTRACT_VALUE - unit_fee)
        limit_price = price
        levels.append((str(price), take))
    if not qty:
        return None, (f"best ask {best} above {limits.max_price}"
                      if best > limits.max_price else "budget exhausted")
    if profit < limits.min_expected_profit:
        return None, f"expected profit ${profit:.4f} below minimum"
    return Plan(market["marketId"], winner["outcomeId"],
                limit_price.quantize(TICK), qty, cost, fee, profit,
                levels), None


# ----------------------------------------------------------------- runner
class Sniper:
    def __init__(self, client, feed, limits, leagues, market_type, log_path,
                 execute=False, catalog_every_s=15.0, clock=time.time):
        self.client = client
        self.feed = feed
        self.limits = limits
        self.leagues = leagues
        self.market_type = market_type
        self.log_path = Path(log_path)
        self.execute = execute
        self.catalog_every_s = catalog_every_s
        self.clock = clock
        self.markets = []
        self._catalog_at = 0.0
        self.fired_markets, self.seen_results, self.spent_today = (
            self._load_state())

    # -- state is rebuilt from the log, so a restart never fires twice
    def _today(self):
        return datetime.fromtimestamp(self.clock(), tz=timezone.utc).date()

    def _load_state(self):
        fired, seen, spent = set(), set(), Decimal(0)
        if not self.log_path.exists():
            return fired, seen, spent
        today = self._today().isoformat()
        with open(self.log_path) as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("env") != self.client.env:
                    continue
                if row.get("result_id"):
                    seen.add(row["result_id"])
                # A failed place may still have reached the engine, so it
                # blocks the market as firmly as a confirmed one.
                if row.get("action") in ("placed", "place_failed"):
                    fired.add(row["market_id"])
                    if (row.get("action") == "placed"
                            and row.get("at", "").startswith(today)):
                        spent += Decimal(row["cost"])
        return fired, seen, spent

    def log(self, **row):
        row["at"] = datetime.fromtimestamp(
            self.clock(), tz=timezone.utc).isoformat()
        row["env"] = self.client.env
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
        print(json.dumps(row, default=str), flush=True)

    def refresh_catalog(self, force=False):
        if not force and self.clock() - self._catalog_at < self.catalog_every_s:
            return
        self.markets = self.client.markets(
            league=self.leagues, eventStatus="OPEN_INGAME",
            marketType=self.market_type)
        self._catalog_at = self.clock()

    def handle(self, result):
        base = {"result_id": result.id, "winner": result.winner,
                "loser": result.loser, "status": result.status,
                "source": result.source}
        if result.id in self.seen_results:
            return
        self.seen_results.add(result.id)
        age = self.clock() - result.completed_at.timestamp()
        if result.status != COMPLETED:
            return self.log(action="skip", reason=f"status {result.status}",
                            **base)
        if age > self.limits.max_result_age_s:
            return self.log(action="skip", reason=f"result {age:.0f}s old",
                            **base)
        found, reason = find_market(result, self.markets)
        if found is None:
            # The catalog may predate the market; look once more.
            self.refresh_catalog(force=True)
            found, reason = find_market(result, self.markets)
        if found is None:
            return self.log(action="skip", reason=reason, **base)
        market, winner, loser = found
        base.update(market_id=market["marketId"],
                    description=market.get("description"))
        if market["marketId"] in self.fired_markets:
            return self.log(action="skip", reason="already fired", **base)
        budget = min(self.limits.max_cost_per_match,
                     self.limits.max_cost_per_day - self.spent_today)
        if budget <= 0:
            return self.log(action="skip", reason="daily cap reached", **base)
        book = self.client.book(market["marketId"])
        plan, reason = plan_order(market, winner, loser, book, self.limits,
                                  budget)
        if plan is None:
            return self.log(action="skip", reason=reason, **base)
        detail = {k: str(v) if isinstance(v, Decimal) else v
                  for k, v in asdict(plan).items() if k != "market_id"}
        self.fired_markets.add(plan.market_id)
        if not self.execute:
            return self.log(action="would_place", **detail, **base)
        client_id = str(uuid.uuid4())
        try:
            placed = self.client.place_order(
                plan.winner_outcome_id, str(plan.limit_price), plan.qty,
                "IOC", client_id)
        except Exception as exc:  # never retry a place: it would double
            return self.log(action="place_failed", error=str(exc),
                            client_id=client_id, **detail, **base)
        self.spent_today += plan.cost
        return self.log(action="placed", order_id=placed.get("orderId"),
                        client_id=client_id, **detail, **base)

    def run(self, poll_s=0.1):
        print(f"sniper: env={self.client.env} execute={self.execute} "
              f"leagues={','.join(self.leagues)} log={self.log_path}",
              flush=True)
        day = self._today()
        while True:
            if self._today() != day:
                day, self.spent_today = self._today(), Decimal(0)
            try:
                self.refresh_catalog()
            except Exception as exc:
                self.log(action="catalog_error", error=str(exc))
            results, errors = self.feed.poll()
            for error in errors:
                self.log(action="feed_error", error=error)
            for result in results:
                try:
                    self.handle(result)
                except Exception as exc:
                    self.log(action="error", result_id=result.id,
                             error=str(exc))
            time.sleep(poll_s)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--feed", required=True,
                        help="JSONL file of finished matches (novig/feeds.py)")
    parser.add_argument("--env", choices=("paper", "live"), default="paper")
    parser.add_argument("--execute", action="store_true",
                        help="place orders; without it, only log them")
    parser.add_argument("--leagues", default="ATP,WTA")
    parser.add_argument("--market-type", default="MONEY")
    parser.add_argument("--min-price", default="0.990")
    parser.add_argument("--max-price", default="0.999")
    parser.add_argument("--max-cost-per-match", default="50")
    parser.add_argument("--max-cost-per-day", default="500")
    parser.add_argument("--min-expected-profit", default="0.05")
    parser.add_argument("--max-result-age", type=float, default=120.0)
    parser.add_argument("--log", default="novig_logs/sniper.jsonl")
    args = parser.parse_args(argv)

    limits = Limits(Decimal(args.min_price), Decimal(args.max_price),
                    Decimal(args.max_cost_per_match),
                    Decimal(args.max_cost_per_day),
                    Decimal(args.min_expected_profit), args.max_result_age)
    if not (Decimal("0") < limits.min_price <= limits.max_price < 1):
        parser.error("need 0 < --min-price <= --max-price < 1")
    if args.env == "live" and args.execute:
        print("LIVE: orders will spend real money.", file=sys.stderr)

    from novig.client import NovigClient
    client = NovigClient.from_env(env=args.env)
    Sniper(client, JsonlResultFeed(args.feed), limits,
           args.leagues.split(","), args.market_type, args.log,
           execute=args.execute).run()


if __name__ == "__main__":
    main()
