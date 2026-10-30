"""Record live tennis match-winner books and lifecycle, then measure them.

The question before any money moves: when a match finishes, what does the
book actually do, and how long is it tradable?

    python -m novig.recorder record --env live          # leave it running
    python -m novig.recorder analyze novig_logs/recording-*.jsonl

``record`` streams the ``book`` channel (which includes ``lifecycle``) for
every live ATP/WTA ``MONEY`` market and writes one JSONL line whenever a
market's top of book or sweepable liquidity changes, plus every lifecycle
transition verbatim. It only reads, so a ``trading::read`` key is enough.

``analyze`` reports per market:

- when a side's ask first reached the jump threshold (default 0.997),
- when Novig sent ``END`` and ``CLOSE``, and the gap between them,
- how much could be bought in the band at ``END`` and after the jump,
- which side Novig graded the winner, so a jump that reversed or a jumped
  side that lost is counted rather than assumed away.
"""

import argparse
import glob
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from novig.sniper import CONTRACT_VALUE

BAND = (Decimal("0.990"), Decimal("0.999"))


# ------------------------------------------------------------------- book
class MarketBook:
    """Resting orders for one market, from a snapshot plus deltas."""

    def __init__(self, outcome_ids):
        self.outcome_ids = list(outcome_ids)
        self.orders = {}
        self.seq = None
        self.resyncing = False

    def load_snapshot(self, book):
        self.orders = {}
        for outcome, rows in (book.get("orders") or {}).items():
            for row in rows:
                order_id = row.get("orderId") or row.get("order")
                self.orders[order_id] = (outcome, Decimal(str(row["price"])),
                                         int(row["qty"]))
        self.seq = book.get("seq")
        self.resyncing = False

    def apply(self, batch):
        """Apply one delta batch. False means a gap: resnapshot."""
        seq = batch.get("seq")
        if self.seq is None:
            return False
        if seq is not None and seq <= self.seq:
            return True
        if seq is not None and seq != self.seq + 1:
            return False
        for delta in batch.get("deltas") or []:
            order_id = delta.get("orderId") or delta.get("order")
            if delta.get("kind") == "add":
                self.orders[order_id] = (delta["outcome"],
                                         Decimal(str(delta["price"])),
                                         int(delta["qty"]))
            elif delta.get("kind") == "remove":
                self.orders.pop(order_id, None)
        self.seq = seq if seq is not None else self.seq
        return True

    def summary(self, band=BAND):
        """Per outcome: its best bid, its ask, and what the band offers.

        Buying outcome X takes bids resting on the *other* outcome, at
        ``1 - bid``.
        """
        bids = defaultdict(list)
        for outcome, price, qty in self.orders.values():
            bids[outcome].append((price, qty))
        out = {}
        for outcome in self.outcome_ids:
            others = [b for o, rows in bids.items() if o != outcome
                      for b in rows]
            own = sorted(bids.get(outcome, []), reverse=True)
            best_other = max((p for p, _ in others), default=None)
            sweep_qty, sweep_cost = 0, Decimal(0)
            for price, qty in others:
                ask = 1 - price
                if band[0] <= ask <= band[1]:
                    sweep_qty += qty
                    sweep_cost += ask * qty * CONTRACT_VALUE
            out[outcome] = {
                "bid": str(own[0][0]) if own else None,
                "ask": str(1 - best_other) if best_other is not None else None,
                "sweep_qty": sweep_qty,
                "sweep_cost": str(sweep_cost.quantize(Decimal("0.01"))),
            }
        return out


# ----------------------------------------------------------------- record
class Recorder:
    def __init__(self, client, leagues, market_type, out_path,
                 catalog_every_s=30.0):
        self.client = client
        self.leagues = leagues
        self.market_type = market_type
        self.out_path = Path(out_path)
        self.catalog_every_s = catalog_every_s
        self.meta = {}       # market id -> catalog row
        self.books = {}      # market id -> MarketBook
        self.last_summary = {}
        self.nonce = 0

    def write(self, **row):
        row["rx"] = int(time.time() * 1000)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "a") as handle:
            handle.write(json.dumps(row, default=str) + "\n")

    def send(self, ws, **verb):
        self.nonce += 1
        ws.send(json.dumps({"nonce": self.nonce, **verb}))

    def refresh(self, ws):
        markets = self.client.markets(league=self.leagues,
                                      eventStatus="OPEN_INGAME",
                                      marketType=self.market_type)
        new = [m for m in markets
               if m["marketId"] not in self.books
               and len(m.get("outcomes") or []) == 2]
        for market in new:
            market_id = market["marketId"]
            self.meta[market_id] = market
            self.books[market_id] = MarketBook(
                o["outcomeId"] for o in market["outcomes"])
            self.write(kind="market", m=market_id, event=market["eventId"],
                       desc=market.get("description"),
                       outcomes={o["outcomeId"]: o["name"]
                                 for o in market["outcomes"]})
        if new:
            self.send(ws, subscribe={"markets": {
                m["marketId"]: "book" for m in new}})

    def handle(self, ws, message):
        frame = json.loads(message)
        if "code" in frame and "snapshot" not in frame:
            self.write(kind="error", frame=frame)
            return
        ts = frame.get("ts")
        for market_id, part in (frame.get("snapshot") or {}).items():
            book = self.books.get(market_id)
            if book is None:
                continue
            if part.get("book"):
                book.load_snapshot(part["book"])
            if part.get("lifecycle"):
                self.write(kind="lifecycle", t=ts, m=market_id,
                           transition="SNAPSHOT",
                           status=part["lifecycle"].get("status"))
            self.emit(market_id, ts)
        for market_id, part in (frame.get("delta") or {}).items():
            book = self.books.get(market_id)
            if book is None:
                continue
            for change in (part.get("lifecycle") or {}).get("deltas") or []:
                self.write(kind="lifecycle", t=ts, m=market_id,
                           transition=change.get("kind"),
                           status=change.get("status"))
                if change.get("kind") == "CLOSE":
                    self.close(ws, market_id)
            if part.get("book") and market_id in self.books:
                if book.resyncing:
                    continue  # the snapshot on its way supersedes these
                if not book.apply(part["book"]):
                    self.write(kind="gap", t=ts, m=market_id)
                    book.resyncing = True
                    self.send(ws, snapshot={"markets": {market_id: "book"}})
                    continue
                self.emit(market_id, ts)

    def emit(self, market_id, ts):
        summary = self.books[market_id].summary()
        if summary != self.last_summary.get(market_id):
            self.last_summary[market_id] = summary
            self.write(kind="top", t=ts, m=market_id, outcomes=summary)

    def close(self, ws, market_id):
        # Keep the watched-market cap free; the grade comes from history.
        self.books.pop(market_id, None)
        self.last_summary.pop(market_id, None)
        self.send(ws, unsubscribe=[f"market:{market_id}"])

    def run(self):
        print(f"recording {','.join(self.leagues)} {self.market_type} "
              f"on {self.client.env} -> {self.out_path}", flush=True)
        while True:
            try:
                with self.client.connect_ws() as ws:
                    # A reconnect starts over: every market resubscribes.
                    self.books.clear()
                    self.last_summary.clear()
                    self.nonce = 0
                    next_refresh = 0.0
                    while True:
                        if time.time() >= next_refresh:
                            self.refresh(ws)
                            next_refresh = time.time() + self.catalog_every_s
                        try:
                            message = ws.recv(timeout=1.0)
                        except TimeoutError:
                            continue
                        self.handle(ws, message)
            except KeyboardInterrupt:
                return
            except Exception as exc:
                self.write(kind="disconnect", error=repr(exc))
                print(f"disconnected: {exc!r}; reconnecting", flush=True)
                time.sleep(5)


# ---------------------------------------------------------------- analyze
def load_recordings(paths):
    markets = {}
    for path in paths:
        with open(path) as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                market_id = row.get("m")
                if not market_id:
                    continue
                market = markets.setdefault(
                    market_id, {"desc": None, "outcomes": {}, "tops": [],
                                "lifecycle": []})
                if row["kind"] == "market":
                    market["desc"] = row.get("desc")
                    market["outcomes"] = row.get("outcomes") or {}
                elif row["kind"] == "top":
                    market["tops"].append(row)
                elif row["kind"] == "lifecycle":
                    market["lifecycle"].append(row)
    return markets


def _when(row):
    return row.get("t") or row.get("rx")


def analyze_market(market, jump=Decimal("0.997"), winner_id=None):
    """One market's settlement window, from its recording."""
    end = next((_when(r) for r in market["lifecycle"]
                if r["transition"] == "END"), None)
    close = next((_when(r) for r in market["lifecycle"]
                  if r["transition"] == "CLOSE"), None)
    jump_at = jumped = None
    for top in market["tops"]:
        for outcome, side in top["outcomes"].items():
            if side.get("ask") and Decimal(side["ask"]) >= jump:
                jump_at, jumped = _when(top), outcome
                break
        if jump_at:
            break

    def sweep_after(start, outcome):
        """Most the band ever offered on ``outcome`` from ``start`` on."""
        if start is None or outcome is None:
            return 0, Decimal(0)
        best = (0, Decimal(0))
        for top in market["tops"]:
            if _when(top) < start or (close and _when(top) > close):
                continue
            side = top["outcomes"].get(outcome) or {}
            best = max(best, (side.get("sweep_qty", 0),
                              Decimal(side.get("sweep_cost", "0"))))
        return best

    reverted = False
    if jump_at:
        for top in market["tops"]:
            ask = (top["outcomes"].get(jumped) or {}).get("ask")
            if _when(top) > jump_at and ask and Decimal(ask) < BAND[0]:
                reverted = True
                break

    name = market["outcomes"].get
    sweep_qty, sweep_cost = sweep_after(jump_at, jumped)
    end_qty, end_cost = sweep_after(end, winner_id or jumped)
    return {
        "desc": market["desc"],
        "jumped": name(jumped) if jumped else None,
        "winner": name(winner_id) if winner_id else None,
        "jumped_side_won": (None if not (jumped and winner_id)
                            else jumped == winner_id),
        "reverted_below_band": reverted,
        "jump_to_end_s": ((end - jump_at) / 1000
                          if end and jump_at else None),
        "end_to_close_s": (close - end) / 1000 if end and close else None,
        "jump_to_close_s": ((close - jump_at) / 1000
                            if close and jump_at else None),
        "sweep_after_jump": {"qty": sweep_qty, "cost": str(sweep_cost)},
        "sweep_after_end": {"qty": end_qty, "cost": str(end_cost)},
    }


def fetch_winner(client, market_id):
    """The outcome id Novig graded ``WIN``, or None if not settled/void."""
    market = client.call("GET", f"/v3/history/markets/{market_id}")
    for outcome in market.get("outcomes") or []:
        if outcome.get("status") == "WIN":
            return outcome["outcomeId"]
    return None


def analyze(paths, client=None, jump=Decimal("0.997")):
    markets = load_recordings(paths)
    rows = []
    for market_id, market in markets.items():
        winner = None
        if client is not None:
            try:
                winner = fetch_winner(client, market_id)
            except Exception as exc:
                print(f"no grade for {market_id}: {exc}")
        rows.append({"market_id": market_id,
                     **analyze_market(market, jump, winner)})
    return rows


def _summarize(rows):
    jumped = [r for r in rows if r["jumped"]]
    graded = [r for r in jumped if r["jumped_side_won"] is not None]
    gaps = sorted(r["end_to_close_s"] for r in rows
                  if r["end_to_close_s"] is not None)
    return {
        "markets": len(rows),
        "jumped": len(jumped),
        "jumped_and_graded": len(graded),
        "jumped_side_lost": sum(1 for r in graded
                                if not r["jumped_side_won"]),
        "reverted_below_band": sum(1 for r in jumped
                                   if r["reverted_below_band"]),
        "with_end_and_close": len(gaps),
        "median_end_to_close_s": gaps[len(gaps) // 2] if gaps else None,
        "markets_with_band_liquidity_after_end": sum(
            1 for r in rows if r["sweep_after_end"]["qty"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--env", choices=("paper", "live"))
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record")
    record.add_argument("--leagues", default="ATP,WTA")
    record.add_argument("--market-type", default="MONEY")
    record.add_argument("--out")
    report = sub.add_parser("analyze")
    report.add_argument("paths", nargs="+")
    report.add_argument("--jump", default="0.997")
    report.add_argument("--no-grades", action="store_true",
                        help="skip fetching Novig's grades")
    args = parser.parse_args(argv)

    from novig.client import NovigClient
    if args.command == "record":
        client = NovigClient.from_env(env=args.env)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = args.out or f"novig_logs/recording-{client.env}-{stamp}.jsonl"
        Recorder(client, args.leagues.split(","), args.market_type,
                 out).run()
        return
    paths = [p for pattern in args.paths for p in glob.glob(pattern)]
    client = None if args.no_grades else NovigClient.from_env(env=args.env)
    rows = analyze(paths, client, Decimal(args.jump))
    for row in rows:
        print(json.dumps(row, default=str))
    print(json.dumps({"summary": _summarize(rows)}, indent=2))


if __name__ == "__main__":
    main()
