# Novig tennis settlement sniper

Buys the winner of a finished tennis match on Novig in the seconds before
Novig's own feed closes the market, inside a 0.990-0.999 price band.

This is separate from the MLB research in the rest of the repo: nothing there
imports from here, and the MLB code still never places a wager.

## Read this first

- **The edge is entirely the result feed.** Novig grades from its own data
  provider. Unless your results reach the sniper before Novig's feed closes
  the market, there is nothing to take. `feeds.py` defines a JSONL hand-off;
  you still need a source fast enough to win that race.
- **Margins are thin, losses are total.** At 0.995 a correct trade earns
  ~0.47% after fees. One wrong result loses the whole stake, which erases
  roughly 200 correct trades.
- **Retirements are never traded.** Novig markets carry a `voids` rule; an
  `FMV` market can settle a retirement at the pre-retirement price, not as a
  win. Only `status: "completed"` results fire.
- **Terms of service.** Trading after a result is known is the kind of thing
  exchanges reserve the right to void and ban. Read Novig's terms before
  going live.
- **Capital is locked until grading**, so turnover is limited by how fast
  Novig settles, not how fast you trade.

## Setup (paper first)

1. Create a paper account and a management key: paper.novig.com -> Profile
   -> Settings -> Novig API. Open a subaccount with its own Ed25519 trading
   key and fund it (see the [quickstart](https://docs.novig.com/api/quickstart.md)).
2. Point the client at the trading key:

   ```bash
   export NOVIG_ENV=paper
   export NOVIG_KEY_ID=<trading key id>
   export NOVIG_PEM=/path/to/trading-key.pem
   pip install requests cryptography
   ```

3. Check the signer and find the real league/market-type names:

   ```bash
   python -m novig.tools echo
   python -m novig.tools leagues
   python -m novig.tools market-types
   python -m novig.tools markets --league ATP,WTA --status OPEN_INGAME
   ```

## Run

```bash
# Dry run: logs every decision as "would_place", places nothing.
python -m novig.sniper --feed results.jsonl

# Paper money.
python -m novig.sniper --feed results.jsonl --execute

# Real money, small caps.
python -m novig.sniper --feed results.jsonl --env live --execute \
  --max-cost-per-match 25 --max-cost-per-day 200
```

Each line appended to `results.jsonl` is one finished match:

```json
{"id": "atp-123", "player1": "Carlos Alcaraz", "player2": "Jannik Sinner",
 "winner": "Carlos Alcaraz", "status": "completed",
 "completed_at": "2026-10-30T18:04:11Z", "source": "my-feed"}
```

Every decision goes to `novig_logs/sniper.jsonl`. The log is also the
sniper's memory: on restart it will not fire again on a market it already
traded.

## What it does per result

1. Skips anything not `completed`, older than `--max-result-age`, or
   doubles.
2. Finds exactly one open live market whose two outcomes match the players
   (accent- and initial-tolerant; two players sharing a surname never match).
3. Reads the book. The winner's ask is `1 - bid` on the loser. If the best
   ask is below `--min-price`, the book disagrees with the feed: skip.
4. Takes every level up to `--max-price` within the per-match and per-day
   budget, as a single IOC order. Never retries a place (a retry would
   double the order).

## Not built yet

- A real result source (the deciding piece).
- Websocket book and lifecycle streaming, which would save one REST round
  trip per trigger and show `END`/`CLOSE` as they happen.
- Fill reconciliation from the private stream or `/v3/portfolio/fills`.
