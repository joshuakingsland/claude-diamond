"""Odds capture for the moneyline, run line, and total.

Ported from a fight-market pipeline that had already been hardened, so the
awkward parts are deliberate:

- Consensus is the median of per-book de-vigged probabilities, never a de-vig
  of independently aggregated prices.
- The consensus the model consumes and the best price the ledger executes on
  are separate numbers. A better sportsbook quote never moves the model input.
- Regions are captured and priced independently, so a research region can
  accumulate history without touching the model.
- Every paired quote is retained with book, region, and timestamp.

Baseball adds three markets rather than one. `spreads` is the run line and
`totals` is the over/under; both carry a `point` that has to travel with the
price, because a -1.5 at -105 and a -1.5 at +130 are different bets and a
+1.5 is a different bet again.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from config import (LEADER_BOOK_KEYS, MARKETS, ODDS_CONSENSUS_VERSION,
                    ODDS_REGIONS, PRICED_ODDS_REGIONS, SPORT_KEY)

API = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"

QUOTE_FIELDS = [
    "snapshot_id", "fetched_at", "event_id", "commence_time", "date",
    "home_team", "away_team", "market", "point", "book_key", "book_title",
    "region", "priced", "book_updated_at", "price_home", "price_away",
    "devig_prob_home",
]

LINE_FIELDS = [
    "date", "commence_time", "event_id", "home_team", "away_team",
    "market", "point", "line_role", "consensus_prob_home", "market_books",
    "market_spread", "consensus_book_keys",
    "consensus_price_home", "consensus_price_away",
    "best_price_home", "best_book_home", "best_price_away", "best_book_away",
    "best_price_home_updated_at", "best_price_away_updated_at",
    "consensus_oldest_updated_at", "consensus_latest_updated_at",
    "leader_prob_home", "leader_books", "follower_prob_home", "follower_books",
    "odds_source", "fetched_at",
]


def _is_future(commence_time, now=None):
    """True only when a start time is present, parseable, and still ahead.

    An unparseable or missing timestamp counts as not-future, so a malformed
    row is dropped rather than silently priced.
    """
    if not commence_time:
        return False
    now = now or datetime.now(timezone.utc)
    try:
        moment = datetime.fromisoformat(str(commence_time).replace("Z", "+00:00"))
    except ValueError:
        return False
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment > now


def american_to_prob(odds):
    odds = float(odds)
    return -odds / (-odds + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def _upper_median(values):
    values = sorted(values)
    return values[len(values) // 2]


def paired_book_quotes(event, region="us"):
    """Return paired two-sided quotes per book, per market, per line point.

    A quote only counts when both sides are present at the same point. A book
    showing one side of a total is not a market, and de-vigging it against a
    different book's other side would invent a price neither offered.
    """
    home = event.get("home_team", "")
    away = event.get("away_team", "")
    paired = []
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            key = market.get("key")
            if key not in MARKETS:
                continue
            sides = {}
            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                point = outcome.get("point")
                price = outcome.get("price")
                if key == "h2h":
                    side = "home" if name == home else "away" if name == away else None
                    point = None
                elif key == "spreads":
                    side = "home" if name == home else "away" if name == away else None
                elif key == "totals":
                    side = {"Over": "home", "Under": "away"}.get(name)
                else:
                    side = None
                if side is None:
                    continue
                # Group a run line or total by the home-side point so the two
                # halves of the same bet meet. Away points are the negation.
                if key == "spreads" and side == "away" and point is not None:
                    group = -float(point)
                else:
                    group = None if point is None else float(point)
                sides.setdefault((key, group), {})[side] = price
            for (market_key, point), prices in sides.items():
                home_price, away_price = prices.get("home"), prices.get("away")
                try:
                    home_price = float(home_price)
                    away_price = float(away_price)
                except (TypeError, ValueError):
                    continue
                if abs(home_price) < 100 or abs(away_price) < 100:
                    continue
                probability_home = american_to_prob(home_price)
                probability_away = american_to_prob(away_price)
                paired.append({
                    "market": market_key,
                    "point": point,
                    "book_key": book.get("key", ""),
                    "book_title": book.get("title", book.get("key", "")),
                    "region": region,
                    "priced": int(region in PRICED_ODDS_REGIONS),
                    "book_updated_at": book.get("last_update", ""),
                    "price_home": home_price,
                    "price_away": away_price,
                    "devig_prob_home": probability_home
                    / (probability_home + probability_away),
                })
    return paired


def priced_quotes(paired):
    """Quotes from regions the model is allowed to price.

    A quote with no region predates multi-region capture and was US, so it
    stays priced.
    """
    return [row for row in paired
            if row.get("region", "us") in PRICED_ODDS_REGIONS]


def leader_split(paired):
    """Market-setting books versus the books that follow them.

    Research provenance only. Leaders are drawn from every captured region so
    Pinnacle counts while `eu` stays unpriced; followers are priced non-leader
    books, so the gap compares setters against the market actually traded.
    """
    leaders = [row["devig_prob_home"] for row in paired
               if row["book_key"] in LEADER_BOOK_KEYS]
    followers = [row["devig_prob_home"] for row in priced_quotes(paired)
                 if row["book_key"] not in LEADER_BOOK_KEYS]
    return {
        "leader_prob_home": (round(float(statistics.median(leaders)), 8)
                             if leaders else ""),
        "leader_books": len(leaders),
        "follower_prob_home": (round(float(statistics.median(followers)), 8)
                               if followers else ""),
        "follower_books": len(followers),
    }


def main_line_points(paired):
    """Choose one well-covered point per market.

    Different books can expose different main points at the same instant.  A
    consensus calculated independently at every point is not a coherent price
    surface: the subsets of books change, and the live policy can end up
    backing both teams -1.5 or two adjacent Overs.  Until a full latent market
    distribution is fitted, the honest executable universe is the modal point
    with the broadest priced-book coverage.

    Ties are resolved toward the cross-book median point, then
    deterministically by the numeric point.  Research-only regions never
    choose the executable line.
    """
    priced = priced_quotes(paired)
    selected = {}
    for market in MARKETS:
        subset = [quote for quote in priced if quote["market"] == market]
        if not subset:
            continue
        if market == "h2h":
            selected[market] = None
            continue
        groups = {}
        for quote in subset:
            groups.setdefault(float(quote["point"]), []).append(quote)
        median = float(statistics.median(
            [float(quote["point"]) for quote in subset]))
        selected[market] = max(
            groups,
            key=lambda point: (
                len({quote["book_key"] for quote in groups[point]}),
                -abs(point - median),
                -abs(point),
                -point,
            ),
        )
    return selected


def consensus_lines(event, paired=None, main_only=True):
    """Consensus rows for one event; executable output is main-line only."""
    paired = paired_book_quotes(event) if paired is None else paired
    selected = main_line_points(paired) if main_only else None
    grouped = {}
    for quote in paired:
        if selected is not None and quote["market"] in selected:
            wanted = selected[quote["market"]]
            point = quote["point"]
            if wanted is None:
                if point is not None:
                    continue
            elif point is None or not math.isclose(float(point), float(wanted)):
                continue
        grouped.setdefault((quote["market"], quote["point"]), []).append(quote)
    rows = []
    for (market, point), quotes in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1] or 0)
    ):
        priced = priced_quotes(quotes)
        if not priced:
            continue
        probabilities = [quote["devig_prob_home"] for quote in priced]
        best_home = max(priced, key=lambda quote: quote["price_home"])
        best_away = max(priced, key=lambda quote: quote["price_away"])
        updates = sorted(str(quote.get("book_updated_at") or "")
                         for quote in priced
                         if quote.get("book_updated_at"))
        rows.append({
            "market": market,
            "point": "" if point is None else point,
            "line_role": "main" if main_only else "offered",
            "consensus_prob_home": round(float(statistics.median(probabilities)), 8),
            "market_books": len(priced),
            "market_spread": round(float(max(probabilities) - min(probabilities)), 8),
            "consensus_book_keys": ",".join(sorted(
                {quote["book_key"] for quote in priced})),
            "consensus_price_home": _upper_median([q["price_home"] for q in priced]),
            "consensus_price_away": _upper_median([q["price_away"] for q in priced]),
            "best_price_home": best_home["price_home"],
            "best_book_home": best_home["book_title"],
            "best_price_away": best_away["price_away"],
            "best_book_away": best_away["book_title"],
            "best_price_home_updated_at": best_home.get("book_updated_at", ""),
            "best_price_away_updated_at": best_away.get("book_updated_at", ""),
            "consensus_oldest_updated_at": updates[0] if updates else "",
            "consensus_latest_updated_at": updates[-1] if updates else "",
            **leader_split(quotes),
        })
    return rows


def fetch_region(key, region, markets=MARKETS, timeout=30):
    """Return ``(events, credits)``, matching ``historical_odds._request``.

    The credit headers used to be discarded here. That left the cheap path
    blind to a budget the expensive path was tracking carefully, which matters
    because both draw on one quota: a backfill at 30 credits a snapshot can
    drain the account, and the only symptom on this side is a capture that
    stops returning data. A missing snapshot cannot be bought back later at
    the live price.
    """
    query = urllib.parse.urlencode({
        "apiKey": key,
        "regions": region,
        "markets": ",".join(markets),
        "oddsFormat": "american",
    })
    request = urllib.request.Request(f"{API}?{query}",
                                     headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response), {
            "credits_used": response.headers.get("x-requests-last"),
            "credits_remaining": response.headers.get("x-requests-remaining"),
        }


def collect_events(key, regions=ODDS_REGIONS, fetch=None):
    """Merge each region's quotes onto one record per event.

    Priced regions are read first so a book listed in more than one region
    keeps its priced quote and the consensus cannot depend on which region
    answered first.

    Returns ``(events, credits)``. ``fetch`` must return ``(events, credits)``
    for one region.
    """
    fetch = fetch_region if fetch is None else fetch
    ordered = ([r for r in regions if r in PRICED_ODDS_REGIONS]
               + [r for r in regions if r not in PRICED_ODDS_REGIONS])
    merged, credits = {}, []
    for region in ordered:
        events, spend = fetch(key, region)
        credits.append({"region": region, **spend})
        for event in events:
            event_id = event.get("id", "")
            slot = merged.setdefault(event_id,
                                     {"event": event, "paired": [], "seen": set()})
            for quote in paired_book_quotes(event, region):
                fingerprint = (quote["book_key"], quote["market"], quote["point"])
                if fingerprint in slot["seen"]:
                    continue
                slot["seen"].add(fingerprint)
                slot["paired"].append(quote)
    return [(slot["event"], slot["paired"]) for slot in merged.values()], credits


def _quote_rows(event, paired, stamp):
    commence = event.get("commence_time", "")
    rows = []
    for quote in paired:
        raw = "|".join(str(value) for value in (
            stamp, event.get("id", ""), quote["region"], quote["book_key"],
            quote["market"], quote["point"], quote["price_home"],
            quote["price_away"], quote["book_updated_at"],
        ))
        rows.append({
            "snapshot_id": hashlib.sha256(raw.encode()).hexdigest()[:20],
            "fetched_at": stamp,
            "event_id": event.get("id", ""),
            "commence_time": commence,
            "date": commence[:10],
            "home_team": event.get("home_team", ""),
            "away_team": event.get("away_team", ""),
            **quote,
        })
    return rows


def _write_atomic(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def append_quote_log(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing.extend(csv.DictReader(handle))
    normalized = [{field: row.get(field, "") for field in QUOTE_FIELDS}
                  for row in existing]
    known = {row["snapshot_id"] for row in normalized}
    for row in rows:
        item = {field: row.get(field, "") for field in QUOTE_FIELDS}
        if item["snapshot_id"] not in known:
            normalized.append(item)
            known.add(item["snapshot_id"])
    _write_atomic(path, QUOTE_FIELDS, normalized)


CREDIT_FIELDS = ["fetched_at", "region", "credits_used", "credits_remaining"]


def record_credits(path, stamp, credits):
    """Append this run's spend and return the lowest remaining balance seen.

    The balance is what the API reports, not a local tally, so spend from any
    other project on the same key shows up here too.
    """
    rows, balances = [], []
    for entry in credits:
        rows.append({
            "fetched_at": stamp,
            "region": entry.get("region", ""),
            "credits_used": entry.get("credits_used") or "",
            "credits_remaining": entry.get("credits_remaining") or "",
        })
        try:
            balances.append(int(entry.get("credits_remaining")))
        except (TypeError, ValueError):
            continue
    if rows:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CREDIT_FIELDS,
                                    extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
    return min(balances) if balances else None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-key", action="store_true")
    parser.add_argument("--lines", default="data/lines_upcoming.csv")
    parser.add_argument("--quotes-dir", default="data/market_quotes")
    parser.add_argument("--credit-log", default="data/credit_log.csv")
    parser.add_argument("--min-credits", type=int, default=0,
                        help="fail the run when fewer credits remain than "
                             "this; 0 disables the floor")
    args = parser.parse_args(argv)

    key = os.environ.get("ODDS_API_KEY")
    if not key:
        if args.require_key:
            raise SystemExit(
                "ODDS_API_KEY is required. Add it as a GitHub Actions secret."
            )
        print("No ODDS_API_KEY set; nothing fetched.")
        if not Path(args.lines).exists():
            _write_atomic(args.lines, LINE_FIELDS, [])
            print(f"Empty {args.lines} template written.")
        return

    stamp = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
    now = datetime.now(timezone.utc)
    rows, quotes, in_play = [], [], 0
    events, credits = collect_events(key)
    for event, paired in events:
        commence = event.get("commence_time", "")
        # Fail closed on anything already under way. The odds endpoint keeps
        # returning a game after first pitch, but those are in-play prices
        # reflecting the current score, which the model cannot see. On a live
        # card they read as enormous edges: four games in the first real
        # capture priced at 0.96, 0.97 and 0.13 home win probability, values
        # that do not occur in a pre-game baseball market.
        if not _is_future(commence, now):
            in_play += 1
            continue
        for line in consensus_lines(event, paired):
            rows.append({
                "date": commence[:10],
                "commence_time": commence,
                "event_id": event.get("id", ""),
                "home_team": event.get("home_team", ""),
                "away_team": event.get("away_team", ""),
                **line,
                "odds_source": f"the-odds-api-{ODDS_CONSENSUS_VERSION}",
                "fetched_at": stamp,
            })
        quotes.extend(_quote_rows(event, paired, stamp))

    _write_atomic(args.lines, LINE_FIELDS, rows)
    quote_path = Path(args.quotes_dir) / f"quotes_{stamp[:7]}.csv"
    append_quote_log(quote_path, quotes)
    priced = sum(int(row.get("priced", 1)) for row in quotes)
    print(f"wrote {len(rows)} consensus lines and appended {len(quotes)} quotes "
          f"({priced} priced from {','.join(PRICED_ODDS_REGIONS)}; "
          f"regions {','.join(ODDS_REGIONS)}; markets {','.join(MARKETS)}; "
          f"{in_play} in-play event(s) rejected)")

    remaining = record_credits(args.credit_log, stamp, credits)
    if remaining is not None:
        print(f"credits remaining: {remaining}")
        if args.min_credits and remaining < args.min_credits:
            raise SystemExit(
                f"credit floor breached: {remaining} left, floor is "
                f"{args.min_credits}. This quota is shared, so the likeliest "
                f"cause is spend from elsewhere. Capture keeps failing until "
                f"the floor is lowered or the plan is topped up."
            )


if __name__ == "__main__":
    main()
