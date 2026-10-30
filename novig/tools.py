"""Setup and inspection commands for a Novig key.

    python -m novig.tools echo                 # proves host, key, clock, signer
    python -m novig.tools leagues              # valid --leagues values
    python -m novig.tools market-types         # valid --market-type values
    python -m novig.tools markets --league ATP,WTA --status OPEN_INGAME
    python -m novig.tools book <market-id>

Reads ``NOVIG_ENV``, ``NOVIG_KEY_ID`` and ``NOVIG_PEM`` (see client.py).
"""

import argparse
import json
import time

from novig.client import NovigClient


def main(argv=None):
    parser = argparse.ArgumentParser(description="Novig API inspection")
    parser.add_argument("--env", choices=("paper", "live"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("echo")
    sub.add_parser("time")
    sub.add_parser("leagues")
    sub.add_parser("market-types")
    markets = sub.add_parser("markets")
    markets.add_argument("--league", default="ATP,WTA")
    markets.add_argument("--status", default="OPEN_INGAME")
    markets.add_argument("--market-type")
    book = sub.add_parser("book")
    book.add_argument("market_id")
    args = parser.parse_args(argv)

    client = NovigClient.from_env(env=args.env)
    if args.command == "echo":
        out = client.echo({"hello": "novig"})
    elif args.command == "time":
        server = client.server_time()
        out = {"server_ms": server,
               "skew_ms": int(time.time() * 1000) - server}
    elif args.command == "leagues":
        out = client.leagues()
    elif args.command == "market-types":
        out = client.market_types()
    elif args.command == "markets":
        out = [{"marketId": m["marketId"], "type": m.get("marketType"),
                "status": m.get("status"),
                "description": m.get("description"),
                "outcomes": [(o["name"], (o.get("bid") or {}).get("price"))
                             for o in m.get("outcomes", [])]}
               for m in client.markets(league=args.league.split(","),
                                       eventStatus=args.status,
                                       marketType=args.market_type)]
    else:
        out = client.book(args.market_id)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
