"""Home-plate umpire assignments, from StatsAPI boxscores.

The umpire is the one input assigned to every game rather than a fraction of
them, which is what makes it worth the ingestion: an effect present in all
13,857 games can matter where an effect present in 55 cannot.

Read this alongside the challenge system. Teams now carry two challenges a
game and keep them when correct, which attacks precisely the calls a zone-bias
model would want to exploit — the obvious misses on the edges. Whatever bias
survives is the bias nobody thought worth challenging. A model fitted on
seasons before challenges will therefore overstate what is available now, so
the test below reports the signal by season rather than pooled, and the
comparison between the pre-challenge seasons and the current one is the
finding rather than a footnote.

One request per game, free and keyless, resumable: game_pks already recorded
are skipped. Fail closed — a boxscore that cannot be parsed is counted and
skipped, never recorded with a blank umpire, because a blank would silently
pool every unparsed game into one phantom official.
"""

import argparse
import csv
import json
import os
import time
import urllib.request
from pathlib import Path

BOX_API = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"

UMPIRE_FIELDS = ["game_pk", "official_date", "hp_umpire_id", "hp_umpire_name"]


def _get(url, timeout=30, attempts=3):
    last = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except Exception as error:  # noqa: BLE001 - retried, then reported
            last = error
            if attempt == attempts - 1:
                break
            time.sleep(1.5 ** attempt)
    raise last


def home_plate_umpire(payload):
    """The official working the plate, or None if the boxscore does not say."""
    for entry in payload.get("officials", []) or []:
        if (entry.get("officialType") or "").strip().lower() == "home plate":
            official = entry.get("official", {}) or {}
            if official.get("id"):
                return official["id"], official.get("fullName", "")
    return None


def run(games, out_path, limit=None, verbose=True):
    out_path = Path(out_path)
    done = set()
    if out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as handle:
            done = {row["game_pk"] for row in csv.DictReader(handle)}
    pending = [g for g in games if str(g["game_pk"]) not in done]
    if limit:
        pending = pending[:limit]
    if verbose:
        print(f"{len(done)} games already recorded; {len(pending)} pending")

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recorded, failed = 0, 0
    with out_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=UMPIRE_FIELDS,
                                extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for game in pending:
            try:
                payload = _get(BOX_API.format(game_pk=game["game_pk"]))
                found = home_plate_umpire(payload)
            except Exception:  # noqa: BLE001 - counted, run continues
                found = None
            if found is None:
                failed += 1
                continue
            umpire_id, name = found
            writer.writerow({
                "game_pk": game["game_pk"],
                "official_date": game.get("official_date", ""),
                "hp_umpire_id": umpire_id,
                "hp_umpire_name": name,
            })
            recorded += 1
            if verbose and recorded % 500 == 0:
                handle.flush()
                print(f"  {recorded} games, {failed} without an umpire")
    if verbose:
        print(f"recorded {recorded} games, {failed} without an umpire")
    return recorded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--out", default="data/umpires.csv")
    parser.add_argument("--seasons", default="")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.games, newline="", encoding="utf-8") as handle:
        games = [row for row in csv.DictReader(handle)
                 if row.get("status") == "Final"]
    if args.seasons:
        wanted = {s.strip() for s in args.seasons.split(",")}
        games = [g for g in games if str(g.get("season")) in wanted]
    run(games, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
