"""Price the upcoming card and set the model beside the market.

Everything else in this repository looks backwards. This is the only path that
produces a number for a game that has not been played, and it is the half that
was missing: `odds.py` was writing tomorrow's prices with nothing to compare
them against.

What this is not. The market comparison in `market.py` says the closing price
beats this model on the moneyline and the run line with intervals excluding
zero. A gap between the model and the market is therefore reported as
`disagreement`, not as edge, and the column is named that way deliberately.
Renaming a disagreement `edge` is the whole failure mode this project exists
to avoid.

Three things keep the live path honest:

**The same builder as training.** `features.build` walks games in date order
and folds a result into state only after emitting its row, so an unplayed game
produces a feature row and contributes nothing. Serving is the same code path
as training rather than a reimplementation that drifts from it.

**Forecast weather, kept apart.** The archive is the training source of truth.
Forecasts land in their own file, because a forecast row for a game that has
since been played would replace reanalysis with a guess, and `weather.py`
skips any game_pk it already holds, so nothing would ever correct it.

**Started games are dropped.** The odds feed keeps returning a game after
first pitch and those prices reflect the current score. Pricing them produces
enormous fictional disagreements, which is the same trap `odds.py` fails
closed on.

Whole-number lines push. A book's two-way price has the push mass removed by
construction, so the model probability is renormalised onto the same basis
before the two are compared; leaving it alone would understate every
whole-number total on the board.
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import MODEL_VERSION
from features import build, load_inputs
from market import match_events_to_games, normalise
from models import RunsModel
from odds import _is_future

CARD_FIELDS = [
    "game_pk", "official_date", "commence_time", "home_team", "away_team",
    "market", "point", "model_prob_home", "market_prob_home", "disagreement",
    "market_books", "market_spread", "consensus_price_home",
    "consensus_price_away",
    "best_price_home", "best_book_home", "best_price_away", "best_book_away",
    "expected_home_runs", "expected_away_runs", "lead_minutes",
    "odds_fetched_at", "model_version", "model_kind", "priced_at",
]


def forecast_weather(games, parks, path, verbose=True):
    """Refetch first-pitch forecasts for the games on the card.

    Refetched rather than resumed: a forecast taken six hours before first
    pitch is not yesterday's forecast, and `weather.py`'s resume logic would
    keep the stale one forever.
    """
    from weather import WEATHER_FIELDS, build_for_games

    rows, failed = build_for_games(games, parks, archive=False, already=set(),
                                   verbose=verbose)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WEATHER_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return pd.DataFrame(rows), failed


def offered_points(lines):
    """The run lines and totals the board is actually quoting.

    Pricing a fixed -1.5 and 8.5 would leave most of the card uncomparable:
    a live board quotes alternate totals from 7 to 10 and both sides of the
    run line.
    """
    runlines, totals = set(), set()
    for row in lines.to_dict("records"):
        if row["market"] == "spreads" and pd.notna(row["point"]):
            runlines.add(float(row["point"]))
        elif row["market"] == "totals" and pd.notna(row["point"]):
            totals.add(float(row["point"]))
    return tuple(sorted(runlines)) or (-1.5,), tuple(sorted(totals)) or (8.5,)


def model_probability(priced, market, point):
    """Model probability for one quoted line, on the market's own basis.

    A whole-number line can push and the book's two-way de-vig has that mass
    removed already, so the model side is renormalised to match. Comparing a
    push-inclusive model number against a push-exclusive market number would
    understate every whole-number bet on the board.
    """
    if market == "h2h":
        return priced["p_home_ml"]
    if market == "spreads":
        column, push_column = f"p_home_rl_{point}", f"push_home_rl_{point}"
    else:
        column, push_column = f"p_over_{point}", f"push_over_{point}"
    if column not in priced:
        return None
    probability = priced[column]
    if push_column in priced:
        remaining = 1.0 - priced[push_column]
        probability = probability.where(remaining <= 0, probability / remaining)
    return probability


def build_card(lines, games, features, kind="glm", now=None, verbose=True):
    """Train on everything played, then price the games still to come."""
    now = now or datetime.now(timezone.utc)
    lines = lines[[_is_future(value, now) for value in lines["commence_time"]]]
    if not len(lines):
        return pd.DataFrame(), {"reason": "no future games on the board"}

    events = lines.drop_duplicates("event_id")[
        ["event_id", "home_team", "away_team", "commence_time"]].copy()
    events["home_key"] = events["home_team"].map(normalise)
    events["away_key"] = events["away_team"].map(normalise)
    events["commence"] = pd.to_datetime(events["commence_time"], utc=True,
                                        errors="coerce")
    events = events.dropna(subset=["commence"])

    schedule = games.copy()
    schedule["home_key"] = schedule["home_team_name"].map(normalise)
    schedule["away_key"] = schedule["away_team_name"].map(normalise)
    schedule["start"] = pd.to_datetime(schedule["game_date_utc"], utc=True,
                                       errors="coerce")
    matched, unmatched = match_events_to_games(events, schedule)

    played = set(games.loc[games["home_score"].notna(), "game_pk"])
    trained = features[features["game_pk"].isin(played)]
    model = RunsModel(kind=kind).fit(trained, games)
    if verbose:
        print(f"trained on {len(trained)} completed games, "
              f"dispersion {model.dispersion[0]:.2f}/"
              f"{model.dispersion[1]:.2f}")

    card_pks = {game_pk for game_pk, _ in matched.values()}
    card = features[features["game_pk"].isin(card_pks)]
    if not len(card):
        return pd.DataFrame(), {"reason": "no card games have feature rows",
                                "unmatched_events": len(unmatched)}
    runlines, totals = offered_points(lines)
    lengths = (card[["game_pk"]]
               .merge(games[["game_pk", "scheduled_innings"]],
                      on="game_pk", how="left")["scheduled_innings"])
    priced = model.price(card, runline_points=runlines, total_points=totals,
                         innings=lengths.to_numpy())
    priced = priced.set_index("game_pk")

    stamp = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
    rows = []
    for row in lines.to_dict("records"):
        target = matched.get(row["event_id"])
        if target is None:
            continue
        game_pk, official_date = target
        if game_pk not in priced.index:
            continue
        point = None if pd.isna(row["point"]) else float(row["point"])
        probability = model_probability(priced.loc[[game_pk]], row["market"],
                                        point)
        if probability is None:
            continue
        model_prob = float(probability.iloc[0])
        market_prob = float(row["consensus_prob_home"])
        commence = pd.to_datetime(row["commence_time"], utc=True)
        rows.append({
            "game_pk": game_pk,
            "official_date": official_date,
            "commence_time": row["commence_time"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "market": row["market"],
            "point": "" if point is None else point,
            "model_prob_home": round(model_prob, 6),
            "market_prob_home": round(market_prob, 6),
            "disagreement": round(model_prob - market_prob, 6),
            "market_books": row["market_books"],
            "market_spread": row["market_spread"],
            "consensus_price_home": row["consensus_price_home"],
            "consensus_price_away": row["consensus_price_away"],
            "best_price_home": row["best_price_home"],
            "best_book_home": row["best_book_home"],
            "best_price_away": row["best_price_away"],
            "best_book_away": row["best_book_away"],
            "expected_home_runs": round(
                float(priced.loc[game_pk, "expected_home_runs"]), 4),
            "expected_away_runs": round(
                float(priced.loc[game_pk, "expected_away_runs"]), 4),
            "lead_minutes": int((commence - now).total_seconds() // 60),
            "odds_fetched_at": row["fetched_at"],
            "model_version": MODEL_VERSION,
            "model_kind": kind,
            "priced_at": stamp,
        })
    summary = {
        "board_rows": int(len(lines)),
        "events": int(len(events)),
        "unmatched_events": len(unmatched),
        "priced_rows": len(rows),
        "dispersion_home": round(float(model.dispersion[0]), 4),
        "dispersion_away": round(float(model.dispersion[1]), 4),
    }
    return pd.DataFrame(rows), summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", default="data/lines_upcoming.csv")
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--weather", default="data/weather.csv")
    parser.add_argument("--forecast", default="data/weather_forecast.csv")
    parser.add_argument("--out", default="data/predictions_upcoming.csv")
    parser.add_argument("--kind", default="glm", choices=["gbm", "glm"])
    parser.add_argument("--skip-forecast", action="store_true",
                        help="reuse the forecast file instead of refetching")
    args = parser.parse_args(argv)

    lines_path = Path(args.lines)
    if not lines_path.exists():
        raise SystemExit(f"{args.lines} not found; run odds.py first")
    lines = pd.read_csv(lines_path)
    if not len(lines):
        print("board is empty; nothing to price")
        Path(args.out).write_text(",".join(CARD_FIELDS) + "\n", encoding="utf-8")
        return

    games, parks, weather, pitching, umpires = load_inputs(args.games,
                                                            args.weather)
    card_games = [game for game in games.to_dict("records")
                  if game.get("home_score") != game.get("home_score")]

    if args.skip_forecast and Path(args.forecast).exists():
        forecast = pd.read_csv(args.forecast)
    else:
        print(f"fetching forecasts for {len(card_games)} unplayed games")
        forecast, failed = forecast_weather(card_games, parks, args.forecast)
        if failed:
            print(f"{len(failed)} venue(s) had no forecast; those games fall "
                  f"back to defaults")

    # Archive first: if a game somehow has both, the reanalysis wins.
    combined = pd.concat([weather, forecast], ignore_index=True)
    combined = combined.drop_duplicates(subset="game_pk", keep="first")

    features = build(games, parks, combined, pitching, umpires)
    card, summary = build_card(lines, games, features, kind=args.kind)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if len(card):
        card.to_csv(args.out, index=False)
    else:
        Path(args.out).write_text(",".join(CARD_FIELDS) + "\n", encoding="utf-8")
    print({key: value for key, value in summary.items()})
    if len(card):
        top = card.reindex(
            card["disagreement"].abs().sort_values(ascending=False).index)
        print("\nlargest disagreements (not edge):")
        for row in top.head(8).to_dict("records"):
            line = "" if row["point"] == "" else f" {row['point']:+g}"
            print(f"  {row['away_team']} @ {row['home_team']} "
                  f"{row['market']}{line}: model {row['model_prob_home']:.3f} "
                  f"vs market {row['market_prob_home']:.3f} "
                  f"({row['disagreement']:+.3f})")
    print(f"\nwrote {len(card)} rows to {args.out}")


if __name__ == "__main__":
    main()
