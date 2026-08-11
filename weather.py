"""Forecast-model weather at first pitch for training and serving.

StatsAPI reports observed conditions, but only once a game is under way or
complete. Training on that and serving on a forecast would fit the model to
information the live path never has — the same train/serve skew that stops a
sharp-book consensus being swapped in mid-flight elsewhere. Open-Meteo's
Historical Forecast API stitches operational forecast runs with the same
variables and response format as the live Forecast API. Training on that
product removes the old realised-reanalysis versus live-forecast skew.
StatsAPI weather is retained only as an independent join check.

Neither endpoint needs an API key.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import time
import urllib.parse
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ARCHIVE_API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"

HOURLY = ("temperature_2m", "relative_humidity_2m", "surface_pressure",
          "wind_speed_10m", "wind_direction_10m", "precipitation")

# Each endpoint serves a bounded window: the historical forecast trails real time by a
# few days, the forecast runs about a fortnight ahead. Requests are batched by
# venue over one date range, so a single game outside the window returns 400
# for that whole request and costs the venue its entire series. Once the
# schedule includes the season still to be played that is every venue, every
# run. Games outside the window are dropped here instead.
ARCHIVE_LAG_DAYS = 2
FORECAST_HORIZON_DAYS = 14

WEATHER_FIELDS = [
    "game_pk", "weather_source", "weather_hour_utc",
    "temp_c", "humidity_pct", "pressure_hpa",
    "wind_speed_ms", "wind_direction_deg", "precip_mm",
    "wind_out_to_center_ms", "wind_left_to_right_ms", "roof_category",
]


def _get(url, timeout=60, attempts=5):
    """GET with exponential backoff.

    A long historical-forecast pull is one request covering years of hourly data,
    so one transient TLS timeout would otherwise discard an entire venue.
    """
    last = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except Exception as error:  # noqa: BLE001 - retried and re-raised
            last = error
            if isinstance(error, urllib.error.HTTPError) \
                    and 400 <= error.code < 500 and error.code != 429:
                break
            if attempt == attempts - 1:
                break
            time.sleep(2 ** attempt)
    raise last


def _round_to_hour(moment):
    return moment.replace(minute=0, second=0, microsecond=0)


def fetch_hourly(latitude, longitude, start_date, end_date, archive=True,
                 timeout=60):
    """Return an hour-indexed weather table for one location and date range."""
    query = urllib.parse.urlencode({
        "latitude": round(float(latitude), 4),
        "longitude": round(float(longitude), 4),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    })
    api = ARCHIVE_API if archive else FORECAST_API
    payload = _get(f"{api}?{query}", timeout=timeout)
    hourly = payload.get("hourly", {})
    times = hourly.get("time", []) or []
    table = {}
    for index, stamp in enumerate(times):
        table[stamp] = {
            name: _at(hourly.get(name), index) for name in HOURLY
        }
    return table


def _at(series, index):
    if not series or index >= len(series):
        return None
    return series[index]


def game_weather(game_pk, first_pitch_utc, park, hourly, archive=True):
    """Resolve one game's first-pitch weather onto the park's own axes."""
    from parks import roof_category, wind_components

    moment = first_pitch_utc
    if isinstance(moment, str):
        moment = datetime.fromisoformat(moment.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    key = _round_to_hour(moment.astimezone(timezone.utc)).strftime("%Y-%m-%dT%H:00")
    reading = hourly.get(key)
    if reading is None:
        return None
    out_to_center, left_to_right = wind_components(
        reading.get("wind_speed_10m"), reading.get("wind_direction_10m"),
        park.get("azimuth_angle"),
    )
    category = roof_category(park.get("roof_type"))
    if category == "dome":
        # A fixed dome is known in advance and the outside wind never reaches
        # the ball. A retractable park keeps its wind reading: whether the
        # roof shut is not knowable at prediction time, so the model gets the
        # category and learns the average attenuation instead.
        out_to_center, left_to_right = 0.0, 0.0
    return {
        "game_pk": game_pk,
        "weather_source": ("open-meteo-historical-forecast" if archive
                           else "open-meteo-forecast"),
        "weather_hour_utc": key,
        "temp_c": reading.get("temperature_2m"),
        "humidity_pct": reading.get("relative_humidity_2m"),
        "pressure_hpa": reading.get("surface_pressure"),
        "wind_speed_ms": reading.get("wind_speed_10m"),
        "wind_direction_deg": reading.get("wind_direction_10m"),
        "precip_mm": reading.get("precipitation"),
        "wind_out_to_center_ms": _round(out_to_center),
        "wind_left_to_right_ms": _round(left_to_right),
        "roof_category": category,
    }


def _round(value, places=3):
    return None if value is None else round(float(value), places)


def air_density_index(temp_c, pressure_hpa, humidity_pct, elevation_m):
    """A single scalar for how far a ball carries, relative to sea level.

    Warmer, lower-pressure, more humid, higher-altitude air is thinner and
    carries the ball further. Values above 1 mean more carry than a standard
    sea-level day. This is a physical index, not a fitted coefficient, so it
    can be computed identically in training and at serve time.
    """
    if temp_c is None or pressure_hpa is None:
        return None
    temperature_k = float(temp_c) + 273.15
    pressure_pa = float(pressure_hpa) * 100.0
    # Vapour pressure lowers density; humid air is lighter than dry air.
    humidity = 0.0 if humidity_pct is None else float(humidity_pct) / 100.0
    saturation = 610.78 * math.exp(17.27 * float(temp_c) / (float(temp_c) + 237.3))
    vapour = humidity * saturation
    density = ((pressure_pa - vapour) / (287.058 * temperature_k)
               + vapour / (461.495 * temperature_k))
    reference = 101325.0 / (287.058 * 288.15)
    if density <= 0:
        return None
    # Elevation already shows up through pressure; it is kept out of the
    # formula to avoid counting the same thin air twice.
    return round(reference / density, 5)


def servable_window(archive, today=None):
    """The date range the chosen endpoint will actually answer for.

    Returned as ``(low, high)`` ISO strings, either of which may be None for
    an open end.
    """
    today = today or datetime.now(timezone.utc).date()
    if archive:
        return None, str(today - timedelta(days=ARCHIVE_LAG_DAYS))
    return (str(today - timedelta(days=1)),
            str(today + timedelta(days=FORECAST_HORIZON_DAYS)))


def build_for_games(games, parks, archive=True, verbose=True, already=None):
    """Fetch weather for many games, one request per park and date span.

    Requests are batched by park and bounded two-year range because Open-Meteo
    returns a full hourly series. Independent chunks run concurrently; this is
    a one-time source migration and subsequent runs skip completed games.

    ``already`` is a set of game_pks that have been fetched on a previous run.
    A venue whose games are all present is skipped, and a venue that fails is
    reported and skipped rather than aborting the run, so the job is
    resumable and a transient network fault costs one venue rather than all
    of them.
    """
    from parks import id_key

    already = already or set()
    low, high = servable_window(archive)
    by_venue, out_of_window = {}, 0
    for game in games:
        venue_id = id_key(game.get("venue_id"))
        if venue_id not in parks:
            continue
        if str(game.get("game_pk")) in already:
            continue
        date = game.get("official_date")
        if not date or (low and date < low) or (high and date > high):
            out_of_window += 1
            continue
        by_venue.setdefault(venue_id, []).append(game)
    rows, missing_hour, failed = [], 0, []
    plans, tasks = {}, []
    for venue_id, venue_games in sorted(by_venue.items()):
        park = parks[venue_id]
        dates = sorted(game["official_date"] for game in venue_games
                       if game.get("official_date"))
        if not dates:
            continue
        # Six variables over five seasons can exceed the historical server's
        # response timeout even though the date range is valid. Two-year
        # chunks bound response size and, more importantly, let one failed
        # archive shard preserve and retry only its games. They are joined in
        # memory before first-pitch lookup, so the output contract is unchanged.
        chunks = {}
        for game in venue_games:
            year = int(str(game["official_date"])[:4])
            chunks.setdefault((year - 2021) // 2, []).append(game)
        chunk_ranges = []
        for chunk_games in chunks.values():
            chunk_dates = sorted(game["official_date"] for game in chunk_games)
            # Pad by a day each side so a late first pitch crossing midnight
            # UTC still finds its hour.
            start = (datetime.fromisoformat(chunk_dates[0])
                     - timedelta(days=1)).date()
            end = (datetime.fromisoformat(chunk_dates[-1])
                   + timedelta(days=1)).date()
            chunk_ranges.append((start, end))
            tasks.append((venue_id, park, start, end))
        plans[venue_id] = (park, venue_games, chunk_ranges)

    hourly_by_venue = {venue_id: {} for venue_id in plans}
    workers = max(1, int(os.environ.get("WEATHER_WORKERS", "6")))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(fetch_hourly, park["latitude"], park["longitude"],
                            str(start), str(end), archive):
            (venue_id, park, start, end)
            for venue_id, park, start, end in tasks
        }
        for future in as_completed(pending):
            venue_id, park, start, end = pending[future]
            try:
                hourly_by_venue[venue_id].update(future.result())
            except Exception as error:  # noqa: BLE001 - retained and retried
                failed.append((venue_id, park["name"], repr(error)[:80]))
                if verbose:
                    print(f"  venue {venue_id} {park['name'][:28]:28s} "
                          f"FAILED {start}..{end}")

    for venue_id, (park, venue_games, chunk_ranges) in sorted(plans.items()):
        hourly = hourly_by_venue[venue_id]
        found = 0
        for game in venue_games:
            row = game_weather(game["game_pk"], game["game_date_utc"], park,
                               hourly, archive=archive)
            if row is None:
                missing_hour += 1
                continue
            rows.append(row)
            found += 1
        if verbose:
            first = min(start for start, _ in chunk_ranges)
            last = max(end for _, end in chunk_ranges)
            print(f"  venue {venue_id} {park['name'][:28]:28s} "
                  f"{found:5d}/{len(venue_games):5d} {first}..{last}")
    missing_park = sum(1 for game in games
                       if id_key(game.get("venue_id")) not in parks)
    if verbose:
        source = "archive" if archive else "forecast"
        print(f"weather rows {len(rows)}, unknown park {missing_park}, "
              f"missing hour {missing_hour}, failed venue-chunks {len(failed)}, "
              f"outside the {source} window {out_of_window}")
        for venue_id, name, error in failed:
            print(f"    retry venue {venue_id} {name}: {error}")
    return rows, failed


def main():
    import csv

    from parks import load_parks

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--out", default="data/weather.csv")
    parser.add_argument("--forecast", action="store_true")
    parser.add_argument(
        "--refresh-source", action="store_true",
        help="replace legacy reanalysis rows as historical forecast rows are "
             "successfully fetched; failed venues retain their old rows",
    )
    args = parser.parse_args()
    with open(args.games, newline="", encoding="utf-8") as handle:
        games = [row for row in csv.DictReader(handle)
                 if row.get("game_date_utc")]
    parks = load_parks()
    path = Path(args.out)
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        print(f"resuming: {len(existing)} rows already present")
    desired_source = ("open-meteo-forecast" if args.forecast
                      else "open-meteo-historical-forecast")
    already = {
        str(row["game_pk"]) for row in existing
        if not args.refresh_source or row.get("weather_source") == desired_source
    }
    rows, failed = build_for_games(games, parks, archive=not args.forecast,
                                   already=already)
    # New rows replace the same game atomically. Legacy rows for a venue that
    # failed this run stay in place and are retried next time; a source
    # migration must never turn a transient API failure into missing history.
    combined_by_game = {str(row["game_pk"]): row for row in existing}
    combined_by_game.update({str(row["game_pk"]): row for row in rows})
    combined = sorted(combined_by_game.values(),
                      key=lambda row: str(row.get("game_pk", "")))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WEATHER_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)
    os.replace(temporary, path)
    print(f"wrote {len(combined)} weather rows to {args.out} "
          f"({len(rows)} new)")
    if failed:
        raise SystemExit(f"{len(failed)} venue(s) failed; rerun to resume")


if __name__ == "__main__":
    main()
