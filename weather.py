"""Weather at first pitch, from one source for both training and serving.

StatsAPI reports observed conditions, but only once a game is under way or
complete. Training on that and serving on a forecast would fit the model to
information the live path never has — the same train/serve skew that stops a
sharp-book consensus being swapped in mid-flight elsewhere. Open-Meteo offers
a reanalysis archive and a forecast with identical variables and units, so it
is the single source of truth for model inputs. StatsAPI weather is retained
only as an independent check that the join is landing on the right game.

Neither endpoint needs an API key.
"""

import argparse
import json
import math
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"

HOURLY = ("temperature_2m", "relative_humidity_2m", "surface_pressure",
          "wind_speed_10m", "wind_direction_10m", "precipitation")

# Each endpoint serves a bounded window: the reanalysis trails real time by a
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

    A long reanalysis pull is a single request covering years of hourly data,
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
        "weather_source": "open-meteo-archive" if archive else "open-meteo-forecast",
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

    Requests are batched by park because Open-Meteo returns a full hourly
    series for a date range, so five seasons of one venue cost a single call.

    ``already`` is a set of game_pks that have been fetched on a previous run.
    A venue whose games are all present is skipped, and a venue that fails is
    reported and skipped rather than aborting the run, so the job is
    resumable and a transient network fault costs one venue rather than all
    of them.
    """
    from parks import venue_key

    already = already or set()
    low, high = servable_window(archive)
    by_venue, out_of_window = {}, 0
    for game in games:
        venue_id = venue_key(game.get("venue_id"))
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
    for venue_id, venue_games in sorted(by_venue.items()):
        park = parks[venue_id]
        dates = sorted(game["official_date"] for game in venue_games
                       if game.get("official_date"))
        if not dates:
            continue
        # Pad by a day each side so a late first pitch crossing midnight UTC
        # still finds its hour.
        start = (datetime.fromisoformat(dates[0]) - timedelta(days=1)).date()
        end = (datetime.fromisoformat(dates[-1]) + timedelta(days=1)).date()
        try:
            hourly = fetch_hourly(park["latitude"], park["longitude"],
                                  str(start), str(end), archive=archive)
        except Exception as error:  # noqa: BLE001 - reported, run continues
            failed.append((venue_id, park["name"], repr(error)[:80]))
            if verbose:
                print(f"  venue {venue_id} {park['name'][:28]:28s} FAILED")
            continue
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
            print(f"  venue {venue_id} {park['name'][:28]:28s} "
                  f"{found:5d}/{len(venue_games):5d} {start}..{end}")
    missing_park = sum(1 for game in games
                       if venue_key(game.get("venue_id")) not in parks)
    if verbose:
        source = "archive" if archive else "forecast"
        print(f"weather rows {len(rows)}, unknown park {missing_park}, "
              f"missing hour {missing_hour}, failed venues {len(failed)}, "
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
    already = {str(row["game_pk"]) for row in existing}
    rows, failed = build_for_games(games, parks, archive=not args.forecast,
                                   already=already)
    combined = existing + rows
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
