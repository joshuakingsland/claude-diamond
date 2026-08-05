"""Venue metadata: coordinates, elevation, orientation, roof, and dimensions.

Everything here comes from the MLB StatsAPI venue endpoint rather than being
transcribed by hand, so park coordinates and orientations cannot drift from
the source through a typo. The fetched table is cached to `data/parks.json`
and is small enough to keep in version control.

`azimuth_angle` is the bearing in degrees from home plate to centre field.
Together with a wind direction it is what makes "blowing out to left" a
number rather than a phrase.
"""

import json
import math
import urllib.request
from pathlib import Path

VENUES_API = "https://statsapi.mlb.com/api/v1/venues"
CACHE = Path("data/parks.json")

# Retractable roofs are only closed some of the time, and StatsAPI does not
# say which state a given game was played in. The reported game-time condition
# is the only signal, so a retractable park is treated as open unless the
# observed condition says otherwise.
CLOSED_CONDITIONS = {"dome", "roof closed"}


def fetch_parks(season, timeout=30):
    """Return venue metadata for every park used in a season."""
    query = (f"{VENUES_API}?sportId=1&season={season}"
             "&hydrate=location,fieldInfo")
    request = urllib.request.Request(query, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    parks = {}
    for venue in payload.get("venues", []):
        location = venue.get("location", {})
        coordinates = location.get("defaultCoordinates", {})
        field = venue.get("fieldInfo", {})
        latitude = coordinates.get("latitude")
        longitude = coordinates.get("longitude")
        if latitude is None or longitude is None:
            continue
        parks[str(venue["id"])] = {
            "venue_id": venue["id"],
            "name": venue.get("name", ""),
            "latitude": float(latitude),
            "longitude": float(longitude),
            "elevation_m": _optional_float(location.get("elevation")),
            "azimuth_angle": _optional_float(location.get("azimuthAngle")),
            "roof_type": field.get("roofType", ""),
            "turf_type": field.get("turfType", ""),
            "capacity": field.get("capacity"),
            "left_line": field.get("leftLine"),
            "center": field.get("center"),
            "right_line": field.get("rightLine"),
        }
    return parks


def _optional_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_parks(path=CACHE):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run `python parks.py --refresh` first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def save_parks(parks, path=CACHE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(parks, indent=1, sort_keys=True), encoding="utf-8")
    return len(parks)


def wind_components(wind_speed, wind_direction, azimuth_angle):
    """Resolve wind onto the park's own axes.

    Returns ``(out_to_center, left_to_right)`` in the same units as
    ``wind_speed``. Positive ``out_to_center`` is wind blowing from home plate
    toward centre field, which carries fly balls out; negative is wind blowing
    in. Positive ``left_to_right`` blows from the left-field line toward right.

    Meteorological wind direction is the direction the wind comes *from*, so
    the direction it blows *toward* is 180 degrees opposed. Getting that
    backwards silently inverts every wind effect in the model, which is why it
    is isolated here and covered by tests.
    """
    if wind_speed is None or wind_direction is None or azimuth_angle is None:
        return None, None
    blowing_toward = (float(wind_direction) + 180.0) % 360.0
    offset = math.radians(blowing_toward - float(azimuth_angle))
    speed = float(wind_speed)
    return speed * math.cos(offset), speed * math.sin(offset)


def roof_category(roof_type):
    """Classify a park as ``open``, ``retractable``, or ``dome``.

    This is the only roof information a model may use, because it is the only
    part known in advance. Whether a retractable roof was actually shut is
    reported by StatsAPI, but only once the game is under way — 1,764 games
    in 2021-2025 show "Roof Closed" after the fact. Feeding that to a model
    would train it on something the live path cannot know three hours before
    first pitch.

    A retractable park instead gets an indicator, and the model learns the
    average wind attenuation there across closed and open nights.
    """
    roof = (roof_type or "").strip().lower()
    if roof == "dome":
        return "dome"
    if roof.startswith("retract"):
        return "retractable"
    return "open"


def observed_roof_closed(condition):
    """Post-hoc roof state, for validating the join only — never a feature."""
    return int((condition or "").strip().lower() in CLOSED_CONDITIONS)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cache", default=str(CACHE))
    args = parser.parse_args()
    if args.refresh or not Path(args.cache).exists():
        parks = fetch_parks(args.season)
        print(f"wrote {save_parks(parks, args.cache)} parks to {args.cache}")
    else:
        print(f"{len(load_parks(args.cache))} parks cached at {args.cache}")


if __name__ == "__main__":
    main()
