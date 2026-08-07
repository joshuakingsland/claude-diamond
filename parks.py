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
            "elevation_m": _feet_to_metres(location.get("elevation")),
            "azimuth_angle": _optional_float(location.get("azimuthAngle")),
            "roof_type": field.get("roofType", ""),
            "turf_type": field.get("turfType", ""),
            "capacity": field.get("capacity"),
            "left_line": field.get("leftLine"),
            "center": field.get("center"),
            "right_line": field.get("rightLine"),
        }
    return parks


FEET_TO_METRES = 0.3048


def _feet_to_metres(value):
    """StatsAPI reports venue elevation in feet, despite the field name here.

    Coors Field comes back as 5190 and Fenway as 21, which are their heights
    in feet; in metres they are 1582 and 6. Storing the raw number under
    `elevation_m` inflated every altitude by 3.28, and `features.py` then
    divided by 1000 to get a nominal `elevation_km` that was really
    kilofeet. This went unnoticed because a separate lookup bug meant
    elevation never reached the model at all — fixing that one exposed this.
    """
    feet = _optional_float(value)
    return None if feet is None else round(feet * FEET_TO_METRES, 2)


def _optional_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_parks_range(seasons, timeout=30, verbose=True):
    """Union of every venue used across ``seasons``.

    One season is not enough. A park retired before the snapshot season is
    simply absent, and its games then fall back to a default 20C still-air day
    that never happened. Tropicana Field and the Oakland Coliseum alone host
    730 games in 2021-2024, and both are gone from a 2025 venue list.
    """
    parks = {}
    for season in seasons:
        found = fetch_parks(season, timeout=timeout)
        added = [key for key in found if key not in parks]
        parks.update(found)
        if verbose:
            print(f"  {season}: {len(found)} venues, {len(added)} new")
    return parks


def id_key(value):
    """A numeric id as the string every lookup table is keyed by.

    Generic, despite living beside the parks: venue ids, team ids and player
    ids all take this path. The name used to say `venue`, which hid that it
    was the fix for a whole class of bug and meant the same mistake was made
    again with starting pitchers.

    Read through pandas an id arrives as ``3313.0`` whenever any row in its
    column is missing, because that types the whole column as float, and
    "3313.0" matches no key. It misses quietly, because callers fall back to
    an empty record rather than raising. It has cost this project twice: an
    `elevation_km` of zero on every row, so Coors Field's mile of altitude
    never reached the model; and every starting pitcher resolving to the
    league average, which made the entire pitcher feature set inert while
    looking perfectly reasonable.
    """
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


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
    parser.add_argument("--seasons", default="2021-2026",
                        help="season or inclusive range, e.g. 2021-2026")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cache", default=str(CACHE))
    args = parser.parse_args()
    if "-" in args.seasons:
        first, last = (int(part) for part in args.seasons.split("-", 1))
        seasons = range(first, last + 1)
    else:
        seasons = [int(args.seasons)]
    if args.refresh or not Path(args.cache).exists():
        print(f"fetching venues for seasons {list(seasons)}")
        parks = fetch_parks_range(seasons)
        print(f"wrote {save_parks(parks, args.cache)} parks to {args.cache}")
    else:
        print(f"{len(load_parks(args.cache))} parks cached at {args.cache}")


if __name__ == "__main__":
    main()
