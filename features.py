"""Point-in-time features for one game, built strictly from earlier games.

The whole file is organised around a single rule: a row for a game on date D
may only use information that existed before D. That is enforced structurally
rather than by care — the builder walks games in chronological order, emits
the feature row from the current state, and only then folds the game's own
result into that state. A feature can therefore never see its own outcome,
and the ordering makes a lookahead bug hard to write rather than easy to
miss.

Season boundaries are respected: team form carries over with heavy
regression, because a team in April is not the team of the previous
September, but ratings are not reset to nothing either.
"""

import csv
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from weather import air_density_index

# League-average runs per team per game, used only as a prior before a team
# has played. It is refined from data as the season goes on.
PRIOR_RUNS = 4.5
PRIOR_WEIGHT = 40.0

ELO_START = 1500.0
ELO_K = 4.0
ELO_HOME_EDGE = 24.0
# Between seasons a team keeps this share of its rating distance from the
# mean. Rosters turn over, but not completely.
ELO_CARRYOVER = 0.70

FORM_WINDOW = 25


class TeamState:
    """Everything known about one team as of the moment before a game."""

    def __init__(self):
        self.elo = ELO_START
        self.runs_for = 0.0
        self.runs_against = 0.0
        self.games = 0
        self.recent_for = deque(maxlen=FORM_WINDOW)
        self.recent_against = deque(maxlen=FORM_WINDOW)
        self.last_date = None
        self.season = None

    def offense(self):
        """Shrunk runs-scored rate; a short record leans on the league prior."""
        return ((self.runs_for + PRIOR_RUNS * PRIOR_WEIGHT)
                / (self.games + PRIOR_WEIGHT))

    def defense(self):
        return ((self.runs_against + PRIOR_RUNS * PRIOR_WEIGHT)
                / (self.games + PRIOR_WEIGHT))

    def recent_offense(self):
        return float(np.mean(self.recent_for)) if self.recent_for else PRIOR_RUNS

    def recent_defense(self):
        return (float(np.mean(self.recent_against))
                if self.recent_against else PRIOR_RUNS)

    def roll_season(self, season):
        if self.season is None:
            self.season = season
            return
        if season == self.season:
            return
        self.elo = ELO_START + (self.elo - ELO_START) * ELO_CARRYOVER
        self.runs_for *= 0.0
        self.runs_against *= 0.0
        self.games = 0
        self.recent_for.clear()
        self.recent_against.clear()
        self.season = season


class PitcherState:
    """Runs allowed by the team in games this pitcher started.

    This is a proxy, not a pitcher's own earned-run average: it includes the
    bullpen that followed him. It is used because it costs nothing — the
    starter's identity is already on every schedule row — and because it is
    point-in-time by construction. A true FIP needs per-start boxscore lines
    and is a separate ingestion job; the column names here do not pretend
    otherwise.
    """

    def __init__(self):
        self.runs_allowed = 0.0
        self.starts = 0
        self.recent = deque(maxlen=10)
        self.last_date = None

    def rate(self):
        return ((self.runs_allowed + PRIOR_RUNS * 10.0)
                / (self.starts + 10.0))

    def recent_rate(self):
        return float(np.mean(self.recent)) if self.recent else PRIOR_RUNS

    def rest(self, date):
        if self.last_date is None:
            return 5
        return int(np.clip((date - self.last_date).days, 0, 15))


class ParkState:
    """Expanding park run factor, from games already played at the venue."""

    def __init__(self):
        self.runs = 0.0
        self.games = 0

    def factor(self, league_mean):
        if self.games < 30 or league_mean <= 0:
            return 1.0
        observed = self.runs / self.games
        raw = observed / league_mean
        # Regress toward neutral; a park factor from 30 games is mostly noise.
        weight = self.games / (self.games + 120.0)
        return 1.0 + (raw - 1.0) * weight


FEATURE_COLUMNS = [
    "home_elo", "away_elo", "elo_diff",
    "home_off", "home_def", "away_off", "away_def",
    "home_recent_off", "home_recent_def",
    "away_recent_off", "away_recent_def",
    "home_rest", "away_rest", "rest_diff",
    "home_games_played", "away_games_played",
    "home_sp_rate", "away_sp_rate", "home_sp_recent", "away_sp_recent",
    "home_sp_starts", "away_sp_starts", "home_sp_rest", "away_sp_rest",
    "park_factor", "elevation_km",
    "temp_c", "air_density_index", "wind_out_to_center_ms",
    "wind_left_to_right_ms", "precip_mm", "roof_retractable", "roof_dome",
    "expected_home_runs_prior", "expected_away_runs_prior",
]


def _rest_days(previous, current):
    if previous is None:
        return 5
    delta = (current - previous).days
    return int(np.clip(delta, 0, 10))


def build(games, parks, weather=None):
    """Return a feature frame aligned to ``games``, in chronological order.

    ``games`` must contain final results; unfinished games are kept so the
    same builder can produce a live card, but they contribute nothing to
    state.
    """
    weather = weather if weather is not None else pd.DataFrame()
    weather_by_game = {}
    if len(weather):
        for row in weather.to_dict("records"):
            weather_by_game[str(row["game_pk"])] = row

    games = games.sort_values(["official_date", "game_pk"]).reset_index(drop=True)
    teams = defaultdict(TeamState)
    pitchers = defaultdict(PitcherState)
    park_states = defaultdict(ParkState)
    league_runs, league_games = 0.0, 0

    rows = []
    for game in games.to_dict("records"):
        season = game["season"]
        home_id, away_id = str(game["home_team_id"]), str(game["away_team_id"])
        home, away = teams[home_id], teams[away_id]
        home.roll_season(season)
        away.roll_season(season)

        date = pd.Timestamp(game["official_date"])
        home_sp = pitchers[str(game.get("home_sp_id"))]
        away_sp = pitchers[str(game.get("away_sp_id"))]
        venue = str(game["venue_id"])
        park = parks.get(venue, {})
        league_mean = (league_runs / league_games) if league_games else PRIOR_RUNS * 2
        park_factor = park_states[venue].factor(league_mean)

        conditions = weather_by_game.get(str(game["game_pk"]), {})
        temp_c = _float(conditions.get("temp_c"))
        density = air_density_index(
            temp_c, _float(conditions.get("pressure_hpa")),
            _float(conditions.get("humidity_pct")),
            park.get("elevation_m"),
        )

        # A simple prior on run scoring: each side's offence against the
        # other's defence, scaled by the park. The model refines this; it is
        # here so the estimator starts from something with baseball in it
        # rather than from zero.
        expected_home = (home.offense() * away.defense() / PRIOR_RUNS) * park_factor
        expected_away = (away.offense() * home.defense() / PRIOR_RUNS) * park_factor

        rows.append({
            "game_pk": game["game_pk"],
            "official_date": game["official_date"],
            "season": season,
            "home_elo": home.elo,
            "away_elo": away.elo,
            "elo_diff": home.elo - away.elo + ELO_HOME_EDGE,
            "home_off": home.offense(),
            "home_def": home.defense(),
            "away_off": away.offense(),
            "away_def": away.defense(),
            "home_recent_off": home.recent_offense(),
            "home_recent_def": home.recent_defense(),
            "away_recent_off": away.recent_offense(),
            "away_recent_def": away.recent_defense(),
            "home_rest": _rest_days(home.last_date, date),
            "away_rest": _rest_days(away.last_date, date),
            "rest_diff": (_rest_days(home.last_date, date)
                          - _rest_days(away.last_date, date)),
            "home_games_played": home.games,
            "away_games_played": away.games,
            "home_sp_rate": home_sp.rate(),
            "away_sp_rate": away_sp.rate(),
            "home_sp_recent": home_sp.recent_rate(),
            "away_sp_recent": away_sp.recent_rate(),
            "home_sp_starts": home_sp.starts,
            "away_sp_starts": away_sp.starts,
            "home_sp_rest": home_sp.rest(date),
            "away_sp_rest": away_sp.rest(date),
            "park_factor": park_factor,
            "elevation_km": (park.get("elevation_m") or 0.0) / 1000.0,
            "temp_c": temp_c if temp_c is not None else 20.0,
            "air_density_index": density if density is not None else 1.0,
            "wind_out_to_center_ms": _float(conditions.get("wind_out_to_center_ms")) or 0.0,
            "wind_left_to_right_ms": _float(conditions.get("wind_left_to_right_ms")) or 0.0,
            "precip_mm": _float(conditions.get("precip_mm")) or 0.0,
            # Roof CATEGORY only. Whether a retractable roof was actually
            # shut is reported after the fact and is not knowable three hours
            # before first pitch, so the model gets the park's category and
            # learns the average wind attenuation there.
            "roof_retractable": int(conditions.get("roof_category") == "retractable"),
            "roof_dome": int(conditions.get("roof_category") == "dome"),
            "expected_home_runs_prior": expected_home,
            "expected_away_runs_prior": expected_away,
        })

        # ---- state update happens only after the row is emitted ----
        home_score, away_score = game.get("home_score"), game.get("away_score")
        if pd.isna(home_score) or pd.isna(away_score):
            continue
        home_score, away_score = float(home_score), float(away_score)

        expected_home_win = 1.0 / (1.0 + 10 ** (-(home.elo - away.elo
                                                  + ELO_HOME_EDGE) / 400.0))
        actual = 1.0 if home_score > away_score else 0.0
        # Margin-aware update: a blowout is more evidence than a one-run game,
        # with diminishing returns so a 15-run night is not fifteen wins.
        margin_weight = np.log1p(abs(home_score - away_score))
        shift = ELO_K * margin_weight * (actual - expected_home_win)
        home.elo += shift
        away.elo -= shift

        for team, scored, allowed in ((home, home_score, away_score),
                                      (away, away_score, home_score)):
            team.runs_for += scored
            team.runs_against += allowed
            team.games += 1
            team.recent_for.append(scored)
            team.recent_against.append(allowed)
            team.last_date = date

        for pitcher, allowed in ((home_sp, away_score), (away_sp, home_score)):
            pitcher.runs_allowed += allowed
            pitcher.starts += 1
            pitcher.recent.append(allowed)
            pitcher.last_date = date

        park_states[venue].runs += home_score + away_score
        park_states[venue].games += 1
        league_runs += home_score + away_score
        league_games += 1

    return pd.DataFrame(rows)


def _float(value):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(result) else result


def load_inputs(games_path="data/games.csv", weather_path="data/weather.csv",
                parks_path="data/parks.json"):
    import json
    from pathlib import Path

    games = pd.read_csv(games_path)
    parks = json.loads(Path(parks_path).read_text(encoding="utf-8"))
    weather = (pd.read_csv(weather_path) if Path(weather_path).exists()
               else pd.DataFrame())
    return games, parks, weather


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--weather", default="data/weather.csv")
    parser.add_argument("--out", default="data/features.csv")
    args = parser.parse_args()
    games, parks, weather = load_inputs(args.games, args.weather)
    frame = build(games, parks, weather)
    frame.to_csv(args.out, index=False)
    print(f"wrote {len(frame)} feature rows and {len(FEATURE_COLUMNS)} "
          f"columns to {args.out}")


if __name__ == "__main__":
    main()
