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

from parks import id_key
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


# Bounds on the divisor used to park-adjust a team's run rates. The point-in-
# time park factor is already regressed toward neutral, so it does not reach
# these in practice; they exist so a thin or strange venue cannot turn one
# game into an outlier that a season of games has to absorb.
MIN_PARK_ADJUSTMENT = 0.75
MAX_PARK_ADJUSTMENT = 1.35


class TeamState:
    """Everything known about one team as of the moment before a game.

    Run rates are stored **park-adjusted**: each game's runs are divided by the
    point-in-time park factor of the venue it was played in, so `offense()` is
    a neutral-park rate rather than a record of where the team happened to
    play.

    Raw rates were the obvious thing and were wrong in a way that compounds.
    `expected_home_runs_prior` multiplies the offence rating by the park
    factor, and the model carries `park_factor` as a feature besides — so a
    rating built from raw runs applies the park once inside itself and again
    outside, and the two do not cancel because a team plays only half its games
    at home.

    What this does and does not buy, measured rather than assumed. On a
    walk-forward over the same 11,495 games it improves the total by 0.00030 of
    log loss with a date-clustered interval of [-0.00047, -0.00014], and leaves
    the moneyline and run line inside noise at +0.00011 and +0.00007, both
    spanning zero. So it is kept for the total and for being right, not for
    being large.

    It does **not** fix the case that prompted it. The model over-predicts
    Colorado road scoring by 0.445 runs a game, and park-adjusting the ratings
    moves that to 0.440 — nothing. The double-count is real in the feature, but
    the estimator was already absorbing most of it through its own coefficients
    on `home_off` and `park_factor`, so removing it at source mostly
    redistributes weight rather than changing the fit. Worth writing down: a
    defect being real in the construction does not mean the model was suffering
    from it.

    Nor is the remainder something to chase. The per-team home/road residual
    split has a year-over-year correlation of +0.008 across 119 team-seasons,
    and Colorado's own runs +1.47, +0.17, -0.05, +0.48, -0.39 — in 2026 they
    are scoring *more* on the road than predicted. There is no persistent trait
    there to model, only a large pooled number carried by 2022 and 2025.
    """

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
        """Shrunk park-adjusted runs-scored rate, on a neutral-park basis."""
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
    bullpen that followed him. It is kept because it costs nothing — the
    starter's identity is already on every schedule row — and because it is
    point-in-time by construction. `PitcherComponents` below adds what it
    cannot see.
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


# Batters faced carried by the league prior before a pitcher has a record of
# his own. A starter faces roughly 24 a start, so this is about eight starts:
# enough that a two-start sample cannot dominate, not so much that a
# established starter is dragged to average.
PRIOR_BATTERS_FACED = 200.0
# Outs a bullpen throws in three days before it is meaningfully taxed. Roughly
# two full games' worth of relief.
BULLPEN_WINDOW_DAYS = 3


class PitcherComponents:
    """A starter's own strikeout, walk and home-run rates, and how deep he goes.

    This is the first thing in the feature set that is not a rearrangement of
    runs scored and allowed. Everything else is downstream of the scoreboard,
    which is precisely what the market has already priced; a starter's own
    contact profile is a different input, and it stabilises far faster than
    runs allowed because it strips out the fielding and bullpen behind him.

    Rates are per batter faced rather than per inning, so a pitcher who is
    pulled early is not rewarded for the outs he never recorded.
    """

    def __init__(self):
        self.batters = 0.0
        self.strikeouts = 0.0
        self.walks = 0.0
        self.home_runs = 0.0
        self.outs = 0.0
        self.starts = 0

    def _rate(self, count, league):
        return ((count + league * PRIOR_BATTERS_FACED)
                / (self.batters + PRIOR_BATTERS_FACED))

    def strikeout_rate(self, league):
        return self._rate(self.strikeouts, league)

    def walk_rate(self, league):
        return self._rate(self.walks, league)

    def home_run_rate(self, league):
        return self._rate(self.home_runs, league)

    def depth(self):
        """Average outs recorded per start; league starters sit near 16."""
        if self.starts < 2:
            return 16.0
        return self.outs / self.starts

    def fold(self, line):
        self.batters += _number(line.get("batters_faced"))
        self.strikeouts += _number(line.get("strike_outs"))
        self.walks += (_number(line.get("walks"))
                       + _number(line.get("hit_batsmen")))
        self.home_runs += _number(line.get("home_runs"))
        self.outs += _number(line.get("outs"))
        self.starts += 1


class BullpenState:
    """Relief quality, and how much of it has been used lately.

    Fatigue is the point. A bullpen that threw six innings yesterday and four
    the day before is a different opponent tonight, and it is knowable before
    first pitch, which is more than can be said for most of what moves a
    baseball game.
    """

    def __init__(self):
        self.runs = 0.0
        self.outs = 0.0
        self.recent = deque()

    def rate(self, league):
        """Runs allowed per 27 outs, shrunk toward the league."""
        prior_outs = 300.0
        return ((self.runs + league * prior_outs / 27.0)
                / (self.outs + prior_outs)) * 27.0

    def workload(self, date):
        """Relief outs thrown in the trailing window."""
        cutoff = date - pd.Timedelta(days=BULLPEN_WINDOW_DAYS)
        return float(sum(outs for day, outs in self.recent if day > cutoff))

    def fold(self, date, outs, runs):
        self.runs += runs
        self.outs += outs
        self.recent.append((date, outs))
        cutoff = date - pd.Timedelta(days=BULLPEN_WINDOW_DAYS * 2)
        while self.recent and self.recent[0][0] <= cutoff:
            self.recent.popleft()


# Umpires work about 140 games a season, so this shrinks a first-season
# umpire most of the way to the league. The weight is deliberately heavy: the
# year-over-year correlation of an umpire's own run tendency is -0.09, so his
# record is close to uninformative about his next game and a light prior would
# hand the model noise dressed as a trait.
PRIOR_UMPIRE_GAMES = 60.0


class UmpireState:
    """Runs and strikeouts in games this umpire has worked.

    Raw observables rather than model residuals. A residual-based feature would
    be circular — the model would be handed its own past errors and would learn
    to undo them in sample.
    """

    def __init__(self):
        self.runs = 0.0
        self.games = 0
        self.strikeouts = 0.0
        # Counted separately from games. A game with no boxscore has no
        # strikeout total, and folding that in as a zero would teach the
        # umpire he calls no strikes — which is what happened, and it drifted
        # further from the truth the more games he worked.
        self.strikeout_games = 0

    def run_rate(self, league):
        return ((self.runs + league * PRIOR_UMPIRE_GAMES)
                / (self.games + PRIOR_UMPIRE_GAMES))

    def strikeout_rate(self, league):
        return ((self.strikeouts + league * PRIOR_UMPIRE_GAMES)
                / (self.strikeout_games + PRIOR_UMPIRE_GAMES))

    def fold(self, runs, strikeouts):
        self.runs += runs
        self.games += 1
        if strikeouts is not None:
            self.strikeouts += strikeouts
            self.strikeout_games += 1


# Batted balls carried by the league prior before a pitcher has a record of
# his own. Expected outcomes stabilise far faster than results -- that is the
# whole reason for using them -- so this prior is much lighter than the 200
# batters faced that `PitcherComponents` needs.
PRIOR_BATTED_BALLS = 60.0
# League anchors, used only until a player has a record. Measured from the
# 2021-2026 aggregate rather than looked up.
LEAGUE_XWOBA = 0.310
LEAGUE_WHIFF_RATE = 0.245
LEAGUE_BARREL_RATE = 0.075


class StatcastState:
    """A player's expected outcomes, accumulated from earlier games only.

    The point of expected statistics is that they say what happened rather
    than what was recorded. `estimated_woba_using_speedangle` scores a batted
    ball by its exit velocity and launch angle, so the fielders, the park and
    the luck fall out, and what is left stabilises in dozens of batted balls
    where runs allowed takes a season.

    That matters here specifically because `PitcherState` is a proxy built
    from *team* runs allowed in games this pitcher started -- it includes the
    bullpen that followed him and the defence behind him. This is the same
    quantity with those removed.

    Sums in, rates out, shrunk toward the league. The caller folds a game in
    only after the row for it has been emitted, so nothing here can see its
    own outcome.
    """

    def __init__(self):
        self.xwoba = 0.0
        self.xwoba_denom = 0.0
        self.whiffs = 0.0
        self.swings = 0.0
        self.barrels = 0.0
        self.batted = 0.0

    def expected_woba(self):
        return ((self.xwoba + LEAGUE_XWOBA * PRIOR_BATTED_BALLS)
                / (self.xwoba_denom + PRIOR_BATTED_BALLS))

    def whiff_rate(self):
        return ((self.whiffs + LEAGUE_WHIFF_RATE * PRIOR_BATTED_BALLS)
                / (self.swings + PRIOR_BATTED_BALLS))

    def barrel_rate(self):
        return ((self.barrels + LEAGUE_BARREL_RATE * PRIOR_BATTED_BALLS)
                / (self.batted + PRIOR_BATTED_BALLS))

    def fold(self, row):
        self.xwoba += _number(row.get("xwoba_sum"))
        self.xwoba_denom += _number(row.get("xwoba_denom"))
        self.whiffs += _number(row.get("whiffs"))
        self.swings += _number(row.get("swings"))
        self.barrels += _number(row.get("barrels"))
        self.batted += _number(row.get("batted_balls"))


class LeagueRates:
    """Running league averages, so a prior is measured rather than assumed."""

    def __init__(self):
        self.batters = 0.0
        self.strikeouts = 0.0
        self.walks = 0.0
        self.home_runs = 0.0
        self.relief_outs = 0.0
        self.relief_runs = 0.0
        # Games the strikeout total above actually covers, which is not every
        # game: a game with no boxscore contributes no lines. Dividing by the
        # schedule instead would understate the league exactly in proportion
        # to how much data is missing.
        self.strikeout_games = 0

    def _rate(self, count, fallback):
        return count / self.batters if self.batters > 5000 else fallback

    @property
    def strikeout(self):
        return self._rate(self.strikeouts, 0.225)

    @property
    def walk(self):
        return self._rate(self.walks, 0.090)

    @property
    def home_run(self):
        return self._rate(self.home_runs, 0.033)

    def strikeouts_per_game(self, fallback=16.0):
        if self.strikeout_games < 200:
            return fallback
        return self.strikeouts / self.strikeout_games

    @property
    def relief(self):
        if self.relief_outs < 3000:
            return 4.2
        return self.relief_runs / self.relief_outs * 27.0

    def fold(self, line):
        self.batters += _number(line.get("batters_faced"))
        self.strikeouts += _number(line.get("strike_outs"))
        self.walks += (_number(line.get("walks"))
                       + _number(line.get("hit_batsmen")))
        self.home_runs += _number(line.get("home_runs"))
        if not int(_number(line.get("is_starter"))):
            self.relief_outs += _number(line.get("outs"))
            self.relief_runs += _number(line.get("runs"))


def _number(value):
    result = _float(value)
    return 0.0 if result is None else result


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
    "home_sp_k_rate", "away_sp_k_rate",
    "home_sp_bb_rate", "away_sp_bb_rate",
    "home_sp_hr_rate", "away_sp_hr_rate",
    "home_sp_depth", "away_sp_depth",
    "home_bp_rate", "away_bp_rate",
    "home_bp_workload", "away_bp_workload",
    "park_factor", "elevation_km",
    # ump_run_rate and ump_k_rate are built and written, deliberately not
    # listed here. Added to the model they made it worse on all three markets
    # with intervals excluding zero, and the GLM had already shrunk them to
    # +0.002 and -0.005 against 0.018 for the park factor. That is not a
    # sample-size problem waiting on more seasons: an umpire's run tendency
    # correlates -0.09 with his own next season, so the trait does not persist
    # and more data cannot rescue it. The columns stay in the table because
    # they are cheap and inspectable; re-enabling them is adding two strings,
    # and the reason not to is written down rather than forgotten.
    "temp_c", "air_density_index", "wind_out_to_center_ms",
    "wind_left_to_right_ms", "precip_mm", "roof_retractable", "roof_dome",
    "expected_home_runs_prior", "expected_away_runs_prior",
]


def _rest_days(previous, current):
    if previous is None:
        return 5
    delta = (current - previous).days
    return int(np.clip(delta, 0, 10))


def build(games, parks, weather=None, pitching=None, umpires=None,
          statcast=None):
    """Return a feature frame aligned to ``games``, in chronological order.

    ``games`` must contain final results; unfinished games are kept so the
    same builder can produce a live card, but they contribute nothing to
    state.

    ``pitching`` is optional per-start boxscore lines. Without it the starter
    and bullpen columns fall back to league priors, so a card can still be
    priced on a day the boxscore ingestion has not caught up — the model then
    sees an average pitcher rather than a wrong one.

    ``umpires`` is optional plate-umpire assignments, treated the same way. The
    plate umpire is not known until the morning of a game, so a card priced
    earlier sees a league-average official, which is the honest default rather
    than a guess.
    """
    weather = weather if weather is not None else pd.DataFrame()
    weather_by_game = {}
    if len(weather):
        for row in weather.to_dict("records"):
            weather_by_game[id_key(row["game_pk"])] = row

    # Keyed by (game, player) for starters and by (game, side) for teams, so
    # the walk can look up either without scanning.
    statcast_pitcher, statcast_team = {}, {}
    if statcast is not None and len(statcast):
        for row in statcast.to_dict("records"):
            game = id_key(row.get("game_pk"))
            if row.get("role") == "pitcher":
                statcast_pitcher[(game, id_key(row.get("player_id")))] = row
            elif "is_home" in row:
                side = "home" if _number(row.get("is_home")) else "away"
                bucket = statcast_team.setdefault((game, side), [])
                bucket.append(row)

    umpire_by_game = {}
    if umpires is not None and len(umpires):
        for row in umpires.to_dict("records"):
            umpire_by_game[id_key(row["game_pk"])] = id_key(row["hp_umpire_id"])

    pitching_by_game = {}
    if pitching is not None and len(pitching):
        for row in pitching.to_dict("records"):
            pitching_by_game.setdefault(id_key(row["game_pk"]), []).append(row)

    # Information order is first-pitch time, not calendar date or game id.
    # Sorting by ``official_date, game_pk`` let a later same-day result update
    # state before an earlier game, and could expose the first game of a
    # doubleheader to the second in the wrong direction.
    games = games.copy()
    starts = (games["game_date_utc"] if "game_date_utc" in games
              else pd.Series(pd.NaT, index=games.index))
    games["_information_time"] = pd.to_datetime(
        starts, utc=True, errors="coerce")
    fallback = pd.to_datetime(games["official_date"], utc=True,
                              errors="coerce")
    games["_information_time"] = games["_information_time"].fillna(fallback)
    games = games.sort_values(
        ["_information_time", "game_pk"]).reset_index(drop=True)
    teams = defaultdict(TeamState)
    pitchers = defaultdict(PitcherState)
    components = defaultdict(PitcherComponents)
    bullpens = defaultdict(BullpenState)
    league = LeagueRates()
    officials = defaultdict(UmpireState)
    starter_shape = defaultdict(StatcastState)
    team_shape = defaultdict(StatcastState)
    park_states = defaultdict(ParkState)
    league_runs, league_games = 0.0, 0

    rows = []
    for game in games.to_dict("records"):
        season = game["season"]
        home_id, away_id = id_key(game["home_team_id"]), id_key(game["away_team_id"])
        home, away = teams[home_id], teams[away_id]
        home.roll_season(season)
        away.roll_season(season)

        date = pd.Timestamp(game["official_date"])
        home_sp = pitchers[id_key(game.get("home_sp_id"))]
        away_sp = pitchers[id_key(game.get("away_sp_id"))]
        home_sp_parts = components[id_key(game.get("home_sp_id"))]
        away_sp_parts = components[id_key(game.get("away_sp_id"))]
        home_pen, away_pen = bullpens[home_id], bullpens[away_id]
        venue = id_key(game["venue_id"])
        park = parks.get(venue, {})
        league_mean = (league_runs / league_games) if league_games else PRIOR_RUNS * 2
        park_factor = park_states[venue].factor(league_mean)

        official = officials[umpire_by_game.get(id_key(game["game_pk"]), "")]
        league_ks = league.strikeouts_per_game()

        conditions = weather_by_game.get(id_key(game["game_pk"]), {})
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
            "home_sp_k_rate": home_sp_parts.strikeout_rate(league.strikeout),
            "away_sp_k_rate": away_sp_parts.strikeout_rate(league.strikeout),
            "home_sp_bb_rate": home_sp_parts.walk_rate(league.walk),
            "away_sp_bb_rate": away_sp_parts.walk_rate(league.walk),
            "home_sp_hr_rate": home_sp_parts.home_run_rate(league.home_run),
            "away_sp_hr_rate": away_sp_parts.home_run_rate(league.home_run),
            "home_sp_depth": home_sp_parts.depth(),
            "away_sp_depth": away_sp_parts.depth(),
            "home_bp_rate": home_pen.rate(league.relief),
            "away_bp_rate": away_pen.rate(league.relief),
            "home_bp_workload": home_pen.workload(date),
            "away_bp_workload": away_pen.workload(date),
            "park_factor": park_factor,
            "elevation_km": (park.get("elevation_m") or 0.0) / 1000.0,
            # Expected outcomes for the two starters and the two lineups, from
            # Statcast. These are the same quantities `home_sp_rate` and
            # `home_off` reach for, with the fielders, the park and the luck
            # taken out -- a batted ball is scored by how hard and at what
            # angle it left the bat rather than by whether it found grass.
            "home_sp_xwoba": starter_shape[
                id_key(game.get("home_sp_id"))].expected_woba(),
            "away_sp_xwoba": starter_shape[
                id_key(game.get("away_sp_id"))].expected_woba(),
            "home_sp_whiff": starter_shape[
                id_key(game.get("home_sp_id"))].whiff_rate(),
            "away_sp_whiff": starter_shape[
                id_key(game.get("away_sp_id"))].whiff_rate(),
            "home_off_xwoba": team_shape[home_id].expected_woba(),
            "away_off_xwoba": team_shape[away_id].expected_woba(),
            "home_off_barrel": team_shape[home_id].barrel_rate(),
            "away_off_barrel": team_shape[away_id].barrel_rate(),
            # The plate umpire, as a run and strikeout environment, expressed
            # as a deviation from the league at that moment. The raw rate is
            # dominated by the league trend rather than by the umpire —
            # strikeouts per game rose across these seasons — so an uncentred
            # version would hand the model a clock instead of an official.
            "ump_run_rate": official.run_rate(league_mean) - league_mean,
            "ump_k_rate": official.strikeout_rate(league_ks) - league_ks,
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

        # Divide by the park before folding in, so what accumulates is what
        # the team would have scored at a neutral venue. `park_factor` here is
        # the point-in-time factor emitted with this game's row and computed
        # from earlier games only, so adjusting with it introduces no lookahead.
        adjustment = float(np.clip(park_factor, MIN_PARK_ADJUSTMENT,
                                   MAX_PARK_ADJUSTMENT))
        for team, scored, allowed in ((home, home_score, away_score),
                                      (away, away_score, home_score)):
            team.runs_for += scored / adjustment
            team.runs_against += allowed / adjustment
            team.games += 1
            team.recent_for.append(scored / adjustment)
            team.recent_against.append(allowed / adjustment)
            team.last_date = date

        for pitcher, allowed in ((home_sp, away_score), (away_sp, home_score)):
            pitcher.runs_allowed += allowed
            pitcher.starts += 1
            pitcher.recent.append(allowed)
            pitcher.last_date = date

        # Boxscore lines fold in here with everything else, after the row for
        # this game has already been emitted, so a starter's own line can
        # never inform the game it was thrown in.
        for line in pitching_by_game.get(id_key(game["game_pk"]), ()):
            league.fold(line)
            team = id_key(line.get("team_id"))
            if int(_number(line.get("is_starter"))):
                components[id_key(line.get("player_id"))].fold(line)
            else:
                bullpens[team].fold(date, _number(line.get("outs")),
                                    _number(line.get("runs")))

        lines = pitching_by_game.get(id_key(game["game_pk"]))
        if lines:
            league.strikeout_games += 1
        official.fold(home_score + away_score,
                      sum(_number(line.get("strike_outs")) for line in lines)
                      if lines else None)

        # Folded here with everything else, after this game's row is already
        # written, so a starter's own outing cannot inform the game he threw
        # it in.
        for side, starter in (("home", game.get("home_sp_id")),
                              ("away", game.get("away_sp_id"))):
            entry = statcast_pitcher.get(
                (id_key(game["game_pk"]), id_key(starter)))
            if entry is not None:
                starter_shape[id_key(starter)].fold(entry)
        for side, team in (("home", home_id), ("away", away_id)):
            for entry in statcast_team.get(
                    (id_key(game["game_pk"]), side), ()):
                team_shape[team].fold(entry)

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
                parks_path="data/parks.json",
                pitching_path="data/pitching.csv",
                umpires_path="data/umpires.csv",
                statcast_path="data/statcast_games.csv"):
    import json
    from pathlib import Path

    games = pd.read_csv(games_path)
    parks = json.loads(Path(parks_path).read_text(encoding="utf-8"))
    weather = (pd.read_csv(weather_path) if Path(weather_path).exists()
               else pd.DataFrame())
    pitching = (pd.read_csv(pitching_path) if Path(pitching_path).exists()
                else pd.DataFrame())
    umpires = (pd.read_csv(umpires_path) if Path(umpires_path).exists()
               else pd.DataFrame())
    statcast = (pd.read_csv(statcast_path) if Path(statcast_path).exists()
                else pd.DataFrame())
    return games, parks, weather, pitching, umpires, statcast


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default="data/games.csv")
    parser.add_argument("--weather", default="data/weather.csv")
    parser.add_argument("--pitching", default="data/pitching.csv")
    parser.add_argument("--umpires", default="data/umpires.csv")
    parser.add_argument("--statcast", default="data/statcast_games.csv")
    parser.add_argument("--out", default="data/features.csv")
    args = parser.parse_args()
    games, parks, weather, pitching, umpires, statcast = load_inputs(
        args.games, args.weather, pitching_path=args.pitching,
        umpires_path=args.umpires, statcast_path=args.statcast)
    frame = build(games, parks, weather, pitching, umpires, statcast)
    frame.to_csv(args.out, index=False)
    print(f"wrote {len(frame)} feature rows and {len(FEATURE_COLUMNS)} "
          f"columns to {args.out}")


if __name__ == "__main__":
    main()
