"""Expected-runs estimators and the pricing layer built on top of them.

Two regressors predict expected runs for the home and away sides. Everything
a bettor actually wants — moneyline, run line, total — is then read off the
joint distribution those two means imply, so the three prices are guaranteed
consistent with one another by construction.

Poisson loss is used rather than squared error because runs are counts with
variance that grows with the mean; squared error would fit the high-scoring
tail at the expense of the 3-2 games that make up most of the schedule.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS
from runs import (DEFAULT_EXTRA_INNING_MARGINS, DEFAULT_INNING_SHAPE,
                  DEFAULT_WALK_OFF_MARGINS,
                  calibrate_extra_innings,
                  calibrate_walk_off,
                  fit_inning_shape, joint_distribution,
                  uncensor_home_mean,
                  moneyline_probability, push_probability,
                  runline_probability, total_over_probability)
from provenance import feature_schema, model_version, repository_revision

# The away side sees the same game from the other dugout. Swapping these
# pairs turns a home-oriented row into an away-oriented one, so one estimator
# can be trained on twice the rows and cannot learn a spurious home/away
# asymmetry beyond what the explicit home-field terms carry.
MIRROR_PAIRS = [
    ("home_elo", "away_elo"),
    ("home_off", "away_off"),
    ("home_def", "away_def"),
    ("home_recent_off", "away_recent_off"),
    ("home_recent_def", "away_recent_def"),
    ("home_rest", "away_rest"),
    ("home_games_played", "away_games_played"),
    ("home_sp_rate", "away_sp_rate"),
    ("home_sp_recent", "away_sp_recent"),
    ("home_sp_starts", "away_sp_starts"),
    ("home_sp_rest", "away_sp_rest"),
    ("home_sp_k_rate", "away_sp_k_rate"),
    ("home_sp_bb_rate", "away_sp_bb_rate"),
    ("home_sp_hr_rate", "away_sp_hr_rate"),
    ("home_sp_depth", "away_sp_depth"),
    ("home_bp_rate", "away_bp_rate"),
    ("home_bp_workload", "away_bp_workload"),
    ("expected_home_runs_prior", "expected_away_runs_prior"),
]
NEGATED = ("elo_diff", "rest_diff", "wind_left_to_right_ms")


def mirror(frame):
    """Return the same games seen from the away dugout."""
    flipped = frame.copy()
    for left, right in MIRROR_PAIRS:
        flipped[left], flipped[right] = frame[right].to_numpy(), frame[left].to_numpy()
    for column in NEGATED:
        if column in flipped:
            flipped[column] = -frame[column].to_numpy()
    return flipped


def _design(frame):
    return frame.reindex(columns=FEATURE_COLUMNS).astype(float).fillna(0.0)


class RunsModel:
    """Predicts expected runs for the batting side of a home-oriented row."""

    def __init__(self, kind="gbm", seed=0):
        self.kind = kind
        self.seed = seed
        self.scaler = None
        self.estimator = None
        self.dispersion = (4.0, 4.0)
        self.shape = DEFAULT_INNING_SHAPE
        self.distribution_family = "inning-zero-inflated-nb-v1"
        self.extra_home_edge = 0.52
        self.extra_margins = DEFAULT_EXTRA_INNING_MARGINS
        self.walk_off_margins = DEFAULT_WALK_OFF_MARGINS

    def fit(self, features, games):
        """Train on every game twice: once per dugout."""
        merged = features.merge(
            games[["game_pk", "home_score", "away_score", "home_win",
                   "innings_played", "scheduled_innings",
                   "home_batted_ninth"]],
            on="game_pk", how="inner",
        )
        merged = merged[merged["home_score"].notna()]
        home_view, away_view = merged, mirror(merged)
        design = pd.concat([_design(home_view), _design(away_view)], axis=0)
        target = np.concatenate([
            merged["home_score"].to_numpy(dtype=float),
            merged["away_score"].to_numpy(dtype=float),
        ])
        if self.kind == "glm":
            self.scaler = StandardScaler().fit(design)
            self.estimator = PoissonRegressor(alpha=1e-3, max_iter=2000)
            self.estimator.fit(self.scaler.transform(design), target)
        else:
            self.estimator = HistGradientBoostingRegressor(
                loss="poisson", max_iter=400, learning_rate=0.05,
                max_leaf_nodes=31, min_samples_leaf=60,
                l2_regularization=1.0, random_state=self.seed,
            )
            self.estimator.fit(design, target)

        # Distribution width must be measured out of sample. The active
        # inning family is fitted on a held-out block; its scoreless share and
        # tail determine width. A former game-level dispersion estimate was
        # dead whenever this family was active and has been removed from the
        # serving fit rather than displayed as false precision.
        self.shape = self._holdout_shape(merged)
        self.extra_home_edge, self.extra_margins = calibrate_extra_innings(merged)
        # Measured from the games rather than assumed, like the extra-innings
        # parameters: a walk-off ends on the go-ahead run most of the time but
        # not always, and the run line is decided at exactly that boundary.
        self.walk_off_margins = calibrate_walk_off(merged)
        return self

    def _holdout_shape(self, merged, folds=4):
        """Fit the active inning-family shape on held-out predictions.

        The earlier implementation also fitted a game-level negative-binomial
        dispersion and displayed it as though pricing used it.  Once the
        inning family is selected, ``run_pieces`` intentionally ignores that
        parameter; tuning and reporting a dead knob made experiments look more
        identified than they were.  The NB path remains a reversible research
        family in ``runs.py``, but the serving model has one active width
        contract: scoreless share plus inning tail.
        """
        merged = merged.sort_values("official_date").reset_index(drop=True)
        cut = int(len(merged) * (folds - 1) / folds)
        if cut < 200 or len(merged) - cut < 200:
            return DEFAULT_INNING_SHAPE
        train, held = merged.iloc[:cut], merged.iloc[cut:]
        probe = RunsModel(kind=self.kind, seed=self.seed + 1)
        probe.scaler, probe.estimator = None, None
        design = pd.concat([_design(train), _design(mirror(train))], axis=0)
        target = np.concatenate([
            train["home_score"].to_numpy(dtype=float),
            train["away_score"].to_numpy(dtype=float),
        ])
        if self.kind == "glm":
            probe.scaler = StandardScaler().fit(design)
            probe.estimator = PoissonRegressor(alpha=1e-3, max_iter=2000)
            probe.estimator.fit(probe.scaler.transform(design), target)
        else:
            probe.estimator = HistGradientBoostingRegressor(
                loss="poisson", max_iter=400, learning_rate=0.05,
                max_leaf_nodes=31, min_samples_leaf=60,
                l2_regularization=1.0, random_state=self.seed + 1,
            )
            probe.estimator.fit(design, target)
        held_design = pd.concat([_design(held), _design(mirror(held))], axis=0)
        predicted = probe._raw(held_design)
        cut = len(held)
        # Away side only. It bats nine innings whatever the score, so its
        # observed spread is run scoring and nothing else; the home side's is
        # run scoring plus the censoring the shape is about to model, and
        # fitting on it would count those innings twice.
        shape = fit_inning_shape(held["away_score"].to_numpy(dtype=float),
                                 predicted[cut:])
        return shape

    def _joint(self, home_mean, away_mean, scheduled):
        """Price each scheduled length separately, then reassemble in order.

        Almost always one group. The loop exists because a seven-inning game
        has a different ninth — there is not one — and a different run
        expectation.
        """
        joint = None
        for length in np.unique(scheduled):
            rows = np.flatnonzero(scheduled == length)
            share = length / 9.0
            # The estimator predicts a nine-inning game; a shorter one scores
            # proportionally less.
            home = home_mean[rows] * share
            away = away_mean[rows] * share
            # The estimator learned a censored home mean, because observed
            # home scores are censored. Recover the full-length mean so the
            # distribution does not truncate the same innings twice.
            home = uncensor_home_mean(
                home, away, self.dispersion, innings=int(length),
                walk_off_margins=self.walk_off_margins,
                extra_inning_home_edge=self.extra_home_edge,
                extra_inning_margins=self.extra_margins,
                shape=self.shape)
            block = joint_distribution(
                home, away, self.dispersion, innings=int(length),
                walk_off_margins=self.walk_off_margins,
                extra_inning_home_edge=self.extra_home_edge,
                extra_inning_margins=self.extra_margins,
                shape=self.shape,
            )
            if joint is None:
                joint = np.empty((len(scheduled),) + block.shape[1:])
            joint[rows] = block
        return joint

    def _raw(self, design):
        if self.kind == "glm":
            return self.estimator.predict(self.scaler.transform(design))
        return self.estimator.predict(design)

    def expected_runs(self, features):
        """Return ``(home_mean, away_mean)`` for home-oriented feature rows."""
        home = self._raw(_design(features))
        away = self._raw(_design(mirror(features)))
        return np.clip(home, 0.2, 20.0), np.clip(away, 0.2, 20.0)

    def price(self, features, runline_points=(-1.5,), total_points=(8.5,),
              innings=None):
        """Return a probability frame for every market and line requested.

        ``innings`` is the scheduled length per game, nine unless a
        doubleheader rule shortened it. 121 games in this dataset were seven
        innings and were being priced as nine, which inflates both sides'
        expected runs by a fifth.
        """
        home_mean, away_mean = self.expected_runs(features)
        scheduled = (np.full(len(features), 9.0) if innings is None
                     else np.asarray(innings, dtype=float))
        scheduled = np.where(np.isfinite(scheduled) & (scheduled > 0),
                             scheduled, 9.0)
        joint = self._joint(home_mean, away_mean, scheduled)
        # Report what was priced. A seven-inning game expects proportionally
        # fewer runs, and showing the nine-inning number beside a total priced
        # for seven would contradict the card it appears on.
        share = scheduled / 9.0
        home_mean, away_mean = home_mean * share, away_mean * share
        out = pd.DataFrame({
            "game_pk": features["game_pk"].to_numpy(),
            "expected_home_runs": home_mean,
            "expected_away_runs": away_mean,
            "p_home_ml": moneyline_probability(joint),
            "scheduled_innings": scheduled,
            "distribution_family": self.distribution_family,
            "dispersion_active": 0,
            "dispersion_home": self.dispersion[0],
            "dispersion_away": self.dispersion[1],
            "inning_scoreless": self.shape[0],
            "inning_tail": self.shape[1],
            "extra_home_edge": self.extra_home_edge,
        })
        for index, value in enumerate(self.extra_margins, 1):
            out[f"extra_margin_{index}"] = value
        for index, value in enumerate(self.walk_off_margins, 1):
            out[f"walk_off_margin_{index}"] = value
        for point in runline_points:
            out[f"p_home_rl_{point}"] = runline_probability(joint, point)
            # Alternate run lines at whole numbers push, exactly as whole
            # totals do. Only +-1.5 is quoted on most cards, so this stayed
            # invisible until the live board offered a -1 and a +2.
            push = push_probability(joint, point, "spreads")
            if np.any(push > 0):
                out[f"push_home_rl_{point}"] = push
        for point in total_points:
            out[f"p_over_{point}"] = total_over_probability(joint, point)
            push = push_probability(joint, point, "totals")
            if np.any(push > 0):
                out[f"push_over_{point}"] = push
        return out


def reprice_requests(requests, predictions):
    """Price arbitrary market points from stored walk-forward distributions.

    Historical validation used to score only home -1.5 and total 8.5 because
    those were the two columns written by ``walk_forward``.  The live path
    priced every point offered by books, so the executed universe was never
    the validated one.  Distribution provenance is now stored beside every
    prediction and this function reconstructs the exact joint distribution at
    whatever main point was actually quoted.
    """
    if not len(requests):
        return requests.assign(model_prob_home=pd.Series(dtype=float),
                               model_push_prob=pd.Series(dtype=float))
    frame = requests.copy().reset_index(drop=True)
    frame["_request_order"] = np.arange(len(frame))
    joined = frame.merge(predictions, on="game_pk", how="left",
                         suffixes=("", "_prediction"))
    required = ("expected_home_runs", "expected_away_runs")
    if any(column not in joined for column in required):
        raise ValueError("predictions do not contain expected-run means")

    defaults = {
        "scheduled_innings": 9.0,
        "dispersion_home": 4.0,
        "dispersion_away": 4.0,
        "inning_scoreless": DEFAULT_INNING_SHAPE[0],
        "inning_tail": DEFAULT_INNING_SHAPE[1],
        "extra_home_edge": 0.52,
    }
    for column, value in defaults.items():
        if column not in joined:
            joined[column] = value
        joined[column] = joined[column].fillna(value)
    for index, value in enumerate(DEFAULT_EXTRA_INNING_MARGINS, 1):
        column = f"extra_margin_{index}"
        if column not in joined:
            joined[column] = value
        joined[column] = joined[column].fillna(value)
    for index, value in enumerate(DEFAULT_WALK_OFF_MARGINS, 1):
        column = f"walk_off_margin_{index}"
        if column not in joined:
            joined[column] = value
        joined[column] = joined[column].fillna(value)

    probabilities = np.full(len(joined), np.nan)
    pushes = np.full(len(joined), np.nan)
    parameter_columns = [
        "scheduled_innings", "dispersion_home", "dispersion_away",
        "inning_scoreless", "inning_tail", "extra_home_edge",
        *[f"extra_margin_{index}" for index in range(1, 5)],
        *[f"walk_off_margin_{index}" for index in range(1, 5)],
        "market", "point",
    ]
    grouped = joined.groupby(parameter_columns, dropna=False, sort=False)
    for _, block in grouped:
        positions = block.index.to_numpy()
        first = block.iloc[0]
        scheduled = int(float(first["scheduled_innings"]))
        dispersion = (float(first["dispersion_home"]),
                      float(first["dispersion_away"]))
        shape = (float(first["inning_scoreless"]),
                 float(first["inning_tail"]))
        extra_margins = tuple(float(first[f"extra_margin_{index}"])
                              for index in range(1, 5))
        walk_off = tuple(float(first[f"walk_off_margin_{index}"])
                         for index in range(1, 5))
        home_target = block["expected_home_runs"].to_numpy(float)
        away_mean = block["expected_away_runs"].to_numpy(float)
        home_mean = uncensor_home_mean(
            home_target, away_mean, dispersion, innings=scheduled,
            walk_off_margins=walk_off,
            extra_inning_home_edge=float(first["extra_home_edge"]),
            extra_inning_margins=extra_margins, shape=shape,
        )
        joint = joint_distribution(
            home_mean, away_mean, dispersion, innings=scheduled,
            walk_off_margins=walk_off,
            extra_inning_home_edge=float(first["extra_home_edge"]),
            extra_inning_margins=extra_margins, shape=shape,
        )
        market = first["market"]
        point = first["point"]
        if market == "h2h":
            probability = moneyline_probability(joint)
            push = np.zeros(len(block))
        elif market == "spreads":
            probability = runline_probability(joint, float(point))
            push = push_probability(joint, float(point), "spreads")
        elif market == "totals":
            probability = total_over_probability(joint, float(point))
            push = push_probability(joint, float(point), "totals")
        else:
            continue
        remaining = 1.0 - np.asarray(push, float)
        probability = np.where(remaining > 0, probability / remaining,
                               np.nan)
        probabilities[positions] = probability
        pushes[positions] = push

    joined["model_prob_home"] = probabilities
    joined["model_push_prob"] = pushes
    return (joined.sort_values("_request_order")
            [list(requests.columns) + ["model_prob_home", "model_push_prob"]]
            .reset_index(drop=True))


def walk_forward(features, games, seasons, kind="gbm", min_train_games=1500,
                 verbose=True):
    """Train on everything before each season, predict that season.

    Retraining per season rather than per game keeps the backtest honest
    without pretending a model would be refit nightly. Every prediction for
    season S comes from a model that has seen only seasons before S.
    """
    features = features.sort_values("official_date")
    predictions = []
    for season in seasons:
        train_features = features[features["season"] < season]
        if len(train_features) < min_train_games:
            if verbose:
                print(f"  {season}: skipped, only {len(train_features)} training games")
            continue
        test_features = features[features["season"] == season]
        if not len(test_features):
            continue
        model = RunsModel(kind=kind).fit(train_features, games)
        lengths = (test_features[["game_pk"]]
                   .merge(games[["game_pk", "scheduled_innings"]],
                          on="game_pk", how="left")["scheduled_innings"])
        priced = model.price(test_features, innings=lengths.to_numpy())
        priced["season"] = season
        revision = repository_revision()
        priced["model_kind"] = kind
        priced["model_revision"] = revision
        priced["feature_schema"] = feature_schema(FEATURE_COLUMNS)
        priced["model_version"] = model_version(kind, FEATURE_COLUMNS,
                                                revision=revision)
        trained_dates = pd.to_datetime(train_features["official_date"],
                                       errors="coerce")
        priced["trained_through"] = (
            str(trained_dates.max().date()) if trained_dates.notna().any()
            else "")
        predictions.append(priced)
        if verbose:
            print(f"  {season}: trained on {len(train_features)} games, "
                  f"predicted {len(test_features)}, distribution "
                  f"{model.distribution_family}, scoreless innings "
                  f"{model.shape[0]:.3f}, tail {model.shape[1]:.3f}")
    if not predictions:
        return pd.DataFrame()
    return pd.concat(predictions, ignore_index=True)
