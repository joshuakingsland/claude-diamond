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
from runs import (calibrate_extra_innings, fit_dispersion, joint_distribution,
                  moneyline_probability, push_probability,
                  runline_probability, total_over_probability)

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
        self.dispersion = 4.0
        self.extra_home_edge = 0.52
        self.extra_total_runs = 1.0

    def fit(self, features, games):
        """Train on every game twice: once per dugout."""
        merged = features.merge(
            games[["game_pk", "home_score", "away_score", "home_win",
                   "innings_played", "scheduled_innings"]],
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

        # Dispersion MUST be measured out of sample. Fitting it on training
        # predictions was the single worst bug in this model: a gradient
        # boosted estimator's in-sample residuals are artificially small, so
        # the method-of-moments fit concluded there was no over-dispersion at
        # all, pinned the size parameter at its ceiling, and produced a run
        # distribution far too tight. Win probabilities then ran from 0.05 to
        # 0.97 on baseball games and the moneyline scored worse than a
        # constant. The estimator was fine; the width was measured wrong.
        self.dispersion = self._holdout_dispersion(merged)
        self.extra_home_edge, self.extra_total_runs = calibrate_extra_innings(merged)
        return self

    def _holdout_dispersion(self, merged, folds=4):
        """Method-of-moments dispersion from out-of-fold residuals only."""
        merged = merged.sort_values("official_date").reset_index(drop=True)
        cut = int(len(merged) * (folds - 1) / folds)
        if cut < 200 or len(merged) - cut < 200:
            return 4.0
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
        held_target = np.concatenate([
            held["home_score"].to_numpy(dtype=float),
            held["away_score"].to_numpy(dtype=float),
        ])
        return fit_dispersion(held_target, probe._raw(held_design))

    def _raw(self, design):
        if self.kind == "glm":
            return self.estimator.predict(self.scaler.transform(design))
        return self.estimator.predict(design)

    def expected_runs(self, features):
        """Return ``(home_mean, away_mean)`` for home-oriented feature rows."""
        home = self._raw(_design(features))
        away = self._raw(_design(mirror(features)))
        return np.clip(home, 0.2, 20.0), np.clip(away, 0.2, 20.0)

    def price(self, features, runline_points=(-1.5,), total_points=(8.5,)):
        """Return a probability frame for every market and line requested."""
        home_mean, away_mean = self.expected_runs(features)
        joint = joint_distribution(
            home_mean, away_mean, self.dispersion,
            extra_inning_home_edge=self.extra_home_edge,
            extra_inning_total_runs=self.extra_total_runs,
        )
        out = pd.DataFrame({
            "game_pk": features["game_pk"].to_numpy(),
            "expected_home_runs": home_mean,
            "expected_away_runs": away_mean,
            "p_home_ml": moneyline_probability(joint),
        })
        for point in runline_points:
            out[f"p_home_rl_{point}"] = runline_probability(joint, point)
        for point in total_points:
            out[f"p_over_{point}"] = total_over_probability(joint, point)
            push = push_probability(joint, point, "totals")
            if np.any(push > 0):
                out[f"push_over_{point}"] = push
        return out


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
        priced = model.price(test_features)
        priced["season"] = season
        priced["dispersion"] = model.dispersion
        predictions.append(priced)
        if verbose:
            print(f"  {season}: trained on {len(train_features)} games, "
                  f"predicted {len(test_features)}, dispersion "
                  f"{model.dispersion:.2f}")
    if not predictions:
        return pd.DataFrame()
    return pd.concat(predictions, ignore_index=True)
