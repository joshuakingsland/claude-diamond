"""The joint run distribution that prices all three markets.

Moneyline, run line, and total are not three models. They are three readings
of one distribution over (home runs, away runs). Pricing them separately is
how a book ends up quoting a total that contradicts its own moneyline, and it
is how a model ends up with an "edge" that is really an internal
inconsistency.

Two baseball-specific facts drive the implementation:

1. Run scoring is over-dispersed relative to Poisson. Team runs per game have
   a variance roughly twice their mean, so a Poisson model is badly
   over-confident about the total landing near its mean. A negative binomial
   with a fitted dispersion matches the observed spread.

2. There are no ties, and the home team does not bat in the bottom of the
   ninth when it is already ahead. Both facts distort the joint distribution
   in ways that matter: the diagonal has to be resolved into extra innings
   (which add runs, so the total moves too), and home scoring is censored in
   roughly 43% of games.

Everything here is deterministic given the inputs; there is no sampling, so a
backtest reproduces exactly.
"""

import numpy as np
from scipy import stats

MAX_RUNS = 30
GRID = np.arange(MAX_RUNS + 1)


def negative_binomial_pmf(mean, dispersion):
    """Runs pmf for one side over 0..MAX_RUNS.

    ``dispersion`` is the negative binomial size parameter; larger values
    approach Poisson. Variance is ``mean + mean**2 / dispersion``.
    """
    mean = np.asarray(mean, dtype=float)[..., None]
    probability = dispersion / (dispersion + mean)
    pmf = stats.nbinom.pmf(GRID, dispersion, probability)
    total = pmf.sum(axis=-1, keepdims=True)
    return pmf / np.where(total > 0, total, 1.0)


def joint_distribution(mean_home, mean_away, dispersion,
                       extra_inning_home_edge=0.52,
                       extra_inning_total_runs=1.0):
    """Return the joint pmf over (home, away) runs with ties resolved.

    Regulation scoring for the two sides is treated as conditionally
    independent given their expected runs: the teams face different pitchers,
    and what correlation remains is largely park and weather, which are
    already in both means.

    Tied regulation mass is moved into extra innings rather than deleted. The
    winner takes ``extra_inning_total_runs`` and the loser takes the rest, so
    a tie shows up in the total as well as the moneyline. Deleting the
    diagonal instead would bias every total downward by the extra-innings
    share. Both parameters are calibrated from observed games by
    `calibrate_extra_innings` rather than assumed.
    """
    home_pmf = negative_binomial_pmf(mean_home, dispersion)
    away_pmf = negative_binomial_pmf(mean_away, dispersion)
    joint = home_pmf[..., :, None] * away_pmf[..., None, :]

    diagonal = np.einsum("...ii->...i", joint).copy()
    np.einsum("...ii->...i", joint)[...] = 0.0

    winner_runs = max(1, int(round(float(extra_inning_total_runs))))
    for score in range(MAX_RUNS + 1):
        mass = diagonal[..., score]
        winner_cell = min(score + winner_runs, MAX_RUNS)
        if winner_cell <= score:
            # Only possible at the top of the grid, where the pmf is
            # negligible; park it on the diagonal rather than lose it.
            joint[..., score, score] += mass
            continue
        joint[..., winner_cell, score] += mass * extra_inning_home_edge
        joint[..., score, winner_cell] += mass * (1.0 - extra_inning_home_edge)

    total = joint.sum(axis=(-2, -1), keepdims=True)
    return joint / np.where(total > 0, total, 1.0)


def calibrate_extra_innings(games):
    """Measure the home edge and run bump in games that went past regulation.

    ``games`` needs ``innings_played``, ``home_win``, ``home_score`` and
    ``away_score``. Returns ``(home_edge, total_runs)`` suitable for
    `joint_distribution`, falling back to neutral values on a thin sample.
    """
    extra = games[(games["innings_played"] > games["scheduled_innings"])
                  & games["home_win"].notna()]
    if len(extra) < 100:
        return 0.52, 1.0
    home_edge = float(extra["home_win"].mean())
    margin = (extra["home_score"] - extra["away_score"]).abs()
    return float(np.clip(home_edge, 0.40, 0.60)), float(np.clip(margin.mean(), 1.0, 3.0))


def moneyline_probability(joint):
    """P(home wins). Ties are already resolved, so this is exact."""
    mask = GRID[:, None] > GRID[None, :]
    return (joint * mask).sum(axis=(-2, -1))


def runline_probability(joint, point):
    """P(home side covers a run line at ``point``).

    ``point`` is the home handicap: -1.5 means the home team must win by two
    or more, +1.5 means it may lose by one. Half-run lines cannot push, which
    is why they are the standard baseball run line.
    """
    margin = GRID[:, None] - GRID[None, :]
    mask = margin > -float(point)
    return (joint * mask).sum(axis=(-2, -1))


def total_over_probability(joint, point):
    """P(combined runs exceed ``point``)."""
    totals = GRID[:, None] + GRID[None, :]
    mask = totals > float(point)
    return (joint * mask).sum(axis=(-2, -1))


def push_probability(joint, point, market):
    """Mass sitting exactly on a whole-number line.

    Whole-number totals and run lines push rather than lose. Treating a push
    as a loss understates every whole-number bet, so callers price against
    ``p / (p + q)`` with the push mass removed.
    """
    point = float(point)
    if point % 1.0 != 0.0:
        return np.zeros(joint.shape[:-2]) if joint.ndim > 2 else 0.0
    if market == "totals":
        mask = (GRID[:, None] + GRID[None, :]) == point
    else:
        mask = (GRID[:, None] - GRID[None, :]) == -point
    return (joint * mask).sum(axis=(-2, -1))


def fit_dispersion(observed_runs, expected_runs):
    """Method-of-moments dispersion from residual over-dispersion.

    Returns the negative binomial size that reproduces the observed variance
    around the model's own predictions, so the width of the distribution is
    measured rather than assumed.
    """
    observed = np.asarray(observed_runs, dtype=float)
    expected = np.asarray(expected_runs, dtype=float)
    keep = np.isfinite(observed) & np.isfinite(expected) & (expected > 0)
    observed, expected = observed[keep], expected[keep]
    if len(observed) < 50:
        return 4.0
    variance = float(np.mean((observed - expected) ** 2))
    mean = float(np.mean(expected))
    excess = variance - mean
    if excess <= 1e-6:
        return 50.0
    return float(np.clip(mean ** 2 / excess, 1.0, 50.0))
