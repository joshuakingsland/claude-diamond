"""The joint run distribution that prices all three markets.

Moneyline, run line, and total are not three models. They are three readings
of one distribution over (home runs, away runs). Pricing them separately is
how a book ends up quoting a total that contradicts its own moneyline, and it
is how a model ends up with an "edge" that is really an internal
inconsistency.

Three baseball-specific facts drive the implementation:

1. Run scoring is over-dispersed relative to Poisson. Team runs per game have
   a variance roughly twice their mean, so a Poisson model is badly
   over-confident about the total landing near its mean. A negative binomial
   with a fitted dispersion matches the observed spread.

2. There are no ties, and the home team does not bat in the bottom of the
   ninth when it is already ahead. Both facts distort the joint distribution
   in ways that matter: the diagonal has to be resolved into extra innings
   (which add runs, so the total moves too), and home scoring is censored in
   roughly 43% of games.

3. The negative binomial gets the width right and the *shape* wrong. Matching
   the mean and the variance uses up both its parameters, and what is left
   over is measurably off: across 11,428 games it puts 22% too little mass on
   an away shutout and 28% too little on a home one, in the same direction at
   every level of the predicted mean. So the family is wrong rather than the
   means being badly spread. The unit where that is fixable is the inning,
   because an inning is mostly a zero — see `inning_pmf`.

The two families meet at `run_pieces`, which answers the same three questions
for either. Everything downstream of it — censoring, walk-offs, ties, extra
innings — is a fact about baseball rather than about a distribution, and does
not know which family it was handed.

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


def _pair(dispersion):
    """Accept one dispersion or a (home, away) pair."""
    if np.isscalar(dispersion):
        return float(dispersion), float(dispersion)
    home, away = dispersion
    return float(home), float(away)


def _convolve(left, right):
    """Distribution of the sum of two independent counts, on the grid.

    Truncated at MAX_RUNS rather than widened. A team scoring more than thirty
    is not a case this repository needs to price, and the mass out there is
    below 1e-9.
    """
    out = np.zeros_like(left)
    for runs in range(MAX_RUNS + 1):
        out[..., runs:] += (left[..., runs][..., None]
                            * right[..., :MAX_RUNS + 1 - runs])
    return out


def _self_convolve(pmf, times):
    """``times``-fold convolution of ``pmf`` with itself, by repeated squaring."""
    result = np.zeros_like(pmf)
    result[..., 0] = 1.0
    base = pmf.copy()
    while times:
        if times & 1:
            result = _convolve(result, base)
        times >>= 1
        if times:
            base = _convolve(base, base)
    return result


def inning_pmf(mean, scoreless, tail):
    """Runs in one half-inning: scoreless, or a zero-truncated negative binomial.

    The reason this exists. A game-level negative binomial matches the mean and
    the variance *by construction* and has no freedom left over for the shape,
    and the shape is measurably wrong: across 11,428 games it puts 22% too
    little mass on an away shutout and 28% too little on a home one, and it is
    wrong in the same direction at every level of the predicted mean, so it is
    the family that is wrong rather than the spread of the means.

    An inning is the unit where that is fixable, because an inning is mostly a
    zero — about 74.7% of them score nothing. P(shutout) is then roughly
    ``scoreless ** 9``, which is enormously sensitive to a quantity the game
    level cannot see. ``scoreless`` is pinned by the observed shutout rate and
    ``tail`` by the observed variance, so both parameters answer a question the
    data asks rather than being tuned.

    The conditional mean is not free: given the inning mean, the mean given
    that the inning scored is fixed at ``mean / (1 - scoreless)``. That is what
    keeps this a reshaping of the distribution rather than a way to smuggle in
    a different expected-runs estimate.
    """
    mean = np.asarray(mean, dtype=float)
    # ``scoreless`` broadcasts against ``mean`` so a whole grid of candidate
    # shapes can be evaluated in one pass; the fit is otherwise dominated by
    # per-call overhead on arrays of 31 numbers.
    scoreless = np.clip(np.asarray(scoreless, dtype=float), 0.0, 0.999)
    conditional = np.maximum(mean, 1e-9) / (1.0 - scoreless)
    # The zero-truncated NB's mean exceeds its parent's, so solve back for the
    # parent that lands on the conditional mean. Contraction; converges flat.
    parent = conditional.copy()
    for _ in range(80):
        parent = conditional * (1.0 - (tail / (tail + parent)) ** tail)
    pmf = stats.nbinom.pmf(GRID, tail, (tail / (tail + parent))[..., None])
    pmf[..., 0] = 0.0
    total = pmf.sum(axis=-1, keepdims=True)
    pmf = pmf / np.where(total > 0, total, 1.0) * (1.0 - scoreless)[..., None]
    pmf[..., 0] = np.broadcast_to(scoreless, conditional.shape)
    return pmf


def run_pieces(mean, dispersion, shape=None, innings=9):
    """``(full game, all but the last inning, the last inning)`` for one side.

    Both families answer the same three questions, which is what lets the
    censoring and extra-innings machinery below stay ignorant of which one it
    was handed.

    With ``shape``, the split is exact by construction: the early piece really
    is eight innings convolved. Without it, the negative binomial's own
    splitting property is used — ``NB(mu, d)`` shares its ``p`` with
    ``NB(8mu/9, 8d/9)`` and ``NB(mu/9, d/9)`` — which is exact for that family
    but has to assume the ninth carries exactly a ninth of the mean.
    """
    mean = np.asarray(mean, dtype=float)
    if shape is None:
        fraction = (innings - 1) / innings
        return (negative_binomial_pmf(mean, dispersion),
                negative_binomial_pmf(mean * fraction, dispersion * fraction),
                negative_binomial_pmf(mean * (1 - fraction),
                                      dispersion * (1 - fraction)))
    scoreless, tail = shape
    last = inning_pmf(mean / innings, scoreless, tail)
    early = _self_convolve(last, innings - 1)
    early = early / early.sum(axis=-1, keepdims=True)
    full = _convolve(early, last)
    full = full / full.sum(axis=-1, keepdims=True)
    return full, early, last


# A walk-off usually ends on the go-ahead run, but not always: a home run with
# runners aboard wins by more. Measured across 1,206 games in which the home
# side batted last and won, 87% win by exactly one. Collapsing all of it onto
# one run overstates that margin and, worse, strips the mass out of the run
# line, which is decided at exactly this boundary.
DEFAULT_WALK_OFF_MARGINS = (0.872, 0.075, 0.037, 0.016)
MAX_WALK_OFF_MARGIN = len(DEFAULT_WALK_OFF_MARGINS)

# Extra innings end by one run 69% of the time. The mean margin is 1.58, which
# rounds to two, and rounding it was the bug: every one of the 8.8% of games
# that go past regulation was being resolved two runs apart, on the exact
# boundary the run line is decided at. Measured over 1,214 games.
DEFAULT_EXTRA_INNING_MARGINS = (0.688, 0.159, 0.086, 0.067)
MAX_EXTRA_INNING_MARGIN = len(DEFAULT_EXTRA_INNING_MARGINS)


def _margin_shares(margin, buckets):
    """Share of wins by 1, 2, ... runs, with the last bucket absorbing the tail."""
    counts = np.zeros(buckets)
    for index in range(buckets):
        runs = index + 1
        counts[index] = float((margin == runs).sum() if runs < buckets
                              else (margin >= runs).sum())
    total = counts.sum()
    return None if total <= 0 else tuple(counts / total)


def calibrate_walk_off(games):
    """Measure the winning margin when the home side bats last and wins.

    ``games`` needs ``home_batted_ninth``, ``home_score`` and ``away_score``.
    Falls back to the measured defaults on a thin sample rather than inventing
    a shape.
    """
    if "home_batted_ninth" not in games:
        return DEFAULT_WALK_OFF_MARGINS
    walked = games[(games["home_batted_ninth"] == 1)
                   & (games["home_score"] > games["away_score"])]
    if len(walked) < 200:
        return DEFAULT_WALK_OFF_MARGINS
    margin = (walked["home_score"] - walked["away_score"]).astype(int)
    shares = _margin_shares(margin, MAX_WALK_OFF_MARGIN)
    return shares or DEFAULT_WALK_OFF_MARGINS


def _censored_home(home_pieces, away_pmf,
                   walk_off_margins=DEFAULT_WALK_OFF_MARGINS):
    """Joint pmf where the home ninth inning happens only if it is needed.

    The home team bats the bottom of the ninth only when it is not already
    ahead, and stops the moment it goes ahead. The model previously gave both
    sides a full nine innings, and the error was visible in exactly the place
    it should be: across 11,428 games it put 14.2% on the home side losing by
    one against 11.1% observed, and 15.2% on winning by one against 17.1%. The
    run line sits on that boundary, which is why it was the worst calibrated
    of the three markets.

    ``home_pieces`` is ``(full, all but the last inning, the last inning)``
    from `run_pieces`. Which family produced them does not matter here, and
    that is deliberate: the censoring rule is a fact about baseball, not about
    the distribution, so it should not have to be reimplemented per family.

    Given the away side finished on ``a``:

    - home led after eight and did not bat: final is ``h8`` for ``h8 > a``
    - home batted and did not pass ``a``: final is the full nine-inning pmf
    - home batted and passed ``a``: the game ends on the winning run, so the
      path lands within a run or so of ``a``, spread by the measured walk-off
      margins rather than collapsed onto ``a + 1``
    """
    home_pmf, eight, ninth = home_pieces
    # survival[k] = P(ninth-inning runs >= k), with a trailing zero so an
    # index one past the grid is defined.
    survival = np.concatenate(
        [np.cumsum(ninth[..., ::-1], axis=-1)[..., ::-1],
         np.zeros(ninth.shape[:-1] + (1,))], axis=-1)

    walk_off = np.zeros_like(eight)
    for scored in range(MAX_RUNS + 1):
        reach = np.arange(scored, MAX_RUNS + 1)
        walk_off[..., reach] += (eight[..., scored, None]
                                 * survival[..., reach - scored + 1])

    home_axis = GRID[:, None]
    away_axis = GRID[None, :]
    joint = np.where(home_axis <= away_axis,
                     home_pmf[..., :, None], eight[..., :, None])
    joint = np.broadcast_to(joint, joint.shape[:-2] + (MAX_RUNS + 1,
                                                       MAX_RUNS + 1)).copy()
    for away_runs in range(MAX_RUNS + 1):
        for index, share in enumerate(walk_off_margins):
            landing = away_runs + index + 1
            if landing > MAX_RUNS:
                # Past the top of the grid the mass is negligible; keep it on
                # the diagonal rather than lose it, where the tie resolution
                # will deal with it.
                joint[..., MAX_RUNS, MAX_RUNS] += walk_off[..., away_runs] * share
                continue
            joint[..., landing, away_runs] += walk_off[..., away_runs] * share
    return joint * away_pmf[..., None, :]


def uncensor_home_mean(target, mean_away, dispersion, innings=9,
                       walk_off_margins=DEFAULT_WALK_OFF_MARGINS,
                       extra_inning_home_edge=0.52,
                       extra_inning_margins=DEFAULT_EXTRA_INNING_MARGINS,
                       shape=None, rounds=6):
    """The full-nine mean whose priced distribution has expectation ``target``.

    The estimator is trained on observed home scores, and those are already
    censored — the home side did not bat the ninth in roughly 43% of them. Its
    output is therefore a censored mean, and feeding it into a distribution
    that censors again truncates twice. Measured over 11,428 games that put the
    implied home mean at 4.31 against 4.45 actual: the home side biased 0.14
    runs low, on the market that matters most.

    The target is the expectation of the *finished* distribution, extra
    innings included, because that is what the estimator was fitted to. A
    fixed point on the ratio converges in a few rounds and each round is one
    build, where bisecting to the same precision would be many.

    A single global inflation factor would not do: censoring bites harder on a
    home favourite, which leads after eight more often. The factor runs from
    about 1.04 for an underdog to 1.07 for a favourite.
    """
    target = np.asarray(target, dtype=float)
    mean_home = target.copy()
    for _ in range(rounds):
        joint = joint_distribution(
            mean_home, mean_away, dispersion, shape=shape,
            extra_inning_home_edge=extra_inning_home_edge,
            extra_inning_margins=extra_inning_margins,
            innings=innings, walk_off_margins=walk_off_margins)
        implied = (joint * GRID[:, None]).sum(axis=(-2, -1))
        mean_home = mean_home * target / np.maximum(implied, 1e-9)
    return mean_home


def joint_distribution(mean_home, mean_away, dispersion,
                       extra_inning_home_edge=0.52,
                       extra_inning_margins=DEFAULT_EXTRA_INNING_MARGINS,
                       censor_home_ninth=True, innings=9,
                       walk_off_margins=DEFAULT_WALK_OFF_MARGINS,
                       shape=None):
    """Return the joint pmf over (home, away) runs with ties resolved.

    ``shape`` selects the family. Given ``(scoreless, tail)`` each side is
    built from innings and ``dispersion`` is unused; given ``None`` each side
    is a game-level negative binomial with the dispersion below. The NB path
    is kept because it is the simpler thing to reason about and because a
    change of family should be reversible by one argument rather than by a
    revert.

    ``dispersion`` may be a single value or a ``(home, away)`` pair. Two is
    the honest choice and one was measurably wrong: home scoring is censored,
    because the home team does not bat in the bottom of the ninth when it is
    already ahead, which happens in roughly 43% of games. A pooled fit is a
    compromise between a censored side and an uncensored one and is too tight
    for both — measured against 11,428 games it understated away variance by
    15.5% and total variance by 8.7%. Fitting each side separately matches the
    observed variance on both.

    Regulation scoring for the two sides is treated as conditionally
    independent given their expected runs: the teams face different pitchers,
    and what correlation remains is largely park and weather, which are
    already in both means. This one is not an assumption of convenience — the
    residual correlation across those same games is +0.003.

    Tied regulation mass is moved into extra innings rather than deleted. The
    winner takes ``extra_inning_total_runs`` and the loser takes the rest, so
    a tie shows up in the total as well as the moneyline. Deleting the
    diagonal instead would bias every total downward by the extra-innings
    share. Both parameters are calibrated from observed games by
    `calibrate_extra_innings` rather than assumed.
    """
    dispersion_home, dispersion_away = _pair(dispersion)
    home_pieces = run_pieces(mean_home, dispersion_home, shape, innings)
    away_pmf = run_pieces(mean_away, dispersion_away, shape, innings)[0]
    if censor_home_ninth:
        joint = _censored_home(home_pieces, away_pmf, walk_off_margins)
    else:
        joint = home_pieces[0][..., :, None] * away_pmf[..., None, :]

    diagonal = np.einsum("...ii->...i", joint).copy()
    np.einsum("...ii->...i", joint)[...] = 0.0

    # Spread across the measured margins rather than one rounded number. The
    # mean extra-inning margin is 1.58, which rounds to two, while 69% of them
    # are decided by one — so rounding put every extra-inning game on the wrong
    # side of the run line.
    for score in range(MAX_RUNS + 1):
        mass = diagonal[..., score]
        for index, share in enumerate(extra_inning_margins):
            winner_cell = min(score + index + 1, MAX_RUNS)
            if winner_cell <= score:
                # Only at the top of the grid, where the pmf is negligible;
                # park it on the diagonal rather than lose it.
                joint[..., score, score] += mass * share
                continue
            joint[..., winner_cell, score] += (mass * share
                                               * extra_inning_home_edge)
            joint[..., score, winner_cell] += (mass * share
                                               * (1.0 - extra_inning_home_edge))

    total = joint.sum(axis=(-2, -1), keepdims=True)
    return joint / np.where(total > 0, total, 1.0)


def calibrate_extra_innings(games):
    """Measure the home edge and the winning margin past regulation.

    ``games`` needs ``innings_played``, ``home_win``, ``home_score`` and
    ``away_score``. Returns ``(home_edge, margins)`` suitable for
    `joint_distribution`, falling back to measured defaults on a thin sample.

    The margin is returned as a distribution rather than a mean. Its mean is
    1.58 and rounding that to two runs resolved every extra-inning game two
    apart, when 69% of them end one apart — 8.8% of the schedule, landing
    exactly on the run line.
    """
    extra = games[(games["innings_played"] > games["scheduled_innings"])
                  & games["home_win"].notna()]
    if len(extra) < 100:
        return 0.52, DEFAULT_EXTRA_INNING_MARGINS
    home_edge = float(extra["home_win"].mean())
    margin = (extra["home_score"] - extra["away_score"]).abs().astype(int)
    shares = _margin_shares(margin, MAX_EXTRA_INNING_MARGIN)
    return (float(np.clip(home_edge, 0.40, 0.60)),
            shares or DEFAULT_EXTRA_INNING_MARGINS)


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


# Bounds on the fitted inning shape. `scoreless` is tight because it is well
# identified -- it lands between 0.745 and 0.750 on every season tried, at
# every value of `tail`. `tail` is wide because it is not: the two parameters
# trade off along a ridge where both moments stay matched, so its fitted value
# wanders (2.0 to 9.5 across four seasons) without the prices moving much.
# Clipping stops an excursion rather than pinning down a real quantity.
SCORELESS_BOUNDS = (0.60, 0.85)
TAIL_BOUNDS = (0.5, 12.0)
DEFAULT_INNING_SHAPE = (0.747, 2.25)


def fit_inning_shape(observed_runs, expected_runs, innings=9):
    """Solve the inning shape from two moments of the observed run scoring.

    Method of moments, like `fit_dispersion`, and out of sample for the same
    reason. Two questions the data can answer, one parameter each:

    - how often does an inning score nothing? -> the observed shutout rate
    - how wide is a game? -> the observed variance around predicted runs

    Fitting by likelihood instead was tried and rejected. The surface in
    ``tail`` is nearly flat, and a search over it returned 2.50, 8.00, 7.75 and
    4.25 on four consecutive seasons with one pinned at the edge of the grid --
    a parameter being fitted to noise, which is the failure this repository has
    already paid for once. Two moments have no surface to get lost on.

    Pass the UNCENSORED side. The away team bats nine innings whatever the
    score, so its observed distribution is run scoring with nothing else mixed
    into it; fitting on the home side would absorb the censoring into the
    shape and then correct for it twice.

    The caller's held-out block is thin in the earliest walk-forward season —
    about 600 games, where P(0) carries a 15% relative standard error, which
    propagates to roughly +-0.013 on ``scoreless`` because P(0) goes as the
    ninth power of it. That season does fit lower than the rest (0.715 against
    0.740-0.755) and it still priced better than the negative binomial did, so
    the noise is tolerated rather than smoothed away. Shrinking it toward a
    prior would mean choosing that prior, and there is nothing to choose it
    from that is not this same data.
    """
    observed = np.asarray(observed_runs, dtype=float)
    expected = np.asarray(expected_runs, dtype=float)
    keep = np.isfinite(observed) & np.isfinite(expected) & (expected > 0)
    observed, expected = observed[keep], expected[keep]
    if len(observed) < 500:
        return DEFAULT_INNING_SHAPE
    target_zero = float((observed == 0).mean())
    target_variance = float(np.mean((observed - expected) ** 2))
    if target_zero <= 0 or target_variance <= 0:
        return DEFAULT_INNING_SHAPE

    # Bucket the means. The pmf is smooth in the mean, so a tenth of a run is
    # far finer than the fit can resolve, and it turns tens of thousands of
    # evaluations into a few dozen.
    buckets, counts = np.unique(np.round(expected, 1), return_counts=True)
    weights = counts / counts.sum()

    candidates = np.arange(SCORELESS_BOUNDS[0], SCORELESS_BOUNDS[1], 0.0025)
    best, best_error = DEFAULT_INNING_SHAPE, np.inf
    for tail in np.arange(TAIL_BOUNDS[0], TAIL_BOUNDS[1] + 1e-9, 0.125):
        # (candidate shapes, mean buckets, runs) in one pass.
        full = run_pieces(buckets, None, (candidates[:, None], tail), innings)[0]
        mean = (full * GRID).sum(axis=-1)
        variance = (full * GRID ** 2).sum(axis=-1) - mean ** 2
        zero = (full[..., 0] * weights).sum(axis=-1)
        spread = (variance * weights).sum(axis=-1)
        # Relative error on both, so neither moment dominates by scale.
        error = ((zero / target_zero - 1.0) ** 2
                 + (spread / target_variance - 1.0) ** 2)
        pick = int(np.argmin(error))
        if error[pick] < best_error:
            best = (float(candidates[pick]), float(tail))
            best_error = float(error[pick])
    return best


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
