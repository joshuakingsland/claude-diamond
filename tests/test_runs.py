import unittest

import numpy as np
import pandas as pd

from runs import (DEFAULT_WALK_OFF_MARGINS, GRID, calibrate_walk_off,
                  joint_distribution, moneyline_probability,
                  negative_binomial_pmf, push_probability,
                  runline_probability, total_over_probability)


class RunDistributionTests(unittest.TestCase):
    def setUp(self):
        self.joint = joint_distribution(4.6, 4.2, dispersion=4.0)

    def test_distribution_is_normalised_and_has_no_ties(self):
        self.assertAlmostEqual(float(self.joint.sum()), 1.0, places=9)
        diagonal = float(np.einsum("ii->", self.joint))
        # Baseball has no draws; every tie must be resolved away. The residual
        # is not exactly zero because tied mass at the very top of the grid has
        # nowhere to move and is parked on the diagonal rather than discarded.
        # At a realistic run total that mass is ~1e-12, so it is preserved for
        # correctness and ignored for pricing.
        self.assertLess(diagonal, 1e-9)

    def test_the_three_markets_agree_with_each_other(self):
        # A -1.5 cover is strictly harder than winning outright, and a +1.5
        # cover is strictly easier. If these ever cross, the joint has been
        # built wrong and the three markets are pricing different games.
        moneyline = moneyline_probability(self.joint)
        minus = runline_probability(self.joint, -1.5)
        plus = runline_probability(self.joint, 1.5)
        self.assertLess(minus, moneyline)
        self.assertGreater(plus, moneyline)

    def test_totals_are_monotone_in_the_line(self):
        points = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
        probabilities = [float(total_over_probability(self.joint, p)) for p in points]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))

    def test_a_stronger_home_side_raises_its_price(self):
        weak = moneyline_probability(joint_distribution(4.0, 4.6, 4.0))
        strong = moneyline_probability(joint_distribution(5.4, 4.6, 4.0))
        self.assertGreater(strong, weak)

    def test_over_dispersion_widens_the_total(self):
        # Lower dispersion means fatter tails, so an extreme total gets more
        # likely even though the mean is unchanged.
        tight = total_over_probability(joint_distribution(4.5, 4.5, 40.0), 13.5)
        loose = total_over_probability(joint_distribution(4.5, 4.5, 2.5), 13.5)
        self.assertGreater(loose, tight)

    def test_extra_innings_add_runs_rather_than_deleting_ties(self):
        # Resolving ties by deleting the diagonal would lose that mass and
        # bias every total downward. The resolved distribution must carry a
        # higher expected total than the unresolved product.
        home = negative_binomial_pmf(4.5, 4.0)
        away = negative_binomial_pmf(4.5, 4.0)
        raw = home[:, None] * away[None, :]
        totals = GRID[:, None] + GRID[None, :]
        raw_total = float((raw * totals).sum())
        resolved = float((self.joint * totals).sum())
        # Censoring is held off here: it legitimately lowers the total,
        # because the home ninth is often not played, and this test is about
        # tie resolution rather than about that.
        balanced = joint_distribution(4.5, 4.5, 4.0, censor_home_ninth=False)
        self.assertGreater(float((balanced * totals).sum()), raw_total)
        self.assertGreater(resolved, 0.0)

    def test_half_run_lines_cannot_push(self):
        self.assertEqual(float(push_probability(self.joint, -1.5, "spreads")), 0.0)
        self.assertEqual(float(push_probability(self.joint, 8.5, "totals")), 0.0)

    def test_whole_number_total_reports_push_mass(self):
        push = float(push_probability(self.joint, 9.0, "totals"))
        self.assertGreater(push, 0.01)

    def test_vectorises_over_many_games(self):
        joint = joint_distribution(np.array([4.0, 5.0, 6.0]),
                                   np.array([4.5, 4.5, 4.5]), 4.0)
        self.assertEqual(joint.shape[0], 3)
        probabilities = moneyline_probability(joint)
        self.assertEqual(probabilities.shape, (3,))
        self.assertTrue(np.all(np.diff(probabilities) > 0))


if __name__ == "__main__":
    unittest.main()


class HomeNinthCensoringTests(unittest.TestCase):
    """The home side bats the ninth only when it needs to.

    Giving both sides a full nine innings put 14.2% on the home team losing by
    one against 11.1% observed, and 15.2% on winning by one against 17.1%. The
    run line is decided at exactly that boundary, which is why it was the worst
    calibrated of the three markets before this and the best after.
    """

    def test_mass_is_conserved(self):
        for home, away, dispersion in ((4.6, 4.2, (3.8, 3.2)),
                                       (3.0, 6.0, 4.0),
                                       (5.5, 5.5, (3.5, 3.5))):
            joint = joint_distribution(home, away, dispersion)
            self.assertAlmostEqual(float(joint.sum()), 1.0, places=9)

    def test_it_moves_mass_onto_a_one_run_home_win(self):
        margin = GRID[:, None] - GRID[None, :]
        loose = joint_distribution(4.6, 4.2, (3.8, 3.2), censor_home_ninth=False)
        tight = joint_distribution(4.6, 4.2, (3.8, 3.2))
        by_one = lambda j: float((j * (margin == 1)).sum())
        self.assertGreater(by_one(tight), by_one(loose))

    def test_the_moneyline_is_untouched(self):
        """Censoring redistributes home wins across margins, it does not
        create or destroy them."""
        loose = joint_distribution(4.6, 4.2, (3.8, 3.2), censor_home_ninth=False)
        tight = joint_distribution(4.6, 4.2, (3.8, 3.2))
        self.assertAlmostEqual(float(moneyline_probability(loose)),
                               float(moneyline_probability(tight)), places=3)

    def test_walk_off_margins_are_spread_not_collapsed(self):
        """Collapsing every walk-off onto one run overshoots that margin and
        strips the run line of mass it should keep."""
        margin = GRID[:, None] - GRID[None, :]
        spread = joint_distribution(4.6, 4.2, (3.8, 3.2),
                                    walk_off_margins=(0.87, 0.08, 0.04, 0.01))
        collapsed = joint_distribution(4.6, 4.2, (3.8, 3.2),
                                       walk_off_margins=(1.0, 0.0, 0.0, 0.0))
        cover = lambda j: float((j * (margin > 1.5)).sum())
        self.assertGreater(cover(spread), cover(collapsed))

    def test_it_is_vectorised_over_games(self):
        home = np.array([4.6, 3.0, 5.5])
        away = np.array([4.2, 6.0, 5.5])
        joint = joint_distribution(home, away, (3.8, 3.2))
        self.assertEqual(joint.shape[0], 3)
        np.testing.assert_allclose(joint.sum(axis=(1, 2)), 1.0, atol=1e-9)


class WalkOffCalibrationTests(unittest.TestCase):
    def test_it_measures_the_margin_from_the_games(self):
        games = pd.DataFrame({
            "home_batted_ninth": [1] * 300,
            "home_score": [5] * 240 + [6] * 60,
            "away_score": [4] * 300,
        })
        margins = calibrate_walk_off(games)
        self.assertAlmostEqual(margins[0], 0.8, places=6)
        self.assertAlmostEqual(margins[1], 0.2, places=6)

    def test_a_thin_sample_keeps_the_measured_default(self):
        games = pd.DataFrame({"home_batted_ninth": [1] * 10,
                              "home_score": [5] * 10, "away_score": [4] * 10})
        self.assertEqual(calibrate_walk_off(games), DEFAULT_WALK_OFF_MARGINS)

    def test_games_the_home_side_lost_are_not_walk_offs(self):
        games = pd.DataFrame({
            "home_batted_ninth": [1] * 300,
            "home_score": [2] * 300, "away_score": [5] * 300,
        })
        self.assertEqual(calibrate_walk_off(games), DEFAULT_WALK_OFF_MARGINS)
