import unittest

import numpy as np

from runs import (GRID, joint_distribution, moneyline_probability,
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
        balanced = joint_distribution(4.5, 4.5, 4.0)
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
