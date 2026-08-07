"""The extreme-bucket test, which exists to refuse a tempting conclusion.

Bucketing a season by disagreement and reading off the widest bucket produces a
spectacular number almost regardless of whether anything is there. These tests
pin the two things that stop it being believed: the search over thresholds is
priced, and the bands are disjoint so a discontinuity cannot hide inside a
cumulative cut.
"""

import unittest

import numpy as np
import pandas as pd

from extremes import band_table, roi, search_correction


def _frame(gaps, market, won, dates=None):
    size = len(gaps)
    return pd.DataFrame({
        "market_key": ["h2h"] * size,
        "official_date": dates or [f"2025-05-{1 + i % 28:02d}" for i in range(size)],
        "gap_pts": gaps,
        "model_prob": np.clip(np.asarray(market) + np.asarray(gaps) / 100, 0, 1),
        "market_prob": market,
        "won": won,
        "books": [8] * size,
    })


class RoiTests(unittest.TestCase):
    def test_winning_exactly_at_the_fair_rate_breaks_even(self):
        self.assertAlmostEqual(roi([1, 0], [0.5, 0.5]), 0.0)

    def test_winning_more_than_the_price_implies_profits(self):
        self.assertGreater(roi([1, 1, 0], [0.5, 0.5, 0.5]), 0)

    def test_a_longshot_that_never_lands_loses_everything(self):
        self.assertAlmostEqual(roi([0, 0, 0], [0.2, 0.2, 0.2]), -1.0)


class BandTests(unittest.TestCase):
    def test_bands_are_disjoint_so_a_swing_cannot_hide(self):
        """A cumulative cut would average these two together; bands must not."""
        gaps = [13.5] * 40 + [14.5] * 40
        market = [0.42] * 40 + [0.39] * 40
        won = [0.0] * 40 + [1.0] * 40
        table = {row["band"]: row for row in band_table(_frame(gaps, market, won))}
        self.assertEqual(table["13-14"]["games"], 40)
        self.assertEqual(table["14-15"]["games"], 40)
        self.assertLess(table["13-14"]["no_vig_roi_pct"], 0)
        self.assertGreater(table["14-15"]["no_vig_roi_pct"], 0)

    def test_every_row_reports_who_was_right(self):
        table = band_table(_frame([1.0] * 30, [0.5] * 30, [1.0, 0.0] * 15))
        self.assertIn("model_says", table[0])
        self.assertIn("market_says", table[0])
        self.assertIn("actually_won", table[0])


class SearchCorrectionTests(unittest.TestCase):
    def test_pure_noise_does_not_look_significant_after_the_search(self):
        """Outcomes drawn from the market price itself: nothing is there.

        The widest cut will still look good — that is the point — so the
        corrected p-value must not be small.
        """
        rng = np.random.default_rng(0)
        size = 1200
        gaps = rng.uniform(0, 20, size)
        market = rng.uniform(0.35, 0.6, size)
        won = (rng.random(size) < market).astype(float)
        result = search_correction(_frame(gaps, market, won), draws=400, seed=1)
        self.assertGreater(result["p_value_after_search"], 0.05)

    def test_the_null_itself_produces_a_flattering_best_cut(self):
        """If searching found nothing on noise there would be nothing to correct."""
        rng = np.random.default_rng(0)
        size = 1200
        gaps = rng.uniform(0, 20, size)
        market = rng.uniform(0.35, 0.6, size)
        won = (rng.random(size) < market).astype(float)
        result = search_correction(_frame(gaps, market, won), draws=400, seed=1)
        self.assertGreater(result["null_best_cut_mean_roi_pct"], 5.0)

    def test_a_real_and_large_effect_still_survives(self):
        """The correction must not be so blunt that nothing could ever pass."""
        rng = np.random.default_rng(2)
        size = 1200
        gaps = rng.uniform(0, 20, size)
        market = np.full(size, 0.40)
        # Wide disagreements win far more often than the price implies.
        probability = np.where(gaps >= 10, 0.75, 0.40)
        won = (rng.random(size) < probability).astype(float)
        result = search_correction(_frame(gaps, market, won), draws=400, seed=1)
        self.assertLess(result["p_value_after_search"], 0.05)


if __name__ == "__main__":
    unittest.main()
