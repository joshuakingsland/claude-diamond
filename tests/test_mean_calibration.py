import unittest
import numpy as np
import pandas as pd

from mean_calibration import (attenuation, calibration, quintile_gaps, stacked,
                              _fits)


def _frame(slope=1.0, seed=0, n=3000):
    """Games whose observed runs relate to predicted by a known slope."""
    rng = np.random.default_rng(seed)
    predicted_home = rng.uniform(3.2, 5.8, n)
    predicted_away = rng.uniform(3.2, 5.8, n)
    centre = 4.5
    true_home = centre + slope * (predicted_home - centre)
    true_away = centre + slope * (predicted_away - centre)
    return pd.DataFrame({
        "game_pk": np.arange(n),
        "season": rng.choice([2023, 2024, 2025], n),
        "official_date": "2025-05-01",
        "home_score": rng.poisson(true_home).astype(float),
        "away_score": rng.poisson(true_away).astype(float),
        "expected_home_runs": predicted_home,
        "expected_away_runs": predicted_away,
        "home_win": 0.0,
        "total_runs": 9.0,
    })


class AttenuationTests(unittest.TestCase):
    def test_calibrated_predictions_give_slope_one(self):
        result = attenuation(_frame(slope=1.0))
        self.assertAlmostEqual(result["pooled"]["slope"], 1.0, delta=0.06)

    def test_over_spread_predictions_are_detected(self):
        # The defect this file exists to measure, planted deliberately.
        result = attenuation(_frame(slope=0.7))
        self.assertAlmostEqual(result["pooled"]["slope"], 0.7, delta=0.06)

    def test_stacking_keeps_every_observation(self):
        frame = _frame()
        observed, predicted, season = stacked(frame)
        self.assertEqual(len(observed), 2 * len(frame))
        self.assertEqual(len(predicted), len(observed))
        self.assertEqual(len(season), len(observed))


class QuintileTests(unittest.TestCase):
    def test_calibrated_predictions_leave_no_pattern_in_the_tails(self):
        buckets = quintile_gaps(_frame(slope=1.0))
        self.assertEqual(len(buckets), 5)
        for bucket in buckets:
            self.assertLess(abs(bucket["gap"]), 0.15)

    def test_over_spread_predictions_show_opposite_signed_tails(self):
        buckets = quintile_gaps(_frame(slope=0.7))
        # Low predictions under-shoot, high predictions over-shoot. Same
        # signature the real walk-forward shows.
        self.assertGreater(buckets[0]["gap"], 0.15)
        self.assertLess(buckets[-1]["gap"], -0.15)


class CalibrationTests(unittest.TestCase):
    def test_a_perfect_forecast_has_slope_one(self):
        rng = np.random.default_rng(3)
        probability = rng.uniform(0.25, 0.75, 20000)
        outcome = (rng.uniform(size=20000) < probability).astype(float)
        result = calibration(probability, outcome)
        self.assertAlmostEqual(result["slope"], 1.0, delta=0.12)

    def test_compressing_a_forecast_reads_as_under_confident(self):
        # The failure mode of a shrunk mean: still calibrated on average,
        # but the slope crosses one because the spread collapsed.
        rng = np.random.default_rng(4)
        truth = rng.uniform(0.2, 0.8, 20000)
        outcome = (rng.uniform(size=20000) < truth).astype(float)
        squashed = 0.5 + 0.6 * (truth - 0.5)
        self.assertGreater(calibration(squashed, outcome)["slope"], 1.2)

    def test_too_few_bins_returns_nothing_rather_than_a_number(self):
        self.assertIsNone(calibration(np.full(40, 0.5), np.zeros(40)))


class RecalibrationMapTests(unittest.TestCase):
    def test_linear_map_recovers_a_planted_slope(self):
        rng = np.random.default_rng(5)
        predicted = rng.uniform(3.0, 6.0, 5000)
        observed = 1.35 + 0.7 * predicted
        mapped = _fits("linear", predicted, observed)(np.array([3.0, 6.0]))
        np.testing.assert_allclose(mapped, [1.35 + 0.7*3.0, 1.35 + 0.7*6.0],
                                   atol=1e-6)

    def test_isotonic_map_never_reorders_two_games(self):
        rng = np.random.default_rng(6)
        predicted = rng.uniform(3.0, 6.0, 4000)
        observed = rng.poisson(0.5 + 0.8 * predicted).astype(float)
        probe = np.linspace(3.0, 6.0, 50)
        mapped = _fits("isotonic", predicted, observed)(probe)
        self.assertTrue(np.all(np.diff(mapped) >= -1e-12))


if __name__ == "__main__":
    unittest.main()
