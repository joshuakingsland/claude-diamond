import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from full_game_movement_evaluation import (CANDIDATE_FEATURES,
                                           ENTRY_PRICE_FEATURES,
                                           MICRO_FEATURES, MOVE_CLIP,
                                           STRUCTURE_FEATURES,
                                           movement_metrics)


class CandidateFamilyTests(unittest.TestCase):
    def test_an_entry_price_free_family_is_offered(self):
        self.assertIn("structure_no_entry_price", CANDIDATE_FEATURES)
        self.assertIn("structure_no_entry_price_plus_model", CANDIDATE_FEATURES)

    def test_the_clean_families_really_exclude_the_entry_price(self):
        for name, features in CANDIDATE_FEATURES.items():
            if not name.startswith("structure_no_entry_price"):
                continue
            for excluded in ENTRY_PRICE_FEATURES:
                self.assertNotIn(excluded, features, msg=name)

    def test_the_clean_family_is_otherwise_the_same(self):
        self.assertEqual(set(MICRO_FEATURES) - set(STRUCTURE_FEATURES),
                         set(ENTRY_PRICE_FEATURES))


class ImpliedClvTests(unittest.TestCase):
    """A relative MSE reduction is not a return, and must not read like one."""

    def _frame(self, actual):
        return pd.DataFrame({"move_logit": actual,
                             "official_date": ["2024-05-01"] * len(actual)})

    def test_clv_is_reported_in_probability_points(self):
        actual = np.full(200, 0.04)
        result = movement_metrics(self._frame(actual), np.full(200, 0.04), draws=50)
        self.assertIn("implied_clv_probability_points", result)
        # 0.04 logit at even money is very close to one probability point.
        self.assertAlmostEqual(result["implied_clv_probability_points"],
                               1.0, delta=0.05)

    def test_predicting_the_wrong_direction_gives_negative_clv(self):
        actual = np.full(200, 0.04)
        result = movement_metrics(self._frame(actual), np.full(200, -0.04), draws=50)
        self.assertLess(result["implied_clv_probability_points"], 0.0)

    def test_a_large_mse_reduction_can_still_be_a_tiny_clv(self):
        # The whole point. A near-perfect fit to a small move is a large
        # relative reduction and a negligible price improvement.
        rng = np.random.default_rng(0)
        actual = rng.normal(0.0, 0.005, 1000)
        result = movement_metrics(self._frame(actual), actual * 0.9, draws=50)
        self.assertGreater(result["relative_mse_reduction"], 0.8)
        self.assertLess(abs(result["implied_clv_probability_points"]), 0.15)


class NoiseReversionArtifactTests(unittest.TestCase):
    """Noise in the entry price reverts by construction, not by market force.

    move = close - entry, so measurement error in `entry` reappears in `move`
    with a negative sign. A candidate carrying the raw entry price can score
    on that alone, and it reproduces perfectly out of sample -- which is why a
    sealed confirmation window did not catch it.
    """

    def _rows(self, n=2000, seed=1, entry_noise=0.05, true_spread=0.005):
        rng = np.random.default_rng(seed)
        # A price with almost no real variation, measured with noise: the
        # totals case, where the book moves the line and not the price.
        true_entry = rng.normal(0.0, true_spread, n)
        observed_entry = true_entry + rng.normal(0.0, entry_noise, n)
        close = true_entry + rng.normal(0.0, true_spread, n)
        frame = pd.DataFrame({
            "entry_logit": observed_entry,
            "abs_entry_logit": np.abs(observed_entry),
            "move_logit": close - observed_entry,
            "official_date": "2024-05-01",
        })
        for feature in STRUCTURE_FEATURES:
            frame[feature] = rng.normal(0.0, 1.0, n)
        return frame

    def _reduction(self, features):
        train, test = self._rows(seed=1), self._rows(seed=2)
        model = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
        model.fit(train[list(features)].to_numpy(float),
                  train["move_logit"].to_numpy(float))
        predicted = np.clip(model.predict(test[list(features)].to_numpy(float)),
                            -MOVE_CLIP, MOVE_CLIP)
        actual = test["move_logit"].to_numpy(float)
        return 1.0 - ((actual - predicted) ** 2).mean() / (actual ** 2).mean()

    def test_pure_noise_produces_a_large_apparent_signal(self):
        # There is no market here at all: the close is independent of the
        # entry noise. The reduction is arithmetic.
        self.assertGreater(self._reduction(MICRO_FEATURES), 0.5)

    def test_the_artifact_survives_a_held_out_season(self):
        # Both frames are drawn independently, so this IS out of sample. A
        # sealed window cannot protect against a mechanical correlation.
        self.assertGreater(self._reduction(MICRO_FEATURES), 0.5)

    def test_removing_the_entry_price_removes_the_artifact(self):
        self.assertLess(abs(self._reduction(STRUCTURE_FEATURES)), 0.05)


if __name__ == "__main__":
    unittest.main()
