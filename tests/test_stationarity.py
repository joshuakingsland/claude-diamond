import unittest

import numpy as np
import pandas as pd

from stationarity import _weights, drift, league_by_season


def _games(rates, per_season=400, seed=0):
    """Seasons whose run environments are set deliberately."""
    rng = np.random.default_rng(seed)
    rows = []
    for season, rate in rates.items():
        home = rng.poisson(rate, per_season).astype(float)
        away = rng.poisson(rate, per_season).astype(float)
        for h, a in zip(home, away):
            rows.append({"season": season, "home_score": h, "away_score": a,
                         "total_runs": h + a})
    return pd.DataFrame(rows)


class RecencyWeightTests(unittest.TestCase):
    def test_the_season_just_gone_carries_full_weight(self):
        w = _weights([2025], target=2026, half_life=1.0)
        self.assertAlmostEqual(float(w[0]), 1.0, places=9)

    def test_one_half_life_back_is_worth_half(self):
        w = _weights([2025, 2024, 2023], target=2026, half_life=1.0)
        np.testing.assert_allclose(w, [1.0, 0.5, 0.25], atol=1e-9)

    def test_a_longer_half_life_decays_more_slowly(self):
        slow = _weights([2022], target=2026, half_life=4.0)
        fast = _weights([2022], target=2026, half_life=1.0)
        self.assertGreater(float(slow[0]), float(fast[0]))

    def test_weights_never_exceed_one_or_go_negative(self):
        w = _weights([2026, 2025, 2020], target=2026, half_life=2.0)
        self.assertTrue(np.all(w <= 1.0) and np.all(w > 0.0))

    def test_a_future_season_is_not_upweighted(self):
        # Clipping matters: without it a season at or after the target would
        # score above one and outweigh everything real.
        self.assertAlmostEqual(float(_weights([2027], 2026, 1.0)[0]), 1.0,
                               places=9)


class DriftTests(unittest.TestCase):
    def test_a_flat_league_shows_no_drift(self):
        result = drift(_games({2023: 4.5, 2024: 4.5, 2025: 4.5}, per_season=3000))
        self.assertLess(result["range"], 0.15)

    def test_a_moving_league_is_detected(self):
        result = drift(_games({2023: 3.8, 2024: 4.5, 2025: 5.2}, per_season=3000))
        self.assertGreater(result["range"], 1.0)

    def test_drift_is_reported_against_the_noise_of_one_game(self):
        # The number that decides whether a regime shift is worth acting on.
        result = drift(_games({2023: 4.4, 2024: 4.5, 2025: 4.6}, per_season=3000))
        self.assertGreater(result["single_game_sd"], 1.0)
        self.assertAlmostEqual(
            result["drift_as_share_of_game_noise"],
            result["range"] / result["single_game_sd"], places=3)

    def test_season_rates_are_per_team_not_per_game(self):
        rates = league_by_season(_games({2025: 4.5}, per_season=4000))
        self.assertAlmostEqual(float(rates.loc[2025]), 4.5, delta=0.15)


if __name__ == "__main__":
    unittest.main()
