import unittest

import numpy as np
import pandas as pd

import devig
from devig import (METHODS, american_to_prob, additive, benchmark, evaluate,
                   outcome_for, power, proportional, shin)


class AmericanPriceTests(unittest.TestCase):
    def test_known_prices(self):
        self.assertAlmostEqual(float(american_to_prob(np.array([100.0]))[0]),
                               0.5, places=9)
        self.assertAlmostEqual(float(american_to_prob(np.array([-110.0]))[0]),
                               110 / 210, places=9)
        self.assertAlmostEqual(float(american_to_prob(np.array([150.0]))[0]),
                               100 / 250, places=9)

    def test_minus_one_hundred_does_not_divide_by_zero(self):
        # np.where evaluates both branches, and the positive formula divides
        # by price + 100. At -100 that is zero.
        with np.errstate(divide="raise", invalid="raise"):
            value = american_to_prob(np.array([-100.0]))
        self.assertAlmostEqual(float(value[0]), 0.5, places=9)
        self.assertTrue(np.isfinite(value).all())


class DevigMethodTests(unittest.TestCase):
    """Every method must return a coherent two-way pair."""

    PAIRS = [(-110.0, -110.0), (-250.0, 200.0), (150.0, -180.0),
             (-1000.0, 600.0), (-105.0, -115.0)]

    def _implied(self, home, away):
        return (american_to_prob(np.array([home])),
                american_to_prob(np.array([away])))

    def test_the_two_sides_sum_to_one(self):
        for home, away in self.PAIRS:
            h, a = self._implied(home, away)
            for name, method in METHODS.items():
                total = float(method(h, a)[0]) + float(method(a, h)[0])
                self.assertAlmostEqual(total, 1.0, places=6,
                                       msg=f"{name} at {home}/{away}")

    def test_a_symmetric_price_is_a_coin_flip_under_every_method(self):
        h, a = self._implied(-110.0, -110.0)
        for name, method in METHODS.items():
            self.assertAlmostEqual(float(method(h, a)[0]), 0.5, places=6,
                                   msg=name)

    def test_methods_order_the_favourite_the_same_way(self):
        # They differ in how much margin comes off each side, never in which
        # side is favoured.
        h, a = self._implied(-250.0, 200.0)
        for name, method in METHODS.items():
            self.assertGreater(float(method(h, a)[0]), 0.5, msg=name)

    def test_power_solves_its_own_equation(self):
        h, a = self._implied(-250.0, 200.0)
        ph, pa = float(power(h, a)[0]), float(power(a, h)[0])
        self.assertAlmostEqual(ph + pa, 1.0, places=6)

    def test_additive_takes_equal_points_from_each_side(self):
        h, a = self._implied(-250.0, 200.0)
        overround = float(h[0] + a[0]) - 1.0
        self.assertAlmostEqual(float(h[0]) - float(additive(h, a)[0]),
                               overround / 2.0, places=9)

    def test_proportional_and_shin_disagree_on_a_lopsided_price(self):
        # If they agreed everywhere the comparison would be vacuous.
        h, a = self._implied(-1000.0, 600.0)
        self.assertGreater(abs(float(proportional(h, a)[0])
                               - float(shin(h, a)[0])), 1e-4)


class OutcomeTests(unittest.TestCase):
    def _frame(self, market, point, home, away):
        return pd.DataFrame([{"market": market, "point": point,
                              "home_score": home, "away_score": away,
                              "home_win": float(home > away),
                              "total_runs": home + away}])

    def test_moneyline(self):
        self.assertEqual(float(outcome_for(self._frame("h2h", np.nan, 5, 3))[0]), 1.0)
        self.assertEqual(float(outcome_for(self._frame("h2h", np.nan, 2, 3))[0]), 0.0)

    def test_run_line(self):
        # -1.5 needs a win by two.
        self.assertEqual(float(outcome_for(self._frame("spreads", -1.5, 5, 3))[0]), 1.0)
        self.assertEqual(float(outcome_for(self._frame("spreads", -1.5, 4, 3))[0]), 0.0)

    def test_totals(self):
        self.assertEqual(float(outcome_for(self._frame("totals", 8.5, 5, 4))[0]), 1.0)
        self.assertEqual(float(outcome_for(self._frame("totals", 8.5, 4, 4))[0]), 0.0)

    def test_a_push_settles_to_nothing_rather_than_a_loss(self):
        self.assertTrue(np.isnan(outcome_for(self._frame("totals", 8.0, 4, 4))[0]))
        self.assertTrue(np.isnan(outcome_for(self._frame("spreads", -1.0, 4, 3))[0]))


def _quotes(books=6, captures=3, games=4):
    """A quote log shaped like the real one, with a distinct id per row."""
    rows, snapshot = [], 0
    for capture in range(captures):
        for game in range(games):
            for book in range(books):
                snapshot += 1
                rows.append({
                    "snapshot_id": f"s{snapshot}",
                    "fetched_at": f"2026-05-0{capture + 1}T18:00:00Z",
                    "event_id": f"e{game}", "commence_time": "2026-05-09T23:00:00Z",
                    "date": "2026-05-09", "home_team": "Home", "away_team": "Away",
                    "market": "h2h", "point": np.nan, "book_key": f"b{book}",
                    "region": "us", "priced": 1,
                    "price_home": -120.0 - book, "price_away": 100.0 + book,
                    "game_pk": game, "home_score": 5.0, "away_score": 3.0,
                    "home_win": 1.0, "total_runs": 8.0,
                })
    return pd.DataFrame(rows)


class ConsensusGroupingTests(unittest.TestCase):
    """A consensus is books within one capture, not one row.

    The first version grouped on snapshot_id, which is a per-quote id, so it
    handed every quote back unchanged and quietly measured single books while
    reporting a consensus.
    """

    def test_consensus_collapses_books_within_a_capture(self):
        frame = _quotes(books=6, captures=3, games=4)
        result, rows = evaluate(frame)
        self.assertEqual(result["quotes"], 6 * 3 * 4)
        self.assertEqual(result["consensus_rows"], 3 * 4)
        self.assertEqual(len(rows), 3 * 4)

    def test_books_per_consensus_is_reported(self):
        result, _ = evaluate(_quotes(books=6, captures=3, games=4))
        self.assertAlmostEqual(result["books_per_consensus"], 6.0, places=6)

    def test_grouping_key_is_the_capture_not_the_quote(self):
        self.assertIn("fetched_at", devig.CAPTURE_KEYS)
        self.assertNotIn("snapshot_id", devig.CAPTURE_KEYS)


class BenchmarkTests(unittest.TestCase):
    def test_a_thin_overlap_reports_itself_rather_than_a_number(self):
        frame = _quotes(books=3, captures=1, games=2)
        frame.loc[frame.book_key == "b0", "book_key"] = "pinnacle"
        result = benchmark(frame)
        self.assertIn("status", result)
        self.assertNotIn("delta", result)

    def test_a_missing_reference_book_is_not_an_error(self):
        result = benchmark(_quotes(), book="nonexistent")
        self.assertIn("no overlap", result["status"])


if __name__ == "__main__":
    unittest.main()
