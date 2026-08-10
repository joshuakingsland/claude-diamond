"""Serving-side arithmetic that has no second chance to be checked.

A backtest gets compared against a known answer. A live card does not, so the
places where the model and the board can quietly end up on different bases are
tested directly.
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from predict_upcoming import main, model_probability, offered_points


def _priced(**columns):
    return pd.DataFrame([columns])


class PushBasisTests(unittest.TestCase):
    """A book's two-way price already has the push mass removed.

    Comparing a push-inclusive model number against a push-exclusive market
    number understates every whole-number line on the board, and a live card
    quotes totals of 7, 8, 9 and 10 alongside the half-run ones.
    """

    def test_a_whole_total_is_renormalised_off_the_push(self):
        priced = _priced(**{"p_over_9.0": 0.45, "push_over_9.0": 0.10})
        value = float(model_probability(priced, "totals", 9.0).iloc[0])
        self.assertAlmostEqual(value, 0.45 / 0.90)

    def test_a_half_total_is_left_alone(self):
        priced = _priced(**{"p_over_8.5": 0.52})
        value = float(model_probability(priced, "totals", 8.5).iloc[0])
        self.assertAlmostEqual(value, 0.52)

    def test_a_whole_run_line_is_renormalised_too(self):
        priced = _priced(**{"p_home_rl_-1.0": 0.40, "push_home_rl_-1.0": 0.08})
        value = float(model_probability(priced, "spreads", -1.0).iloc[0])
        self.assertAlmostEqual(value, 0.40 / 0.92)

    def test_the_moneyline_ignores_the_point(self):
        priced = _priced(p_home_ml=0.55)
        self.assertAlmostEqual(
            float(model_probability(priced, "h2h", None).iloc[0]), 0.55)

    def test_a_line_the_model_did_not_price_returns_nothing(self):
        priced = _priced(**{"p_over_8.5": 0.5})
        self.assertIsNone(model_probability(priced, "totals", 12.5))


class OfferedPointTests(unittest.TestCase):
    def test_every_quoted_point_is_collected(self):
        lines = pd.DataFrame([
            {"market": "h2h", "point": None},
            {"market": "spreads", "point": -1.5},
            {"market": "spreads", "point": 1.5},
            {"market": "totals", "point": 7.0},
            {"market": "totals", "point": 8.5},
        ])
        runlines, totals = offered_points(lines)
        self.assertEqual(runlines, (-1.5, 1.5))
        self.assertEqual(totals, (7.0, 8.5))

    def test_an_empty_board_falls_back_to_the_standard_lines(self):
        lines = pd.DataFrame([{"market": "h2h", "point": None}])
        self.assertEqual(offered_points(lines), ((-1.5,), (8.5,)))


if __name__ == "__main__":
    unittest.main()


class EmptyCardTests(unittest.TestCase):
    """An empty card must not look the same as an unpriceable one.

    The distinction is the whole point: the season went missing from
    games.csv, every event failed to match, and the workflow stayed green for
    two days because writing a header-only file is not an error.
    """

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.lines = os.path.join(self.directory, "lines.csv")
        self.out = os.path.join(self.directory, "card.csv")
        # A slice of the real table rather than a synthetic one, so the
        # columns and dtypes are the ones the builder actually meets -- and
        # small, because a full build is half a minute.
        self.games = os.path.join(self.directory, "games.csv")
        pd.read_csv("data/games.csv").head(400).to_csv(self.games, index=False)

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _run(self, lines):
        pd.DataFrame(lines).to_csv(self.lines, index=False)
        return main(["--lines", self.lines, "--out", self.out,
                     "--games", self.games, "--skip-forecast"])

    def test_an_empty_board_is_not_an_error(self):
        # Nothing quoted is an ordinary off day, not a failure.
        pd.DataFrame(columns=["event_id"]).to_csv(self.lines, index=False)
        main(["--lines", self.lines, "--out", self.out,
              "--games", self.games, "--skip-forecast"])
        self.assertTrue(os.path.exists(self.out))

    def test_a_board_that_prices_nothing_fails_loudly(self):
        future = (datetime.now(timezone.utc) + timedelta(hours=6)
                  ).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = [{
            "event_id": "e1", "home_team": "Nowhere Nobodies",
            "away_team": "Elsewhere Nothings", "commence_time": future,
            "market": "h2h", "point": "", "consensus_prob_home": 0.5,
            "market_books": 5, "market_spread": 0.01,
            "consensus_price_home": -110, "consensus_price_away": -110,
            "best_price_home": -105, "best_book_home": "a",
            "best_price_away": -105, "best_book_away": "b",
            "fetched_at": future,
        }]
        with self.assertRaises(SystemExit) as caught:
            self._run(rows)
        self.assertIn("none could be priced", str(caught.exception))
