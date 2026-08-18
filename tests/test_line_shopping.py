"""The failure modes that would make a line-shopping study lie.

Every one of these is a way to manufacture an edge out of nothing: a panel
that grows, a bet chosen after the fact, a close taken from the same capture
the bet deviated from. They are tested rather than asserted in a docstring.
"""

import unittest

import numpy as np
import pandas as pd

import line_shopping as ls
from devig import american_to_prob


def quote(event, market, book, captured, home, away, point=np.nan,
          commence="2026-08-05T22:00:00Z", priced=1):
    return {"event_id": event, "market": market, "point": point,
            "book_key": book, "fetched_at": captured, "priced": priced,
            "commence_time": commence, "price_home": home,
            "price_away": away}


def panel(captured, prices, event="g1", market="h2h", **kw):
    """One capture of a whole book panel, priced as (home, away) per book."""
    return [quote(event, market, f"book{i}", captured, home, away, **kw)
            for i, (home, away) in enumerate(prices)]


class AmericanBestPriceTests(unittest.TestCase):
    """`captures` takes the numeric max as the best price. That must hold."""

    PRICES = [-500.0, -250.0, -110.0, -105.0, 100.0, 105.0, 250.0]

    def test_the_maximum_american_price_is_the_cheapest_break_even(self):
        # The ordering holds inside the negatives, inside the positives, and
        # across the sign change, so no sorting rule beyond `max` is needed.
        prices = np.array(self.PRICES)
        implied = american_to_prob(prices)
        self.assertEqual(float(prices.max()), self.PRICES[-1])
        self.assertAlmostEqual(float(implied.min()),
                               float(american_to_prob(np.array([250.0]))[0]))
        # Monotone decreasing throughout, not merely at the endpoints.
        self.assertTrue(np.all(np.diff(implied) < 0))

    def test_the_one_place_the_ordering_ties(self):
        # -100 and +100 are the same price. `max` picks +100; both imply a
        # half, so the tie costs nothing.
        pair = american_to_prob(np.array([-100.0, 100.0]))
        self.assertAlmostEqual(float(pair[0]), float(pair[1]), places=12)


class PanelTests(unittest.TestCase):
    """The best of N prices rises with N. A growing panel invents an edge."""

    def test_a_book_that_arrives_late_is_excluded(self):
        rows = []
        for i in range(10):
            rows += panel(f"2026-08-05T2{i}:00:00Z", [(-110.0, -110.0)] * 6)
        # A book present for only the last capture: 10% coverage.
        rows.append(quote("g1", "h2h", "latecomer", "2026-08-05T29:00:00Z",
                          -110.0, -110.0))
        books = ls.panel_books(pd.DataFrame(rows))
        self.assertNotIn("latecomer", books)
        self.assertEqual(len(books), 6)

    def test_unpriced_books_are_not_shoppable(self):
        rows = []
        for i in range(10):
            rows += panel(f"2026-08-05T2{i}:00:00Z", [(-110.0, -110.0)] * 6)
            rows.append(quote("g1", "h2h", "sharp", f"2026-08-05T2{i}:00:00Z",
                              -104.0, -104.0, priced=0))
        books = ls.panel_books(pd.DataFrame(rows))
        self.assertNotIn("sharp", books)


class SelectionTests(unittest.TestCase):
    """A shopper watching live takes the first qualifying price, not the best."""

    def _book(self, rows):
        frame = pd.DataFrame(rows)
        return ls.captures(ls.prepare(frame, ls.panel_books(frame)))

    def test_the_first_qualifying_capture_is_taken_not_the_biggest(self):
        commence = "2026-08-05T22:00:00Z"
        rows = []
        # 21:00 — one book is well off the consensus. Qualifies.
        rows += panel("2026-08-05T21:00:00Z",
                      [(-110.0, -110.0)] * 5 + [(120.0, -160.0)],
                      commence=commence)
        # 21:30 — the same book is off by even more. Must not be preferred.
        rows += panel("2026-08-05T21:30:00Z",
                      [(-110.0, -110.0)] * 5 + [(200.0, -260.0)],
                      commence=commence)
        picks = ls.opportunities(self._book(rows), threshold=0.005)
        self.assertEqual(len(picks), 1)
        self.assertEqual(picks.iloc[0]["fetched_at"], "2026-08-05T21:00:00Z")
        self.assertEqual(float(picks.iloc[0]["price"]), 120.0)

    def test_a_market_in_agreement_produces_no_bet(self):
        rows = panel("2026-08-05T21:00:00Z", [(-110.0, -110.0)] * 6)
        self.assertEqual(len(ls.opportunities(self._book(rows),
                                              threshold=0.005)), 0)


class CloseTests(unittest.TestCase):
    """A bet must be scored against a capture strictly later than its own."""

    def _tables(self, rows):
        frame = ls.prepare(pd.DataFrame(rows), ls.panel_books(pd.DataFrame(rows)))
        return ls.captures(frame), ls.closes(frame)

    def _rows(self, second_capture=True):
        commence = "2026-08-05T22:00:00Z"
        rows = panel("2026-08-05T21:00:00Z",
                     [(-110.0, -110.0)] * 5 + [(120.0, -160.0)],
                     commence=commence)
        if second_capture:
            rows += panel("2026-08-05T21:45:00Z", [(-110.0, -110.0)] * 6,
                          commence=commence)
        return rows

    def test_a_bet_at_the_last_capture_has_no_close_and_is_dropped(self):
        # Its consensus and its close would be the same number, so its closing
        # line value would equal the deviation it was selected on. That is not
        # a result, it is the entry condition read back.
        book, close = self._tables(self._rows(second_capture=False))
        picks = ls.opportunities(book, threshold=0.005)
        self.assertEqual(len(picks), 1)
        self.assertEqual(len(ls.settle(picks, close)), 0)

    def test_a_bet_with_a_later_capture_is_scored_against_it(self):
        book, close = self._tables(self._rows())
        settled = ls.settle(ls.opportunities(book, threshold=0.005), close)
        self.assertEqual(len(settled), 1)
        row = settled.iloc[0]
        self.assertEqual(row["side"], "home")
        # Took +120 (break-even 0.4545) into a close that stayed at -110
        # (fair 0.5). Positive closing line value, net of the vig paid.
        self.assertAlmostEqual(float(row["close_probability"]), 0.5, places=6)
        self.assertAlmostEqual(float(row["break_even"]), 100 / 220, places=6)
        self.assertGreater(float(row["clv"]), 0.0)

    def test_a_price_worse_than_the_close_gives_negative_value(self):
        commence = "2026-08-05T22:00:00Z"
        rows = panel("2026-08-05T21:00:00Z", [(-200.0, 170.0)] * 6,
                     commence=commence)
        rows += panel("2026-08-05T21:45:00Z", [(-110.0, -110.0)] * 6,
                      commence=commence)
        book, close = self._tables(rows)
        # No selection: take the home side at the first capture regardless.
        picks = ls.routine(book)
        settled = ls.settle(picks, close)
        home = settled[settled["side"] == "home"].iloc[0]
        self.assertLess(float(home["clv"]), 0.0)


class UnconditionalIdentityTests(unittest.TestCase):
    """The no-selection arm is arithmetic, and the code must not hide that."""

    def test_both_sides_sum_to_one_minus_the_best_overround(self):
        commence = "2026-08-05T22:00:00Z"
        rows = panel("2026-08-05T21:00:00Z",
                     [(-130.0, 115.0), (-125.0, 108.0), (-140.0, 120.0),
                      (-120.0, 104.0), (-135.0, 118.0), (-128.0, 112.0)],
                     commence=commence)
        rows += panel("2026-08-05T21:45:00Z", [(-160.0, 140.0)] * 6,
                      commence=commence)
        frame = pd.DataFrame(rows)
        prepared = ls.prepare(frame, ls.panel_books(frame))
        book = ls.captures(prepared)
        settled = ls.settle(ls.routine(book), ls.closes(prepared))
        self.assertEqual(len(settled), 2)
        # Whatever the close did, the two sides sum to the same number.
        total = float(settled["clv"].sum())
        overround = float(settled["best_overround"].iloc[0])
        self.assertAlmostEqual(total, 1.0 - overround, places=12)


if __name__ == "__main__":
    unittest.main()
