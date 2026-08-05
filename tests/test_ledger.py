"""The staking policy, which until now was written down and never applied.

Every constant these tests exercise sat in `config.py` unreferenced by any
code. A control that is declared and never enforced reads to a later auditor
as a control that was in force, so each gate gets a test that makes it bite.
"""

import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from config import (EDGE_RULE, GAME_DAY_STAKE_CAP, MAX_EXECUTION_DEVIATION,
                    MAX_ODDS_AGE_MINUTES, MIN_LOCK_LEAD_MINUTES,
                    MIN_MARKET_BOOKS)
from ledger import (_outcome, payout, screen, settle, staked_by_day,
                    summarise)

NOW = datetime(2026, 8, 5, 18, 0, tzinfo=timezone.utc)


def _card(**overrides):
    row = {
        "game_pk": 1, "official_date": "2026-08-05",
        "commence_time": "2026-08-05T22:00:00Z",
        "home_team": "Home Nine", "away_team": "Away Nine",
        "market": "h2h", "point": "",
        "model_prob_home": 0.60, "market_prob_home": 0.50,
        "disagreement": 0.10,
        "market_books": 8, "market_spread": 0.01,
        "consensus_price_home": -110.0, "consensus_price_away": -110.0,
        "best_price_home": -105.0, "best_book_home": "Book A",
        "best_price_away": -105.0, "best_book_away": "Book B",
        "lead_minutes": 240, "odds_fetched_at": "2026-08-05T17:45:00Z",
        "model_version": "diamond-v0", "model_kind": "glm",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _gates(rejections):
    return {rejection["gate"] for rejection in rejections}


class GateTests(unittest.TestCase):
    def test_a_clear_disagreement_becomes_a_wager(self):
        wagers, _ = screen(_card(), now=NOW)
        self.assertEqual(len(wagers), 1)
        self.assertEqual(wagers[0]["side"], "home")

    def test_a_disagreement_below_the_edge_rule_is_rejected(self):
        wagers, rejections = screen(
            _card(disagreement=EDGE_RULE / 2), now=NOW)
        self.assertEqual(wagers, [])
        self.assertIn("below_edge_rule", _gates(rejections))

    def test_a_thin_market_is_rejected(self):
        wagers, rejections = screen(
            _card(market_books=MIN_MARKET_BOOKS - 1), now=NOW)
        self.assertEqual(wagers, [])
        self.assertIn("too_few_books", _gates(rejections))

    def test_a_quote_outside_the_lock_window_is_rejected(self):
        """Lineups post about three hours out; earlier is not a lock."""
        wagers, rejections = screen(
            _card(lead_minutes=MIN_LOCK_LEAD_MINUTES - 1), now=NOW)
        self.assertEqual(wagers, [])
        self.assertIn("outside_lock_window", _gates(rejections))

    def test_a_stale_quote_is_rejected(self):
        old = NOW - timedelta(minutes=MAX_ODDS_AGE_MINUTES + 10)
        wagers, rejections = screen(
            _card(odds_fetched_at=old.strftime("%Y-%m-%dT%H:%M:%SZ")), now=NOW)
        self.assertEqual(wagers, [])
        self.assertIn("stale_quote", _gates(rejections))

    def test_a_price_far_better_than_consensus_is_a_broken_quote(self):
        """Line shopping is worth a point or two, not twenty."""
        wagers, rejections = screen(
            _card(consensus_price_home=-200.0, best_price_home=400.0), now=NOW)
        self.assertEqual(wagers, [])
        self.assertIn("execution_deviation", _gates(rejections))

    def test_a_negative_disagreement_backs_the_away_side(self):
        wagers, _ = screen(
            _card(model_prob_home=0.40, disagreement=-0.10), now=NOW)
        self.assertEqual(wagers[0]["side"], "away")
        self.assertAlmostEqual(wagers[0]["model_prob"], 0.60)

    def test_an_already_open_wager_is_not_reopened(self):
        wagers, _ = screen(_card(), now=NOW)
        again, rejections = screen(
            _card(), open_ids={wagers[0]["wager_id"]}, now=NOW)
        self.assertEqual(again, [])
        self.assertIn("already_locked", _gates(rejections))

    def test_a_wide_market_is_flagged_but_still_taken(self):
        """Config calls this a warning; a test that drops its awkward rows
        measures nothing."""
        wagers, _ = screen(_card(market_spread=0.20), now=NOW)
        self.assertEqual(len(wagers), 1)
        self.assertEqual(wagers[0]["wide_market"], 1)


class DayCapTests(unittest.TestCase):
    def test_the_cap_holds_and_keeps_the_widest_disagreements(self):
        rows = []
        for index in range(GAME_DAY_STAKE_CAP + 3):
            rows.append(_card(game_pk=index, market="h2h",
                              disagreement=0.05 + index / 100.0,
                              model_prob_home=0.5 + 0.05 + index / 100.0))
        card = pd.concat(rows, ignore_index=True)
        wagers, rejections = screen(card, now=NOW)
        self.assertEqual(sum(w["stake"] for w in wagers), GAME_DAY_STAKE_CAP)
        self.assertIn("day_cap", _gates(rejections))
        # The cap must not be handed out in file order.
        taken = {wager["game_pk"] for wager in wagers}
        self.assertEqual(taken, {5, 4, 3})


class CapAcrossRunsTests(unittest.TestCase):
    """The cap has to survive the ledger, not just the call.

    The capture workflow screens the same card every hour. A cap computed
    fresh per call grants a whole new allowance each run — thirteen runs
    against a three-unit cap is thirty-nine units on a day limited to three,
    and it cannot show up in a single-run test. It had already happened in
    production before this was caught: six units on one day.
    """

    def _big_card(self):
        rows = [_card(game_pk=index, disagreement=0.05 + index / 100.0,
                      model_prob_home=0.55 + index / 100.0)
                for index in range(GAME_DAY_STAKE_CAP + 3)]
        return pd.concat(rows, ignore_index=True)

    def test_a_full_day_blocks_every_later_run(self):
        card = self._big_card()
        first, _ = screen(card, now=NOW)
        self.assertEqual(sum(w["stake"] for w in first), GAME_DAY_STAKE_CAP)

        ledger = pd.DataFrame(first)
        second, rejections = screen(
            card, open_ids=set(ledger["wager_id"]),
            prior_stakes=staked_by_day(ledger), now=NOW)
        self.assertEqual(second, [])
        self.assertIn("day_cap", _gates(rejections))

    def test_a_partly_used_day_allows_only_the_remainder(self):
        card = self._big_card()
        wagers, _ = screen(card, prior_stakes={"2026-08-05": 2.0}, now=NOW)
        self.assertEqual(sum(w["stake"] for w in wagers),
                         GAME_DAY_STAKE_CAP - 2.0)

    def test_another_day_is_unaffected(self):
        card = self._big_card()
        wagers, _ = screen(card, prior_stakes={"2026-09-01": 99.0}, now=NOW)
        self.assertEqual(sum(w["stake"] for w in wagers), GAME_DAY_STAKE_CAP)

    def test_stakes_are_totalled_off_a_reloaded_ledger(self):
        ledger = pd.DataFrame([
            {"official_date": "2026-08-05", "stake": 1.0},
            {"official_date": "2026-08-05", "stake": 1.0},
            {"official_date": "2026-08-06", "stake": 1.0},
        ])
        self.assertEqual(staked_by_day(ledger),
                         {"2026-08-05": 2.0, "2026-08-06": 1.0})


class SettlementTests(unittest.TestCase):
    def _wager(self, **overrides):
        row = {"market": "h2h", "point": "", "side": "home"}
        row.update(overrides)
        return row

    def test_moneyline_settles_on_the_winner(self):
        game = {"home_score": 5.0, "away_score": 3.0}
        self.assertEqual(_outcome(self._wager(), game), "win")
        self.assertEqual(_outcome(self._wager(side="away"), game), "loss")

    def test_a_whole_number_total_can_push(self):
        game = {"home_score": 4.0, "away_score": 5.0}
        wager = self._wager(market="totals", point=9.0)
        self.assertEqual(_outcome(wager, game), "push")

    def test_the_over_is_the_home_side_of_a_total(self):
        game = {"home_score": 6.0, "away_score": 5.0}
        self.assertEqual(
            _outcome(self._wager(market="totals", point=8.5), game), "win")
        self.assertEqual(
            _outcome(self._wager(market="totals", point=8.5, side="away"), game),
            "loss")

    def test_the_run_line_uses_the_home_handicap(self):
        game = {"home_score": 5.0, "away_score": 3.0}
        self.assertEqual(
            _outcome(self._wager(market="spreads", point=-1.5), game), "win")
        game = {"home_score": 4.0, "away_score": 3.0}
        self.assertEqual(
            _outcome(self._wager(market="spreads", point=-1.5), game), "loss")

    def test_an_unplayed_game_stays_open(self):
        ledger = pd.DataFrame([{
            "game_pk": 1, "market": "h2h", "point": "", "side": "home",
            "stake": 1.0, "price": -110.0, "outcome": "", "profit": "",
            "settled_at": "",
        }])
        games = pd.DataFrame([{"game_pk": 1, "home_score": None,
                               "away_score": None}])
        settled_ledger, count = settle(ledger, games)
        self.assertEqual(count, 0)
        self.assertEqual(settled_ledger.iloc[0]["outcome"], "")


class WagerIdTests(unittest.TestCase):
    """The id is the dedupe key; it must not depend on how pandas typed a
    column.

    A moneyline row has no point, and a card read back through pandas delivers
    that as NaN. Formatting NaN into the key gave `1|h2h|nan|away`, stable only
    while the column keeps typing as float — otherwise every moneyline id
    changes and every open moneyline wager is locked again.
    """

    def test_a_missing_point_is_empty_however_it_arrives(self):
        blank = screen(_card(market="h2h", point=""), now=NOW)[0][0]["wager_id"]
        nan = screen(_card(market="h2h", point=float("nan")),
                     now=NOW)[0][0]["wager_id"]
        none = screen(_card(market="h2h", point=None), now=NOW)[0][0]["wager_id"]
        self.assertEqual(blank, nan)
        self.assertEqual(blank, none)
        self.assertNotIn("nan", blank)

    def test_a_real_point_still_reaches_the_id(self):
        wager = screen(_card(market="totals", point=8.5), now=NOW)[0][0]
        self.assertIn("8.5", wager["wager_id"])

    def test_a_reloaded_moneyline_wager_is_not_locked_twice(self):
        import tempfile

        wagers, _ = screen(_card(market="h2h", point=float("nan")), now=NOW)
        with tempfile.NamedTemporaryFile(suffix=".csv") as handle:
            pd.DataFrame(wagers).to_csv(handle.name, index=False)
            reloaded = pd.read_csv(handle.name)
        open_ids = set(reloaded["wager_id"].astype(str))
        again, rejections = screen(_card(market="h2h", point=""),
                                   open_ids=open_ids, now=NOW)
        self.assertEqual(again, [])
        self.assertIn("already_locked", _gates(rejections))


class RoundTripTests(unittest.TestCase):
    """An empty CSV column comes back as NaN, and NaN is truthy.

    Tested through an actual save and load, because in memory the ledger holds
    empty strings and the bug cannot appear: settlement skipped every open
    wager forever and the summary called them settled with no result.
    """

    def test_open_wagers_survive_a_save_and_load(self):
        import tempfile

        wagers, _ = screen(_card(), now=NOW)
        frame = pd.DataFrame(wagers)
        with tempfile.NamedTemporaryFile(suffix=".csv") as handle:
            frame.to_csv(handle.name, index=False)
            reloaded = pd.read_csv(handle.name)
        self.assertTrue(pd.isna(reloaded.iloc[0]["outcome"]))
        self.assertEqual(summarise(reloaded)["open"], 1)
        self.assertEqual(summarise(reloaded)["settled"], 0)

    def test_a_reloaded_open_wager_still_settles(self):
        import tempfile

        wagers, _ = screen(_card(), now=NOW)
        frame = pd.DataFrame(wagers)
        with tempfile.NamedTemporaryFile(suffix=".csv") as handle:
            frame.to_csv(handle.name, index=False)
            reloaded = pd.read_csv(handle.name)
        games = pd.DataFrame([{"game_pk": 1, "home_score": 5.0,
                               "away_score": 3.0}])
        settled_ledger, count = settle(reloaded, games)
        self.assertEqual(count, 1)
        self.assertEqual(settled_ledger.iloc[0]["outcome"], "win")
        self.assertGreater(float(settled_ledger.iloc[0]["profit"]), 0)


class PayoutTests(unittest.TestCase):
    def test_underdog_and_favourite_payouts(self):
        self.assertAlmostEqual(payout(150), 1.5)
        self.assertAlmostEqual(payout(-200), 0.5)

    def test_summary_reports_roi_over_settled_wagers_only(self):
        ledger = pd.DataFrame([
            {"market": "h2h", "stake": 1.0, "outcome": "win", "profit": 0.9},
            {"market": "h2h", "stake": 1.0, "outcome": "loss", "profit": -1.0},
            {"market": "h2h", "stake": 1.0, "outcome": "", "profit": ""},
        ])
        report = summarise(ledger)
        self.assertEqual(report["settled"], 2)
        self.assertEqual(report["open"], 1)
        self.assertAlmostEqual(report["roi"], -0.05)


if __name__ == "__main__":
    unittest.main()
