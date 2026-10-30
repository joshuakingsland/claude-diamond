import json
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from novig.feeds import JsonlResultFeed, parse_result
from novig.sniper import Limits, Sniper, find_market, names_match, plan_order

NOW = datetime(2026, 10, 30, 18, 0, 0, tzinfo=timezone.utc).timestamp()


def market(market_id="m1", a="Carlos Alcaraz", b="Jannik Sinner",
           status="OPEN"):
    return {"marketId": market_id, "status": status, "marketType": "MONEY",
            "description": f"{a} vs {b}",
            "fee": {"coefficient": "0.06", "charged": "WHEN_LIVE"},
            "outcomes": [{"outcomeId": f"{market_id}-a", "name": a},
                         {"outcomeId": f"{market_id}-b", "name": b}]}


def result(winner="Alcaraz C.", loser="Sinner J.", status="completed",
           age_s=5, result_id="r1"):
    return parse_result({
        "id": result_id, "player1": winner, "player2": loser,
        "winner": winner, "status": status,
        "completed_at": (NOW - age_s) * 1000})


def book(loser_bids):
    return {"orders": {"m1-b": [{"price": p, "qty": q}
                                for p, q in loser_bids]}}


class NameTests(unittest.TestCase):
    def test_common_feed_spellings_match(self):
        for feed in ("Alcaraz C.", "C. Alcaraz", "Alcaraz, Carlos",
                     "carlos alcaraz", "ALCARAZ"):
            with self.subTest(feed):
                self.assertTrue(names_match(feed, "Carlos Alcaraz"))

    def test_accents_and_hyphens(self):
        self.assertTrue(names_match("Auger-Aliassime F.",
                                    "Félix Auger-Aliassime"))
        self.assertTrue(names_match("Monfils G.", "Gaël Monfils"))

    def test_same_surname_different_player_does_not_match(self):
        self.assertFalse(names_match("Mischa Zverev", "Alexander Zverev"))
        self.assertFalse(names_match("Zverev M.", "Alexander Zverev"))
        self.assertFalse(names_match("Williams V.", "Serena Williams"))

    def test_doubles_never_match(self):
        self.assertFalse(names_match("Ram/Salisbury", "Rajeev Ram"))


class FindMarketTests(unittest.TestCase):
    def test_finds_winner_and_loser_either_order(self):
        found, _ = find_market(result("Sinner J.", "Alcaraz C."), [market()])
        self.assertEqual(found[1]["name"], "Jannik Sinner")
        self.assertEqual(found[2]["name"], "Carlos Alcaraz")

    def test_closed_market_is_ignored(self):
        found, reason = find_market(result(), [market(status="CLOSED")])
        self.assertIsNone(found)
        self.assertIn("no open market", reason)

    def test_two_matching_markets_is_ambiguous(self):
        found, reason = find_market(result(), [market("m1"), market("m2")])
        self.assertIsNone(found)
        self.assertIn("ambiguous", reason)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.market = market()
        self.winner, self.loser = self.market["outcomes"]
        self.limits = Limits(min_expected_profit=Decimal("0"))

    def plan(self, bids, budget="50"):
        return plan_order(self.market, self.winner, self.loser, book(bids),
                          self.limits, Decimal(budget))

    def test_takes_levels_inside_band_only(self):
        # Loser bids 0.008 and 0.002 -> asks 0.992 and 0.998; 0.0005 is
        # not a valid tick and 0.000 would be 1.000, so cap at 0.999.
        plan, _ = self.plan([("0.008", 1000), ("0.002", 2000),
                             ("0.000", 5000)])
        self.assertEqual(plan.qty, 3000)
        self.assertEqual(plan.limit_price, Decimal("0.998"))
        self.assertEqual(plan.levels, [("0.992", 1000), ("0.998", 2000)])

    def test_ask_below_band_means_feed_disagrees(self):
        plan, reason = self.plan([("0.050", 1000)])
        self.assertIsNone(plan)
        self.assertIn("disagrees", reason)

    def test_profit_is_net_of_taker_fee(self):
        plan, _ = self.plan([("0.005", 10000)], budget="200")
        self.assertEqual(plan.qty, 10000)
        gross = Decimal("10000") * Decimal("0.005") * Decimal("0.01")
        fee = Decimal("0.06") * Decimal("0.995") * Decimal("0.005") * 100
        self.assertEqual(plan.fee, fee)
        self.assertEqual(plan.expected_profit, gross - fee)
        self.assertGreater(plan.expected_profit, 0)

    def test_budget_caps_quantity(self):
        plan, _ = self.plan([("0.005", 1_000_000)], budget="10")
        self.assertLessEqual(plan.cost + plan.fee, Decimal("10"))
        self.assertEqual(plan.qty, 1004)

    def test_no_bids(self):
        self.assertEqual(self.plan([])[1], "no resting bid on the loser")


class FakeClient:
    env = "paper"

    def __init__(self, markets, book_):
        self._markets = markets
        self._book = book_
        self.placed = []

    def markets(self, **_):
        return self._markets

    def book(self, _):
        return self._book

    def place_order(self, outcome_id, price, qty, tif, client_id):
        self.placed.append((outcome_id, price, qty, tif))
        return {"orderId": "o1", "clientId": client_id}


class SniperTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.log = Path(self.dir.name) / "log.jsonl"
        self.client = FakeClient([market()], book([("0.005", 2000)]))

    def tearDown(self):
        self.dir.cleanup()

    def sniper(self, execute=True):
        sniper = Sniper(self.client, None, Limits(), ["ATP"], "MONEY",
                        self.log, execute=execute, clock=lambda: NOW)
        sniper.refresh_catalog(force=True)
        return sniper

    def actions(self):
        return [json.loads(line)["action"]
                for line in self.log.read_text().splitlines()]

    def test_places_one_ioc_and_never_twice(self):
        sniper = self.sniper()
        sniper.handle(result())
        sniper.handle(result(result_id="r2"))
        self.assertEqual(self.client.placed,
                         [("m1-a", "0.995", 2000, "IOC")])
        self.assertEqual(self.actions(), ["placed", "skip"])

    def test_restart_remembers_fired_markets(self):
        self.sniper().handle(result())
        self.sniper().handle(result(result_id="r2"))
        self.assertEqual(len(self.client.placed), 1)

    def test_retirement_is_never_traded(self):
        self.sniper().handle(result(status="retired"))
        self.assertEqual(self.client.placed, [])

    def test_stale_result_is_never_traded(self):
        self.sniper().handle(result(age_s=600))
        self.assertEqual(self.client.placed, [])

    def test_dry_run_places_nothing(self):
        self.sniper(execute=False).handle(result())
        self.assertEqual(self.client.placed, [])
        self.assertEqual(self.actions(), ["would_place"])

    def test_daily_cap(self):
        sniper = self.sniper()
        sniper.spent_today = Decimal("500")
        sniper.handle(result())
        self.assertEqual(self.client.placed, [])


class FeedTests(unittest.TestCase):
    def test_tails_new_complete_lines_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feed.jsonl"
            path.write_text('{"old": true}\n')
            feed = JsonlResultFeed(path)
            self.assertEqual(feed.poll(), ([], []))
            line = json.dumps({"player1": "A B", "player2": "C D",
                               "winner": "A B", "status": "completed",
                               "completed_at": "2026-10-30T18:00:00Z"})
            with open(path, "a") as handle:
                handle.write(line + "\n" + line[:10])
            results, errors = feed.poll()
            self.assertEqual([r.winner for r in results], ["A B"])
            self.assertEqual(errors, [])

    def test_winner_must_be_a_player(self):
        with self.assertRaises(ValueError):
            parse_result({"player1": "A", "player2": "B", "winner": "C",
                          "status": "completed", "completed_at": 0})


if __name__ == "__main__":
    unittest.main()
