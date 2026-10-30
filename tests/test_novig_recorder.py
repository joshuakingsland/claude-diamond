import json
import tempfile
import unittest
from pathlib import Path

from novig.recorder import (MarketBook, Recorder, _summarize, analyze,
                            analyze_market, load_recordings)


def snapshot(seq=10):
    return {"seq": seq, "orders": {
        "a": [{"order": "o1", "price": "0.990", "qty": 500}],
        "b": [{"order": "o2", "price": "0.005", "qty": 300},
              {"order": "o3", "price": "0.002", "qty": 200},
              {"order": "o4", "price": "0.0", "qty": 0}],
    }}


class BookTests(unittest.TestCase):
    def setUp(self):
        self.book = MarketBook(["a", "b"])
        self.book.load_snapshot(snapshot())

    def test_summary_prices_each_side_from_the_other(self):
        top = self.book.summary()
        self.assertEqual(top["a"]["ask"], "0.995")
        self.assertEqual(top["a"]["bid"], "0.990")
        # 0.995 and 0.998 are in the band; ask 1.000 from a 0.0 bid is not.
        self.assertEqual(top["a"]["sweep_qty"], 500)
        self.assertEqual(top["a"]["sweep_cost"], "4.98")
        self.assertEqual(top["b"]["ask"], "0.010")
        self.assertEqual(top["b"]["sweep_qty"], 0)

    def test_deltas_apply_in_order(self):
        self.assertTrue(self.book.apply({"seq": 11, "deltas": [
            {"kind": "remove", "order": "o2", "reason": "fill"},
            {"kind": "add", "orderId": "o5", "outcome": "b",
             "price": "0.001", "qty": 50}]}))
        top = self.book.summary()
        self.assertEqual(top["a"]["ask"], "0.998")
        self.assertEqual(top["a"]["sweep_qty"], 250)

    def test_gap_is_reported_and_old_seq_ignored(self):
        self.assertTrue(self.book.apply({"seq": 9, "deltas": [
            {"kind": "remove", "order": "o1"}]}))
        self.assertIn("o1", self.book.orders)
        self.assertFalse(self.book.apply({"seq": 12, "deltas": []}))


class FakeWs:
    def __init__(self):
        self.sent = []

    def send(self, text):
        self.sent.append(json.loads(text))


class FakeClient:
    env = "paper"

    def markets(self, **_):
        return [{"marketId": "m1", "eventId": "e1", "description": "A v B",
                 "outcomes": [{"outcomeId": "a", "name": "A"},
                              {"outcomeId": "b", "name": "B"}]}]


class RecorderTests(unittest.TestCase):
    def test_records_tops_lifecycle_and_resyncs_on_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rec.jsonl"
            recorder = Recorder(FakeClient(), ["ATP"], "MONEY", out)
            ws = FakeWs()
            recorder.refresh(ws)
            self.assertEqual(ws.sent[0]["subscribe"],
                             {"markets": {"m1": "book"}})
            recorder.handle(ws, json.dumps({"ts": 1000, "snapshot": {"m1": {
                "book": snapshot(), "lifecycle": {"seq": 1,
                                                  "status": "OPEN"}}}}))
            recorder.handle(ws, json.dumps({"ts": 2000, "delta": {"m1": {
                "book": {"seq": 13, "deltas": []}}}}))
            self.assertEqual(ws.sent[-1]["snapshot"],
                             {"markets": {"m1": "book"}})
            recorder.handle(ws, json.dumps({"ts": 3000, "delta": {"m1": {
                "lifecycle": {"seq": 2, "deltas": [
                    {"kind": "END", "status": "OPEN"},
                    {"kind": "CLOSE", "status": "CLOSED"}]}}}}))
            self.assertEqual(ws.sent[-1]["unsubscribe"], ["market:m1"])
            kinds = [json.loads(line)["kind"]
                     for line in out.read_text().splitlines()]
            self.assertEqual(kinds, ["market", "lifecycle", "top", "gap",
                                     "lifecycle", "lifecycle"])


def recording(path):
    rows = [
        {"kind": "market", "m": "m1", "desc": "A v B",
         "outcomes": {"a": "A", "b": "B"}},
        {"kind": "top", "m": "m1", "t": 1000, "outcomes": {
            "a": {"ask": "0.992", "sweep_qty": 900, "sweep_cost": "8.93"},
            "b": {"ask": "0.020", "sweep_qty": 0, "sweep_cost": "0"}}},
        {"kind": "top", "m": "m1", "t": 3000, "outcomes": {
            "a": {"ask": "0.997", "sweep_qty": 400, "sweep_cost": "3.99"},
            "b": {"ask": "0.010", "sweep_qty": 0, "sweep_cost": "0"}}},
        {"kind": "lifecycle", "m": "m1", "t": 5000, "transition": "END"},
        {"kind": "top", "m": "m1", "t": 6000, "outcomes": {
            "a": {"ask": "0.998", "sweep_qty": 150, "sweep_cost": "1.50"},
            "b": {"ask": "0.010", "sweep_qty": 0, "sweep_cost": "0"}}},
        {"kind": "lifecycle", "m": "m1", "t": 9000, "transition": "CLOSE"},
    ]
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


class AnalyzeTests(unittest.TestCase):
    def test_window_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rec.jsonl"
            recording(path)
            market = load_recordings([path])["m1"]
        row = analyze_market(market, winner_id="a")
        self.assertEqual(row["jumped"], "A")
        self.assertTrue(row["jumped_side_won"])
        self.assertEqual(row["jump_to_end_s"], 2.0)
        self.assertEqual(row["end_to_close_s"], 4.0)
        self.assertEqual(row["sweep_after_jump"]["qty"], 400)
        self.assertEqual(row["sweep_after_end"]["qty"], 150)
        self.assertFalse(row["reverted_below_band"])

    def test_jumped_side_losing_is_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rec.jsonl"
            recording(path)

            class Grades:
                def call(self, *_):
                    return {"outcomes": [{"outcomeId": "a", "status": "LOSS"},
                                         {"outcomeId": "b", "status": "WIN"}]}

            summary = _summarize(analyze([path], Grades()))
        self.assertEqual(summary["jumped_side_lost"], 1)
        self.assertEqual(summary["median_end_to_close_s"], 4.0)


if __name__ == "__main__":
    unittest.main()
