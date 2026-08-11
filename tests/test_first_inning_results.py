import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from first_inning_report import build_report
from first_inning_results import label_game, match_game, run


EVENT = {
    "event_id": "odds-1", "status": "offered",
    "commence_time": "2026-08-10T23:07:00Z",
    "home_team": "Toronto Blue Jays", "away_team": "Boston Red Sox",
}


def _game(home=0, away=1):
    return {
        "gamePk": 1, "officialDate": "2026-08-10",
        "gameDate": "2026-08-10T23:08:00Z",
        "status": {"abstractGameState": "Final"},
        "teams": {
            "home": {"team": {"name": "Toronto Blue Jays"}},
            "away": {"team": {"name": "Boston Red Sox"}},
        },
        "linescore": {"innings": [{"num": 1,
                                     "home": {"runs": home},
                                     "away": {"runs": away}}]},
    }


class FirstInningResultTests(unittest.TestCase):
    def test_first_inning_label_uses_the_actual_two_halves(self):
        label, status = label_game(_game(home=2, away=0))
        self.assertEqual(status, "final")
        self.assertEqual(label["first_inning_total"], 2)
        self.assertEqual(label["yrfi"], 1)
        self.assertEqual(label["nrfi"], 0)

    def test_unfinished_game_is_not_labelled_as_an_nrfi(self):
        game = _game()
        game["status"] = {"abstractGameState": "Live"}
        self.assertEqual(label_game(game), (None, "not_final"))

    def test_sides_and_start_time_disambiguate_the_game(self):
        event = {key: value for key, value in EVENT.items() if key != "event_id"}
        other = _game()
        other["gameDate"] = "2026-08-11T23:08:00Z"
        self.assertEqual(match_game(event, [other, _game()])["gamePk"], 1)

    def test_run_writes_a_settled_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit, output = root / "audit.csv", root / "results.csv"
            with audit.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=EVENT)
                writer.writeheader()
                writer.writerow(EVENT)
            def fetch(start, end):
                self.assertEqual(start, end)
                return [_game()]
            rows = run(audit, output, fetch=fetch)
            self.assertEqual(rows[0]["yrfi"], 1)
            self.assertEqual(rows[0]["game_pk"], 1)


class FirstInningMarketReportTests(unittest.TestCase):
    def test_report_is_a_market_baseline_not_a_model_claim(self):
        quotes = pd.DataFrame([
            {"event_id": "a", "market": "totals_1st_1_innings", "point": 0.5,
             "book_key": "a", "devig_prob_home": 0.45},
            {"event_id": "a", "market": "totals_1st_1_innings", "point": 0.5,
             "book_key": "b", "devig_prob_home": 0.55},
        ])
        results = pd.DataFrame([{"event_id": "a", "yrfi": 1,
                                 "result_status": "final"}])
        report = build_report(quotes, results)
        self.assertEqual(report["status"], "market_baseline_only")
        self.assertEqual(report["events"], 1)
        self.assertEqual(report["market_mean_yrfi_probability"], 0.5)
