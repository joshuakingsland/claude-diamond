import unittest

import numpy as np
import pandas as pd

from first_inning_open_evaluation import (
    BOOK_FEATURES,
    CONFIRMATION_YEAR,
    MICRO_FEATURES,
    OPEN_SAFE_CONTEXT,
    SELECTION_YEAR,
    TRAIN_YEAR,
    archive_coverage,
    build_open_snapshots,
    evaluate,
)


MARKET = "totals_1st_1_innings"


def _quote(event_id, fetched_at, book, probability, over=-105, under=-115):
    return {
        "event_id": event_id,
        "fetched_at": fetched_at,
        "market": MARKET,
        "point": 0.5,
        "book_key": book,
        "devig_prob_home": probability,
        "price_home": over,
        "price_away": under,
    }


def _audit(event_id, commence, requested, returned, status="offered"):
    return {
        "event_id": event_id,
        "commence_time": commence,
        "requested_snapshot": requested,
        "returned_snapshot": returned,
        "status": status,
    }


class OpeningSnapshotTests(unittest.TestCase):
    def test_earliest_qualified_rung_is_the_open(self):
        quotes = pd.DataFrame([
            _quote("game", "2025-06-01T00:00:00Z", "a", 0.47),
            _quote("game", "2025-06-01T00:00:00Z", "b", 0.49),
            _quote("game", "2025-06-01T23:00:00Z", "a", 0.52),
            _quote("game", "2025-06-01T23:00:00Z", "b", 0.54),
        ])
        audit = pd.DataFrame([
            _audit("game", "2025-06-02T00:00:00Z",
                   "2025-06-01T00:00:00Z", "2025-06-01T00:00:00Z"),
            _audit("game", "2025-06-02T00:00:00Z",
                   "2025-06-01T23:00:00Z", "2025-06-01T23:00:00Z"),
        ])
        opening, integrity = build_open_snapshots(
            quotes, audit, leads=(1440, 60))
        self.assertEqual(len(opening), 1)
        self.assertEqual(opening.iloc[0]["open_requested_lead_minutes"], 1440)
        self.assertAlmostEqual(opening.iloc[0]["open_prob_yrfi"], 0.48)
        self.assertEqual(integrity["qualified_open_events"], 1)

    def test_one_book_early_rung_falls_forward_to_first_broad_quote(self):
        quotes = pd.DataFrame([
            _quote("game", "2025-06-01T00:00:00Z", "a", 0.47),
            _quote("game", "2025-06-01T23:00:00Z", "a", 0.52),
            _quote("game", "2025-06-01T23:00:00Z", "b", 0.54),
        ])
        audit = pd.DataFrame([
            _audit("game", "2025-06-02T00:00:00Z",
                   "2025-06-01T00:00:00Z", "2025-06-01T00:00:00Z"),
            _audit("game", "2025-06-02T00:00:00Z",
                   "2025-06-01T23:00:00Z", "2025-06-01T23:00:00Z"),
        ])
        opening, _ = build_open_snapshots(quotes, audit, leads=(1440, 60))
        self.assertEqual(opening.iloc[0]["open_requested_lead_minutes"], 60)
        self.assertAlmostEqual(opening.iloc[0]["open_prob_yrfi"], 0.53)


class OpeningArchiveCoverageTests(unittest.TestCase):
    def test_no_offer_and_failed_rows_count_as_completed_attempts(self):
        close = pd.DataFrame([
            {"event_id": "a", "commence_time": "2023-06-02T00:00:00Z"},
        ])
        opened = pd.DataFrame([
            _audit("a", "2023-06-02T00:00:00Z",
                   "2023-06-01T00:00:00Z", "", "no_offer"),
            _audit("a", "2023-06-02T00:00:00Z",
                   "2023-06-01T23:00:00Z", "", "failed"),
        ])
        coverage = archive_coverage(opened, close, leads=(1440, 60))
        season = coverage[str(TRAIN_YEAR)]
        self.assertTrue(season["complete"])
        self.assertEqual(season["open_no_offer"], 1)
        self.assertEqual(season["open_failed"], 1)

    def test_missing_rung_keeps_archive_incomplete(self):
        close = pd.DataFrame([
            {"event_id": "a", "commence_time": "2023-06-02T00:00:00Z"},
        ])
        opened = pd.DataFrame([
            _audit("a", "2023-06-02T00:00:00Z",
                   "2023-06-01T00:00:00Z", "", "no_offer"),
        ])
        coverage = archive_coverage(opened, close, leads=(1440, 60))
        self.assertFalse(coverage[str(TRAIN_YEAR)]["complete"])
        self.assertEqual(coverage[str(TRAIN_YEAR)]["missing_event_rungs"], 1)

    def test_ineligible_close_events_do_not_create_paid_requirements(self):
        close = pd.DataFrame([
            {"event_id": "eligible",
             "commence_time": "2023-06-02T00:00:00Z"},
            {"event_id": "thin-close",
             "commence_time": "2023-06-03T00:00:00Z"},
        ])
        opened = pd.DataFrame([
            _audit("eligible", "2023-06-02T00:00:00Z",
                   "2023-06-01T00:00:00Z", "", "no_offer"),
        ])
        coverage = archive_coverage(
            opened, close, leads=(1440,), eligible_event_ids={"eligible"})
        season = coverage[str(TRAIN_YEAR)]
        self.assertTrue(season["complete"])
        self.assertEqual(season["closing_events"], 1)
        self.assertEqual(season["expected_open_attempts"], 1)


class OpeningEvaluationGateTests(unittest.TestCase):
    def test_confirmation_stays_sealed_until_development_archive_complete(self):
        rows = pd.DataFrame({
            "season": [TRAIN_YEAR, SELECTION_YEAR, CONFIRMATION_YEAR],
        })
        coverage = {
            str(TRAIN_YEAR): {"complete": True},
            str(SELECTION_YEAR): {"complete": False},
            str(CONFIRMATION_YEAR): {"complete": True},
        }
        report = evaluate(rows, coverage, {}, draws=10)
        self.assertEqual(
            report["status"], "awaiting_complete_2023_2024_opening_archive")
        self.assertNotIn("confirmation_2025", report)
        self.assertEqual(report["bets_placed"], 0)

    def test_development_signal_locks_before_incomplete_confirmation(self):
        rows = []
        for year in (TRAIN_YEAR, SELECTION_YEAR):
            for index in range(450):
                signal = -1.0 if index % 2 else 1.0
                open_probability = 0.5
                close_logit = 0.15 * signal
                close_probability = 1.0 / (1.0 + np.exp(-close_logit))
                row = {
                    "event_id": f"{year}-{index}",
                    "game_pk": year * 1000 + index,
                    "official_date": (
                        f"{year}-{5 + (index // 196):02d}-{index % 28 + 1:02d}"),
                    "season": year,
                    "yrfi": float(close_probability >= 0.5),
                    "open_prob_yrfi": open_probability,
                    "close_prob_yrfi": close_probability,
                    "open_logit": 0.0,
                    "close_logit": close_logit,
                    "move_logit": close_logit,
                    "best_price_yrfi": 110.0,
                    "best_price_nrfi": 110.0,
                    "best_book_yrfi": "betmgm",
                    "best_book_nrfi": "fanduel",
                }
                for feature in set(MICRO_FEATURES + BOOK_FEATURES
                                   + OPEN_SAFE_CONTEXT):
                    row.setdefault(feature, 0.0)
                row.update({
                    "open_books": 4.0,
                    "open_lead_hours": 24.0,
                    "book_betmgm_deviation": signal,
                    "book_betmgm_present": 1.0,
                    "book_betmgm_staleness_minutes": 1.0,
                })
                rows.append(row)
        coverage = {
            str(TRAIN_YEAR): {"complete": True},
            str(SELECTION_YEAR): {"complete": True},
            str(CONFIRMATION_YEAR): {"complete": False},
        }
        report = evaluate(pd.DataFrame(rows), coverage, {}, draws=20)
        self.assertTrue(report["development_signal"])
        self.assertEqual(
            report["status"], "candidate_locked_2025_price_confirmation_sealed")
        self.assertNotIn("confirmation_2025", report)


if __name__ == "__main__":
    unittest.main()
