import unittest

import numpy as np
import pandas as pd

import features
from features import FEATURE_COLUMNS, build
from parks import roof_category, wind_components


def _games(n=60, seed=0):
    """A small synthetic season with four teams and two parks."""
    rng = np.random.default_rng(seed)
    teams = [108, 111, 112, 113]
    rows = []
    for index in range(n):
        home, away = teams[index % 4], teams[(index // 4 + 1) % 4]
        if home == away:
            away = teams[(index // 4 + 2) % 4]
        date = pd.Timestamp("2024-04-01") + pd.Timedelta(days=index // 2)
        rows.append({
            "game_pk": 100000 + index,
            "official_date": date.strftime("%Y-%m-%d"),
            "game_date_utc": date.strftime("%Y-%m-%dT23:00:00Z"),
            "season": 2024,
            "home_team_id": home, "away_team_id": away,
            "home_score": int(rng.integers(0, 10)),
            "away_score": int(rng.integers(0, 10)),
            "venue_id": 1 if index % 2 else 19,
            "scheduled_innings": 9, "innings_played": 9,
        })
    frame = pd.DataFrame(rows)
    frame["home_win"] = (frame["home_score"] > frame["away_score"]).astype(int)
    frame["total_runs"] = frame["home_score"] + frame["away_score"]
    return frame


PARKS = {
    "1": {"name": "A", "latitude": 33.8, "longitude": -117.9, "elevation_m": 151,
          "azimuth_angle": 43.6, "roof_type": "Open"},
    "19": {"name": "B", "latitude": 39.7, "longitude": -105.0, "elevation_m": 1580,
           "azimuth_angle": 5.0, "roof_type": "Open"},
}


class NoLookaheadTests(unittest.TestCase):
    """The property the whole project rests on."""

    def test_changing_a_result_cannot_move_any_earlier_feature_row(self):
        games = _games()
        baseline = build(games, PARKS).sort_values("game_pk").reset_index(drop=True)

        # Rewrite one game's result into an extreme blowout. Every feature row
        # for a game at or before it must be byte-identical; only later rows
        # may respond. A lookahead bug shows up here immediately.
        target_index = 40
        tampered = games.copy()
        tampered.loc[target_index, "home_score"] = 30
        tampered.loc[target_index, "away_score"] = 0
        tampered.loc[target_index, "home_win"] = 1
        after = build(tampered, PARKS).sort_values("game_pk").reset_index(drop=True)

        target_pk = games.loc[target_index, "game_pk"]
        cutoff = baseline.index[baseline["game_pk"] == target_pk][0]
        before = baseline.loc[:cutoff, FEATURE_COLUMNS]
        pd.testing.assert_frame_equal(before, after.loc[:cutoff, FEATURE_COLUMNS])

        # And the tampering must actually have had an effect later on,
        # otherwise this test would pass on a builder that ignores results.
        tail = baseline.loc[cutoff + 1:, "home_elo"]
        self.assertFalse(np.allclose(tail, after.loc[cutoff + 1:, "home_elo"]))

    def test_first_game_of_a_team_carries_no_history(self):
        frame = build(_games(), PARKS).sort_values("game_pk")
        first = frame.iloc[0]
        self.assertEqual(first["home_games_played"], 0)
        self.assertEqual(first["away_games_played"], 0)
        self.assertAlmostEqual(first["home_off"], features.PRIOR_RUNS, places=6)
        self.assertAlmostEqual(first["home_elo"], features.ELO_START, places=6)

    def test_park_factor_starts_neutral_before_evidence_exists(self):
        frame = build(_games(), PARKS).sort_values("game_pk")
        self.assertAlmostEqual(frame.iloc[0]["park_factor"], 1.0, places=9)

    def test_every_declared_feature_column_is_produced(self):
        frame = build(_games(), PARKS)
        missing = [column for column in FEATURE_COLUMNS if column not in frame]
        self.assertEqual(missing, [])
        self.assertFalse(frame[FEATURE_COLUMNS].isna().any().any())


class WindGeometryTests(unittest.TestCase):
    """A sign error here silently inverts every wind effect in the model."""

    def test_wind_from_home_plate_blows_out_to_center(self):
        # Park faces due north; wind coming FROM the south blows out to centre.
        out, across = wind_components(10.0, 180.0, 0.0)
        self.assertAlmostEqual(out, 10.0, places=6)
        self.assertAlmostEqual(across, 0.0, places=6)

    def test_wind_from_center_blows_in(self):
        out, _ = wind_components(10.0, 0.0, 0.0)
        self.assertAlmostEqual(out, -10.0, places=6)

    def test_crosswind_has_no_out_to_center_component(self):
        out, across = wind_components(10.0, 270.0, 0.0)
        self.assertAlmostEqual(out, 0.0, places=6)
        self.assertAlmostEqual(across, 10.0, places=6)

    def test_park_orientation_rotates_the_answer(self):
        # Same meteorological wind, park rotated 90 degrees: what was blowing
        # straight out now blows across.
        out, _ = wind_components(10.0, 180.0, 90.0)
        self.assertAlmostEqual(out, 0.0, places=6)

    def test_missing_inputs_yield_no_components(self):
        self.assertEqual(wind_components(None, 180.0, 0.0), (None, None))
        self.assertEqual(wind_components(5.0, None, 0.0), (None, None))
        self.assertEqual(wind_components(5.0, 180.0, None), (None, None))


class RoofCategoryTests(unittest.TestCase):
    def test_only_forecastable_categories_are_produced(self):
        self.assertEqual(roof_category("Dome"), "dome")
        self.assertEqual(roof_category("Retractable"), "retractable")
        self.assertEqual(roof_category("Open"), "open")
        self.assertEqual(roof_category(None), "open")


if __name__ == "__main__":
    unittest.main()
