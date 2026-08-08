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
            # Starter ids, so the boxscore fold has something to key on. Real
            # data agrees with the scheduled probable 99.8% of the time; the
            # rest are late scratches, which the live path cannot know either.
            "home_sp_id": 900 + home, "away_sp_id": 900 + away,
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


def _pitching(games, seed=1):
    """Boxscore lines for the synthetic season: one starter plus two relievers."""
    rng = np.random.default_rng(seed)
    rows = []
    for game in games.to_dict("records"):
        for team, runs in ((game["home_team_id"], game["away_score"]),
                           (game["away_team_id"], game["home_score"])):
            rows.append({
                "game_pk": game["game_pk"], "official_date": game["official_date"],
                "team_id": team, "player_id": 900 + team, "is_starter": 1,
                "outs": 18, "batters_faced": 24, "hits": 5,
                "runs": max(runs - 1, 0), "earned_runs": max(runs - 1, 0),
                "home_runs": int(rng.integers(0, 3)), "walks": int(rng.integers(0, 4)),
                "hit_batsmen": 0, "strike_outs": int(rng.integers(2, 11)),
                "pitches": 90, "strikes": 60,
            })
            rows.append({
                "game_pk": game["game_pk"], "official_date": game["official_date"],
                "team_id": team, "player_id": 800 + team, "is_starter": 0,
                "outs": 9, "batters_faced": 12, "hits": 2, "runs": 1,
                "earned_runs": 1, "home_runs": 0, "walks": 1, "hit_batsmen": 0,
                "strike_outs": 3, "pitches": 40, "strikes": 25,
            })
    return pd.DataFrame(rows)


class PitchingLookaheadTests(unittest.TestCase):
    """A starter's own line must not inform the game it was thrown in.

    The boxscore fold is the newest state in the builder and the easiest place
    to reintroduce lookahead, because unlike a score it is tempting to attach
    a pitching line to the game it describes.
    """

    def test_changing_a_pitching_line_cannot_move_its_own_or_earlier_rows(self):
        games = _games()
        pitching = _pitching(games)
        base = build(games, PARKS, None, pitching)

        target = games["game_pk"].iloc[30]
        tampered = pitching.copy()
        mask = (tampered["game_pk"] == target) & (tampered["is_starter"] == 1)
        tampered.loc[mask, "strike_outs"] = 20
        tampered.loc[mask, "home_runs"] = 9
        after = build(games, PARKS, None, tampered)

        upto = base["game_pk"] <= target
        pd.testing.assert_frame_equal(base[upto], after[upto])
        # and later rows must move, or the fold is doing nothing at all
        self.assertFalse(base[~upto].equals(after[~upto]))

    def test_pitching_is_optional_and_falls_back_to_league_priors(self):
        games = _games()
        frame = build(games, PARKS, None, None)
        for column in ("home_sp_k_rate", "home_bp_rate", "home_bp_workload"):
            self.assertIn(column, frame.columns)
        self.assertTrue(frame["home_sp_k_rate"].notna().all())
        # With no boxscores every starter looks league-average.
        self.assertEqual(frame["home_sp_k_rate"].nunique(), 1)

    def test_boxscores_actually_separate_pitchers(self):
        games = _games()
        frame = build(games, PARKS, None, _pitching(games))
        self.assertGreater(frame["home_sp_k_rate"].nunique(), 1)
        self.assertGreater(frame["home_bp_workload"].max(), 0)


def _umpires(games, alternating=True):
    """Two umpires alternating, so a tendency can accumulate for each."""
    return pd.DataFrame([
        {"game_pk": g["game_pk"], "official_date": g["official_date"],
         "hp_umpire_id": 700 + (index % 2 if alternating else 0),
         "hp_umpire_name": "Test Ump"}
        for index, g in enumerate(games.to_dict("records"))])


class UmpireLookaheadTests(unittest.TestCase):
    """An umpire's own game must not inform the row for that game."""

    def test_changing_a_games_runs_cannot_move_its_own_umpire_feature(self):
        games = _games()
        umpires = _umpires(games)
        base = build(games, PARKS, None, None, umpires)

        target_index = 30
        target = games["game_pk"].iloc[target_index]
        tampered = games.copy()
        tampered.loc[target_index, "home_score"] = 30
        tampered["home_win"] = (tampered.home_score > tampered.away_score).astype(int)
        tampered["total_runs"] = tampered.home_score + tampered.away_score
        after = build(tampered, PARKS, None, None, umpires)

        upto = base["game_pk"] <= target
        pd.testing.assert_series_equal(base[upto].ump_run_rate,
                                       after[upto].ump_run_rate)
        self.assertFalse(base[~upto].ump_run_rate
                         .equals(after[~upto].ump_run_rate))

    def test_umpires_are_optional_and_default_to_league_average(self):
        frame = build(_games(), PARKS, None, None, None)
        for column in ("ump_run_rate", "ump_k_rate"):
            self.assertIn(column, frame.columns)
            # With no assignments every game shares one unknown official, so
            # the deviation from the league is a constant.
            self.assertLessEqual(frame[column].std(), 1e-9)

    def test_two_umpires_separate_once_they_have_a_record(self):
        games = _games(n=120)
        frame = build(games, PARKS, None, None, _umpires(games))
        self.assertGreater(frame.ump_run_rate.tail(40).std(), 0.0)

    def test_the_feature_is_a_deviation_not_a_level(self):
        """Centred on the league, so it cannot smuggle in the era instead."""
        games = _games(n=120)
        frame = build(games, PARKS, None, None, _umpires(games))
        self.assertLess(abs(float(frame.ump_run_rate.mean())), 1.0)
