import unittest

import numpy as np
import pandas as pd

from statcast import BARREL_CODE, PITCHER_FIELDS, aggregate, season_days


def _pitches():
    """Four pitches in one game: a called strike, a whiff, a barrel, a walk."""
    return pd.DataFrame([
        {"game_pk": 1, "game_date": "2026-05-01", "pitcher": 100, "batter": 200,
         "inning_topbot": "Top", "description": "called_strike", "events": None, "zone": 5,
         "bb_type": None, "launch_speed_angle": None, "launch_speed": None,
         "estimated_woba_using_speedangle": None, "woba_value": None,
         "woba_denom": None, "release_speed": 94.0,
         "release_spin_rate": 2400.0, "delta_run_exp": -0.04},
        {"game_pk": 1, "game_date": "2026-05-01", "pitcher": 100, "batter": 200,
         "inning_topbot": "Top", "description": "swinging_strike", "events": "strikeout", "zone": 14,
         "bb_type": None, "launch_speed_angle": None, "launch_speed": None,
         "estimated_woba_using_speedangle": 0.0, "woba_value": 0.0,
         "woba_denom": 1.0, "release_speed": 95.0,
         "release_spin_rate": 2500.0, "delta_run_exp": -0.2},
        {"game_pk": 1, "game_date": "2026-05-01", "pitcher": 100, "batter": 201,
         "inning_topbot": "Top", "description": "hit_into_play", "events": "home_run", "zone": 5,
         "bb_type": "fly_ball", "launch_speed_angle": BARREL_CODE,
         "launch_speed": 108.0, "estimated_woba_using_speedangle": 1.85,
         "woba_value": 2.0, "woba_denom": 1.0, "release_speed": 92.0,
         "release_spin_rate": 2200.0, "delta_run_exp": 1.4},
        {"game_pk": 1, "game_date": "2026-05-01", "pitcher": 100, "batter": 202,
         "inning_topbot": "Top", "description": "ball", "events": "walk", "zone": 13,
         "bb_type": None, "launch_speed_angle": None, "launch_speed": None,
         "estimated_woba_using_speedangle": 0.69, "woba_value": 0.69,
         "woba_denom": 1.0, "release_speed": 93.0,
         "release_spin_rate": 2300.0, "delta_run_exp": 0.3},
    ])


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.rows = aggregate(_pitches())
        self.pitcher = self.rows[self.rows.role == "pitcher"].iloc[0]

    def test_both_roles_are_produced(self):
        self.assertEqual(set(self.rows.role), {"pitcher", "batter"})

    def test_the_schema_is_stable(self):
        self.assertEqual(list(self.rows.columns), PITCHER_FIELDS)

    def test_every_pitch_is_counted_exactly_once_per_role(self):
        for role in ("pitcher", "batter"):
            block = self.rows[self.rows.role == role]
            self.assertEqual(block.pitches.sum(), 4)

    def test_swings_and_whiffs_are_distinguished(self):
        # A swinging strike and a ball-in-play are both swings; only the first
        # is a whiff. Getting this backwards would invert the best feature.
        self.assertEqual(self.pitcher.swings, 2)
        self.assertEqual(self.pitcher.whiffs, 1)

    def test_a_called_strike_is_not_a_swing(self):
        self.assertEqual(self.pitcher.called_strikes, 1)

    def test_zone_counts_only_the_nine_inner_zones(self):
        # Zones 11-14 are outside the strike zone; counting them would make
        # every pitcher look like a control artist.
        self.assertEqual(self.pitcher.in_zone, 2)

    def test_barrels_come_from_the_statcast_code(self):
        self.assertEqual(self.pitcher.barrels, 1)
        self.assertEqual(self.pitcher.batted_balls, 1)

    def test_plate_appearances_are_rows_carrying_an_event(self):
        self.assertEqual(self.pitcher.batters_faced, 3)

    def test_expected_woba_keeps_its_own_denominator(self):
        # xwOBA is undefined on most pitches, so its denominator is counted
        # from the rows that carry one rather than assumed to be the pitch
        # count or the plate-appearance count.
        self.assertEqual(self.pitcher.xwoba_denom, 3)
        self.assertAlmostEqual(self.pitcher.xwoba_sum, 0.0 + 1.85 + 0.69,
                               places=6)

    def test_sums_are_stored_not_rates(self):
        # features.py pools across games in date order; a rate stored here
        # would decide the window before the point-in-time walk gets to.
        self.assertIn("launch_speed_sum", PITCHER_FIELDS)
        self.assertIn("launch_speed_count", PITCHER_FIELDS)
        self.assertNotIn("launch_speed_mean", PITCHER_FIELDS)

    def test_an_empty_day_yields_an_empty_frame_with_the_schema(self):
        empty = aggregate(pd.DataFrame())
        self.assertEqual(len(empty), 0)
        self.assertEqual(list(empty.columns), PITCHER_FIELDS)

    def test_identifiers_stay_integers(self):
        # A float player_id is how the pitcher features were once inert:
        # 669373.0 never matches 669373.
        self.assertEqual(self.rows.player_id.dtype.kind, "i")
        self.assertEqual(self.rows.game_pk.dtype.kind, "i")


class SideAttributionTests(unittest.TestCase):
    """A batter row with no side is a row with no team."""

    def test_the_home_team_pitches_in_the_top_and_bats_in_the_bottom(self):
        rows = aggregate(_pitches())
        pitcher = rows[rows.role == "pitcher"].iloc[0]
        batter = rows[rows.role == "batter"].iloc[0]
        self.assertEqual(pitcher.is_home, 1.0)
        self.assertEqual(batter.is_home, 0.0)

    def test_a_feed_without_the_half_inning_stops_rather_than_guesses(self):
        # Defaulting would hand every batter to the away side: a wrong answer
        # that looks exactly like data.
        frame = _pitches().drop(columns=["inning_topbot"])
        with self.assertRaises(KeyError) as caught:
            aggregate(frame)
        self.assertIn("inning_topbot", str(caught.exception))


class SeasonDayTests(unittest.TestCase):
    def test_days_stop_at_today_for_the_current_season(self):
        import datetime as _dt
        days = list(season_days(2026, today=_dt.date(2026, 4, 10)))
        self.assertEqual(days[0], "2026-03-01")
        self.assertEqual(days[-1], "2026-04-10")

    def test_a_finished_season_runs_to_its_own_end(self):
        import datetime as _dt
        days = list(season_days(2024, today=_dt.date(2026, 4, 10)))
        self.assertEqual(days[-1], "2024-11-15")


if __name__ == "__main__":
    unittest.main()


class ResumeTests(unittest.TestCase):
    """A bounded run must reach today, not spin on the same off day."""

    def setUp(self):
        import tempfile
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_a_settled_off_day_is_never_refetched(self):
        from statcast import existing_days, record_empty
        record_empty(self.directory, "2021-03-01")
        self.assertIn("2021-03-01", existing_days(self.directory))

    def test_the_marker_file_is_not_read_as_data(self):
        # no_games.csv lives in the same directory as the shards; counting it
        # as a shard would put a game_date column with no rows into the frame.
        from csv_collection import read_csv_collection
        from statcast import record_empty
        record_empty(self.directory, "2021-03-01")
        frame = read_csv_collection(self.directory)
        self.assertNotIn("player_id", frame.columns)

    def test_a_bounded_run_advances_instead_of_spinning(self):
        """The bug this exists for.

        417 dates in 2021-2026 have no games. Left pending, a run capped at
        three dates refetched 2021-03-01 every time and never reached today.
        """
        import statcast
        calls = []

        def empty_fetch(day, game_type=statcast.GAME_TYPE):
            calls.append(day)
            return pd.DataFrame()

        original = statcast.fetch_day
        statcast.fetch_day = empty_fetch
        try:
            import datetime as _dt
            today = _dt.date(2021, 6, 1)
            for _ in range(3):
                statcast.run([2021], self.directory, limit=3, pause=0,
                             verbose=False, today=today)
        finally:
            statcast.fetch_day = original
        # Nine distinct dates attempted across three capped runs, not three
        # dates attempted three times.
        self.assertEqual(len(calls), 9)
        self.assertEqual(len(set(calls)), 9)

    def test_a_recent_empty_date_stays_pending(self):
        # Savant lags about a day, so an empty answer for yesterday means
        # "not published", not "no games".
        import statcast
        import datetime as _dt
        original = statcast.fetch_day
        statcast.fetch_day = lambda day, game_type=None: pd.DataFrame()
        try:
            today = _dt.date(2021, 6, 1)
            statcast.run([2021], self.directory, limit=200, pause=0,
                         verbose=False, today=today)
        finally:
            statcast.fetch_day = original
        settled = statcast.settled_empty(self.directory)
        self.assertIn("2021-03-01", settled)
        self.assertNotIn("2021-06-01", settled)
