"""Each Open-Meteo endpoint answers for a bounded window.

Requests are batched by venue over one date range, so a single game outside
the window returns 400 for that whole request and costs the venue its entire
series. That stayed invisible while the schedule held only completed seasons;
the moment the current season was ingested, every venue asked for an end date
in the future and every venue failed.
"""

import datetime
import unittest

from weather import (ARCHIVE_LAG_DAYS, build_for_games, game_weather,
                     servable_window)

TODAY = datetime.date(2026, 8, 5)


class WindowTests(unittest.TestCase):
    def test_the_archive_stops_short_of_today(self):
        low, high = servable_window(True, TODAY)
        self.assertIsNone(low)
        self.assertEqual(high, str(TODAY - datetime.timedelta(
            days=ARCHIVE_LAG_DAYS)))

    def test_the_forecast_covers_a_window_around_today(self):
        low, high = servable_window(False, TODAY)
        self.assertLess(low, str(TODAY))
        self.assertGreater(high, str(TODAY))

    def test_training_rows_are_labelled_historical_forecast(self):
        hourly = {"2026-07-01T18:00": {
            "temperature_2m": 20.0, "relative_humidity_2m": 50.0,
            "surface_pressure": 1010.0, "wind_speed_10m": 1.0,
            "wind_direction_10m": 90.0, "precipitation": 0.0,
        }}
        park = {"azimuth_angle": 45.0, "roof_type": "Open"}
        row = game_weather(1, "2026-07-01T18:00:00Z", park, hourly,
                           archive=True)
        self.assertEqual(row["weather_source"],
                         "open-meteo-historical-forecast")


class FilterTests(unittest.TestCase):
    """A future game must not drag its venue's whole request out of range."""

    def _games(self):
        return [
            {"game_pk": 1, "venue_id": "7", "official_date": "2026-07-01",
             "game_date_utc": "2026-07-01T18:00:00Z"},
            {"game_pk": 2, "venue_id": "7", "official_date": "2026-09-20",
             "game_date_utc": "2026-09-20T18:00:00Z"},
        ]

    def test_games_beyond_the_archive_are_dropped_before_the_request(self):
        calls = []

        def fake_fetch(latitude, longitude, start, end, archive=True,
                       timeout=60):
            calls.append((start, end))
            return {"2026-07-01T18:00": {"temperature_2m": 20.0,
                                         "wind_speed_10m": 1.0,
                                         "wind_direction_10m": 90.0,
                                         "relative_humidity_2m": 50,
                                         "surface_pressure": 1010.0,
                                         "precipitation": 0.0}}

        import weather
        original = weather.fetch_hourly
        weather.fetch_hourly = fake_fetch
        try:
            parks = {"7": {"name": "Test Park", "latitude": 39.0,
                           "longitude": -94.6, "azimuth_angle": 45.0,
                           "roof_type": "Open", "elevation_m": 200.0}}
            rows, failed = build_for_games(self._games(), parks,
                                           archive=True, verbose=False)
        finally:
            weather.fetch_hourly = original

        self.assertEqual(failed, [])
        self.assertEqual(len(calls), 1)
        # The September game must not appear in the requested range.
        self.assertLess(calls[0][1], "2026-09-20")
        self.assertEqual([row["game_pk"] for row in rows], [1])


if __name__ == "__main__":
    unittest.main()
