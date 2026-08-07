"""Two bugs that hid each other.

Park lookups missed on every game because a venue id read through pandas
arrives as a float, so `elevation_km` was zero on all 14,580 feature rows.
That in turn concealed the units: StatsAPI reports elevation in feet under a
field this code calls `elevation_m`. Fixing the lookup alone would have handed
the model an altitude scale inflated by 3.28 — worse than the zero it
replaced, because it looks plausible.
"""

import unittest

from parks import _feet_to_metres, id_key


class VenueKeyTests(unittest.TestCase):
    def test_a_float_venue_id_matches_the_parks_key(self):
        """pandas types the column as float because two games have no venue."""
        self.assertEqual(id_key(3313.0), "3313")

    def test_an_integer_or_string_id_is_unchanged(self):
        self.assertEqual(id_key(3313), "3313")
        self.assertEqual(id_key("3313"), "3313")

    def test_a_real_park_is_found_through_the_key(self):
        parks = {"3313": {"name": "Yankee Stadium"}}
        self.assertIn(id_key(3313.0), parks)

    def test_an_id_that_only_looks_like_a_float_is_not_truncated(self):
        self.assertEqual(id_key("3313.05"), "3313.05")


class ElevationTests(unittest.TestCase):
    """Coors Field is 5,190 feet, which is 1,582 metres, not 5,190."""

    def test_coors_field_converts_to_metres(self):
        self.assertAlmostEqual(_feet_to_metres(5190), 1581.91, places=2)

    def test_a_sea_level_park_stays_near_zero(self):
        self.assertAlmostEqual(_feet_to_metres(21), 6.4, places=2)

    def test_a_missing_elevation_stays_missing(self):
        self.assertIsNone(_feet_to_metres(None))
        self.assertIsNone(_feet_to_metres(""))


if __name__ == "__main__":
    unittest.main()
