import unittest

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS
from models import MIRROR_PAIRS, NEGATED, mirror


class MirrorTests(unittest.TestCase):
    """Training sees every game from both dugouts, so the swap must be total.

    A column that is home/away specific but missing from MIRROR_PAIRS silently
    attaches the wrong team's or pitcher's numbers to half the training rows.
    Nothing errors; the model just gets quietly worse.
    """

    def _row(self):
        values = {column: float(index + 1)
                  for index, column in enumerate(FEATURE_COLUMNS)}
        return pd.DataFrame([values])

    def test_every_home_column_has_an_away_partner(self):
        paired = {name for pair in MIRROR_PAIRS for name in pair}
        unpaired = []
        for column in FEATURE_COLUMNS:
            if column.startswith(("home_", "away_")) and column not in paired:
                unpaired.append(column)
            if (column.startswith(("expected_home", "expected_away"))
                    and column not in paired):
                unpaired.append(column)
        self.assertEqual(sorted(set(unpaired)), [])

    def test_mirroring_twice_returns_the_original(self):
        row = self._row()
        pd.testing.assert_frame_equal(
            mirror(mirror(row))[FEATURE_COLUMNS], row[FEATURE_COLUMNS]
        )

    def test_paired_columns_actually_swap(self):
        row = self._row()
        flipped = mirror(row)
        for left, right in MIRROR_PAIRS:
            self.assertEqual(flipped.iloc[0][left], row.iloc[0][right], left)
            self.assertEqual(flipped.iloc[0][right], row.iloc[0][left], right)

    def test_directional_columns_change_sign(self):
        row = self._row()
        flipped = mirror(row)
        for column in NEGATED:
            self.assertEqual(flipped.iloc[0][column], -row.iloc[0][column], column)

    def test_symmetric_columns_are_untouched(self):
        row = self._row()
        flipped = mirror(row)
        for column in ("park_factor", "temp_c", "air_density_index",
                       "wind_out_to_center_ms", "elevation_km", "roof_dome"):
            self.assertEqual(flipped.iloc[0][column], row.iloc[0][column], column)


if __name__ == "__main__":
    unittest.main()
