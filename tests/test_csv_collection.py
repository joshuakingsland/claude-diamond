import tempfile
import unittest
from pathlib import Path

import pandas as pd

from csv_collection import read_csv_collection, yearly_part


class CsvCollectionTests(unittest.TestCase):
    def test_reads_parts_in_one_logical_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([{"id": 1}]).to_csv(root / "quotes_2023.csv", index=False)
            pd.DataFrame([{"id": 2}]).to_csv(root / "quotes_2024.csv", index=False)
            self.assertEqual(read_csv_collection(root)["id"].tolist(), [1, 2])

    def test_yearly_part_preserves_an_explicit_test_file(self):
        target = Path("tmp/quotes.csv")
        self.assertEqual(yearly_part(target, "2024-05-01T20:00:00Z"), target)

    def test_yearly_part_uses_event_year(self):
        self.assertEqual(
            yearly_part("data/quotes", "2024-05-01T20:00:00Z"),
            Path("data/quotes/quotes_2024.csv"),
        )

    def test_yearly_part_can_separate_snapshot_roles(self):
        self.assertEqual(
            yearly_part("data/quotes", "2024-05-01T20:00:00Z",
                        prefix="quotes_early"),
            Path("data/quotes/quotes_early_2024.csv"),
        )


if __name__ == "__main__":
    unittest.main()
