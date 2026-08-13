"""Small helpers for CSV datasets split into deterministic part files."""

from pathlib import Path

import pandas as pd


def csv_parts(path):
    """Return one CSV or every CSV directly inside a dataset directory."""
    target = Path(path)
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(item for item in target.glob("*.csv") if item.is_file())
    return []


def read_csv_collection(path):
    """Read a single CSV or a sharded directory as one logical frame."""
    parts = csv_parts(path)
    if not parts:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(part) for part in parts), ignore_index=True)


def yearly_part(path, timestamp, prefix="quotes"):
    """Choose a stable yearly part; preserve explicit CSV paths for tests."""
    target = Path(path)
    if target.suffix.lower() == ".csv":
        return target
    year = str(timestamp)[:4]
    if len(year) != 4 or not year.isdigit():
        raise ValueError(f"cannot derive quote year from {timestamp!r}")
    return target / f"{prefix}_{year}.csv"
