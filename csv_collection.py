"""Small helpers for CSV datasets split into deterministic part files."""

import glob
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


# GitHub rejects any file of 100 MB outright, and warns from 50 MB. A shard is
# rolled well before either: the quote log grows about 94 MB a day under two
# bursts, so a monthly shard reached 138 MB and the push was refused after the
# capture had already been paid for. Rolling at 40 MB leaves room for a slate
# larger than any yet seen without a second thought.
MAX_SHARD_BYTES = 40 * 1024 * 1024


def dated_part(path, timestamp, prefix="quotes", max_bytes=MAX_SHARD_BYTES):
    """Choose a shard for this timestamp, rolling before it can grow too large.

    Named by day rather than by month, and rolled to `.p2`, `.p3` and so on
    once a day's shard passes ``max_bytes``. Two guarantees matter and neither
    is negotiable: no file ever approaches a size the remote will refuse, and
    the name is a pure function of the timestamp and what is already on disk,
    so two runs appending in the same second land in the same place and
    `merge_data.py` can union them.

    Explicit `.csv` paths pass through untouched, which keeps tests and
    one-off exports pointing where they were told to.
    """
    target = Path(path)
    if target.suffix.lower() == ".csv":
        return target
    day = str(timestamp)[:10]
    if len(day) != 10 or day[4] != "-" or day[7] != "-":
        raise ValueError(f"cannot derive a quote day from {timestamp!r}")
    shard = target / f"{prefix}_{day}.csv"
    part = 1
    while shard.exists() and shard.stat().st_size >= max_bytes:
        part += 1
        shard = target / f"{prefix}_{day}.p{part}.csv"
    return shard


def read_quote_shards(pattern="data/market_quotes/*.csv", key="snapshot_id"):
    """Read every quote shard as one frame, counting each quote once.

    Overlapping shards are not hypothetical. When the log was resharded from
    monthly to daily files, a workflow run that had checked out the older code
    pushed its monthly file back alongside the new daily ones, and for a few
    minutes every quote in the repository appeared twice. Nothing failed:
    `panel_books` still returned eleven books, the studies still ran, and every
    count they reported was double. That is the worst kind of breakage here —
    silent, plausible, and wrong.

    De-duplicating on `snapshot_id` costs a moment and makes the shard layout
    an implementation detail rather than something every reader must get right.
    """
    frames = [pd.read_csv(path) for path in sorted(glob.glob(str(pattern)))]
    if not frames:
        return pd.DataFrame()
    quotes = pd.concat(frames, ignore_index=True)
    if key in quotes.columns:
        quotes = quotes.drop_duplicates(subset=[key], keep="first")
        quotes = quotes.reset_index(drop=True)
    return quotes


def yearly_part(path, timestamp, prefix="quotes"):
    """Choose a stable yearly part; preserve explicit CSV paths for tests."""
    target = Path(path)
    if target.suffix.lower() == ".csv":
        return target
    year = str(timestamp)[:4]
    if len(year) != 4 or not year.isdigit():
        raise ValueError(f"cannot derive quote year from {timestamp!r}")
    return target / f"{prefix}_{year}.csv"
