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


# Hours per shard. Deliberately a clock division and not a size, for a reason
# worth stating plainly, because the size version looked obviously right and
# was not.
#
# The first attempt rolled to a new part once a shard passed 40 MB. It could
# not work. A shard is written on a runner, but what lands in the repository is
# `merge_data.py`'s union of every runner that committed in the meantime. Two
# bursts each wrote about 35 MB locally, neither crossed 40 MB, neither rolled,
# and the union committed at 70 MB. A local size check cannot bound a merged
# file — it cannot see the rows another job is writing at the same moment.
#
# Three hours is a pure function of the timestamp, so concurrent writers agree
# on the destination without observing anything. The bound is then the real
# traffic in those hours: the heaviest day yet, 21 August, carried 124 MB, so a
# three-hour block on such a day is about 15 MB. Even at triple that volume no
# shard approaches the 100 MB the remote refuses.
SHARD_HOURS = 3


def dated_part(path, timestamp, prefix="quotes", block_hours=SHARD_HOURS):
    """Choose a shard for this timestamp. Deterministic; never reads the disk.

    Named `quotes_2026-08-21T00.csv` for the block beginning at midnight, `T03`
    for the next, and so on. Two runners polling the same minute derive the same
    name from the timestamp alone, so their rows land in one file and
    `merge_data.py` unions them.

    Nothing here consults file sizes. That is the whole design: the previous
    version rolled on size, could only measure its own runner's copy, and so
    let a 70 MB file into the repository while believing the cap was 40.

    Explicit `.csv` paths pass through untouched, which keeps tests and
    one-off exports pointing where they were told to.
    """
    target = Path(path)
    if target.suffix.lower() == ".csv":
        return target
    stamp = str(timestamp)
    day = stamp[:10]
    if len(day) != 10 or day[4] != "-" or day[7] != "-":
        raise ValueError(f"cannot derive a quote day from {timestamp!r}")
    try:
        hour = int(stamp[11:13])
    except (ValueError, IndexError):
        raise ValueError(f"cannot derive a quote hour from {timestamp!r}")
    if not 0 <= hour <= 23:
        raise ValueError(f"hour out of range in {timestamp!r}")
    block = (hour // block_hours) * block_hours
    return target / f"{prefix}_{day}T{block:02d}.csv"


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
