"""Merge this run's data files onto whatever landed while it was working.

Two capture runs can execute at once. GitHub fires scheduled workflows at
irregular times and sometimes bunches several together, and the obvious guard —
a concurrency group — does not queue them: only one run may be pending, so a
third arriving cancels the second outright. Two captures were lost that way
before this existed, and a lost capture cannot be re-bought at the live price.

Serialising was the wrong lever. What actually needed protecting was the
commit, because `git pull --rebase` conflicts whenever both runs touched the
same CSV, and a conflict fails the run with the snapshot already paid for.

So the runs are allowed to overlap and the files are merged by meaning
instead:

- **Append-only logs** are unioned on a key. Every quote, credit reading,
  wager and rejection from both runs survives. Taking one side wholesale
  would silently drop the other run's rows, which is the data loss the
  concurrency group was there to prevent.
- **Regenerated snapshots** — the board, the card, the page — are a picture of
  one moment, and the later picture simply replaces the earlier one. Merging
  them row by row would splice two different moments into a state that never
  existed.

Ours wins on a key collision, because ours is the newer read: a wager the
other run wrote open may have been settled in this one.
"""

import argparse
import csv
import io
import shutil
from pathlib import Path

# Append-only logs, and the columns that identify a row. A key must be stable
# across runs — anything derived from when the file was written would make
# every row look new and the union would grow without bound.
UNION_KEYS = {
    "data/credit_log.csv": ("fetched_at", "region"),
    "data/paper_ledger.csv": ("wager_id",),
    "data/paper_rejections.csv": ("screened_at", "game_pk", "market", "point",
                                  "side", "gate"),
    "data/historical_quotes.csv": ("snapshot_id",),
    "data/historical_manifest.csv": ("requested_date",),
    "data/full_game_event_audit.csv": ("audit_id",),
    "data/first_inning_quotes.csv": ("snapshot_id",),
    "data/first_inning_audit.csv": ("audit_id",),
    "data/first_inning_open_quotes.csv": ("snapshot_id",),
    "data/first_inning_open_audit.csv": ("audit_id",),
    "data/first_inning_results.csv": ("event_id",),
    "data/schedule_snapshots.csv": ("snapshot_id",),
    "data/lineup_snapshots.csv": ("snapshot_id",),
    "data/clv_signals.csv": ("wager_id",),
}

# Monthly quote logs are named by month, so they are matched by shape.
QUOTE_LOG_DIR = "data/market_quotes"
QUOTE_LOG_KEY = ("snapshot_id",)
FULL_GAME_QUOTE_DIR = "data/full_game_event_quotes"


def union_key(relative):
    if relative in UNION_KEYS:
        return UNION_KEYS[relative]
    if relative.startswith(QUOTE_LOG_DIR + "/") and relative.endswith(".csv"):
        return QUOTE_LOG_KEY
    if (relative.startswith(FULL_GAME_QUOTE_DIR + "/")
            and relative.endswith(".csv")):
        return QUOTE_LOG_KEY
    return None


def _read(path):
    path = Path(path)
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def merge_csv(ours_path, theirs_path, key):
    """Union two CSVs on ``key``. Ours wins a collision; their order leads."""
    their_fields, their_rows = _read(theirs_path)
    our_fields, our_rows = _read(ours_path)
    fields = their_fields or our_fields
    if not fields:
        return None, 0
    # A column added by a newer version of the writer must not be dropped.
    for name in our_fields:
        if name not in fields:
            fields.append(name)

    def identity(row):
        return tuple(str(row.get(column, "")) for column in key)

    merged, seen = [], {}
    for row in their_rows:
        seen[identity(row)] = len(merged)
        merged.append(row)
    added = 0
    for row in our_rows:
        marker = identity(row)
        if marker in seen:
            merged[seen[marker]] = row
        else:
            seen[marker] = len(merged)
            merged.append(row)
            added += 1
    return (fields, merged), added


def merge_tree(ours_root, targets, verbose=True):
    """Merge every file under ``ours_root`` into the working tree."""
    ours_root = Path(ours_root)
    unioned, replaced, unchanged = 0, 0, 0
    for source in sorted(ours_root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(ours_root).as_posix()
        if targets and not any(relative == t or relative.startswith(t + "/")
                               for t in targets):
            continue
        destination = Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        key = union_key(relative)
        if key is None:
            shutil.copy2(source, destination)
            replaced += 1
            continue
        result, added = merge_csv(source, destination, key)
        if result is None:
            shutil.copy2(source, destination)
            replaced += 1
            continue
        fields, rows = result
        # newline="" throughout, on the render, the comparison and the write.
        # The csv writer emits \r\n; universal-newline translation on either
        # side would rewrite every file forever by comparing \r\n against \n.
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        rendered = buffer.getvalue()
        # Only touch the file if the merge actually changed it. Every burst
        # commit used to rewrite all nineteen shards -- about 318 MB -- because
        # the write was unconditional. Git then stored a fresh blob for each,
        # and the repository grew roughly 50 MB a day for data nobody changed.
        try:
            if destination.exists():
                with destination.open(newline="", encoding="utf-8") as handle:
                    if handle.read() == rendered:
                        unchanged += 1
                        continue
        except (OSError, UnicodeDecodeError):
            pass
        with destination.open("w", newline="", encoding="utf-8") as handle:
            handle.write(rendered)
        unioned += 1
        if verbose and added:
            print(f"  union {relative}: +{added} row(s) from this run")
    if verbose:
        print(f"merged {unioned} append-only file(s), "
              f"replaced {replaced} regenerated file(s), "
              f"left {unchanged} unchanged")
    return unioned, replaced


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", required=True,
                        help="directory holding this run's generated files")
    parser.add_argument("--targets", nargs="*", default=["data", "docs"],
                        help="paths to merge, relative to the repository root")
    args = parser.parse_args(argv)
    merge_tree(args.ours, args.targets)


if __name__ == "__main__":
    main()
