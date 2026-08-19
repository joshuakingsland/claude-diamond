"""Reproducible identifiers for model decisions.

The ledger used to record every materially different build as ``diamond-v0``.
That makes a forward test impossible to audit: a result cannot be tied back to
the code and feature contract that produced it.  The identifier here is stable
for a checkout and changes when either the revision or ordered feature schema
changes.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

from config import MODEL_FAMILY


def repository_revision():
    """Return the deployed revision, including a stable local-worktree hash.

    CI runs from a commit and therefore records the SHA directly. During local
    research, generated artifacts must not claim to come from the unchanged
    HEAD while the model code is modified. Only source/config/test files enter
    the worktree hash, so rewriting predictions does not recursively change
    their own version.
    """
    supplied = os.environ.get("GITHUB_SHA") or os.environ.get("MODEL_REVISION")
    if supplied:
        return supplied[:12]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=2,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            stderr=subprocess.DEVNULL, text=True, timeout=3,
        )
        relevant = []
        extensions = {".py", ".sh", ".yml", ".yaml", ".toml", ".txt", ".md"}
        for line in status.splitlines():
            path_text = line[3:].split(" -> ")[-1]
            path = Path(path_text)
            if path.suffix.lower() not in extensions:
                continue
            relevant.append(path)
        if relevant:
            digest = hashlib.sha256()
            for path in sorted(set(relevant), key=lambda item: item.as_posix()):
                digest.update(path.as_posix().encode())
                if path.exists() and path.is_file():
                    digest.update(path.read_bytes())
                else:
                    digest.update(b"<deleted>")
            return f"{revision}-wip{digest.hexdigest()[:8]}"
        return revision
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def feature_schema(columns):
    payload = "\n".join(str(column) for column in columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def model_version(kind, columns, revision=None):
    revision = revision or repository_revision()
    return (f"{MODEL_FAMILY}-{kind}-{revision}-"
            f"features-{feature_schema(columns)}")


def merge_report(path, fresh, indent=2):
    """Write a report without destroying blocks this run did not compute.

    A plain overwrite is wrong whenever some of a report's blocks come from an
    optional expensive pass. `stationarity.py` computes `walk_forward` only
    under `--walk-forward` and `mean_calibration.py` computes `corrections`
    only under `--fit`, and running either without its flag replaced a complete
    report with a partial one — silently, and with an exit code of zero. Both
    reports lost a block that way during an audit, which is how this was found.

    This is the same rule `results.merge_table` applies to the results table
    after a stale `--seasons` argument deleted a whole season: a fetch that did
    not run must not be able to delete what an earlier one produced.

    A carried block is marked rather than passed off as current, because a
    stale number presented as fresh is the failure this repository cares most
    about. Returns the written document.
    """
    path = Path(path)
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    document = dict(fresh)
    carried = [key for key in existing if key not in document]
    for key in carried:
        block = existing[key]
        if isinstance(block, dict):
            block = dict(block)
            block["carried_over"] = (
                "not recomputed by this run; rerun with the flag that "
                "produces it to refresh")
        document[key] = block
    if carried:
        document["carried_over_blocks"] = sorted(carried)
    path.write_text(json.dumps(document, indent=indent), encoding="utf-8")
    return document
