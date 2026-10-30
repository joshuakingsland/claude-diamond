"""Where finished tennis matches come from.

The strategy is only as fast as this feed. Novig grades off its own data
provider; the window exists only when a result reaches us before Novig's
feed closes the market. Any source can drive the sniper by appending one
JSON object per line to a file:

    {"id": "atp-123", "player1": "Carlos Alcaraz", "player2": "Jannik Sinner",
     "winner": "Carlos Alcaraz", "status": "completed",
     "completed_at": "2026-10-30T18:04:11Z", "source": "my-feed"}

``status`` is ``completed`` for a match played out. Anything else
(``retired``, ``walkover``, ``abandoned``, ``suspended``) is recorded and
never traded: a retirement can grade at fair market value rather than as a
win, so a 0.995 contract can settle at whatever the book said before the
retirement.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone

COMPLETED = "completed"


@dataclass(frozen=True)
class MatchResult:
    id: str
    player1: str
    player2: str
    winner: str
    status: str
    completed_at: datetime
    source: str = ""

    @property
    def loser(self):
        return self.player2 if self.winner == self.player1 else self.player1


def _parse_time(value):
    if isinstance(value, (int, float)):
        # Unix seconds or milliseconds.
        seconds = value / 1000 if value > 1e11 else value
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"completed_at needs a timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def parse_result(record):
    """A ``MatchResult`` from one feed record, or ``ValueError``."""
    try:
        player1 = str(record["player1"]).strip()
        player2 = str(record["player2"]).strip()
        winner = str(record["winner"]).strip()
        status = str(record["status"]).strip().lower()
        completed_at = _parse_time(record["completed_at"])
    except KeyError as missing:
        raise ValueError(f"result is missing {missing}") from None
    if not player1 or not player2 or player1 == player2:
        raise ValueError("a result needs two distinct players")
    if winner not in (player1, player2):
        raise ValueError(f"winner {winner!r} is neither player")
    result_id = str(record.get("id") or
                    f"{player1}|{player2}|{completed_at.isoformat()}")
    return MatchResult(result_id, player1, player2, winner, status,
                       completed_at, str(record.get("source", "")))


class JsonlResultFeed:
    """Tails a JSONL file, yielding each new line once.

    Starts at the end of the file by default, so restarting the sniper never
    replays results it has already seen.
    """

    def __init__(self, path, from_start=False):
        self.path = path
        self.offset = None if not from_start else 0
        self._partial = b""

    def poll(self):
        """``(results, errors)`` appended since the last poll."""
        try:
            with open(self.path, "rb") as handle:
                if self.offset is None:
                    handle.seek(0, 2)
                    self.offset = handle.tell()
                    return [], []
                handle.seek(0, 2)
                if handle.tell() < self.offset:
                    # Truncated or rotated: start over from the top.
                    self.offset = 0
                    self._partial = b""
                handle.seek(self.offset)
                chunk = handle.read()
                self.offset = handle.tell()
        except FileNotFoundError:
            self.offset = self.offset or 0
            return [], []
        lines = (self._partial + chunk).split(b"\n")
        # A writer mid-line leaves an unterminated tail; keep it for later.
        self._partial = lines.pop()
        results, errors = [], []
        for line in lines:
            if not line.strip():
                continue
            try:
                results.append(parse_result(json.loads(line)))
            except (ValueError, TypeError) as exc:
                errors.append(f"{exc}: {line[:200]!r}")
        return results, errors
