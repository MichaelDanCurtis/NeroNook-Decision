"""Event-driven watch lifecycle, independent of model prediction and execution.

Events arrive from an eventual SDK adapter, not an ambient recorder. Host receipt
time and grant identity must be supplied by that adapter, not by module content.
"""
from dataclasses import dataclass, field
import math

@dataclass(frozen=True)
class Event:
    source: str
    subject: str
    revision: int
    kind: str
    received_at: float
    expires_at: float
    episode: str = "initial"

@dataclass
class Watch:
    id: str
    source: str
    subject: str
    granted_epoch: int
    kinds: frozenset[str] = frozenset({"completed", "failed", "needs_input"})
    last_revision: int = -1
    terminal_seen: set[tuple[str, str]] = field(default_factory=set)
    active: bool = True

class ContinualPolicy:
    """No model can override scope, freshness, pause, revocation or deduplication."""
    def __init__(self):
        self.paused = False
        self.watches: dict[str, Watch] = {}

    def add(self, watch: Watch):
        if not watch.id or len(self.watches) >= 64 or watch.id in self.watches:
            raise ValueError("Watch limit, duplicate or missing ID")
        self.watches[watch.id] = watch

    def observe(self, watch_id, event: Event, *, now: float, granted_epoch: int, quiet=False):
        watch = self.watches.get(watch_id)
        if self.paused or not watch or not watch.active:
            return "ignore"
        if granted_epoch != watch.granted_epoch:
            watch.active = False
            return "revoked"
        if (event.source, event.subject) != (watch.source, watch.subject):
            return "ignore"
        if not all(math.isfinite(t) for t in [event.received_at, event.expires_at, now]) or not event.received_at <= now < event.expires_at:
            return "stale"
        if type(event.revision) is not int or event.revision < 0 or event.revision <= watch.last_revision:
            return "duplicate"
        watch.last_revision = event.revision
        if event.kind not in watch.kinds:
            return "observe"
        identity = (event.kind, "terminal" if event.kind == "completed" else event.episode)
        if identity in watch.terminal_seen:
            return "duplicate"
        if len(watch.terminal_seen) >= 128:
            watch.active = False
            return "watch_limit"
        watch.terminal_seen.add(identity)
        # A quiet-mode event remains in the activity inbox; it is not spoken or replayed later.
        return "inbox" if quiet else "suggest_notification"
