"""In-memory context boundary for future native capture/provider adapters.

This module does not capture anything. Source identity, generation, protection
flags and timestamps must be attached by a trusted adapter, never by captured
text. Raw screen pixels are intentionally outside this prototype's contract.
"""
from dataclasses import dataclass
import math

KINDS = frozenset({"app_metadata", "window_text", "module_state", "file_excerpt"})


@dataclass(frozen=True)
class ContextGrant:
    source: str  # Exact adapter/session-bound app, window or chosen workspace.
    generation: int
    expires_at: float
    kinds: frozenset[str]
    allow_cloud: bool = False


@dataclass(frozen=True)
class Observation:
    source: str
    generation: int
    revision: int
    kind: str
    observed_at: float
    expires_at: float
    text: str
    protected: bool = False


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


class ContextBuffer:
    """A bounded, transient view with separate local and cloud eligibility."""
    def __init__(self):
        self.paused = False
        self._grants: dict[str, ContextGrant] = {}
        self._last_generation: dict[str, int] = {}
        self._observations: dict[tuple[str, str], Observation] = {}

    def grant(self, grant: ContextGrant):
        if (not isinstance(grant.source, str) or not 0 < len(grant.source) <= 256 or
                type(grant.generation) is not int or grant.generation < 0 or
                not finite_number(grant.expires_at) or
                not isinstance(grant.kinds, frozenset) or not grant.kinds or
                not grant.kinds <= KINDS or type(grant.allow_cloud) is not bool):
            raise ValueError("Invalid context grant")
        if grant.generation <= self._last_generation.get(grant.source, -1):
            raise ValueError("A changed scope requires a new generation")
        if grant.source not in self._last_generation and len(self._last_generation) >= 64:
            raise ValueError("Context source limit")
        self.revoke(grant.source)
        self._last_generation[grant.source] = grant.generation
        self._grants[grant.source] = grant

    def revoke(self, source: str):
        self._grants.pop(source, None)
        self._observations = {k: v for k, v in self._observations.items() if k[0] != source}

    def pause(self):
        self.paused = True
        self._observations.clear()
        self._grants.clear()

    def resume(self):
        # Native adapters must renew their session generations after restarting;
        # an in-flight result from before Pause cannot become fresh context.
        self.paused = False

    def _eligible(self, observation: Observation, now: float, *, cloud=False):
        grant = self._grants.get(observation.source)
        return (not self.paused and grant is not None and
                observation.generation == grant.generation and
                observation.kind in grant.kinds and
                observation.observed_at <= now < min(observation.expires_at, grant.expires_at) and
                (not cloud or grant.allow_cloud))

    def ingest(self, observation: Observation, *, now: float) -> bool:
        if (not isinstance(observation.source, str) or
                type(observation.generation) is not int or
                type(observation.revision) is not int or observation.revision < 0 or
                not isinstance(observation.kind, str) or observation.kind not in KINDS or
                not all(finite_number(t) for t in [observation.observed_at, observation.expires_at, now]) or
                type(observation.protected) is not bool or observation.protected or
                not isinstance(observation.text, str) or not observation.text or
                len(observation.text.encode("utf-8")) > 8192 or
                not self._eligible(observation, now)):
            return False
        self._purge(now)
        key = (observation.source, observation.kind)
        previous = self._observations.get(key)
        if previous and observation.revision <= previous.revision:
            return False
        if key not in self._observations and len(self._observations) >= 32:
            return False
        self._observations[key] = observation
        return True

    def _purge(self, now):
        self._observations = {k: v for k, v in self._observations.items() if self._eligible(v, now)}

    def snapshot(self, *, now: float, destination="local") -> list[dict]:
        if not finite_number(now) or destination not in {"local", "cloud"}:
            raise ValueError("Invalid context destination or clock")
        self._purge(now)
        return [
            {"source": item.source, "kind": item.kind, "revision": item.revision,
             "generation": item.generation, "observed_at": item.observed_at,
             "expires_at": min(item.expires_at, self._grants[item.source].expires_at),
             "text": item.text, "authority": "observed_data"}
            for item in self._observations.values()
            if self._eligible(item, now, cloud=destination == "cloud")
        ]
