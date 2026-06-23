import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass
class SessionEntry:
    history: list[dict[str, Any]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.monotonic)


class SessionStore:
    def __init__(self, max_sessions: int = 1024, ttl_seconds: int = 3600):
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._lock = RLock()
        self._entries: dict[tuple[str, str], SessionEntry] = {}

    def get_history(self, doc_id: str, session_id: str | None) -> list[dict[str, Any]]:
        if not session_id:
            return []
        with self._lock:
            self._purge_expired_locked()
            entry = self._entries.get((doc_id, session_id))
            if entry is None:
                return []
            entry.updated_at = time.monotonic()
            return list(entry.history)

    def append_messages(
        self,
        doc_id: str,
        session_id: str | None,
        messages: list[dict[str, Any]],
    ) -> None:
        if not session_id or not messages:
            return
        with self._lock:
            self._purge_expired_locked()
            key = (doc_id, session_id)
            entry = self._entries.setdefault(key, SessionEntry())
            entry.history.extend(messages)
            entry.updated_at = time.monotonic()
            self._evict_overflow_locked()

    def _purge_expired_locked(self) -> None:
        if self.ttl_seconds <= 0:
            return
        now = time.monotonic()
        expired = [
            key
            for key, entry in self._entries.items()
            if now - entry.updated_at > self.ttl_seconds
        ]
        for key in expired:
            self._entries.pop(key, None)

    def _evict_overflow_locked(self) -> None:
        overflow = len(self._entries) - self.max_sessions
        if overflow <= 0:
            return
        oldest_keys = sorted(
            self._entries,
            key=lambda key: self._entries[key].updated_at,
        )[:overflow]
        for key in oldest_keys:
            self._entries.pop(key, None)
