import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


# memory：门控后的好回合，喂回图做多轮上下文；display：完整对话，给前端展示。
@dataclass
class SessionEntry:
    memory: list[dict[str, Any]] = field(default_factory=list)
    display: list[dict[str, Any]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.monotonic)


# TTL 默认 7 天：多对话场景下别让会话一小时就被淘汰；仍是内存，重启会丢。
class SessionStore:
    # TTL 默认 7 天：多对话场景下别让会话一小时就被淘汰；仍是内存，重启会丢。
    def __init__(self, max_sessions: int = 1024, ttl_seconds: int = 604800):
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._lock = RLock()
        self._entries: dict[tuple[str, str], SessionEntry] = {}

    # 记录结果。
    def record(
        self,
        doc_id: str,
        session_id: str | None,
        memory_messages: list[dict[str, Any]],
        display_messages: list[dict[str, Any]],
    ) -> None:
        # 记忆与展示分开累积：记忆可能为空（答案被门控），展示只要有问答就留。
        if not session_id or (not memory_messages and not display_messages):
            return
        with self._lock:
            self._purge_expired_locked()
            entry = self._entries.setdefault((doc_id, session_id), SessionEntry())
            entry.memory.extend(memory_messages or [])
            entry.display.extend(display_messages or [])
            entry.updated_at = time.monotonic()
            self._evict_overflow_locked()

    # 获取 history。
    def get_history(self, doc_id: str, session_id: str | None) -> list[dict[str, Any]]:
        # 图输入：只取门控后的记忆回合。
        return self._read(doc_id, session_id, "memory")

    # 获取 display。
    def get_display(self, doc_id: str, session_id: str | None) -> list[dict[str, Any]]:
        # 前端展示：取完整对话。
        return self._read(doc_id, session_id, "display")

    # 读取。
    def _read(self, doc_id: str, session_id: str | None, field_name: str) -> list:
        if not session_id:
            return []
        with self._lock:
            self._purge_expired_locked()
            entry = self._entries.get((doc_id, session_id))
            if entry is None:
                return []
            entry.updated_at = time.monotonic()
            return list(getattr(entry, field_name))

    # 清理。
    def clear(self, doc_id: str, session_id: str | None) -> None:
        # 删除某会话的全部历史。
        if not session_id:
            return
        with self._lock:
            self._entries.pop((doc_id, session_id), None)

    # 清理 kb。
    def clear_kb(self, doc_id: str) -> None:
        # 删库时连带清掉该 KB 下所有会话，避免同名新库复用 doc_id 后捡到旧历史。
        with self._lock:
            for key in [k for k in self._entries if k[0] == doc_id]:
                self._entries.pop(key, None)

    # 列出 sessions。
    def list_sessions(self, doc_id: str) -> list[dict[str, Any]]:
        # 列出某库下的会话，title 取展示历史里首条用户消息，按最近活跃排序。
        with self._lock:
            self._purge_expired_locked()
            sessions = []
            for (entry_doc, session_id), entry in self._entries.items():
                if entry_doc != doc_id:
                    continue
                title = next(
                    (
                        t.get("content", "")
                        for t in entry.display
                        if t.get("role") == "user"
                    ),
                    "",
                )
                sessions.append(
                    {
                        "session_id": session_id,
                        "title": (title.strip()[:40] or "新对话"),
                        "message_count": len(entry.display),
                        "_updated_at": entry.updated_at,
                    }
                )
            sessions.sort(key=lambda s: s["_updated_at"], reverse=True)
            for s in sessions:
                s.pop("_updated_at")
            return sessions

    # 统计已记录回答数。
    def answer_count(self, doc_id: str) -> int:
        with self._lock:
            self._purge_expired_locked()
            return sum(
                1
                for (entry_doc, _session_id), entry in self._entries.items()
                if entry_doc == doc_id
                for message in entry.display
                if message.get("role") == "assistant"
            )

    # 清理 expired locked。
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

    # 淘汰溢出记录locked。
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
