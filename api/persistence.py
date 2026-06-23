import json
import os
import sqlite3
import time
from threading import RLock
from typing import Any


def _connect(db_path: str) -> sqlite3.Connection:
    # 单连接跨线程复用：WAL 提升并发读写、busy_timeout 等锁而非立刻报错；外层用 RLock 串行化。
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


class SqliteSessionStore:
    # SessionStore 的落盘版：接口与内存版一致，但 updated_at 用墙钟，重启后 TTL 仍有效。
    def __init__(
        self,
        db_path: str,
        max_sessions: int = 1024,
        ttl_seconds: int = 604800,
    ):
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._lock = RLock()
        self._conn = _connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "doc_id TEXT, session_id TEXT, memory TEXT, display TEXT, "
            "updated_at REAL, PRIMARY KEY (doc_id, session_id))"
        )
        self._conn.commit()

    def record(
        self,
        doc_id: str,
        session_id: str | None,
        memory_messages: list[dict[str, Any]],
        display_messages: list[dict[str, Any]],
    ) -> None:
        # 读改写追加：记忆可能为空（答案被门控），展示只要有问答就留。
        if not session_id or (not memory_messages and not display_messages):
            return
        with self._lock:
            self._purge_expired_locked()
            row = self._conn.execute(
                "SELECT memory, display FROM sessions WHERE doc_id=? AND session_id=?",
                (doc_id, session_id),
            ).fetchone()
            memory = json.loads(row[0]) if row else []
            display = json.loads(row[1]) if row else []
            memory.extend(memory_messages or [])
            display.extend(display_messages or [])
            self._conn.execute(
                "INSERT INTO sessions (doc_id, session_id, memory, display, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(doc_id, session_id) DO UPDATE SET "
                "memory=excluded.memory, display=excluded.display, "
                "updated_at=excluded.updated_at",
                (
                    doc_id,
                    session_id,
                    json.dumps(memory, ensure_ascii=False),
                    json.dumps(display, ensure_ascii=False),
                    time.time(),
                ),
            )
            self._evict_overflow_locked()
            self._conn.commit()

    def get_history(self, doc_id: str, session_id: str | None) -> list[dict[str, Any]]:
        # 图输入：只取门控后的记忆回合。
        return self._read(doc_id, session_id, "memory")

    def get_display(self, doc_id: str, session_id: str | None) -> list[dict[str, Any]]:
        # 前端展示：取完整对话。
        return self._read(doc_id, session_id, "display")

    def _read(self, doc_id: str, session_id: str | None, column: str) -> list:
        if not session_id:
            return []
        with self._lock:
            self._purge_expired_locked()
            row = self._conn.execute(
                f"SELECT {column} FROM sessions WHERE doc_id=? AND session_id=?",
                (doc_id, session_id),
            ).fetchone()
            if row is None:
                return []
            self._conn.execute(
                "UPDATE sessions SET updated_at=? WHERE doc_id=? AND session_id=?",
                (time.time(), doc_id, session_id),
            )
            self._conn.commit()
            return json.loads(row[0])

    def clear(self, doc_id: str, session_id: str | None) -> None:
        # 删除某会话的全部历史。
        if not session_id:
            return
        with self._lock:
            self._conn.execute(
                "DELETE FROM sessions WHERE doc_id=? AND session_id=?",
                (doc_id, session_id),
            )
            self._conn.commit()

    def list_sessions(self, doc_id: str) -> list[dict[str, Any]]:
        # 列出某库下的会话，title 取展示历史里首条用户消息，按最近活跃排序。
        with self._lock:
            self._purge_expired_locked()
            rows = self._conn.execute(
                "SELECT session_id, display FROM sessions WHERE doc_id=? "
                "ORDER BY updated_at DESC",
                (doc_id,),
            ).fetchall()
            sessions = []
            for session_id, display_json in rows:
                display = json.loads(display_json)
                title = next(
                    (t.get("content", "") for t in display if t.get("role") == "user"),
                    "",
                )
                sessions.append(
                    {
                        "session_id": session_id,
                        "title": (title.strip()[:40] or "新对话"),
                        "message_count": len(display),
                    }
                )
            return sessions

    def _purge_expired_locked(self) -> None:
        if self.ttl_seconds <= 0:
            return
        self._conn.execute(
            "DELETE FROM sessions WHERE updated_at < ?",
            (time.time() - self.ttl_seconds,),
        )

    def _evict_overflow_locked(self) -> None:
        # 超出上限按最旧活跃淘汰，和内存版语义一致。
        count = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        overflow = count - self.max_sessions
        if overflow <= 0:
            return
        self._conn.execute(
            "DELETE FROM sessions WHERE rowid IN ("
            "SELECT rowid FROM sessions ORDER BY updated_at ASC LIMIT ?)",
            (overflow,),
        )


# 非终态：进程重启时这些任务的线程已没了，必须协调为失败，避免前端永远轮询 pending。
_NON_TERMINAL_STATUS = ("pending", "running")


class SqliteJobStore:
    # 入库任务记录的落盘版：整条 record 存 JSON，status 单列出来便于孤儿协调。
    def __init__(self, db_path: str):
        self._lock = RLock()
        self._conn = _connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS index_jobs ("
            "job_id TEXT PRIMARY KEY, status TEXT, data TEXT)"
        )
        self._conn.commit()
        self.reconcile_orphans()

    def create(self, record: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO index_jobs (job_id, status, data) VALUES (?, ?, ?)",
                (
                    record["job_id"],
                    record["status"],
                    json.dumps(record, ensure_ascii=False),
                ),
            )
            self._conn.commit()

    def update(self, job_id: str, **fields: Any) -> None:
        # 读改写整条记录，status 列同步更新。
        with self._lock:
            record = self._get_locked(job_id)
            if record is None:
                return
            record.update(fields)
            self._conn.execute(
                "UPDATE index_jobs SET status=?, data=? WHERE job_id=?",
                (record["status"], json.dumps(record, ensure_ascii=False), job_id),
            )
            self._conn.commit()

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            return self._get_locked(job_id)

    def _get_locked(self, job_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT data FROM index_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def reconcile_orphans(self) -> None:
        # 启动时把上次进程残留的 pending/running 任务标记为失败：线程不可能复活。
        with self._lock:
            rows = self._conn.execute(
                "SELECT job_id FROM index_jobs WHERE status IN (?, ?)",
                _NON_TERMINAL_STATUS,
            ).fetchall()
            for (job_id,) in rows:
                self.update(
                    job_id,
                    status="failed",
                    error_code="INGEST_FAILED",
                    message="服务重启，任务中断",
                )


class InMemoryJobStore:
    # IndexJobManager 默认记录存储：纯内存 dict，保持原有非持久行为，便于测试隔离。
    def __init__(self):
        self._lock = RLock()
        self._jobs: dict[str, dict] = {}

    def create(self, record: dict) -> None:
        with self._lock:
            self._jobs[record["job_id"]] = dict(record)

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return dict(record) if record else None

    def reconcile_orphans(self) -> None:
        # 内存版启动即空，无孤儿可协调。
        return
