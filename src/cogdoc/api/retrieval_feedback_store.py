from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from threading import RLock
from typing import Any, TypeAlias
from uuid import uuid4

from cogdoc.agents.feedback_understanding import feedback_target_items
from cogdoc.config.settings import get_settings
from cogdoc.api.persistence import connect_sqlite


# 空文件修改时间标记。
_MISSING_MTIME = -1.0

_RetrievalFeedbackRows: TypeAlias = list[dict[str, Any]]
_TargetChunks: TypeAlias = list[dict[str, str]]


# 返回当前协调世界时时间字符串。
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 归一化查询文本。
def normalize_query_text(query: str) -> str:
    normalized = unicodedata.normalize("NFKC", query)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return "".join(ch.lower() if "A" <= ch <= "Z" else ch for ch in normalized)


# 计算查询哈希。
def query_hash(query: str) -> str:
    return hashlib.sha256(normalize_query_text(query).encode("utf-8")).hexdigest()


# 读取必填文本。
def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    return value.strip() if isinstance(value, str) else ""


# 反馈类型对应的手动权重。
def _feedback_weight(feedback: str, rating: int | None = None) -> tuple[int, float]:
    if rating is not None:
        score = int(rating) - 3
        return score, score * 0.12
    if feedback == "thumbs_up":
        return 1, 0.2
    if feedback == "thumbs_down":
        return -1, -0.35
    if feedback == "correction":
        return -2, -0.55
    return 0, 0.0


# 只把明确归因于检索的负反馈用于惩罚引用分块，避免错罚正确证据。
def _attributed_feedback_weight(payload: dict[str, Any]) -> tuple[int, float]:
    if payload.get("skip_retrieval_feedback") is True:
        return 0, 0.0
    user_score, weight_delta = _feedback_weight(
        str(payload.get("feedback") or ""), payload.get("rating")
    )
    if weight_delta < 0 and payload.get("feedback_type") != "bad_retrieval":
        return 0, 0.0
    return user_score, weight_delta


# 从反馈载荷抽取被评价的分块。
def _target_chunks(payload: dict[str, Any]) -> _TargetChunks:
    return [
        {
            "chunk_id": item["chunk_id"],
            "source_type": item["source_type"],
        }
        for item in feedback_target_items(payload)
    ]


# 读取调权记录的目标分块，兼容旧版单 chunk 记录。
def _record_targets(row: dict[str, Any]) -> _TargetChunks:
    raw_targets = row.get("target_chunks")
    targets = []
    seen = set()
    if isinstance(raw_targets, list):
        for item in raw_targets:
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id") or "")
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            targets.append(
                {
                    "chunk_id": chunk_id,
                    "source_type": str(item.get("source_type") or "document"),
                }
            )
    chunk_id = str(row.get("chunk_id") or "")
    if chunk_id and chunk_id not in seen:
        targets.append(
            {
                "chunk_id": chunk_id,
                "source_type": str(row.get("source_type") or "document"),
            }
        )
    return targets


# 同一次反馈只作为一条调权展示和统计。
def _feedback_group_key(row: dict[str, Any]) -> str:
    return str(row.get("feedback_id") or row.get("retrieval_feedback_id") or "")


def _aggregate_retrieval_feedback_group(
    rows: _RetrievalFeedbackRows,
) -> dict[str, Any]:
    latest = sorted(
        rows, key=lambda row: str(row.get("created_at") or ""), reverse=True
    )[0]
    targets = []
    seen = set()
    for row in rows:
        for target in _record_targets(row):
            chunk_id = target["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            targets.append(target)
    source_types = sorted({target["source_type"] for target in targets})
    enabled = any(row.get("enabled") is True for row in rows)
    disabled_rows = [row for row in rows if row.get("enabled") is False]
    aggregated = dict(latest)
    aggregated["enabled"] = enabled
    aggregated["target_chunks"] = targets
    aggregated["chunk_count"] = len(targets)
    aggregated["chunk_id"] = (
        targets[0]["chunk_id"] if targets else latest.get("chunk_id")
    )
    aggregated["source_type"] = (
        source_types[0]
        if len(source_types) == 1
        else ("mixed" if source_types else None)
    )
    if not enabled and disabled_rows:
        disabled_latest = sorted(
            disabled_rows,
            key=lambda row: str(row.get("disabled_at") or row.get("created_at") or ""),
            reverse=True,
        )[0]
        aggregated["disabled_at"] = disabled_latest.get("disabled_at")
        aggregated["disabled_by"] = disabled_latest.get("disabled_by")
        aggregated["disable_reason"] = disabled_latest.get("disable_reason")
    return aggregated


# 检索反馈存储，逐行追加并按标识折叠最新状态。
class RetrievalFeedbackStore:
    def __init__(self, path: str | None = None):
        self._path = path or get_settings().retrieval_feedback_path
        self._lock = RLock()
        self._cache_mtime: float | None = None
        self._cache_latest: dict[str, dict[str, Any]] | None = None
        os.makedirs(os.path.dirname(self._path), exist_ok=True)

    # 从用户反馈生成调权记录。
    def record_from_feedback(
        self, feedback_id: str, payload: dict[str, Any]
    ) -> _RetrievalFeedbackRows:
        kb_id = _required_text(payload, "kb_id")
        query_text = _required_text(payload, "query")
        if not kb_id or not query_text:
            return []
        user_score, weight_delta = _attributed_feedback_weight(payload)
        if weight_delta == 0:
            return []
        targets = _target_chunks(payload)
        if not targets:
            return []

        now = _now_iso()
        q_hash = query_hash(query_text)
        with self._lock:
            source_types = sorted({target["source_type"] for target in targets})
            record = {
                "retrieval_feedback_id": uuid4().hex,
                "feedback_id": feedback_id,
                "kb_id": kb_id,
                "query_hash": q_hash,
                "query_text": query_text,
                "chunk_id": targets[0]["chunk_id"],
                "source_type": (source_types[0] if len(source_types) == 1 else "mixed"),
                "target_chunks": targets,
                "chunk_count": len(targets),
                "trace_id": payload.get("trace_id"),
                "user_score": user_score,
                "agent_score": None,
                "agent_reason": None,
                "weight_delta": weight_delta,
                "confidence": 1.0,
                "enabled": True,
                "disabled_at": None,
                "disabled_by": None,
                "disable_reason": None,
                "created_at": now,
            }
            self._append(record)
        return [record]

    # 读取某查询下启用的调权。
    def boosts_for_query(self, kb_id: str, query: str) -> dict[str, float]:
        q_hash = query_hash(query)
        boosts: dict[str, float] = {}
        with self._lock:
            rows = list(self._latest_cached().values())
        for row in rows:
            if (
                row.get("kb_id") == kb_id
                and row.get("query_hash") == q_hash
                and row.get("enabled") is True
            ):
                for target in _record_targets(row):
                    chunk_id = target["chunk_id"]
                    boosts[chunk_id] = boosts.get(chunk_id, 0.0) + float(
                        row.get("weight_delta") or 0.0
                    ) * float(row.get("confidence") or 1.0)
        return boosts

    # 列出检索反馈。
    def list(
        self,
        *,
        kb_id: str,
        enabled: bool | None = None,
        limit: int = 100,
    ) -> _RetrievalFeedbackRows:
        with self._lock:
            rows = list(self._latest_cached().values())
        rows = [row for row in rows if row.get("kb_id") == kb_id]
        grouped: dict[str, _RetrievalFeedbackRows] = {}
        for row in rows:
            grouped.setdefault(_feedback_group_key(row), []).append(row)
        aggregated = [
            _aggregate_retrieval_feedback_group(group) for group in grouped.values()
        ]
        if enabled is not None:
            aggregated = [row for row in aggregated if row.get("enabled") is enabled]
        aggregated.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
        return aggregated[:limit]

    # 统计检索调权反馈。
    def counts(self, *, kb_id: str) -> dict[str, int]:
        with self._lock:
            rows = list(self._latest_cached().values())
        rows = [row for row in rows if row.get("kb_id") == kb_id]
        grouped: dict[str, _RetrievalFeedbackRows] = {}
        for row in rows:
            grouped.setdefault(_feedback_group_key(row), []).append(row)
        aggregated = [
            _aggregate_retrieval_feedback_group(group) for group in grouped.values()
        ]
        enabled = sum(1 for row in aggregated if row.get("enabled") is True)
        disabled = sum(1 for row in aggregated if row.get("enabled") is False)
        return {
            "total": len(aggregated),
            "enabled": enabled,
            "disabled": disabled,
        }

    # 删除某 KB 的检索调权记录，避免同名重建后旧调权继续生效。
    def clear_kb(self, kb_id: str) -> None:
        with self._lock:
            rows = [row for row in self._read_history() if row.get("kb_id") != kb_id]
            self._rewrite_history(rows)

    # 导出折叠后的当前快照，供 JSONL 与 SQLite 存储之间迁移。
    def export_records(self) -> _RetrievalFeedbackRows:
        with self._lock:
            return [
                json.loads(json.dumps(row, ensure_ascii=False))
                for row in self._latest_cached().values()
            ]

    # 按稳定标识追加新快照；相同快照重复导入不会扩张历史。
    def import_records(self, records: _RetrievalFeedbackRows) -> dict[str, int]:
        incoming = [
            json.loads(json.dumps(record, ensure_ascii=False)) for record in records
        ]
        for record in incoming:
            if not str(record.get("retrieval_feedback_id") or ""):
                raise ValueError("retrieval_feedback_id is required")
        with self._lock:
            latest = dict(self._latest_cached())
            changed = 0
            for record in incoming:
                record_id = str(record["retrieval_feedback_id"])
                if latest.get(record_id) == record:
                    continue
                self._append(record)
                latest[record_id] = record
                changed += 1
            return {"imported": changed, "skipped": len(incoming) - changed}

    # 设置启用状态。
    def set_enabled(
        self,
        retrieval_feedback_id: str,
        enabled: bool,
        *,
        actor: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            latest = self._latest_cached()
            current = latest.get(retrieval_feedback_id)
            if current is None:
                return None
            group_key = _feedback_group_key(current)
            group = [
                row for row in latest.values() if _feedback_group_key(row) == group_key
            ]
            updated_rows = []
            disabled_at = _now_iso() if not enabled else None
            for row in group:
                updated = {**row, "enabled": enabled}
                if enabled:
                    updated["disabled_at"] = None
                    updated["disabled_by"] = None
                    updated["disable_reason"] = None
                else:
                    updated["disabled_at"] = disabled_at
                    updated["disabled_by"] = actor
                    updated["disable_reason"] = reason
                self._append(updated)
                updated_rows.append(updated)
            return _aggregate_retrieval_feedback_group(updated_rows)

    # 读取带缓存的最新快照。
    def _latest_cached(self) -> dict[str, dict[str, Any]]:
        mtime = (
            os.path.getmtime(self._path)
            if os.path.exists(self._path)
            else _MISSING_MTIME
        )
        if self._cache_mtime == mtime and self._cache_latest is not None:
            return self._cache_latest
        latest = self._latest()
        self._cache_mtime = mtime
        self._cache_latest = latest
        return latest

    # 读取最新快照。
    def _latest(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self._read_history():
            row_id = str(row.get("retrieval_feedback_id") or "")
            if row_id:
                latest[row_id] = row
        return latest

    # 读取全部历史快照。
    def _read_history(self) -> _RetrievalFeedbackRows:
        rows: _RetrievalFeedbackRows = []
        if not os.path.exists(self._path):
            return rows
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    # 追加。
    def _append(self, entry: dict[str, Any]) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._cache_mtime = None
        self._cache_latest = None

    # 重写历史。
    def _rewrite_history(self, rows: _RetrievalFeedbackRows) -> None:
        tmp_path = f"{self._path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, self._path)
        self._cache_mtime = None
        self._cache_latest = None


def _enabled_value(value: Any) -> int | None:
    if value is True:
        return 1
    if value is False:
        return 0
    return None


# 检索调权反馈的 SQLite 适配器；每个标识保存最新快照。
class SqliteRetrievalFeedbackStore(RetrievalFeedbackStore):
    def __init__(self, db_path: str):
        self._lock = RLock()
        self._conn = connect_sqlite(db_path)
        self._closed = False
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS retrieval_feedback_records ("
            "retrieval_feedback_id TEXT PRIMARY KEY, "
            "feedback_group_key TEXT NOT NULL, kb_id TEXT, query_hash TEXT, "
            "enabled INTEGER, created_at TEXT NOT NULL, data TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_retrieval_feedback_kb_created "
            "ON retrieval_feedback_records(kb_id, created_at DESC)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_retrieval_feedback_boosts "
            "ON retrieval_feedback_records(kb_id, query_hash, enabled)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_retrieval_feedback_group "
            "ON retrieval_feedback_records(feedback_group_key)"
        )

    # 从用户反馈生成调权记录。
    def record_from_feedback(
        self, feedback_id: str, payload: dict[str, Any]
    ) -> _RetrievalFeedbackRows:
        kb_id = _required_text(payload, "kb_id")
        query_text = _required_text(payload, "query")
        if not kb_id or not query_text:
            return []
        user_score, weight_delta = _attributed_feedback_weight(payload)
        if weight_delta == 0:
            return []
        targets = _target_chunks(payload)
        if not targets:
            return []
        source_types = sorted({target["source_type"] for target in targets})
        record = {
            "retrieval_feedback_id": uuid4().hex,
            "feedback_id": feedback_id,
            "kb_id": kb_id,
            "query_hash": query_hash(query_text),
            "query_text": query_text,
            "chunk_id": targets[0]["chunk_id"],
            "source_type": source_types[0] if len(source_types) == 1 else "mixed",
            "target_chunks": targets,
            "chunk_count": len(targets),
            "trace_id": payload.get("trace_id"),
            "user_score": user_score,
            "agent_score": None,
            "agent_reason": None,
            "weight_delta": weight_delta,
            "confidence": 1.0,
            "enabled": True,
            "disabled_at": None,
            "disabled_by": None,
            "disable_reason": None,
            "created_at": _now_iso(),
        }
        with self._lock:
            self._begin_locked()
            try:
                self._upsert_locked(record)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return [record]

    # 读取某查询下启用的调权。
    def boosts_for_query(self, kb_id: str, query: str) -> dict[str, float]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM retrieval_feedback_records "
                "WHERE kb_id=? AND query_hash=? AND enabled=?",
                (kb_id, query_hash(query), 1),
            ).fetchall()
        boosts: dict[str, float] = {}
        for (raw_data,) in rows:
            row = json.loads(raw_data)
            for target in _record_targets(row):
                chunk_id = target["chunk_id"]
                boosts[chunk_id] = boosts.get(chunk_id, 0.0) + float(
                    row.get("weight_delta") or 0.0
                ) * float(row.get("confidence") or 1.0)
        return boosts

    # 列出检索反馈。
    def list(
        self,
        *,
        kb_id: str,
        enabled: bool | None = None,
        limit: int = 100,
    ) -> _RetrievalFeedbackRows:
        with self._lock:
            raw_rows = self._conn.execute(
                "SELECT data FROM retrieval_feedback_records "
                "WHERE kb_id=? ORDER BY rowid ASC",
                (kb_id,),
            ).fetchall()
        grouped: dict[str, _RetrievalFeedbackRows] = {}
        for (raw_data,) in raw_rows:
            row = json.loads(raw_data)
            grouped.setdefault(_feedback_group_key(row), []).append(row)
        aggregated = [
            _aggregate_retrieval_feedback_group(group) for group in grouped.values()
        ]
        if enabled is not None:
            aggregated = [row for row in aggregated if row.get("enabled") is enabled]
        aggregated.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
        return aggregated[:limit]

    # 统计检索调权反馈。
    def counts(self, *, kb_id: str) -> dict[str, int]:
        with self._lock:
            raw_rows = self._conn.execute(
                "SELECT data FROM retrieval_feedback_records WHERE kb_id=?",
                (kb_id,),
            ).fetchall()
        grouped: dict[str, _RetrievalFeedbackRows] = {}
        for (raw_data,) in raw_rows:
            row = json.loads(raw_data)
            grouped.setdefault(_feedback_group_key(row), []).append(row)
        aggregated = [
            _aggregate_retrieval_feedback_group(group) for group in grouped.values()
        ]
        enabled = sum(1 for row in aggregated if row.get("enabled") is True)
        disabled = sum(1 for row in aggregated if row.get("enabled") is False)
        return {
            "total": len(aggregated),
            "enabled": enabled,
            "disabled": disabled,
        }

    # 删除某 KB 的全部调权记录。
    def clear_kb(self, kb_id: str) -> None:
        with self._lock:
            self._begin_locked()
            try:
                self._conn.execute(
                    "DELETE FROM retrieval_feedback_records WHERE kb_id=?", (kb_id,)
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # 设置同一次反馈下所有记录的启用状态。
    def set_enabled(
        self,
        retrieval_feedback_id: str,
        enabled: bool,
        *,
        actor: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            self._begin_locked()
            try:
                current_raw = self._conn.execute(
                    "SELECT data FROM retrieval_feedback_records "
                    "WHERE retrieval_feedback_id=?",
                    (retrieval_feedback_id,),
                ).fetchone()
                if current_raw is None:
                    self._conn.execute("COMMIT")
                    return None
                current = json.loads(current_raw[0])
                group_key = _feedback_group_key(current)
                group_rows = self._conn.execute(
                    "SELECT data FROM retrieval_feedback_records "
                    "WHERE feedback_group_key=? ORDER BY rowid ASC",
                    (group_key,),
                ).fetchall()
                disabled_at = _now_iso() if not enabled else None
                updated_rows = []
                for (raw_data,) in group_rows:
                    updated = {**json.loads(raw_data), "enabled": enabled}
                    if enabled:
                        updated["disabled_at"] = None
                        updated["disabled_by"] = None
                        updated["disable_reason"] = None
                    else:
                        updated["disabled_at"] = disabled_at
                        updated["disabled_by"] = actor
                        updated["disable_reason"] = reason
                    self._upsert_locked(updated)
                    updated_rows.append(updated)
                self._conn.execute("COMMIT")
                return _aggregate_retrieval_feedback_group(updated_rows)
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # 导出折叠后的当前快照，按首次写入顺序保持稳定。
    def export_records(self) -> _RetrievalFeedbackRows:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM retrieval_feedback_records ORDER BY rowid ASC"
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    # 事务性 upsert；同一批数据可安全重复导入。
    def import_records(self, records: _RetrievalFeedbackRows) -> dict[str, int]:
        incoming = [
            json.loads(json.dumps(record, ensure_ascii=False)) for record in records
        ]
        for record in incoming:
            if not str(record.get("retrieval_feedback_id") or ""):
                raise ValueError("retrieval_feedback_id is required")
        with self._lock:
            self._begin_locked()
            try:
                changed = 0
                for record in incoming:
                    existing = self._conn.execute(
                        "SELECT data FROM retrieval_feedback_records "
                        "WHERE retrieval_feedback_id=?",
                        (str(record["retrieval_feedback_id"]),),
                    ).fetchone()
                    if existing is not None and json.loads(existing[0]) == record:
                        continue
                    self._upsert_locked(record)
                    changed += 1
                self._conn.execute("COMMIT")
                return {"imported": changed, "skipped": len(incoming) - changed}
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # 关闭数据库连接；允许清理路径安全地重复调用。
    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True

    def _begin_locked(self) -> None:
        self._conn.execute("BEGIN IMMEDIATE")

    def _upsert_locked(self, record: dict[str, Any]) -> None:
        record_id = str(record["retrieval_feedback_id"])
        self._conn.execute(
            "INSERT INTO retrieval_feedback_records ("
            "retrieval_feedback_id, feedback_group_key, kb_id, query_hash, "
            "enabled, created_at, data) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(retrieval_feedback_id) DO UPDATE SET "
            "feedback_group_key=excluded.feedback_group_key, "
            "kb_id=excluded.kb_id, query_hash=excluded.query_hash, "
            "enabled=excluded.enabled, created_at=excluded.created_at, "
            "data=excluded.data",
            (
                record_id,
                _feedback_group_key(record),
                record.get("kb_id"),
                record.get("query_hash"),
                _enabled_value(record.get("enabled")),
                str(record.get("created_at") or ""),
                json.dumps(record, ensure_ascii=False),
            ),
        )
