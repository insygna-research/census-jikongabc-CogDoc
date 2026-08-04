from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4

from cogdoc.config.settings import get_settings
from cogdoc.api.persistence import connect_sqlite


# 空文件修改时间标记。
_MISSING_MTIME = -1.0


# 返回当前协调世界时时间字符串。
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 反馈理解结果存储。
class FeedbackAnalysisStore:
    def __init__(self, path: str | None = None):
        self._path = path or get_settings().feedback_analysis_path
        self._lock = RLock()
        self._cache_mtime: float | None = None
        self._cache_rows: list[dict[str, Any]] | None = None
        directory = os.path.dirname(self._path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    # 记录分析结果。
    def record(
        self, feedback_id: str, payload: dict[str, Any], analysis: dict[str, Any]
    ) -> dict[str, Any]:
        entry = {
            "feedback_analysis_id": uuid4().hex,
            "feedback_id": feedback_id,
            "kb_id": payload.get("kb_id"),
            "trace_id": payload.get("trace_id"),
            "query": payload.get("query"),
            "created_at": _now_iso(),
            **analysis,
        }
        with self._lock:
            self._append(entry)
        return entry

    # 查询分析结果。
    def list(
        self,
        *,
        kb_id: str,
        feedback_id: str | None = None,
        trace_id: str | None = None,
        recommended_action: str | None = None,
        needs_review: bool | None = None,
        min_confidence: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._read_all()
        rows = [row for row in rows if row.get("kb_id") == kb_id]
        if feedback_id is not None:
            rows = [row for row in rows if row.get("feedback_id") == feedback_id]
        if trace_id is not None:
            rows = [row for row in rows if row.get("trace_id") == trace_id]
        if recommended_action is not None:
            rows = [
                row
                for row in rows
                if row.get("recommended_action") == recommended_action
            ]
        if needs_review is not None:
            rows = [row for row in rows if row.get("needs_review") is needs_review]
        if min_confidence is not None:
            rows = [
                row
                for row in rows
                if float(row.get("confidence") or 0.0) >= min_confidence
            ]
        rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
        return rows[:limit]

    # 统计反馈理解队列。
    def counts(self, *, kb_id: str) -> dict[str, dict[str, int] | int]:
        with self._lock:
            rows = self._read_all()
        rows = [row for row in rows if row.get("kb_id") == kb_id]
        by_action: dict[str, int] = {}
        by_type: dict[str, int] = {}
        needs_review = 0
        for row in rows:
            action = str(row.get("recommended_action") or "unknown")
            feedback_type = str(row.get("feedback_type") or "unknown")
            by_action[action] = by_action.get(action, 0) + 1
            by_type[feedback_type] = by_type.get(feedback_type, 0) + 1
            if row.get("needs_review") is True:
                needs_review += 1
        return {
            "total": len(rows),
            "needs_review": needs_review,
            "by_action": by_action,
            "by_type": by_type,
        }

    # 删除某 KB 的反馈分析记录，避免同名重建后继承旧审核队列。
    def clear_kb(self, kb_id: str) -> None:
        with self._lock:
            rows = [row for row in self._read_all() if row.get("kb_id") != kb_id]
            self._rewrite(rows)

    # 导出当前记录，供 JSONL 与 SQLite 存储之间迁移。
    def export_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                json.loads(json.dumps(row, ensure_ascii=False))
                for row in self._read_all()
            ]

    # 按稳定标识导入或更新记录；重复导入不会产生重复行。
    def import_records(self, records: list[dict[str, Any]]) -> dict[str, int]:
        incoming = [
            json.loads(json.dumps(record, ensure_ascii=False)) for record in records
        ]
        for record in incoming:
            if not str(record.get("feedback_analysis_id") or ""):
                raise ValueError("feedback_analysis_id is required")
        with self._lock:
            rows = [dict(row) for row in self._read_all()]
            positions = {
                str(row.get("feedback_analysis_id")): index
                for index, row in enumerate(rows)
                if row.get("feedback_analysis_id")
            }
            changed = 0
            for record in incoming:
                record_id = str(record["feedback_analysis_id"])
                position = positions.get(record_id)
                if position is None:
                    positions[record_id] = len(rows)
                    rows.append(record)
                    changed += 1
                elif rows[position] != record:
                    rows[position] = record
                    changed += 1
            if changed:
                self._rewrite(rows)
            return {"imported": changed, "skipped": len(incoming) - changed}

    # 读取全部分析结果。
    def _read_all(self) -> list[dict[str, Any]]:
        mtime = (
            os.path.getmtime(self._path)
            if os.path.exists(self._path)
            else _MISSING_MTIME
        )
        if self._cache_mtime == mtime and self._cache_rows is not None:
            return self._cache_rows
        if not os.path.exists(self._path):
            self._cache_mtime = mtime
            self._cache_rows = []
            return []
        rows = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        self._cache_mtime = mtime
        self._cache_rows = rows
        return rows

    # 追加。
    def _append(self, entry: dict[str, Any]) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._cache_mtime = None
        self._cache_rows = None

    # 重写记录。
    def _rewrite(self, rows: list[dict[str, Any]]) -> None:
        tmp_path = f"{self._path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, self._path)
        self._cache_mtime = None
        self._cache_rows = None


def _needs_review_value(value: Any) -> int | None:
    if value is True:
        return 1
    if value is False:
        return 0
    return None


# 反馈理解结果的 SQLite 适配器；整条 JSON 保真，索引列仅用于查询。
class SqliteFeedbackAnalysisStore(FeedbackAnalysisStore):
    def __init__(self, db_path: str):
        self._lock = RLock()
        self._conn = connect_sqlite(db_path)
        self._closed = False
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS feedback_analysis_records ("
            "feedback_analysis_id TEXT PRIMARY KEY, "
            "feedback_id TEXT, kb_id TEXT, trace_id TEXT, "
            "recommended_action TEXT, needs_review INTEGER, "
            "confidence REAL NOT NULL, created_at TEXT NOT NULL, "
            "data TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_analysis_kb_created "
            "ON feedback_analysis_records(kb_id, created_at DESC)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_analysis_filters "
            "ON feedback_analysis_records("
            "kb_id, feedback_id, trace_id, recommended_action, needs_review)"
        )

    # 记录分析结果。
    def record(
        self, feedback_id: str, payload: dict[str, Any], analysis: dict[str, Any]
    ) -> dict[str, Any]:
        entry = {
            "feedback_analysis_id": uuid4().hex,
            "feedback_id": feedback_id,
            "kb_id": payload.get("kb_id"),
            "trace_id": payload.get("trace_id"),
            "query": payload.get("query"),
            "created_at": _now_iso(),
            **analysis,
        }
        with self._lock:
            self._begin_locked()
            try:
                self._upsert_locked(entry)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return entry

    # 查询分析结果。
    def list(
        self,
        *,
        kb_id: str,
        feedback_id: str | None = None,
        trace_id: str | None = None,
        recommended_action: str | None = None,
        needs_review: bool | None = None,
        min_confidence: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses = ["kb_id=?"]
        params: list[Any] = [kb_id]
        if feedback_id is not None:
            clauses.append("feedback_id=?")
            params.append(feedback_id)
        if trace_id is not None:
            clauses.append("trace_id=?")
            params.append(trace_id)
        if recommended_action is not None:
            clauses.append("recommended_action=?")
            params.append(recommended_action)
        if needs_review is not None:
            clauses.append("needs_review=?")
            params.append(1 if needs_review else 0)
        if min_confidence is not None:
            clauses.append("confidence>=?")
            params.append(min_confidence)
        query = (
            "SELECT data FROM feedback_analysis_records WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at DESC, rowid ASC"
        )
        if limit >= 0:
            query += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        decoded = [json.loads(row[0]) for row in rows]
        return decoded if limit >= 0 else decoded[:limit]

    # 统计反馈理解队列。
    def counts(self, *, kb_id: str) -> dict[str, dict[str, int] | int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM feedback_analysis_records WHERE kb_id=?",
                (kb_id,),
            ).fetchall()
        by_action: dict[str, int] = {}
        by_type: dict[str, int] = {}
        needs_review = 0
        for (raw_data,) in rows:
            row = json.loads(raw_data)
            action = str(row.get("recommended_action") or "unknown")
            feedback_type = str(row.get("feedback_type") or "unknown")
            by_action[action] = by_action.get(action, 0) + 1
            by_type[feedback_type] = by_type.get(feedback_type, 0) + 1
            if row.get("needs_review") is True:
                needs_review += 1
        return {
            "total": len(rows),
            "needs_review": needs_review,
            "by_action": by_action,
            "by_type": by_type,
        }

    # 删除某 KB 的全部分析记录。
    def clear_kb(self, kb_id: str) -> None:
        with self._lock:
            self._begin_locked()
            try:
                self._conn.execute(
                    "DELETE FROM feedback_analysis_records WHERE kb_id=?", (kb_id,)
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # 导出当前记录，按首次写入顺序保持稳定。
    def export_records(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM feedback_analysis_records ORDER BY rowid ASC"
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    # 事务性 upsert；同一批数据可安全重复导入。
    def import_records(self, records: list[dict[str, Any]]) -> dict[str, int]:
        incoming = [
            json.loads(json.dumps(record, ensure_ascii=False)) for record in records
        ]
        for record in incoming:
            if not str(record.get("feedback_analysis_id") or ""):
                raise ValueError("feedback_analysis_id is required")
        with self._lock:
            self._begin_locked()
            try:
                changed = 0
                for record in incoming:
                    existing = self._conn.execute(
                        "SELECT data FROM feedback_analysis_records "
                        "WHERE feedback_analysis_id=?",
                        (str(record["feedback_analysis_id"]),),
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
        self._conn.execute(
            "INSERT INTO feedback_analysis_records ("
            "feedback_analysis_id, feedback_id, kb_id, trace_id, "
            "recommended_action, needs_review, confidence, created_at, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(feedback_analysis_id) DO UPDATE SET "
            "feedback_id=excluded.feedback_id, kb_id=excluded.kb_id, "
            "trace_id=excluded.trace_id, "
            "recommended_action=excluded.recommended_action, "
            "needs_review=excluded.needs_review, confidence=excluded.confidence, "
            "created_at=excluded.created_at, data=excluded.data",
            (
                str(record["feedback_analysis_id"]),
                record.get("feedback_id"),
                record.get("kb_id"),
                record.get("trace_id"),
                record.get("recommended_action"),
                _needs_review_value(record.get("needs_review")),
                float(record.get("confidence") or 0.0),
                str(record.get("created_at") or ""),
                json.dumps(record, ensure_ascii=False),
            ),
        )
