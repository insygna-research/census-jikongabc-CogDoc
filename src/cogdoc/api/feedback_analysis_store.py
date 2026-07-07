from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4

from cogdoc.config.settings import get_settings


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
