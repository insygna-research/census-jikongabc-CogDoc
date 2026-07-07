import hashlib
import json
import os
import re
import unicodedata
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


# 从反馈载荷抽取被评价的分块。
def _target_chunks(payload: dict[str, Any]) -> list[dict[str, str]]:
    targets = []
    seen = set()
    for field in ("citations", "evidence"):
        for item in payload.get(field) or []:
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
    return targets


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
    ) -> list[dict[str, Any]]:
        kb_id = _required_text(payload, "kb_id")
        query_text = _required_text(payload, "query")
        if not kb_id or not query_text:
            return []
        user_score, weight_delta = _feedback_weight(
            str(payload.get("feedback") or ""), payload.get("rating")
        )
        if weight_delta == 0:
            return []
        targets = _target_chunks(payload)
        if not targets:
            return []

        now = _now_iso()
        q_hash = query_hash(query_text)
        records = []
        with self._lock:
            for target in targets:
                record = {
                    "retrieval_feedback_id": uuid4().hex,
                    "feedback_id": feedback_id,
                    "kb_id": kb_id,
                    "query_hash": q_hash,
                    "query_text": query_text,
                    "chunk_id": target["chunk_id"],
                    "source_type": target["source_type"],
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
                records.append(record)
        return records

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
                chunk_id = str(row.get("chunk_id") or "")
                if chunk_id:
                    boosts[chunk_id] = boosts.get(chunk_id, 0.0) + float(
                        row.get("weight_delta") or 0.0
                    ) * float(row.get("confidence") or 1.0)
        return boosts

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
            current = self._latest_cached().get(retrieval_feedback_id)
            if current is None:
                return None
            updated = {**current, "enabled": enabled}
            if enabled:
                updated["disabled_at"] = None
                updated["disabled_by"] = None
                updated["disable_reason"] = None
            else:
                updated["disabled_at"] = _now_iso()
                updated["disabled_by"] = actor
                updated["disable_reason"] = reason
            self._append(updated)
            return updated

    # 读取带缓存的最新快照。
    def _latest_cached(self) -> dict[str, dict[str, Any]]:
        mtime = os.path.getmtime(self._path) if os.path.exists(self._path) else _MISSING_MTIME
        if self._cache_mtime == mtime and self._cache_latest is not None:
            return self._cache_latest
        latest = self._latest()
        self._cache_mtime = mtime
        self._cache_latest = latest
        return latest

    # 读取最新快照。
    def _latest(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        if not os.path.exists(self._path):
            return latest
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = str(row.get("retrieval_feedback_id") or "")
                if row_id:
                    latest[row_id] = row
        return latest

    # 追加。
    def _append(self, entry: dict[str, Any]) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._cache_mtime = None
        self._cache_latest = None
