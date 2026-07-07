from __future__ import annotations

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


ACTIVE_STATUSES = {"pending", "approved", "stale"}
VALID_STATUSES = ACTIVE_STATUSES | {"rejected", "archived"}
ALLOWED_BINDING_UPDATE_FIELDS = {
    "related_document_id",
    "related_source",
    "related_source_sha256",
    "related_chunk_ids",
}


# 返回当前协调世界时时间字符串。
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 归一化知识正文，供精确去重与后续相似检测打底。
def normalize_knowledge_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", normalized).strip()


# 计算归一化知识哈希。
def normalized_knowledge_hash(text: str) -> str:
    return hashlib.sha256(normalize_knowledge_text(text).encode("utf-8")).hexdigest()


# 逐行对象格式派生知识存储：追加快照，读取时按知识标识折叠最新状态。
class DerivedKnowledgeStore:
    def __init__(self, path: str | None = None):
        self._path = path or get_settings().derived_knowledge_path
        self._lock = RLock()
        os.makedirs(os.path.dirname(self._path), exist_ok=True)

    # 创建知识；同库精确归一化哈希已存在时返回现有主记录。
    def create(self, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        text = str(payload.get("text") or "").strip()
        if not text:
            raise ValueError("text must not be blank")
        kb_id = str(payload.get("kb_id") or "").strip()
        if not kb_id:
            raise ValueError("kb_id must not be blank")

        normalized_text = normalize_knowledge_text(text)
        normalized_hash = normalized_knowledge_hash(text)
        with self._lock:
            existing = self._find_duplicate(kb_id, normalized_hash)
            if existing is not None:
                return existing, True
            now = _now_iso()
            entry = {
                "knowledge_id": f"K{uuid4().hex[:12]}",
                "kb_id": kb_id,
                "text": text,
                "normalized_text": normalized_text,
                "normalized_hash": normalized_hash,
                "version": int(payload.get("version") or 1),
                "previous_version_id": payload.get("previous_version_id"),
                "conflict_group_id": payload.get("conflict_group_id"),
                "related_document_id": payload.get("related_document_id"),
                "related_source": payload.get("related_source"),
                "related_source_sha256": payload.get("related_source_sha256"),
                "related_chunk_ids": list(payload.get("related_chunk_ids") or []),
                "source_note": payload.get("source_note"),
                "certainty": payload.get("certainty") or "medium",
                "status": payload.get("status") or "pending",
                "origin": payload.get("origin") or "manual_entry",
                "created_from_trace_id": payload.get("created_from_trace_id"),
                "created_by": payload.get("created_by"),
                "created_at": now,
                "updated_at": now,
                "archived_at": None,
                "reviewed_by": None,
                "reviewed_at": None,
                "review_note": None,
            }
            self._append(entry)
            return entry, False

    # 查询最新知识快照。
    def list(
        self,
        *,
        kb_id: str,
        status: str | None = None,
        document_id: str | None = None,
        origin: str | None = None,
        created_by: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._latest().values())
        rows = [row for row in rows if row.get("kb_id") == kb_id]
        if status is not None:
            rows = [row for row in rows if row.get("status") == status]
        if document_id is not None:
            rows = [
                row
                for row in rows
                if row.get("related_document_id") == document_id
                or row.get("related_source") == document_id
            ]
        if origin is not None:
            rows = [row for row in rows if row.get("origin") == origin]
        if created_by is not None:
            rows = [row for row in rows if row.get("created_by") == created_by]
        if created_after is not None:
            rows = [
                row for row in rows if str(row.get("created_at", "")) >= created_after
            ]
        if created_before is not None:
            rows = [
                row for row in rows if str(row.get("created_at", "")) <= created_before
            ]
        return sorted(
            rows, key=lambda row: str(row.get("created_at", "")), reverse=True
        )

    # 统计知识审核队列。
    def counts(
        self,
        *,
        kb_id: str,
        document_id: str | None = None,
        origin: str | None = None,
        created_by: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> dict[str, dict[str, int] | int]:
        rows = self.list(
            kb_id=kb_id,
            document_id=document_id,
            origin=origin,
            created_by=created_by,
            created_after=created_after,
            created_before=created_before,
        )
        by_status: dict[str, int] = {}
        by_origin: dict[str, int] = {}
        for row in rows:
            status = str(row.get("status") or "unknown")
            row_origin = str(row.get("origin") or "unknown")
            by_status[status] = by_status.get(status, 0) + 1
            by_origin[row_origin] = by_origin.get(row_origin, 0) + 1
        return {
            "total": len(rows),
            "by_status": by_status,
            "by_origin": by_origin,
        }

    # 修改审核状态，保留历史快照。
    def set_status(
        self,
        knowledge_id: str,
        status: str,
        *,
        actor: str | None = None,
        note: str | None = None,
        binding_updates: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status}")
        with self._lock:
            current = self._latest().get(knowledge_id)
            if current is None:
                return None
            updated = {**current}
            now = _now_iso()
            updated["status"] = status
            updated["updated_at"] = now
            updated["reviewed_by"] = actor
            updated["reviewed_at"] = now
            updated["review_note"] = note
            if binding_updates:
                for key, value in binding_updates.items():
                    if key in ALLOWED_BINDING_UPDATE_FIELDS and value is not None:
                        updated[key] = value
            if status == "archived":
                updated["archived_at"] = now
            self._append(updated)
            return updated

    # 批量修改审核状态。
    def batch_set_status(
        self,
        knowledge_ids: list[str],
        status: str,
        *,
        actor: str | None = None,
        note: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        updated, missing = [], []
        for knowledge_id in knowledge_ids:
            row = self.set_status(knowledge_id, status, actor=actor, note=note)
            if row is None:
                missing.append(knowledge_id)
            else:
                updated.append(row)
        return updated, missing

    # 文档哈希变化后，将绑定旧哈希的派生知识标记为过期。
    def mark_stale_for_source(
        self, kb_id: str, source: str, old_source_sha256: str
    ) -> list[dict[str, Any]]:
        stale = []
        with self._lock:
            rows = list(self._latest().values())
        for row in rows:
            if (
                row.get("kb_id") == kb_id
                and row.get("related_source") == source
                and row.get("related_source_sha256") == old_source_sha256
                and row.get("status") == "approved"
            ):
                updated = self.set_status(row["knowledge_id"], "stale")
                if updated is not None:
                    stale.append(updated)
        return stale

    # 查找同库未归档的精确重复记录。
    def _find_duplicate(
        self, kb_id: str, normalized_hash: str
    ) -> dict[str, Any] | None:
        for row in self._latest().values():
            if (
                row.get("kb_id") == kb_id
                and row.get("normalized_hash") == normalized_hash
                and row.get("status") in ACTIVE_STATUSES
            ):
                return row
        return None

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
                knowledge_id = str(row.get("knowledge_id") or "")
                if knowledge_id:
                    latest[knowledge_id] = row
        return latest

    # 追加。
    def _append(self, entry: dict[str, Any]) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
