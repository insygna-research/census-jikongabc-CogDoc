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
REVISION_SOURCE_STATUSES = {"approved", "stale"}
SIMILARITY_CONFLICT_THRESHOLD = 0.72
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


# 提取相似度计算用的字词片段。
def _similarity_terms(text: str) -> set[str]:
    compact = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalize_knowledge_text(text))
    if len(compact) <= 2:
        return {compact} if compact else set()
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


# 计算文本片段重叠度。
def _text_similarity(left: str, right: str) -> float:
    left_terms = _similarity_terms(left)
    right_terms = _similarity_terms(right)
    if not left_terms or not right_terms:
        return 0.0
    overlap = len(left_terms & right_terms)
    jaccard = overlap / len(left_terms | right_terms)
    containment = overlap / min(len(left_terms), len(right_terms))
    return max(jaccard, containment)


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
            similar = self._find_similar(kb_id, normalized_text)
            conflict_group_id = None
            if similar:
                conflict_group_id = self._ensure_conflict_group(similar)
            now = _now_iso()
            entry = {
                "knowledge_id": f"K{uuid4().hex[:12]}",
                "kb_id": kb_id,
                "text": text,
                "normalized_text": normalized_text,
                "normalized_hash": normalized_hash,
                "version": int(payload.get("version") or 1),
                "previous_version_id": payload.get("previous_version_id"),
                "conflict_group_id": conflict_group_id,
                "related_document_id": payload.get("related_document_id"),
                "related_source": payload.get("related_source"),
                "related_source_sha256": payload.get("related_source_sha256"),
                "related_chunk_ids": list(payload.get("related_chunk_ids") or []),
                "source_note": payload.get("source_note"),
                "certainty": payload.get("certainty") or "medium",
                "status": "pending" if similar else payload.get("status") or "pending",
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

    # 创建修订版本，不覆盖原知识。
    def revise(
        self, knowledge_id: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        text = str(payload.get("text") or "").strip()
        if not text:
            raise ValueError("text must not be blank")
        with self._lock:
            current = self._latest().get(knowledge_id)
            if current is None:
                return None
            if current.get("status") not in REVISION_SOURCE_STATUSES:
                raise ValueError(
                    f"knowledge with status {current.get('status')} cannot be revised"
                )
            normalized_text = normalize_knowledge_text(text)
            normalized_hash = normalized_knowledge_hash(text)
            existing = self._find_duplicate(str(current["kb_id"]), normalized_hash)
            if existing is not None:
                raise ValueError(
                    f"duplicate active knowledge exists: {existing['knowledge_id']}"
                )
            now = _now_iso()
            entry = {
                "knowledge_id": f"K{uuid4().hex[:12]}",
                "kb_id": current["kb_id"],
                "text": text,
                "normalized_text": normalized_text,
                "normalized_hash": normalized_hash,
                "version": int(current.get("version") or 1) + 1,
                "previous_version_id": current["knowledge_id"],
                "conflict_group_id": current.get("conflict_group_id"),
                "related_document_id": payload.get(
                    "related_document_id", current.get("related_document_id")
                ),
                "related_source": payload.get(
                    "related_source", current.get("related_source")
                ),
                "related_source_sha256": payload.get(
                    "related_source_sha256", current.get("related_source_sha256")
                ),
                "related_chunk_ids": list(
                    payload.get("related_chunk_ids", current.get("related_chunk_ids"))
                    or []
                ),
                "source_note": payload.get("source_note", current.get("source_note")),
                "certainty": payload.get("certainty") or current.get("certainty"),
                "status": payload.get("status") or "pending",
                "origin": current.get("origin") or "manual_entry",
                "created_from_trace_id": payload.get("created_from_trace_id"),
                "created_by": payload.get("created_by"),
                "created_at": now,
                "updated_at": now,
                "archived_at": None,
                "reviewed_by": None,
                "reviewed_at": None,
                "review_note": payload.get("review_note"),
            }
            self._append(entry)
            if entry["status"] == "approved":
                self._archive_previous_version(
                    entry, payload.get("created_by"), entry["knowledge_id"]
                )
            return entry

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

    # 查询同一冲突组的其他知识。
    def conflicts_for(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        group_id = row.get("conflict_group_id")
        if not group_id:
            return []
        with self._lock:
            rows = list(self._latest().values())
        conflicts = [
            item
            for item in rows
            if item.get("conflict_group_id") == group_id
            and item.get("knowledge_id") != row.get("knowledge_id")
            and item.get("status") in ACTIVE_STATUSES
        ]
        conflicts.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return conflicts

    # 统计曾过期知识的复核完成情况。
    def stale_review_counts(self, *, kb_id: str) -> dict[str, int]:
        with self._lock:
            history = self._read_history()
            latest = self._latest()
        stale_ids = {
            str(row.get("knowledge_id"))
            for row in history
            if row.get("kb_id") == kb_id
            and row.get("status") == "stale"
            and row.get("knowledge_id")
        }
        reviewed = sum(
            1
            for knowledge_id in stale_ids
            if latest.get(knowledge_id, {}).get("status") != "stale"
        )
        return {"total": len(stale_ids), "reviewed": reviewed}

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
            if status == "approved" and current.get("previous_version_id"):
                self._archive_previous_version(current, actor, knowledge_id)
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

    # 查找同库活跃相似知识。
    def _find_similar(self, kb_id: str, normalized_text: str) -> list[dict[str, Any]]:
        rows = []
        for row in self._latest().values():
            if row.get("kb_id") != kb_id or row.get("status") not in ACTIVE_STATUSES:
                continue
            score = _text_similarity(
                normalized_text, str(row.get("normalized_text") or "")
            )
            if score >= SIMILARITY_CONFLICT_THRESHOLD:
                rows.append({**row, "similarity": round(score, 4)})
        rows.sort(key=lambda item: float(item.get("similarity") or 0.0), reverse=True)
        return rows

    # 确保相似知识属于同一个冲突组。
    def _ensure_conflict_group(self, rows: list[dict[str, Any]]) -> str:
        group_id = next(
            (
                str(row.get("conflict_group_id"))
                for row in rows
                if row.get("conflict_group_id")
            ),
            "",
        )
        if not group_id:
            group_id = f"C{uuid4().hex[:12]}"
        now = _now_iso()
        for row in rows:
            if row.get("conflict_group_id") == group_id:
                continue
            updated = {
                **row,
                "conflict_group_id": group_id,
                "updated_at": now,
            }
            updated.pop("similarity", None)
            self._append(updated)
        return group_id

    # 新版本通过后归档旧版本。
    def _archive_previous_version(
        self, current: dict[str, Any], actor: str | None, replacement_id: str
    ) -> None:
        previous_id = str(current.get("previous_version_id") or "")
        if not previous_id:
            return
        previous = self._latest().get(previous_id)
        if previous is None or previous.get("status") == "archived":
            return
        now = _now_iso()
        archived = {
            **previous,
            "status": "archived",
            "updated_at": now,
            "archived_at": now,
            "reviewed_by": actor,
            "reviewed_at": now,
            "review_note": f"由新版本 {replacement_id} 替代",
        }
        self._append(archived)

    # 读取最新快照。
    def _latest(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self._read_history():
            knowledge_id = str(row.get("knowledge_id") or "")
            if knowledge_id:
                latest[knowledge_id] = row
        return latest

    # 读取全部历史快照。
    def _read_history(self) -> list[dict[str, Any]]:
        rows = []
        if not os.path.exists(self._path):
            return rows
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        return rows

    # 追加。
    def _append(self, entry: dict[str, Any]) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
