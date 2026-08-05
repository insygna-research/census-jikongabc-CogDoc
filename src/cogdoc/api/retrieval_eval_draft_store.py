from __future__ import annotations

import json
import os
from threading import RLock
from typing import Any, Mapping, TypeAlias

from cogdoc.api.persistence import connect_sqlite
from cogdoc.tools.eval.retrieval_eval_drafts import (
    DatasetPartition,
    DraftStatus,
    RetrievalEvalDraft,
    apply_review_annotations,
    approve_draft,
    draft_snapshot_identity_key,
    export_retrieval_eval_case,
    reject_draft,
)


_DraftRecord: TypeAlias = dict[str, Any]
_DraftRows: TypeAlias = list[_DraftRecord]
_EvalCases: TypeAlias = list[dict[str, Any]]
_MISSING_MTIME = -1.0


class DraftRevisionConflictError(ValueError):
    """A reviewer submitted a stale draft revision."""


def _check_expected_revision(
    record: Mapping[str, Any], expected_revision: int | None
) -> None:
    if expected_revision is None:
        return
    actual_revision = int(record.get("revision") or 0)
    if actual_revision != expected_revision:
        raise DraftRevisionConflictError(
            f"draft revision conflict: expected {expected_revision}, "
            f"found {actual_revision}"
        )


def _record(draft: RetrievalEvalDraft | Mapping[str, Any]) -> _DraftRecord:
    model = (
        draft
        if isinstance(draft, RetrievalEvalDraft)
        else RetrievalEvalDraft.model_validate(draft)
    )
    return model.model_dump(mode="json")


def _clone(record: Mapping[str, Any]) -> _DraftRecord:
    return json.loads(json.dumps(record, ensure_ascii=False))


def _status_value(status: DraftStatus | str | None) -> str | None:
    return DraftStatus(status).value if status is not None else None


def _partition_value(
    partition: DatasetPartition | str | None,
) -> str | None:
    return DatasetPartition(partition).value if partition is not None else None


class RetrievalEvalDraftStore:
    """Atomic JSONL store for reviewable retrieval-eval annotations."""

    def __init__(self, path: str):
        self._path = path
        self._lock = RLock()
        self._closed = False
        self._cache_mtime: float | None = None
        self._cache_rows: _DraftRows | None = None
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def save(self, draft: RetrievalEvalDraft | Mapping[str, Any]) -> _DraftRecord:
        incoming = _record(draft)
        with self._lock:
            self._ensure_open()
            rows = [_clone(row) for row in self._read_all_locked()]
            changed = self._merge_record(rows, incoming)
            if changed:
                self._rewrite_locked(rows)
        return _clone(incoming)

    def ensure(self, draft: RetrievalEvalDraft | Mapping[str, Any]) -> _DraftRecord:
        """Insert a stable proposal once without overwriting prior review work."""

        incoming = _record(draft)
        incoming_snapshot_key = draft_snapshot_identity_key(incoming)
        with self._lock:
            self._ensure_open()
            rows = [_clone(row) for row in self._read_all_locked()]
            for row in rows:
                if (
                    row["draft_id"] == incoming["draft_id"]
                    or (row["dedupe_key"] == incoming["dedupe_key"])
                    or (draft_snapshot_identity_key(row) == incoming_snapshot_key)
                ):
                    return _clone(row)
            rows.append(_clone(incoming))
            self._rewrite_locked(rows)
        return _clone(incoming)

    def get(self, draft_id: str) -> _DraftRecord | None:
        with self._lock:
            self._ensure_open()
            for row in self._read_all_locked():
                if row["draft_id"] == draft_id:
                    return _clone(row)
        return None

    def list(
        self,
        *,
        kb_id: str | None = None,
        status: DraftStatus | str | None = None,
        dataset_partition: DatasetPartition | str | None = None,
        limit: int = 100,
    ) -> _DraftRows:
        status_value = _status_value(status)
        partition_value = _partition_value(dataset_partition)
        with self._lock:
            self._ensure_open()
            rows = [_clone(row) for row in self._read_all_locked()]
        if kb_id is not None:
            rows = [row for row in rows if row["kb_id"] == kb_id]
        if status_value is not None:
            rows = [row for row in rows if row["status"] == status_value]
        if partition_value is not None:
            rows = [row for row in rows if row["dataset_partition"] == partition_value]
        rows.sort(key=lambda row: (row["updated_at"], row["draft_id"]), reverse=True)
        return rows[: max(0, limit)]

    def approve(
        self,
        draft_id: str,
        *,
        reviewer: str,
        annotations: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
        now: str | None = None,
    ) -> _DraftRecord:
        with self._lock:
            self._ensure_open()
            rows = [_clone(row) for row in self._read_all_locked()]
            position = self._position(rows, draft_id)
            candidate = rows[position]
            _check_expected_revision(candidate, expected_revision)
            if annotations is not None:
                candidate = _record(apply_review_annotations(candidate, annotations))
            updated = _record(approve_draft(candidate, reviewer=reviewer, now=now))
            rows[position] = updated
            self._rewrite_locked(rows)
            return _clone(updated)

    def review(
        self,
        draft_id: str,
        *,
        decision: DraftStatus | str,
        reviewer: str,
        annotations: Mapping[str, Any] | None = None,
        reason: str = "",
        expected_revision: int | None = None,
        now: str | None = None,
    ) -> _DraftRecord:
        resolved = DraftStatus(decision)
        if resolved is DraftStatus.APPROVED:
            return self.approve(
                draft_id,
                reviewer=reviewer,
                annotations=annotations,
                expected_revision=expected_revision,
                now=now,
            )
        if resolved is DraftStatus.REJECTED:
            if annotations:
                raise ValueError("rejected reviews cannot contain gold annotations")
            return self.reject(
                draft_id,
                reviewer=reviewer,
                reason=reason,
                expected_revision=expected_revision,
                now=now,
            )
        raise ValueError("review decision must be approved or rejected")

    def reject(
        self,
        draft_id: str,
        *,
        reviewer: str,
        reason: str,
        expected_revision: int | None = None,
        now: str | None = None,
    ) -> _DraftRecord:
        with self._lock:
            self._ensure_open()
            rows = [_clone(row) for row in self._read_all_locked()]
            position = self._position(rows, draft_id)
            _check_expected_revision(rows[position], expected_revision)
            updated = _record(
                reject_draft(rows[position], reviewer=reviewer, reason=reason, now=now)
            )
            rows[position] = updated
            self._rewrite_locked(rows)
            return _clone(updated)

    def clear_kb(self, kb_id: str) -> None:
        with self._lock:
            self._ensure_open()
            rows = [
                _clone(row) for row in self._read_all_locked() if row["kb_id"] != kb_id
            ]
            self._rewrite_locked(rows)

    def export_eval_cases(
        self, *, dataset_partition: DatasetPartition | str
    ) -> _EvalCases:
        """Return approved rows for one partition; never mutate a gate file."""

        approved = self.list(
            status=DraftStatus.APPROVED,
            dataset_partition=dataset_partition,
            limit=2**31 - 1,
        )
        approved.sort(key=lambda row: row["draft_id"])
        return [export_retrieval_eval_case(row) for row in approved]

    def export_records(self) -> _DraftRows:
        with self._lock:
            self._ensure_open()
            return [_clone(row) for row in self._read_all_locked()]

    def import_records(self, records: _DraftRows) -> dict[str, int]:
        # Validate and clone the complete batch before taking a lock or writing.
        incoming = [_record(record) for record in records]
        with self._lock:
            self._ensure_open()
            rows = [_clone(row) for row in self._read_all_locked()]
            changed = 0
            for record in incoming:
                changed += int(self._merge_record(rows, record))
            if changed:
                self._rewrite_locked(rows)
        return {"imported": changed, "skipped": len(incoming) - changed}

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._cache_rows = None
            self._cache_mtime = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("retrieval eval draft store is closed")

    @staticmethod
    def _position(rows: _DraftRows, draft_id: str) -> int:
        for position, row in enumerate(rows):
            if row["draft_id"] == draft_id:
                return position
        raise KeyError(draft_id)

    @staticmethod
    def _merge_record(rows: _DraftRows, incoming: _DraftRecord) -> bool:
        for position, row in enumerate(rows):
            if (
                row["dedupe_key"] == incoming["dedupe_key"]
                and row["draft_id"] != incoming["draft_id"]
            ):
                raise ValueError("dedupe_key is already bound to another draft_id")
            if row["draft_id"] == incoming["draft_id"]:
                if row == incoming:
                    return False
                rows[position] = _clone(incoming)
                return True
        rows.append(_clone(incoming))
        return True

    def _read_all_locked(self) -> _DraftRows:
        mtime = (
            os.path.getmtime(self._path)
            if os.path.exists(self._path)
            else _MISSING_MTIME
        )
        if self._cache_mtime == mtime and self._cache_rows is not None:
            return self._cache_rows
        if not os.path.exists(self._path):
            rows: _DraftRows = []
        else:
            rows = []
            with open(self._path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        rows.append(_record(json.loads(line)))
        self._cache_mtime = mtime
        self._cache_rows = rows
        return rows

    def _rewrite_locked(self, rows: _DraftRows) -> None:
        temporary_path = f"{self._path}.tmp"
        try:
            with open(temporary_path, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        self._cache_mtime = None
        self._cache_rows = None


class SqliteRetrievalEvalDraftStore(RetrievalEvalDraftStore):
    """SQLite adapter with the exact JSONL store contract."""

    def __init__(self, db_path: str):
        self._lock = RLock()
        self._closed = False
        self._conn = connect_sqlite(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS retrieval_eval_drafts ("
            "draft_id TEXT PRIMARY KEY, dedupe_key TEXT NOT NULL UNIQUE, "
            "kb_id TEXT NOT NULL, status TEXT NOT NULL, "
            "dataset_partition TEXT NOT NULL, updated_at TEXT NOT NULL, "
            "data TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_retrieval_eval_drafts_queue "
            "ON retrieval_eval_drafts("
            "kb_id, dataset_partition, status, updated_at DESC)"
        )

    def save(self, draft: RetrievalEvalDraft | Mapping[str, Any]) -> _DraftRecord:
        incoming = _record(draft)
        with self._lock:
            self._ensure_open()
            self._begin_locked()
            try:
                self._upsert_locked(incoming)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return _clone(incoming)

    def ensure(self, draft: RetrievalEvalDraft | Mapping[str, Any]) -> _DraftRecord:
        incoming = _record(draft)
        incoming_snapshot_key = draft_snapshot_identity_key(incoming)
        with self._lock:
            self._ensure_open()
            self._begin_locked()
            try:
                existing = self._conn.execute(
                    "SELECT data FROM retrieval_eval_drafts "
                    "WHERE draft_id=? OR dedupe_key=?",
                    (incoming["draft_id"], incoming["dedupe_key"]),
                ).fetchone()
                if existing is not None:
                    result = _record(json.loads(existing[0]))
                else:
                    candidates = self._conn.execute(
                        "SELECT data FROM retrieval_eval_drafts "
                        "WHERE kb_id=? AND dataset_partition=?",
                        (incoming["kb_id"], incoming["dataset_partition"]),
                    ).fetchall()
                    result = None
                    for (raw_candidate,) in candidates:
                        candidate = _record(json.loads(raw_candidate))
                        if (
                            draft_snapshot_identity_key(candidate)
                            == incoming_snapshot_key
                        ):
                            result = candidate
                            break
                    if result is None:
                        self._upsert_locked(incoming)
                        result = incoming
                self._conn.execute("COMMIT")
                assert result is not None
                return _clone(result)
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def get(self, draft_id: str) -> _DraftRecord | None:
        with self._lock:
            self._ensure_open()
            row = self._conn.execute(
                "SELECT data FROM retrieval_eval_drafts WHERE draft_id=?",
                (draft_id,),
            ).fetchone()
        return _record(json.loads(row[0])) if row is not None else None

    def list(
        self,
        *,
        kb_id: str | None = None,
        status: DraftStatus | str | None = None,
        dataset_partition: DatasetPartition | str | None = None,
        limit: int = 100,
    ) -> _DraftRows:
        clauses = []
        params: list[Any] = []
        if kb_id is not None:
            clauses.append("kb_id=?")
            params.append(kb_id)
        status_value = _status_value(status)
        if status_value is not None:
            clauses.append("status=?")
            params.append(status_value)
        partition_value = _partition_value(dataset_partition)
        if partition_value is not None:
            clauses.append("dataset_partition=?")
            params.append(partition_value)
        query = "SELECT data FROM retrieval_eval_drafts"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, draft_id DESC LIMIT ?"
        params.append(max(0, limit))
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(query, tuple(params)).fetchall()
        return [_record(json.loads(row[0])) for row in rows]

    def approve(
        self,
        draft_id: str,
        *,
        reviewer: str,
        annotations: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
        now: str | None = None,
    ) -> _DraftRecord:
        return self._review(
            draft_id,
            lambda row: approve_draft(
                apply_review_annotations(row, annotations)
                if annotations is not None
                else row,
                reviewer=reviewer,
                now=now,
            ),
            expected_revision=expected_revision,
        )

    def review(
        self,
        draft_id: str,
        *,
        decision: DraftStatus | str,
        reviewer: str,
        annotations: Mapping[str, Any] | None = None,
        reason: str = "",
        expected_revision: int | None = None,
        now: str | None = None,
    ) -> _DraftRecord:
        resolved = DraftStatus(decision)
        if resolved is DraftStatus.APPROVED:
            return self.approve(
                draft_id,
                reviewer=reviewer,
                annotations=annotations,
                expected_revision=expected_revision,
                now=now,
            )
        if resolved is DraftStatus.REJECTED:
            if annotations:
                raise ValueError("rejected reviews cannot contain gold annotations")
            return self.reject(
                draft_id,
                reviewer=reviewer,
                reason=reason,
                expected_revision=expected_revision,
                now=now,
            )
        raise ValueError("review decision must be approved or rejected")

    def reject(
        self,
        draft_id: str,
        *,
        reviewer: str,
        reason: str,
        expected_revision: int | None = None,
        now: str | None = None,
    ) -> _DraftRecord:
        return self._review(
            draft_id,
            lambda row: reject_draft(row, reviewer=reviewer, reason=reason, now=now),
            expected_revision=expected_revision,
        )

    def clear_kb(self, kb_id: str) -> None:
        with self._lock:
            self._ensure_open()
            self._begin_locked()
            try:
                self._conn.execute(
                    "DELETE FROM retrieval_eval_drafts WHERE kb_id=?", (kb_id,)
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def export_records(self) -> _DraftRows:
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(
                "SELECT data FROM retrieval_eval_drafts ORDER BY rowid ASC"
            ).fetchall()
        return [_record(json.loads(row[0])) for row in rows]

    def import_records(self, records: _DraftRows) -> dict[str, int]:
        incoming = [_record(record) for record in records]
        with self._lock:
            self._ensure_open()
            self._begin_locked()
            try:
                changed = 0
                for record in incoming:
                    existing = self._conn.execute(
                        "SELECT data FROM retrieval_eval_drafts WHERE draft_id=?",
                        (record["draft_id"],),
                    ).fetchone()
                    if (
                        existing is not None
                        and _record(json.loads(existing[0])) == record
                    ):
                        continue
                    self._upsert_locked(record)
                    changed += 1
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return {"imported": changed, "skipped": len(incoming) - changed}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True

    def _review(
        self,
        draft_id: str,
        transition: Any,
        *,
        expected_revision: int | None,
    ) -> _DraftRecord:
        with self._lock:
            self._ensure_open()
            self._begin_locked()
            try:
                row = self._conn.execute(
                    "SELECT data FROM retrieval_eval_drafts WHERE draft_id=?",
                    (draft_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(draft_id)
                current = _record(json.loads(row[0]))
                _check_expected_revision(current, expected_revision)
                updated = _record(transition(current))
                self._upsert_locked(updated)
                self._conn.execute("COMMIT")
                return _clone(updated)
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def _begin_locked(self) -> None:
        self._conn.execute("BEGIN IMMEDIATE")

    def _upsert_locked(self, record: _DraftRecord) -> None:
        self._conn.execute(
            "INSERT INTO retrieval_eval_drafts ("
            "draft_id, dedupe_key, kb_id, status, dataset_partition, updated_at, data"
            ") VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(draft_id) DO UPDATE SET "
            "dedupe_key=excluded.dedupe_key, kb_id=excluded.kb_id, "
            "status=excluded.status, "
            "dataset_partition=excluded.dataset_partition, "
            "updated_at=excluded.updated_at, data=excluded.data",
            (
                record["draft_id"],
                record["dedupe_key"],
                record["kb_id"],
                record["status"],
                record["dataset_partition"],
                record["updated_at"],
                json.dumps(record, ensure_ascii=False),
            ),
        )
