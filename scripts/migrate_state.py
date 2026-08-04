#!/usr/bin/env python3
"""Atomically migrate JSONL state stores into the shared CogDoc state.db."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from typing import Any

from cogdoc.api.derived_knowledge_store import (
    DerivedKnowledgeStore,
    SqliteDerivedKnowledgeStore,
)
from cogdoc.api.feedback_analysis_store import (
    FeedbackAnalysisStore,
    SqliteFeedbackAnalysisStore,
)
from cogdoc.api.feedback_store import FeedbackStore, SqliteFeedbackStore
from cogdoc.api.retrieval_feedback_store import (
    RetrievalFeedbackStore,
    SqliteRetrievalFeedbackStore,
)
from cogdoc.config.settings import get_settings
from cogdoc.service.process_lock import (
    acquire_single_instance_lock,
    release_single_instance_lock,
)


MIGRATION_VERSION = 1


class MigrationError(RuntimeError):
    pass


def _canonical_records(records: list[dict[str, Any]]) -> list[str]:
    return sorted(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    )


def _record_digest(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for encoded in _canonical_records(records):
        digest.update(encoded.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _source_stores(data_dir: Path) -> tuple[dict[str, Any], list[Any]]:
    feedback_jsonl = data_dir / "feedback" / "feedback.jsonl"
    legacy_feedback_db = data_dir / "feedback" / "feedback.db"
    close_after: list[Any] = []
    if legacy_feedback_db.is_file():
        feedback = SqliteFeedbackStore(
            str(legacy_feedback_db),
            feedback_path=str(feedback_jsonl),
            bad_cases_path=str(data_dir / "feedback" / "bad_cases.jsonl"),
            export_jsonl=False,
        )
        close_after.append(feedback)
    else:
        feedback = FeedbackStore(
            str(feedback_jsonl),
            str(data_dir / "feedback" / "bad_cases.jsonl"),
        )
    return {
        "feedback": feedback,
        "feedback_analysis": FeedbackAnalysisStore(
            str(data_dir / "feedback" / "feedback_analysis.jsonl")
        ),
        "derived_knowledge": DerivedKnowledgeStore(
            str(data_dir / "knowledge" / "derived_knowledge.jsonl")
        ),
        "retrieval_feedback": RetrievalFeedbackStore(
            str(data_dir / "feedback" / "retrieval_feedback.jsonl")
        ),
    }, close_after


def _target_stores(db_path: Path, scratch: Path) -> dict[str, Any]:
    return {
        "feedback": SqliteFeedbackStore(
            str(db_path),
            feedback_path=str(scratch / "empty-feedback.jsonl"),
            bad_cases_path=str(scratch / "empty-bad-cases.jsonl"),
            export_jsonl=False,
        ),
        "feedback_analysis": SqliteFeedbackAnalysisStore(str(db_path)),
        "derived_knowledge": SqliteDerivedKnowledgeStore(str(db_path)),
        "retrieval_feedback": SqliteRetrievalFeedbackStore(str(db_path)),
    }


def _close_stores(stores: list[Any] | dict[str, Any]) -> None:
    values = stores.values() if isinstance(stores, dict) else stores
    for store in values:
        close = getattr(store, "close", None)
        if callable(close):
            close()
            continue
        connection = getattr(store, "_conn", None)
        if connection is not None:
            connection.close()


def _checkpoint(path: Path) -> None:
    if not path.exists():
        return
    connection = sqlite3.connect(str(path), timeout=30)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.commit()
    finally:
        connection.close()


def _clone_database(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    target = sqlite3.connect(str(destination), timeout=30)
    try:
        if source.is_file():
            original = sqlite3.connect(str(source), timeout=30)
            try:
                original.backup(target)
            finally:
                original.close()
        target.commit()
    finally:
        target.close()


def _write_migration_marker(
    db_path: Path, source_digest: str, counts: dict[str, int]
) -> None:
    connection = sqlite3.connect(str(db_path), timeout=30)
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS state_migrations ("
            "migration_id TEXT PRIMARY KEY, version INTEGER NOT NULL, "
            "created_at TEXT NOT NULL, source_digest TEXT NOT NULL, counts_json TEXT NOT NULL)"
        )
        migration_id = f"unified-state-v{MIGRATION_VERSION}:{source_digest}"
        connection.execute(
            "INSERT OR REPLACE INTO state_migrations "
            "(migration_id, version, created_at, source_digest, counts_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                migration_id,
                MIGRATION_VERSION,
                datetime.now(timezone.utc).isoformat(),
                source_digest,
                json.dumps(counts, sort_keys=True),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _snapshot(stores: dict[str, Any]) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    records = {name: store.export_records() for name, store in stores.items()}
    summary = {
        name: {"count": len(rows), "sha256": _record_digest(rows)}
        for name, rows in records.items()
    }
    return records, summary


def _combined_digest(summary: dict[str, dict]) -> str:
    encoded = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def migrate_state(
    data_dir: Path,
    *,
    apply: bool = False,
    verify_only: bool = False,
) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    target_db = data_dir / "state.db"
    lock = acquire_single_instance_lock(str(data_dir / "kb" / ".cogdoc.lock"))
    if lock is None:
        raise MigrationError("another CogDoc process holds the data-directory lock")

    source_stores: dict[str, Any] = {}
    source_close: list[Any] = []
    try:
        source_stores, source_close = _source_stores(data_dir)
        source_records, source_summary = _snapshot(source_stores)
        source_digest = _combined_digest(source_summary)

        if verify_only and not target_db.is_file():
            raise MigrationError("state.db does not exist for verification")

        with tempfile.TemporaryDirectory(
            prefix=".state-migration-", dir=data_dir
        ) as temporary_dir:
            scratch = Path(temporary_dir)
            candidate = scratch / "candidate.db"
            if verify_only:
                _clone_database(target_db, candidate)
            else:
                _checkpoint(target_db)
                _clone_database(target_db, candidate)

            target_stores = _target_stores(candidate, scratch)
            try:
                if not verify_only:
                    for name, records in source_records.items():
                        target_stores[name].import_records(records)
                target_records, target_summary = _snapshot(target_stores)
            finally:
                _close_stores(target_stores)

            mismatches = {
                name: {
                    "source": source_summary[name],
                    "target": target_summary[name],
                }
                for name in source_summary
                if _canonical_records(source_records[name])
                != _canonical_records(target_records[name])
            }
            if mismatches:
                raise MigrationError(
                    "migrated records do not match source: "
                    + ", ".join(sorted(mismatches))
                )

            counts = {name: item["count"] for name, item in source_summary.items()}
            _write_migration_marker(candidate, source_digest, counts)
            _checkpoint(candidate)

            backup_path: Path | None = None
            if apply:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                if target_db.is_file():
                    backup_path = target_db.with_name(
                        f"{target_db.name}.pre-unified-{timestamp}.bak"
                    )
                    _clone_database(target_db, backup_path)
                for suffix in ("-wal", "-shm"):
                    Path(str(target_db) + suffix).unlink(missing_ok=True)
                os.replace(candidate, target_db)

            return {
                "ok": True,
                "operation": "verify" if verify_only else ("apply" if apply else "dry-run"),
                "data_dir": str(data_dir),
                "database": str(target_db),
                "source_digest": source_digest,
                "stores": source_summary,
                "backup": str(backup_path) if backup_path else None,
            }
    finally:
        _close_stores(source_close)
        release_single_instance_lock(lock)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(get_settings().cogdoc_data_dir)
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    try:
        report = migrate_state(
            args.data_dir, apply=args.apply, verify_only=args.verify_only
        )
        encoded = json.dumps(report, ensure_ascii=False, indent=2)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.json.with_name(args.json.name + ".tmp")
            temporary.write_text(encoded + "\n", encoding="utf-8")
            os.replace(temporary, args.json)
        print(encoded)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": type(exc).__name__, "message": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
