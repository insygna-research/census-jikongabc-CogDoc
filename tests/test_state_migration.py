import json
from pathlib import Path
import sqlite3

from cogdoc.api.derived_knowledge_store import (
    DerivedKnowledgeStore,
    SqliteDerivedKnowledgeStore,
)
from cogdoc.api.feedback_analysis_store import SqliteFeedbackAnalysisStore
from cogdoc.api.feedback_store import SqliteFeedbackStore
from cogdoc.api.retrieval_feedback_store import SqliteRetrievalFeedbackStore
from scripts.migrate_state import migrate_state


CREATED_AT = "2026-08-04T00:00:00+00:00"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _seed_jsonl_state(data_dir: Path) -> None:
    _write_jsonl(
        data_dir / "feedback" / "feedback.jsonl",
        [
            {
                "feedback_id": "feedback-1",
                "created_at": CREATED_AT,
                "kb_id": "kb",
                "trace_id": "trace",
                "feedback": "thumbs_down",
                "feedback_type": "quick",
            }
        ],
    )
    _write_jsonl(
        data_dir / "feedback" / "feedback_analysis.jsonl",
        [
            {
                "feedback_analysis_id": "analysis-1",
                "feedback_id": "feedback-1",
                "kb_id": "kb",
                "trace_id": "trace",
                "recommended_action": "adjust_retrieval",
                "needs_review": True,
                "confidence": 0.8,
                "created_at": CREATED_AT,
            }
        ],
    )
    _write_jsonl(
        data_dir / "feedback" / "retrieval_feedback.jsonl",
        [
            {
                "retrieval_feedback_id": "retrieval-1",
                "feedback_group_key": "kb:query",
                "feedback_id": "feedback-1",
                "kb_id": "kb",
                "query_hash": "query",
                "enabled": True,
                "created_at": CREATED_AT,
            }
        ],
    )
    knowledge = DerivedKnowledgeStore(
        str(data_dir / "knowledge" / "derived_knowledge.jsonl")
    )
    knowledge.create(
        {
            "kb_id": "kb",
            "text": "migrated knowledge",
            "origin": "manual",
            "created_by": "migration-test",
        }
    )


def test_state_migration_is_atomic_verifiable_and_idempotent(tmp_path):
    data_dir = tmp_path / "data"
    _seed_jsonl_state(data_dir)
    state_db = data_dir / "state.db"
    state_db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(state_db)
    connection.execute("CREATE TABLE sentinel (value TEXT)")
    connection.execute("INSERT INTO sentinel VALUES ('preserved')")
    connection.commit()
    connection.close()

    dry_run = migrate_state(data_dir)
    assert dry_run["operation"] == "dry-run"
    connection = sqlite3.connect(state_db)
    assert connection.execute(
        "SELECT name FROM sqlite_master WHERE name='derived_knowledge_events'"
    ).fetchone() is None
    connection.close()

    applied = migrate_state(data_dir, apply=True)
    assert applied["operation"] == "apply"
    assert applied["backup"]
    assert Path(applied["backup"]).is_file()
    assert {name: item["count"] for name, item in applied["stores"].items()} == {
        "feedback": 1,
        "feedback_analysis": 1,
        "derived_knowledge": 1,
        "retrieval_feedback": 1,
    }

    connection = sqlite3.connect(state_db)
    assert connection.execute("SELECT value FROM sentinel").fetchone()[0] == "preserved"
    connection.close()
    assert len(SqliteFeedbackStore(str(state_db), export_jsonl=False).export_records()) == 1
    assert len(SqliteFeedbackAnalysisStore(str(state_db)).export_records()) == 1
    assert len(SqliteDerivedKnowledgeStore(str(state_db)).export_records()) == 1
    assert len(SqliteRetrievalFeedbackStore(str(state_db)).export_records()) == 1

    verified = migrate_state(data_dir, verify_only=True)
    assert verified["operation"] == "verify"
    repeated = migrate_state(data_dir, apply=True)
    assert repeated["stores"] == applied["stores"]
