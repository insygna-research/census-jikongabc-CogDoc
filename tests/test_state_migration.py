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
from cogdoc.api.retrieval_eval_draft_store import (
    RetrievalEvalDraftStore,
    SqliteRetrievalEvalDraftStore,
)
from cogdoc.tools.eval.retrieval_eval_drafts import (
    EvidenceUnitDraft,
    EvidenceUnitTask,
    create_pending_draft,
)
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
    draft_store = RetrievalEvalDraftStore(
        str(data_dir / "feedback" / "retrieval_eval_drafts.jsonl")
    )
    draft_store.ensure(
        create_pending_draft(
            kb_id="kb",
            query="总结报名规则",
            units=[
                EvidenceUnitDraft(
                    unit_id="eligibility-summary",
                    task_kind=EvidenceUnitTask.SUMMARY_SECTION,
                    label="报名资格",
                    retrieval_query="报名资格 年龄 条件",
                    recovery_query="参赛者 年龄限制",
                    source="rules.pdf",
                    dimension_id="eligibility",
                )
            ],
            index_generation="generation-1",
            index_build_version="hybrid-v2",
            chunk_identity_version="chunk-v5",
            source_versions=[{"source": "rules.pdf", "sha256": "sha-rules"}],
            origin_feedback_id="feedback-1",
            now=CREATED_AT,
        )
    )
    draft_store.close()


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
    assert (
        connection.execute(
            "SELECT name FROM sqlite_master WHERE name='derived_knowledge_events'"
        ).fetchone()
        is None
    )
    assert (
        connection.execute(
            "SELECT name FROM sqlite_master WHERE name='retrieval_eval_drafts'"
        ).fetchone()
        is None
    )
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
        "retrieval_eval_drafts": 1,
    }

    connection = sqlite3.connect(state_db)
    assert connection.execute("SELECT value FROM sentinel").fetchone()[0] == "preserved"
    connection.close()
    assert (
        len(SqliteFeedbackStore(str(state_db), export_jsonl=False).export_records())
        == 1
    )
    assert len(SqliteFeedbackAnalysisStore(str(state_db)).export_records()) == 1
    assert len(SqliteDerivedKnowledgeStore(str(state_db)).export_records()) == 1
    assert len(SqliteRetrievalFeedbackStore(str(state_db)).export_records()) == 1
    migrated_drafts = SqliteRetrievalEvalDraftStore(str(state_db)).export_records()
    assert len(migrated_drafts) == 1
    assert migrated_drafts[0]["status"] == "pending"
    assert migrated_drafts[0]["units"][0]["task_kind"] == "summary_section"

    verified = migrate_state(data_dir, verify_only=True)
    assert verified["operation"] == "verify"
    repeated = migrate_state(data_dir, apply=True)
    assert repeated["stores"] == applied["stores"]
    reopened_drafts = SqliteRetrievalEvalDraftStore(str(state_db)).export_records()
    assert reopened_drafts == migrated_drafts
    connection = sqlite3.connect(state_db)
    assert (
        connection.execute("SELECT COUNT(*) FROM state_migrations").fetchone()[0] == 1
    )
    assert connection.execute("SELECT version FROM state_migrations").fetchone()[0] == 2
    connection.close()
