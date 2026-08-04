from cogdoc.api.feedback_analysis_store import (
    FeedbackAnalysisStore,
    SqliteFeedbackAnalysisStore,
)
from cogdoc.api.retrieval_feedback_store import (
    RetrievalFeedbackStore,
    SqliteRetrievalFeedbackStore,
)
from cogdoc.api.persistence import connect_sqlite


def _analysis(action: str, *, needs_review: bool, confidence: float) -> dict:
    return {
        "feedback_type": "correction",
        "sentiment": "negative",
        "target": {
            "chunk_ids": ["chunk-1"],
            "sources": [{"name": "说明书", "page": 3}],
            "source_type": "document",
        },
        "extracted_claim": "修正后的嵌套 JSON",
        "recommended_action": action,
        "weight_delta": -0.55,
        "confidence": confidence,
        "needs_review": needs_review,
    }


def test_sqlite_feedback_analysis_store_parity_and_idempotent_migration(tmp_path):
    legacy = FeedbackAnalysisStore(path=str(tmp_path / "analysis.jsonl"))
    first = legacy.record(
        "feedback-1",
        {"kb_id": "kb-1", "trace_id": "trace-1", "query": "原问题"},
        _analysis("adjust_retrieval", needs_review=True, confidence=0.8),
    )
    legacy.record(
        "feedback-2",
        {"kb_id": "kb-1", "trace_id": "trace-2", "query": "另一个问题"},
        _analysis("record_only", needs_review=False, confidence=0.4),
    )

    db_path = str(tmp_path / "state.db")
    store = SqliteFeedbackAnalysisStore(db_path)
    assert isinstance(store, FeedbackAnalysisStore)
    exported = legacy.export_records()
    assert legacy.import_records(exported) == {"imported": 0, "skipped": 2}
    assert store.import_records(exported) == {"imported": 2, "skipped": 0}
    assert store.import_records(exported) == {"imported": 0, "skipped": 2}
    assert store.counts(kb_id="kb-1") == legacy.counts(kb_id="kb-1")
    assert store.list(
        kb_id="kb-1",
        recommended_action="adjust_retrieval",
        needs_review=True,
        min_confidence=0.5,
    ) == [first]
    assert store.export_records()[0]["target"] == first["target"]
    store.close()
    store.close()

    reopened = SqliteFeedbackAnalysisStore(db_path)
    assert reopened.list(kb_id="kb-1", feedback_id="feedback-1") == [first]
    reopened.clear_kb("kb-1")
    assert reopened.counts(kb_id="kb-1")["total"] == 0
    reopened.close()


def _retrieval_payload() -> dict:
    return {
        "kb_id": "kb-1",
        "query": "  API   LIMIT  ",
        "feedback": "thumbs_down",
        "trace_id": "trace-1",
        "citations": [
            {"chunk_id": "chunk-1", "source_type": "document"},
            {"chunk_id": "chunk-2", "source_type": "feedback_knowledge"},
        ],
        "evidence": [{"chunk_id": "chunk-1", "source_type": "document"}],
    }


def test_sqlite_retrieval_feedback_store_parity_enable_and_shared_database(tmp_path):
    legacy = RetrievalFeedbackStore(path=str(tmp_path / "retrieval.jsonl"))
    original = legacy.record_from_feedback("feedback-1", _retrieval_payload())[0]

    db_path = str(tmp_path / "state.db")
    store = SqliteRetrievalFeedbackStore(db_path)
    assert isinstance(store, RetrievalFeedbackStore)
    exported = legacy.export_records()
    assert legacy.import_records(exported) == {"imported": 0, "skipped": 1}
    assert store.import_records(exported) == {"imported": 1, "skipped": 0}
    assert store.import_records(exported) == {"imported": 0, "skipped": 1}
    assert store.boosts_for_query("kb-1", "api limit") == {
        "chunk-1": -0.35,
        "chunk-2": -0.35,
    }
    listed = store.list(kb_id="kb-1")
    assert listed[0]["target_chunks"] == original["target_chunks"]
    assert listed[0]["chunk_count"] == 2
    assert store.counts(kb_id="kb-1") == {
        "total": 1,
        "enabled": 1,
        "disabled": 0,
    }

    disabled = store.set_enabled(
        original["retrieval_feedback_id"],
        False,
        actor="reviewer",
        reason="误标",
    )
    assert disabled is not None
    assert disabled["disabled_by"] == "reviewer"
    assert store.boosts_for_query("kb-1", "API LIMIT") == {}
    assert store.counts(kb_id="kb-1")["disabled"] == 1
    store.close()
    store.close()

    reopened = SqliteRetrievalFeedbackStore(db_path)
    enabled = reopened.set_enabled(original["retrieval_feedback_id"], True)
    assert enabled is not None
    assert enabled["disabled_at"] is None
    assert reopened.boosts_for_query("kb-1", "API LIMIT")["chunk-1"] == -0.35
    reopened.clear_kb("kb-1")
    assert reopened.list(kb_id="kb-1") == []
    reopened.close()


def test_feedback_sqlite_stores_share_public_connection_policy(tmp_path):
    db_path = str(tmp_path / "shared-state.db")
    connection = connect_sqlite(db_path, busy_timeout_ms=1375)
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert connection.execute("PRAGMA synchronous").fetchone()[0] == 1
    assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 1375
    connection.close()

    analysis = SqliteFeedbackAnalysisStore(db_path)
    retrieval = SqliteRetrievalFeedbackStore(db_path)
    inspector = connect_sqlite(db_path)
    table_names = {
        row[0]
        for row in inspector.execute(
            "SELECT name FROM sqlite_master WHERE type=?", ("table",)
        )
    }
    inspector.close()
    assert "feedback_analysis_records" in table_names
    assert "retrieval_feedback_records" in table_names
    analysis.close()
    retrieval.close()
