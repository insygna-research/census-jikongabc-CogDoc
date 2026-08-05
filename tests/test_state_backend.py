from cogdoc.api.app import (
    _default_feedback_analysis_store,
    _default_feedback_store,
    _default_knowledge_store,
    _default_retrieval_feedback_store,
    _default_retrieval_eval_draft_store,
)
from cogdoc.api.derived_knowledge_store import SqliteDerivedKnowledgeStore
from cogdoc.api.feedback_analysis_store import SqliteFeedbackAnalysisStore
from cogdoc.api.feedback_store import SqliteFeedbackStore
from cogdoc.api.retrieval_feedback_store import SqliteRetrievalFeedbackStore
from cogdoc.api.retrieval_eval_draft_store import SqliteRetrievalEvalDraftStore
from cogdoc.config.settings import get_settings


def test_unified_sqlite_backend_uses_one_state_database(monkeypatch, tmp_path):
    monkeypatch.setenv("COGDOC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("COGDOC_STATE_BACKEND", "sqlite")
    get_settings.cache_clear()
    stores = [
        _default_feedback_store(),
        _default_feedback_analysis_store(),
        _default_knowledge_store(),
        _default_retrieval_feedback_store(),
        _default_retrieval_eval_draft_store(),
    ]
    try:
        assert isinstance(stores[0], SqliteFeedbackStore)
        assert isinstance(stores[1], SqliteFeedbackAnalysisStore)
        assert isinstance(stores[2], SqliteDerivedKnowledgeStore)
        assert isinstance(stores[3], SqliteRetrievalFeedbackStore)
        assert isinstance(stores[4], SqliteRetrievalEvalDraftStore)
        database_paths = {
            store._conn.execute("PRAGMA database_list").fetchone()[2]
            for store in stores
        }
        assert database_paths == {str(tmp_path / "state.db")}
    finally:
        for store in stores:
            store._conn.close()
        get_settings.cache_clear()
