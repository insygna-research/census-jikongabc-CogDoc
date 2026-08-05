from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from cogdoc.api.app import create_app
from cogdoc.api.derived_knowledge_store import DerivedKnowledgeStore
from cogdoc.api.feedback_analysis_store import FeedbackAnalysisStore
from cogdoc.api.feedback_store import FeedbackStore
from cogdoc.api.ingest import KnowledgeBaseRegistry
from cogdoc.api.retrieval_feedback_store import RetrievalFeedbackStore
from cogdoc.api.retrieval_eval_draft_store import RetrievalEvalDraftStore
from cogdoc.tools.retriever.derived_knowledge import DerivedKnowledgeRetriever


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _make_app(tmp_path, monkeypatch):
    import cogdoc.api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    registry = KnowledgeBaseRegistry(
        registry_path=str(tmp_path / "registry.json"),
        source_dir_for=lambda kb_id: str(tmp_path / "sources" / kb_id),
    )
    app = create_app(
        kb_registry=registry,
        feedback_store=FeedbackStore(
            feedback_path=str(tmp_path / "feedback.jsonl"),
            bad_cases_path=str(tmp_path / "bad_cases.jsonl"),
        ),
        feedback_analysis_store=FeedbackAnalysisStore(
            path=str(tmp_path / "feedback_analysis.jsonl")
        ),
        knowledge_store=DerivedKnowledgeStore(path=str(tmp_path / "knowledge.jsonl")),
        retrieval_feedback_store=RetrievalFeedbackStore(
            path=str(tmp_path / "retrieval_feedback.jsonl")
        ),
        retrieval_eval_draft_store=RetrievalEvalDraftStore(
            path=str(tmp_path / "retrieval_eval_drafts.jsonl")
        ),
    )
    # 本验收刻意走真实词法检索，避免后台向量索引刷新引入异步竞态。
    app.state.derived_knowledge_index_auto_refresh = False
    return app


@asynccontextmanager
async def _client(app):
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            yield client


@pytest.mark.anyio
async def test_feedback_knowledge_review_and_retrieval_lifecycle_e2e(
    tmp_path, monkeypatch
):
    app = _make_app(tmp_path, monkeypatch)
    retriever = DerivedKnowledgeRetriever(app.state.knowledge_store, enable_index=False)
    correction = {
        "trace_id": "feedback-knowledge-e2e",
        "feedback": "correction",
        "kb_id": "kb",
        "query": "旧版差旅费用怎么处理？",
        "answer": "差旅费用可以月底统一处理。",
        "correction_text": "差旅报销需要在七天内提交。",
        "feedback_text": "原回答引用了旧规则。",
        "save_as_knowledge": True,
        "skip_retrieval_feedback": True,
        "citations": [
            {"chunk_id": "policy-chunk-2", "source": "policy.pdf", "page": 2}
        ],
        "related_source_sha256": "sha-old",
        "related_page_start": 2,
        "related_page_end": 2,
        "related_chunk_text_hash": "chunk-hash",
        "related_anchor_text": "差旅报销",
        "created_by": "reviewer",
    }
    follow_up_query = "差旅报销提交期限"

    async with _client(app) as client:
        created = await client.post("/v1/feedback", json=correction)
        duplicate = await client.post("/v1/feedback", json=correction)

        assert created.status_code == 201
        assert duplicate.status_code == 201
        knowledge_id = created.json()["knowledge_id"]
        assert created.json()["knowledge_status"] == "pending"
        assert duplicate.json()["knowledge_id"] == knowledge_id
        assert duplicate.json()["knowledge_deduplicated"] is True
        assert len(app.state.knowledge_store.list(kb_id="kb")) == 1
        assert retriever.search("kb", follow_up_query, top_k=5) == []

        approved = await client.post(
            f"/v1/knowledge/{knowledge_id}/approve",
            json={"actor": "admin", "note": "规则已核验"},
        )
        approved_again = await client.post(
            f"/v1/knowledge/{knowledge_id}/approve",
            json={"actor": "admin", "note": "规则已核验"},
        )

        assert approved.status_code == 200
        assert approved_again.status_code == 200
        assert approved_again.json()["knowledge_id"] == knowledge_id
        assert len(app.state.knowledge_store.list(kb_id="kb")) == 1
        hits = retriever.search("kb", follow_up_query, top_k=5)
        assert len(hits) == 1
        hit = hits[0]
        assert hit["text"] == correction["correction_text"]
        assert hit["meta"]["chunk_id"] == f"knowledge:{knowledge_id}"
        assert hit["meta"]["knowledge_id"] == knowledge_id
        assert hit["meta"]["source_type"] == "derived_knowledge"
        assert hit["meta"]["source"] == f"knowledge:{knowledge_id}"
        assert hit["meta"]["related_source"] == "policy.pdf"
        assert hit["meta"]["related_chunk_ids"] == ["policy-chunk-2"]
        assert hit["meta"]["page"] == 2
        assert hit["meta"]["page_start"] == 2
        assert hit["meta"]["page_end"] == 2
        assert hit["meta"]["related_chunk_text_hash"] == "chunk-hash"
        assert hit["meta"]["related_anchor_text"] == "差旅报销"
        assert hit["retrieval"]["status_filter"] == "approved"

        stale = app.state.knowledge_store.mark_stale_for_source(
            "kb", "policy.pdf", "sha-old"
        )
        assert [row["knowledge_id"] for row in stale] == [knowledge_id]
        assert retriever.search("kb", follow_up_query, top_k=5) == []

        restored = await client.post(
            f"/v1/knowledge/{knowledge_id}/approve",
            json={
                "actor": "admin",
                "note": "新文档规则已复核",
                "related_source_sha256": "sha-new",
            },
        )
        assert restored.status_code == 200
        assert retriever.search("kb", follow_up_query, top_k=5)

        rejected = await client.post(
            f"/v1/knowledge/{knowledge_id}/reject",
            json={"actor": "admin", "note": "暂不采用"},
        )
        assert rejected.status_code == 200
        assert rejected.json()["status"] == "rejected"
        assert retriever.search("kb", follow_up_query, top_k=5) == []

        await client.post(
            f"/v1/knowledge/{knowledge_id}/approve", json={"actor": "admin"}
        )
        archived = await client.post(
            f"/v1/knowledge/{knowledge_id}/archive",
            json={"actor": "admin", "note": "规则退役"},
        )
        archived_again = await client.post(
            f"/v1/knowledge/{knowledge_id}/archive",
            json={"actor": "admin", "note": "规则退役"},
        )
        assert archived.status_code == 200
        assert archived_again.status_code == 200
        assert archived_again.json()["status"] == "archived"
        assert len(app.state.knowledge_store.list(kb_id="kb")) == 1
        assert retriever.search("kb", follow_up_query, top_k=5) == []
