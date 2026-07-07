from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from cogdoc.api.app import create_app
from cogdoc.api.derived_knowledge_store import DerivedKnowledgeStore
from cogdoc.api.feedback_analysis_store import FeedbackAnalysisStore
from cogdoc.api.retrieval_feedback_store import RetrievalFeedbackStore


# 声明异步测试使用的后端。
@pytest.fixture
def anyio_backend():
    return "asyncio"


# 构造应用。
def _make_app(tmp_path, monkeypatch):
    import cogdoc.api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    store = DerivedKnowledgeStore(path=str(tmp_path / "knowledge.jsonl"))
    feedback_analysis_store = FeedbackAnalysisStore(
        path=str(tmp_path / "feedback_analysis.jsonl")
    )
    retrieval_feedback_store = RetrievalFeedbackStore(
        path=str(tmp_path / "retrieval_feedback.jsonl")
    )
    return create_app(
        knowledge_store=store,
        feedback_analysis_store=feedback_analysis_store,
        retrieval_feedback_store=retrieval_feedback_store,
    )


# 构造测试客户端。
@asynccontextmanager
async def _client(app):
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            yield client


# 验证手工知识生命周期场景。
@pytest.mark.anyio
async def test_manual_knowledge_lifecycle(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)

    async with _client(app) as client:
        create = await client.post(
            "/v1/knowledge",
            json={
                "kb_id": "kb",
                "text": "入职审批需要直属经理确认。",
                "related_source": "hr.pdf",
                "related_source_sha256": "sha-old",
                "related_chunk_ids": ["c1"],
                "source_note": "HR 手工确认",
                "certainty": "high",
                "created_by": "reviewer",
            },
        )
        assert create.status_code == 201
        row = create.json()["knowledge"]
        assert row["status"] == "pending"
        assert row["origin"] == "manual_entry"
        knowledge_id = row["knowledge_id"]

        pending = await client.get(
            "/v1/knowledge", params={"kb_id": "kb", "status": "pending"}
        )
        assert pending.status_code == 200
        assert [item["knowledge_id"] for item in pending.json()["knowledge"]] == [
            knowledge_id
        ]

        approved = await client.post(
            f"/v1/knowledge/{knowledge_id}/approve",
            json={"actor": "admin", "note": "确认有效"},
        )
        assert approved.status_code == 200
        assert approved.json()["status"] == "approved"
        assert approved.json()["reviewed_by"] == "admin"

        archived = await client.post(
            f"/v1/knowledge/{knowledge_id}/archive", json={"actor": "admin"}
        )
        assert archived.status_code == 200
        assert archived.json()["status"] == "archived"
        assert archived.json()["archived_at"]


# 验证精确重复返回现有记录场景。
@pytest.mark.anyio
async def test_exact_duplicate_returns_existing_knowledge(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)

    async with _client(app) as client:
        first = await client.post(
            "/v1/knowledge", json={"kb_id": "kb", "text": "A   B"}
        )
        second = await client.post("/v1/knowledge", json={"kb_id": "kb", "text": "A B"})

        assert first.status_code == 201
        assert second.status_code == 201
        assert second.json()["deduplicated"] is True
        assert (
            first.json()["knowledge"]["knowledge_id"]
            == second.json()["knowledge"]["knowledge_id"]
        )


# 验证保存回答来源可以创建待审核知识场景。
@pytest.mark.anyio
async def test_saved_answer_origin_creates_pending_knowledge(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)

    async with _client(app) as client:
        created = await client.post(
            "/v1/knowledge",
            json={
                "kb_id": "kb",
                "text": "系统回答中的高价值结论。",
                "origin": "saved_answer",
                "created_from_trace_id": "trace-1",
                "related_chunk_ids": ["c1", "c2"],
                "source_note": "保存自问答",
            },
        )

        assert created.status_code == 201
        row = created.json()["knowledge"]
        assert row["origin"] == "saved_answer"
        assert row["status"] == "pending"
        assert row["created_from_trace_id"] == "trace-1"
        assert row["related_chunk_ids"] == ["c1", "c2"]


# 验证审核队列摘要聚合多类待处理事项场景。
@pytest.mark.anyio
async def test_review_queue_summary_counts_pending_work(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)

    app.state.feedback_analysis_store.record(
        "fb1",
        {"kb_id": "kb", "trace_id": "t1", "query": "问题"},
        {
            "feedback_type": "correction",
            "sentiment": "negative",
            "target": {"chunk_ids": [], "sources": [], "source_type": "none"},
            "extracted_claim": "正确说法",
            "recommended_action": "create_pending_knowledge",
            "weight_delta": -0.55,
            "confidence": 0.72,
            "needs_review": True,
        },
    )
    app.state.retrieval_feedback_store.record_from_feedback(
        "fb2",
        {
            "kb_id": "kb",
            "query": "问题",
            "feedback": "thumbs_down",
            "citations": [{"chunk_id": "c1", "source": "a.pdf"}],
        },
    )

    async with _client(app) as client:
        await client.post(
            "/v1/knowledge",
            json={"kb_id": "kb", "text": "待审核知识。"},
        )
        await client.post(
            "/v1/knowledge",
            json={
                "kb_id": "kb",
                "text": "保存答案知识。",
                "origin": "saved_answer",
                "enable_immediately": True,
            },
        )
        summary = await client.get("/v1/review-queue", params={"kb_id": "kb"})

    assert summary.status_code == 200
    body = summary.json()
    assert body["knowledge"]["pending"] == 1
    assert body["knowledge"]["approved"] == 1
    assert body["knowledge_origin"]["saved_answer"] == 1
    assert body["feedback_analysis"]["create_pending_knowledge"] == 1
    assert body["feedback_analysis"]["needs_review"] == 1
    assert body["retrieval_feedback"]["enabled"] == 1


# 验证批量审核报告缺失标识场景。
@pytest.mark.anyio
async def test_batch_review_reports_missing_ids(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)

    async with _client(app) as client:
        created = await client.post(
            "/v1/knowledge", json={"kb_id": "kb", "text": "知识 A"}
        )
        knowledge_id = created.json()["knowledge"]["knowledge_id"]

        batch = await client.post(
            "/v1/knowledge/batch-approve",
            json={"knowledge_ids": [knowledge_id, "missing"], "actor": "admin"},
        )

        assert batch.status_code == 200
        body = batch.json()
        assert [item["knowledge_id"] for item in body["updated"]] == [knowledge_id]
        assert body["missing_ids"] == ["missing"]
