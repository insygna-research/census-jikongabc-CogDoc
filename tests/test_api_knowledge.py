from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from cogdoc.api.app import create_app
from cogdoc.api.derived_knowledge_store import DerivedKnowledgeStore
from cogdoc.api.feedback_analysis_store import FeedbackAnalysisStore
from cogdoc.api.feedback_store import FeedbackStore
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
    feedback_store = FeedbackStore(
        feedback_path=str(tmp_path / "feedback.jsonl"),
        bad_cases_path=str(tmp_path / "bad_cases.jsonl"),
    )
    retrieval_feedback_store = RetrievalFeedbackStore(
        path=str(tmp_path / "retrieval_feedback.jsonl")
    )
    return create_app(
        knowledge_store=store,
        feedback_store=feedback_store,
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
                "related_page_start": 2,
                "related_page_end": 3,
                "related_chunk_text_hash": "hash-old",
                "related_anchor_text": "审批规则",
                "source_note": "HR 手工确认",
                "certainty": "high",
                "created_by": "reviewer",
            },
        )
        assert create.status_code == 201
        row = create.json()["knowledge"]
        assert row["status"] == "pending"
        assert row["origin"] == "manual_entry"
        assert row["related_page_start"] == 2
        assert row["related_page_end"] == 3
        assert row["related_chunk_text_hash"] == "hash-old"
        assert row["related_anchor_text"] == "审批规则"
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


# 验证相似知识会进入冲突组等待审核。
@pytest.mark.anyio
async def test_similar_knowledge_creates_conflict_group(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)

    async with _client(app) as client:
        first = await client.post(
            "/v1/knowledge",
            json={
                "kb_id": "kb",
                "text": "差旅报销需要七天内提交。",
                "enable_immediately": True,
            },
        )
        second = await client.post(
            "/v1/knowledge",
            json={
                "kb_id": "kb",
                "text": "差旅报销需要7天内提交。",
                "enable_immediately": True,
            },
        )
        listed = await client.get("/v1/knowledge", params={"kb_id": "kb"})
        conflict_only = await client.get(
            "/v1/knowledge", params={"kb_id": "kb", "has_conflict": True}
        )
        summary = await client.get("/v1/review-queue", params={"kb_id": "kb"})

    assert first.status_code == 201
    assert second.status_code == 201
    body = second.json()
    assert body["knowledge"]["status"] == "pending"
    assert body["knowledge"]["conflict_group_id"]
    assert body["requires_review"] is True
    assert (
        body["conflicts"][0]["knowledge_id"]
        == first.json()["knowledge"]["knowledge_id"]
    )
    rows = listed.json()["knowledge"]
    groups = {row["knowledge_id"]: row["conflict_group_id"] for row in rows}
    assert len(set(groups.values())) == 1
    conflict_group_id = body["knowledge"]["conflict_group_id"]
    conflict_rows = conflict_only.json()["knowledge"]
    assert {row["knowledge_id"] for row in conflict_rows} == set(groups)

    async with _client(app) as client:
        grouped = await client.get(
            "/v1/knowledge",
            params={"kb_id": "kb", "conflict_group_id": conflict_group_id},
        )

    assert grouped.status_code == 200
    assert {row["knowledge_id"] for row in grouped.json()["knowledge"]} == set(groups)
    assert summary.status_code == 200
    assert summary.json()["knowledge_conflicts"] == {
        "total": 2,
        "groups": 1,
        "pending": 1,
        "stale": 0,
    }


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


# 验证待审核知识创建会发送回调场景。
@pytest.mark.anyio
async def test_pending_knowledge_creation_emits_webhook(
    tmp_path, monkeypatch, webhook_dispatcher
):
    import cogdoc.api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    app = create_app(
        knowledge_store=DerivedKnowledgeStore(path=str(tmp_path / "knowledge.jsonl")),
        feedback_store=FeedbackStore(
            feedback_path=str(tmp_path / "feedback.jsonl"),
            bad_cases_path=str(tmp_path / "bad_cases.jsonl"),
        ),
        feedback_analysis_store=FeedbackAnalysisStore(
            path=str(tmp_path / "feedback_analysis.jsonl")
        ),
        retrieval_feedback_store=RetrievalFeedbackStore(
            path=str(tmp_path / "retrieval_feedback.jsonl")
        ),
        webhook_dispatcher=webhook_dispatcher,
    )

    async with _client(app) as client:
        pending = await client.post(
            "/v1/knowledge", json={"kb_id": "kb", "text": "待审核知识。"}
        )
        approved = await client.post(
            "/v1/knowledge",
            json={
                "kb_id": "kb",
                "text": "直接通过知识。",
                "enable_immediately": True,
            },
        )

    assert pending.status_code == 201
    assert approved.status_code == 201
    assert [event for event, _ in webhook_dispatcher.events] == [
        "knowledge.pending_created"
    ]
    payload = webhook_dispatcher.events[0][1]
    assert payload["source"] == "knowledge_create"
    assert payload["knowledge"]["status"] == "pending"
    assert payload["knowledge"]["text"] == "待审核知识。"


# 验证过期知识复核通过时刷新绑定场景。
@pytest.mark.anyio
async def test_stale_knowledge_approve_refreshes_binding(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)

    async with _client(app) as client:
        created = await client.post(
            "/v1/knowledge",
            json={
                "kb_id": "kb",
                "text": "旧文档中的规则。",
                "related_document_id": "doc-old",
                "related_source": "policy.pdf",
                "related_source_sha256": "sha-old",
                "related_chunk_ids": ["old-c1"],
                "related_page_start": 1,
                "related_page_end": 1,
                "related_chunk_text_hash": "hash-old",
                "related_anchor_text": "旧锚点",
                "enable_immediately": True,
            },
        )
        knowledge_id = created.json()["knowledge"]["knowledge_id"]
        app.state.knowledge_store.set_status(knowledge_id, "stale")

        approved = await client.post(
            f"/v1/knowledge/{knowledge_id}/approve",
            json={
                "actor": "admin",
                "note": "新版文档确认仍有效",
                "related_document_id": "doc-new",
                "related_source": "policy.pdf",
                "related_source_sha256": "sha-new",
                "related_chunk_ids": ["new-c1", "new-c2"],
                "related_page_start": 4,
                "related_page_end": 5,
                "related_chunk_text_hash": "hash-new",
                "related_anchor_text": "新版锚点",
            },
        )

    assert approved.status_code == 200
    row = approved.json()
    assert row["status"] == "approved"
    assert row["related_document_id"] == "doc-new"
    assert row["related_source_sha256"] == "sha-new"
    assert row["related_chunk_ids"] == ["new-c1", "new-c2"]
    assert row["related_page_start"] == 4
    assert row["related_page_end"] == 5
    assert row["related_chunk_text_hash"] == "hash-new"
    assert row["related_anchor_text"] == "新版锚点"
    assert row["reviewed_by"] == "admin"
    assert row["review_note"] == "新版文档确认仍有效"


# 验证绑定更新不会覆盖非绑定字段。
def test_knowledge_binding_updates_are_allowlisted(tmp_path):
    store = DerivedKnowledgeStore(path=str(tmp_path / "knowledge.jsonl"))
    row, _ = store.create(
        {
            "kb_id": "kb",
            "text": "知识",
            "status": "approved",
            "related_source_sha256": "sha-old",
        }
    )

    updated = store.set_status(
        row["knowledge_id"],
        "approved",
        binding_updates={
            "status": "rejected",
            "related_source_sha256": "sha-new",
            "related_page_start": 9,
            "related_chunk_text_hash": "hash-new",
        },
    )

    assert updated["status"] == "approved"
    assert updated["related_source_sha256"] == "sha-new"
    assert updated["related_page_start"] == 9
    assert updated["related_chunk_text_hash"] == "hash-new"


# 验证知识修订创建新版本且通过后归档旧版本。
@pytest.mark.anyio
async def test_knowledge_revision_supersedes_previous_version(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)

    async with _client(app) as client:
        created = await client.post(
            "/v1/knowledge",
            json={
                "kb_id": "kb",
                "text": "旧规则。",
                "related_source": "policy.pdf",
                "related_page_start": 1,
                "related_page_end": 2,
                "related_anchor_text": "旧规则锚点",
                "enable_immediately": True,
            },
        )
        previous = created.json()["knowledge"]
        revised = await client.post(
            f"/v1/knowledge/{previous['knowledge_id']}/revise",
            json={
                "text": "新规则。",
                "related_source": "policy-v2.pdf",
                "related_chunk_ids": ["c2"],
                "related_page_start": 5,
                "related_page_end": 6,
                "related_chunk_text_hash": "hash-v2",
                "related_anchor_text": "新规则锚点",
                "source_note": "人工修订",
                "created_by": "admin",
            },
        )
        revision = revised.json()["knowledge"]
        approved = await client.post(
            f"/v1/knowledge/{revision['knowledge_id']}/approve",
            json={"actor": "admin", "note": "新版确认"},
        )
        archived = await client.get(
            "/v1/knowledge", params={"kb_id": "kb", "status": "archived"}
        )

    assert created.status_code == 201
    assert revised.status_code == 201
    assert revision["version"] == 2
    assert revision["previous_version_id"] == previous["knowledge_id"]
    assert revision["status"] == "pending"
    assert revision["related_source"] == "policy-v2.pdf"
    assert revision["related_chunk_ids"] == ["c2"]
    assert revision["related_page_start"] == 5
    assert revision["related_page_end"] == 6
    assert revision["related_chunk_text_hash"] == "hash-v2"
    assert revision["related_anchor_text"] == "新规则锚点"
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"
    archived_rows = archived.json()["knowledge"]
    assert [row["knowledge_id"] for row in archived_rows] == [previous["knowledge_id"]]
    assert archived_rows[0]["review_note"].startswith("由新版本 ")


# 验证立即启用修订版本会归档旧版本。
@pytest.mark.anyio
async def test_knowledge_revision_enable_immediately_archives_previous(
    tmp_path, monkeypatch
):
    app = _make_app(tmp_path, monkeypatch)

    async with _client(app) as client:
        created = await client.post(
            "/v1/knowledge",
            json={"kb_id": "kb", "text": "旧规则。", "enable_immediately": True},
        )
        previous = created.json()["knowledge"]
        revised = await client.post(
            f"/v1/knowledge/{previous['knowledge_id']}/revise",
            json={
                "text": "新规则。",
                "enable_immediately": True,
                "created_by": "admin",
            },
        )
        archived = await client.get(
            "/v1/knowledge", params={"kb_id": "kb", "status": "archived"}
        )

    assert revised.status_code == 201
    assert revised.json()["knowledge"]["status"] == "approved"
    archived_rows = archived.json()["knowledge"]
    assert [row["knowledge_id"] for row in archived_rows] == [previous["knowledge_id"]]


# 验证待审核和驳回知识不能修订。
@pytest.mark.anyio
async def test_knowledge_revision_rejects_non_reviewed_statuses(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)

    async with _client(app) as client:
        pending = await client.post(
            "/v1/knowledge", json={"kb_id": "kb", "text": "待审核知识。"}
        )
        pending_id = pending.json()["knowledge"]["knowledge_id"]
        pending_revision = await client.post(
            f"/v1/knowledge/{pending_id}/revise", json={"text": "新知识。"}
        )
        await client.post(f"/v1/knowledge/{pending_id}/reject", json={})
        rejected_revision = await client.post(
            f"/v1/knowledge/{pending_id}/revise", json={"text": "新知识。"}
        )

    assert pending_revision.status_code == 400
    assert rejected_revision.status_code == 400


# 验证知识修订拒绝活跃重复文本。
@pytest.mark.anyio
async def test_knowledge_revision_rejects_duplicate_active_text(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)

    async with _client(app) as client:
        first = await client.post(
            "/v1/knowledge",
            json={"kb_id": "kb", "text": "第一条。", "enable_immediately": True},
        )
        await client.post(
            "/v1/knowledge",
            json={"kb_id": "kb", "text": "第二条。", "enable_immediately": True},
        )
        knowledge_id = first.json()["knowledge"]["knowledge_id"]
        duplicate = await client.post(
            f"/v1/knowledge/{knowledge_id}/revise", json={"text": "第二条。"}
        )

    assert duplicate.status_code == 400


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
    app.state.feedback_store.record(
        {
            "kb_id": "kb",
            "trace_id": "t0",
            "query": "问题",
            "feedback": "thumbs_down",
        }
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
        export = await client.get(
            "/v1/review-queue/export",
            params={
                "kb_id": "kb",
                "knowledge_origin": "manual_entry",
                "limit": 50,
            },
        )

    assert summary.status_code == 200
    body = summary.json()
    assert body["knowledge"]["pending"] == 1
    assert body["knowledge"]["approved"] == 1
    assert body["knowledge_origin"]["saved_answer"] == 1
    assert body["feedback_counts"]["total"] == 1
    assert body["feedback_counts"]["bad_cases"] == 1
    assert body["feedback_analysis"]["create_pending_knowledge"] == 1
    assert body["feedback_analysis"]["needs_review"] == 1
    assert body["retrieval_feedback"]["enabled"] == 1
    assert export.status_code == 200
    exported = export.json()
    assert exported["kb_id"] == "kb"
    assert exported["summary"]["feedback_counts"]["bad_cases"] == 1
    assert exported["summary"]["knowledge_origin"]["manual_entry"] == 1
    assert len(exported["pending_knowledge"]) == 1
    assert exported["pending_knowledge"][0]["text"] == "待审核知识。"
    assert exported["stale_knowledge"] == []
    assert len(exported["feedback_analysis_needs_review"]) == 1
    assert len(exported["retrieval_feedback_enabled"]) == 1
    assert len(exported["feedback_bad_cases"]) == 1


# 验证待审核计数和反馈闭环指标场景。
@pytest.mark.anyio
async def test_review_metrics_endpoints_report_feedback_loop(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)

    app.state.feedback_store.record(
        {"kb_id": "kb", "trace_id": "t1", "feedback": "thumbs_down"}
    )
    app.state.feedback_store.record(
        {"kb_id": "kb", "trace_id": "t2", "feedback": "correction"}
    )
    app.state.feedback_analysis_store.record(
        "fb1",
        {"kb_id": "kb", "trace_id": "t2", "query": "问题"},
        {
            "feedback_type": "correction",
            "sentiment": "negative",
            "target": {"chunk_ids": [], "sources": [], "source_type": "none"},
            "extracted_claim": "正确说法",
            "recommended_action": "create_pending_knowledge",
            "weight_delta": -0.55,
            "confidence": 0.9,
            "needs_review": True,
        },
    )
    records = app.state.retrieval_feedback_store.record_from_feedback(
        "fb2",
        {
            "kb_id": "kb",
            "query": "问题",
            "feedback": "thumbs_down",
            "citations": [{"chunk_id": "c1", "source": "a.pdf"}],
        },
    )
    app.state.retrieval_feedback_store.set_enabled(
        records[0]["retrieval_feedback_id"], False
    )

    async with _client(app) as client:
        approved = await client.post(
            "/v1/knowledge",
            json={"kb_id": "kb", "text": "已通过知识。", "enable_immediately": True},
        )
        rejected = await client.post(
            "/v1/knowledge", json={"kb_id": "kb", "text": "驳回知识。"}
        )
        await client.post(
            f"/v1/knowledge/{rejected.json()['knowledge']['knowledge_id']}/reject",
            json={},
        )
        stale_id = approved.json()["knowledge"]["knowledge_id"]
        app.state.knowledge_store.set_status(stale_id, "stale")
        pending_count = await client.get(
            "/v1/knowledge/pending-count", params={"kb_id": "kb"}
        )
        metrics = await client.get(
            "/v1/feedback-loop-metrics",
            params={"kb_id": "kb", "answer_count": 4},
        )
        app.state.knowledge_store.set_status(stale_id, "approved")
        reviewed_metrics = await client.get(
            "/v1/feedback-loop-metrics", params={"kb_id": "kb"}
        )

    assert pending_count.status_code == 200
    assert pending_count.json()["stale"] == 1
    assert pending_count.json()["feedback_analysis_needs_review"] == 1
    assert pending_count.json()["total"] == 2
    assert metrics.status_code == 200
    body = metrics.json()
    assert body["counts"]["feedback_total"] == 2
    assert body["counts"]["negative_feedback_total"] == 2
    assert body["rates"]["feedback_rate"] == 0.5
    assert body["rates"]["negative_feedback_rate"] == 0.5
    assert body["rates"]["pending_rejection_rate"] == 0.5
    assert body["rates"]["feedback_to_pending_rate"] == 1.0
    assert body["rates"]["retrieval_feedback_rollback_rate"] == 1.0
    assert body["rates"]["stale_review_completion_rate"] == 0.0
    assert reviewed_metrics.json()["rates"]["stale_review_completion_rate"] == 1.0


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


# 验证批量审核拒绝绑定字段场景。
@pytest.mark.anyio
async def test_batch_review_rejects_binding_fields(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)

    async with _client(app) as client:
        created = await client.post(
            "/v1/knowledge", json={"kb_id": "kb", "text": "知识 A"}
        )
        knowledge_id = created.json()["knowledge"]["knowledge_id"]

        batch = await client.post(
            "/v1/knowledge/batch-approve",
            json={
                "knowledge_ids": [knowledge_id],
                "actor": "admin",
                "related_source_sha256": "sha-new",
            },
        )

    assert batch.status_code == 422
