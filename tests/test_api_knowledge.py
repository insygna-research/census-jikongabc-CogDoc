from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from cogdoc.api.app import create_app
from cogdoc.api.derived_knowledge_store import DerivedKnowledgeStore


# 声明异步测试使用的后端。
@pytest.fixture
def anyio_backend():
    return "asyncio"


# 构造应用。
def _make_app(tmp_path, monkeypatch):
    import cogdoc.api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    store = DerivedKnowledgeStore(path=str(tmp_path / "knowledge.jsonl"))
    return create_app(knowledge_store=store)


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
