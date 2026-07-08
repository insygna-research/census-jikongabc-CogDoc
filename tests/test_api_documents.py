import asyncio
import time
from types import SimpleNamespace
import pytest
from httpx import ASGITransport, AsyncClient
from cogdoc.api.app import create_app
from cogdoc.api.derived_knowledge_store import DerivedKnowledgeStore
from cogdoc.api.feedback_analysis_store import FeedbackAnalysisStore
from cogdoc.api.feedback_store import FeedbackStore
from cogdoc.api.ingest import IndexJobManager, KnowledgeBaseRegistry
from cogdoc.api.retrieval_feedback_store import RetrievalFeedbackStore


# 指定异步测试后端。
@pytest.fixture
def anyio_backend():
    return "asyncio"


# 模拟成功ingest。
def _ok_ingest(kb_id, source_dir):
    return SimpleNamespace(document_count=1, chunk_count=3)


# 构造应用。
def _make_app(tmp_path, ingest_fn=_ok_ingest, monkeypatch=None):
    if monkeypatch is not None:
        import cogdoc.api.app as app_module

        monkeypatch.setattr(app_module, "configure_logging", lambda: None)

    # 返回目录for。
    def source_dir_for(kb_id: str) -> str:
        return str(tmp_path / "kb" / kb_id / "sources")

    registry = KnowledgeBaseRegistry(
        registry_path=str(tmp_path / "registry.json"), source_dir_for=source_dir_for
    )
    jobs = IndexJobManager(
        ingest_fn=ingest_fn,
        source_dir_for=source_dir_for,
        kb_exists=registry.exists,
    )
    app = create_app(
        kb_registry=registry,
        index_jobs=jobs,
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
    )
    return app, source_dir_for


# 创建测试客户端。
async def _client(app):
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver")


# 等待任务。
async def _wait_job(client, job_id, timeout=2.0):
    deadline = time.time() + timeout
    resp = await client.get(f"/v1/index-jobs/{job_id}")
    while time.time() < deadline and resp.json()["status"] in ("pending", "running"):
        await asyncio.sleep(0.02)
        resp = await client.get(f"/v1/index-jobs/{job_id}")
    return resp


# 验证 registry corrupt quarantines and fails closed。
def test_registry_corrupt_quarantines_and_fails_closed(tmp_path):
    from cogdoc.api.ingest import RegistryCorruptError

    reg_path = tmp_path / "registry.json"
    reg_path.write_text("{ 半截损坏的 json", encoding="utf-8")

    # 损坏的 registry 不再静默退回空表（否则现存 KB 全消失、同名重建复用旧数据），而是隔离并 fail-closed 抛错。
    with pytest.raises(RegistryCorruptError):
        KnowledgeBaseRegistry(
            registry_path=str(reg_path), source_dir_for=lambda kb: str(tmp_path / kb)
        )
    # 损坏文件被改名留存供人工恢复。
    assert list(tmp_path.glob("registry.json.corrupt-*"))


# 验证 create rolls back when lifecycle finalize fails。
def test_create_rolls_back_when_lifecycle_finalize_fails(tmp_path, monkeypatch):
    from cogdoc.api.ingest import KnowledgeBaseRegistry

    source_dir = tmp_path / "kb" / "sources"
    registry = KnowledgeBaseRegistry(
        registry_path=str(tmp_path / "registry.json"),
        source_dir_for=lambda _: str(source_dir),
    )
    monkeypatch.setattr(
        "cogdoc.api.ingest.shared_lifecycle_store",
        lambda: type(
            "BrokenLifecycle",
            (),
            {"set": lambda *args: (_ for _ in ()).throw(OSError("disk"))},
        )(),
    )
    with pytest.raises(OSError, match="disk"):
        registry.create("kb")
    assert not registry.exists("kb")
    assert not source_dir.parent.exists()


# 验证 create list get knowledge base。
@pytest.mark.anyio
async def test_create_list_get_knowledge_base(tmp_path, monkeypatch):
    app, _ = _make_app(tmp_path, monkeypatch=monkeypatch)
    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            created = await client.post("/v1/knowledge-bases", json={"kb_id": "papers"})
            dup = await client.post("/v1/knowledge-bases", json={"kb_id": "papers"})
            listed = await client.get("/v1/knowledge-bases")
            got = await client.get("/v1/knowledge-bases/papers")
            missing = await client.get("/v1/knowledge-bases/nope")

    assert created.status_code == 201
    assert created.json()["kb_id"] == "papers"
    assert created.json()["tenant_id"] == "default"
    assert dup.status_code == 409 and dup.json()["error_code"] == "KB_EXISTS"
    assert [kb["kb_id"] for kb in listed.json()] == ["papers"]
    assert got.status_code == 200 and got.json()["document_count"] == 0
    assert missing.status_code == 404 and missing.json()["error_code"] == "KB_NOT_FOUND"


# 验证 delete knowledge base。
@pytest.mark.anyio
async def test_delete_knowledge_base(tmp_path, monkeypatch):
    import os

    app, source_dir_for = _make_app(tmp_path, monkeypatch=monkeypatch)
    import cogdoc.api.routes.documents as docs_module

    monkeypatch.setattr(
        docs_module, "delete_kb_index_transactional", lambda kb_id: None
    )

    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            await client.post("/v1/knowledge-bases", json={"kb_id": "kb"})
            deleted = await client.delete("/v1/knowledge-bases/kb")
            after = await client.get("/v1/knowledge-bases")
            missing = await client.delete("/v1/knowledge-bases/ghost")

    assert deleted.status_code == 204
    assert after.json() == []
    assert not os.path.exists(os.path.dirname(source_dir_for("kb")))
    assert missing.status_code == 404 and missing.json()["error_code"] == "KB_NOT_FOUND"


# 验证删除 KB 会清理审核队列状态，同名重建不继承旧反馈。
@pytest.mark.anyio
async def test_delete_recreated_kb_clears_review_state(tmp_path, monkeypatch):
    app, _ = _make_app(tmp_path, monkeypatch=monkeypatch)
    import cogdoc.api.routes.documents as docs_module

    monkeypatch.setattr(
        docs_module, "delete_kb_index_transactional", lambda kb_id: None
    )

    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            await client.post("/v1/knowledge-bases", json={"kb_id": "kb"})
            await client.post(
                "/v1/knowledge", json={"kb_id": "kb", "text": "待审核知识。"}
            )
            app.state.feedback_store.record(
                {
                    "kb_id": "kb",
                    "trace_id": "t1",
                    "feedback": "thumbs_down",
                    "query": "问题",
                }
            )
            app.state.feedback_analysis_store.record(
                "fb1",
                {"kb_id": "kb", "trace_id": "t1", "query": "问题"},
                {
                    "feedback_type": "correction",
                    "sentiment": "negative",
                    "target": {
                        "chunk_ids": ["c1"],
                        "sources": ["a.pdf"],
                        "source_type": "document",
                    },
                    "extracted_claim": "正确说法",
                    "recommended_action": "create_pending_knowledge",
                    "weight_delta": -0.55,
                    "confidence": 0.9,
                    "needs_review": True,
                },
            )
            app.state.retrieval_feedback_store.record_from_feedback(
                "fb1",
                {
                    "kb_id": "kb",
                    "query": "问题",
                    "feedback": "thumbs_down",
                    "citations": [{"chunk_id": "c1"}],
                },
            )

            before = await client.get("/v1/review-queue", params={"kb_id": "kb"})
            deleted = await client.delete("/v1/knowledge-bases/kb")
            recreated = await client.post("/v1/knowledge-bases", json={"kb_id": "kb"})
            summary = await client.get("/v1/review-queue", params={"kb_id": "kb"})
            pending_count = await client.get(
                "/v1/knowledge/pending-count", params={"kb_id": "kb"}
            )

    assert before.json()["feedback_counts"]["total"] == 1
    assert before.json()["retrieval_feedback"]["enabled"] == 1
    assert deleted.status_code == 204
    assert recreated.status_code == 201
    body = summary.json()
    assert body["knowledge"] == {}
    assert body["feedback_counts"]["total"] == 0
    assert body["feedback_counts"]["bad_cases"] == 0
    assert body["feedback_analysis"]["needs_review"] == 0
    assert body["retrieval_feedback"]["enabled"] == 0
    assert pending_count.json()["total"] == 0


# 验证 delete kb cleanup failure keeps kb。
@pytest.mark.anyio
async def test_delete_kb_cleanup_failure_keeps_kb(tmp_path, monkeypatch):
    from cogdoc.service.ingest_service import KBCleanupError

    app, _ = _make_app(tmp_path, monkeypatch=monkeypatch)
    import cogdoc.api.routes.documents as docs_module

    # 模拟失败路径。
    def boom(kb_id):
        raise KBCleanupError("部分代清理失败")

    monkeypatch.setattr(docs_module, "delete_kb_index_transactional", boom)

    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            await client.post("/v1/knowledge-bases", json={"kb_id": "kb"})
            resp = await client.delete("/v1/knowledge-bases/kb")
            after = await client.get("/v1/knowledge-bases")

    # 清理失败：返回可重试错误，KB 仍存在于 registry。
    assert resp.status_code == 500
    assert resp.json()["error_code"] == "KB_CLEANUP_FAILED"
    assert [kb["kb_id"] for kb in after.json()] == ["kb"]


# 验证 create kb rejects overlong id。
@pytest.mark.anyio
async def test_create_kb_rejects_overlong_id(tmp_path, monkeypatch):
    app, _ = _make_app(tmp_path, monkeypatch=monkeypatch)
    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            too_long = await client.post(
                "/v1/knowledge-bases", json={"kb_id": "x" * 57}
            )
            boundary = await client.post(
                "/v1/knowledge-bases", json={"kb_id": "y" * 56}
            )

    # 超过 56 字符会让 col-{kb_id} 截断撞库，必须在契约层挡掉。
    assert too_long.status_code == 422
    assert boundary.status_code == 201


# 验证 upload triggers job until succeeded。
@pytest.mark.anyio
async def test_upload_triggers_job_until_succeeded(tmp_path, monkeypatch):
    app, source_dir_for = _make_app(tmp_path, monkeypatch=monkeypatch)
    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            await client.post("/v1/knowledge-bases", json={"kb_id": "kb"})
            up = await client.post(
                "/v1/knowledge-bases/kb/documents",
                files={"file": ("a.pdf", b"%PDF-1.4 fake content", "application/pdf")},
            )
            job_id = up.json()["job_id"]
            done = await _wait_job(client, job_id)

    assert up.status_code == 202 and up.json()["job_id"]
    assert up.json()["status"] in ("pending", "running", "succeeded")
    assert done.json()["status"] == "succeeded"
    assert done.json()["document_count"] == 1 and done.json()["chunk_count"] == 3
    import os

    assert os.path.exists(os.path.join(source_dir_for("kb"), "a.pdf"))


# 验证 upload rejects bad inputs。
@pytest.mark.anyio
async def test_upload_rejects_bad_inputs(tmp_path, monkeypatch):
    app, _ = _make_app(tmp_path, monkeypatch=monkeypatch)
    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            to_missing = await client.post(
                "/v1/knowledge-bases/nope/documents",
                files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
            )
            await client.post("/v1/knowledge-bases", json={"kb_id": "kb"})
            not_pdf_ext = await client.post(
                "/v1/knowledge-bases/kb/documents",
                files={"file": ("a.txt", b"%PDF-1.4", "text/plain")},
            )
            bad_magic = await client.post(
                "/v1/knowledge-bases/kb/documents",
                files={"file": ("a.pdf", b"not a pdf", "application/pdf")},
            )

    assert to_missing.status_code == 404
    assert (
        not_pdf_ext.status_code == 400
        and not_pdf_ext.json()["error_code"] == "INVALID_PDF"
    )
    assert (
        bad_magic.status_code == 400 and bad_magic.json()["error_code"] == "INVALID_PDF"
    )


# 验证 upload rejects oversize。
@pytest.mark.anyio
async def test_upload_rejects_oversize(tmp_path, monkeypatch):
    app, _ = _make_app(tmp_path, monkeypatch=monkeypatch)
    import cogdoc.api.routes.documents as docs_module

    monkeypatch.setattr(
        docs_module, "get_settings", lambda: SimpleNamespace(max_upload_mb=0)
    )
    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            await client.post("/v1/knowledge-bases", json={"kb_id": "kb"})
            resp = await client.post(
                "/v1/knowledge-bases/kb/documents",
                files={"file": ("a.pdf", b"%PDF-1.4 data", "application/pdf")},
            )

    assert resp.status_code == 413 and resp.json()["error_code"] == "FILE_TOO_LARGE"


# 验证 delete document and job not found。
@pytest.mark.anyio
async def test_delete_document_and_job_not_found(tmp_path, monkeypatch):
    app, source_dir_for = _make_app(tmp_path, monkeypatch=monkeypatch)
    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            await client.post("/v1/knowledge-bases", json={"kb_id": "kb"})
            up = await client.post(
                "/v1/knowledge-bases/kb/documents",
                files={"file": ("a.pdf", b"%PDF-1.4 data", "application/pdf")},
            )
            # 文件写入在 executor 内异步完成；等 upload job 结束后文件才落盘。
            await _wait_job(client, up.json()["job_id"])
            deleted = await client.delete("/v1/knowledge-bases/kb/documents/a.pdf")
            delete_missing = await client.delete(
                "/v1/knowledge-bases/kb/documents/ghost.pdf"
            )
            job_missing = await client.get("/v1/index-jobs/does-not-exist")

    # delete_document 始终 202；文档不存在时 job 以 DOCUMENT_NOT_FOUND 状态失败。
    assert deleted.status_code == 202
    assert delete_missing.status_code == 202
    assert job_missing.status_code == 404
    assert job_missing.json()["error_code"] == "JOB_NOT_FOUND"


# 验证 delete missing document job fails with not found。
@pytest.mark.anyio
async def test_delete_missing_document_job_fails_with_not_found(tmp_path, monkeypatch):
    app, _ = _make_app(tmp_path, monkeypatch=monkeypatch)
    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            await client.post("/v1/knowledge-bases", json={"kb_id": "kb"})
            resp = await client.delete("/v1/knowledge-bases/kb/documents/ghost.pdf")
            assert resp.status_code == 202
            done = await _wait_job(client, resp.json()["job_id"])
    assert done.json()["status"] == "failed"
    assert done.json()["error_code"] == "DOCUMENT_NOT_FOUND"


# 验证 ingest failure marks job failed。
@pytest.mark.anyio
async def test_ingest_failure_marks_job_failed(tmp_path, monkeypatch):
    # 模拟失败路径。
    def boom(kb_id, source_dir):
        raise ValueError("解析崩了")

    app, _ = _make_app(tmp_path, ingest_fn=boom, monkeypatch=monkeypatch)
    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            await client.post("/v1/knowledge-bases", json={"kb_id": "kb"})
            up = await client.post(
                "/v1/knowledge-bases/kb/documents",
                files={"file": ("a.pdf", b"%PDF-1.4 data", "application/pdf")},
            )
            done = await _wait_job(client, up.json()["job_id"])

    assert done.json()["status"] == "failed"
    assert done.json()["error_code"] == "INGEST_FAILED"
