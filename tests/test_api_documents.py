import asyncio
import time
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from api.app import create_app
from api.ingest import IndexJobManager, KnowledgeBaseRegistry


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _ok_ingest(kb_id, source_dir):
    return SimpleNamespace(document_count=1, chunk_count=3)


def _make_app(tmp_path, ingest_fn=_ok_ingest, monkeypatch=None):
    if monkeypatch is not None:
        import api.app as app_module

        monkeypatch.setattr(app_module, "configure_logging", lambda: None)

    def source_dir_for(kb_id: str) -> str:
        return str(tmp_path / "kb" / kb_id / "sources")

    registry = KnowledgeBaseRegistry(
        registry_path=str(tmp_path / "registry.json"), source_dir_for=source_dir_for
    )
    jobs = IndexJobManager(ingest_fn=ingest_fn, source_dir_for=source_dir_for)
    app = create_app(kb_registry=registry, index_jobs=jobs)
    return app, source_dir_for


async def _client(app):
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver")


async def _wait_job(client, job_id, timeout=2.0):
    deadline = time.time() + timeout
    resp = await client.get(f"/v1/index-jobs/{job_id}")
    while time.time() < deadline and resp.json()["status"] in ("pending", "running"):
        await asyncio.sleep(0.02)
        resp = await client.get(f"/v1/index-jobs/{job_id}")
    return resp


def test_registry_recovers_from_corrupt_file(tmp_path):
    reg_path = tmp_path / "registry.json"
    reg_path.write_text("{ 半截损坏的 json", encoding="utf-8")

    # 损坏的 registry 不应让构造（即 create_app 启动）抛异常。
    registry = KnowledgeBaseRegistry(
        registry_path=str(reg_path), source_dir_for=lambda kb: str(tmp_path / kb)
    )
    assert registry.list() == []

    registry.create("kb1")
    reloaded = KnowledgeBaseRegistry(
        registry_path=str(reg_path), source_dir_for=lambda kb: str(tmp_path / kb)
    )
    assert [r["kb_id"] for r in reloaded.list()] == ["kb1"]


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
    assert not_pdf_ext.status_code == 400 and not_pdf_ext.json()["error_code"] == "INVALID_PDF"
    assert bad_magic.status_code == 400 and bad_magic.json()["error_code"] == "INVALID_PDF"


@pytest.mark.anyio
async def test_upload_rejects_oversize(tmp_path, monkeypatch):
    app, _ = _make_app(tmp_path, monkeypatch=monkeypatch)
    import api.routes.documents as docs_module

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


@pytest.mark.anyio
async def test_delete_document_and_job_not_found(tmp_path, monkeypatch):
    app, source_dir_for = _make_app(tmp_path, monkeypatch=monkeypatch)
    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            await client.post("/v1/knowledge-bases", json={"kb_id": "kb"})
            await client.post(
                "/v1/knowledge-bases/kb/documents",
                files={"file": ("a.pdf", b"%PDF-1.4 data", "application/pdf")},
            )
            deleted = await client.delete("/v1/knowledge-bases/kb/documents/a.pdf")
            delete_missing = await client.delete(
                "/v1/knowledge-bases/kb/documents/ghost.pdf"
            )
            job_missing = await client.get("/v1/index-jobs/does-not-exist")

    assert deleted.status_code == 202
    assert delete_missing.status_code == 404
    assert delete_missing.json()["error_code"] == "DOCUMENT_NOT_FOUND"
    assert job_missing.status_code == 404
    assert job_missing.json()["error_code"] == "JOB_NOT_FOUND"


@pytest.mark.anyio
async def test_ingest_failure_marks_job_failed(tmp_path, monkeypatch):
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
