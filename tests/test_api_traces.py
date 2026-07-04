import json
import pytest
from httpx import ASGITransport, AsyncClient
from cogdoc.api.app import create_app
from cogdoc.observability.trace import build_trace_payload, build_trace_step


# 声明异步测试使用的后端。
@pytest.fixture
def anyio_backend():
    return "asyncio"


# 创建测试客户端。
async def _get_trace(app, trace_id: str):
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.get(f"/v1/traces/{trace_id}")


# 验证接口返回已导出的跟踪文件。
@pytest.mark.anyio
async def test_trace_endpoint_returns_exported_trace(tmp_path, monkeypatch):
    import cogdoc.api.app as app_module
    import cogdoc.api.routes.traces as traces_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    monkeypatch.setattr(
        traces_module, "trace_path", lambda trace_id: tmp_path / f"{trace_id}.json"
    )
    step = build_trace_step("intent_router", {"task_type": "qa"}, 1.0)
    payload = build_trace_payload(
        "trace-1",
        "req-1",
        "qa",
        [step],
        status="ok",
        duration_ms=2.0,
        config={"doc_id": "kb"},
    )
    (tmp_path / "trace-1.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    app = create_app()

    response = await _get_trace(app, "trace-1")

    body = response.json()
    assert response.status_code == 200
    assert body["schema_version"] == "v1"
    assert body["trace_id"] == "trace-1"
    assert body["status"] == "ok"
    assert body["config"]["doc_id"] == "kb"
    assert body["summary"]["step_count"] == 1
    assert body["steps"][0]["node_name"] == "intent_router"


# 验证缺失跟踪文件返回稳定错误。
@pytest.mark.anyio
async def test_trace_endpoint_returns_not_found_for_missing_trace(tmp_path, monkeypatch):
    import cogdoc.api.app as app_module
    import cogdoc.api.routes.traces as traces_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    monkeypatch.setattr(
        traces_module, "trace_path", lambda trace_id: tmp_path / f"{trace_id}.json"
    )
    app = create_app()

    response = await _get_trace(app, "missing")

    body = response.json()
    assert response.status_code == 404
    assert body["error_code"] == "TRACE_NOT_FOUND"


# 验证接口兼容旧版跟踪文件。
@pytest.mark.anyio
async def test_trace_endpoint_normalizes_legacy_trace(tmp_path, monkeypatch):
    import cogdoc.api.app as app_module
    import cogdoc.api.routes.traces as traces_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    monkeypatch.setattr(
        traces_module, "trace_path", lambda trace_id: tmp_path / f"{trace_id}.json"
    )
    (tmp_path / "legacy.json").write_text(
        json.dumps(
            {
                "trace_id": "legacy",
                "request_id": "req-legacy",
                "task_type": "qa",
                "steps": [{"node_name": "intent_router"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    app = create_app()

    response = await _get_trace(app, "legacy")

    body = response.json()
    assert response.status_code == 200
    assert body["schema_version"] == "v1"
    assert body["status"] == "ok"
    assert body["summary"]["step_count"] == 1


# 验证非法标识不会落到文件系统路径。
@pytest.mark.anyio
async def test_trace_endpoint_rejects_unsafe_trace_id(tmp_path, monkeypatch):
    import cogdoc.api.app as app_module
    import cogdoc.api.routes.traces as traces_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    calls = []
    monkeypatch.setattr(
        traces_module,
        "trace_path",
        lambda trace_id: calls.append(trace_id) or tmp_path / f"{trace_id}.json",
    )
    app = create_app()

    response = await _get_trace(app, "bad.trace")

    body = response.json()
    assert response.status_code == 404
    assert body["error_code"] == "TRACE_NOT_FOUND"
    assert calls == []


# 验证损坏跟踪文件返回稳定错误。
@pytest.mark.anyio
async def test_trace_endpoint_handles_corrupt_trace_file(tmp_path, monkeypatch):
    import cogdoc.api.app as app_module
    import cogdoc.api.routes.traces as traces_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    monkeypatch.setattr(
        traces_module, "trace_path", lambda trace_id: tmp_path / f"{trace_id}.json"
    )
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    app = create_app()

    response = await _get_trace(app, "bad")

    body = response.json()
    assert response.status_code == 500
    assert body["error_code"] == "INTERNAL_ERROR"
