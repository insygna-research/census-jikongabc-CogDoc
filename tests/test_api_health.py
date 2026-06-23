import json
import pytest
from httpx import ASGITransport, AsyncClient
from api.app import _unhandled_error_response, create_app
from api.session_store import SessionStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def _client(app, raise_app_exceptions=True):
    transport = ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
    return AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.anyio
async def test_healthz_is_ok(monkeypatch):
    import api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    app = create_app(session_store=SessionStore())

    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            resp = await client.get("/healthz")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_readyz_reports_native_dependency(monkeypatch):
    import api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    app = create_app(session_store=SessionStore())

    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            resp = await client.get("/readyz")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "rust_core": True}


@pytest.mark.anyio
async def test_readyz_returns_503_when_native_missing(monkeypatch):
    import api.app as app_module
    import api.routes.health as health_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)

    def _missing(*symbols):
        raise RuntimeError("rust_core 未安装")

    monkeypatch.setattr(health_module, "ensure_rust_core", _missing)
    app = create_app(session_store=SessionStore())

    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            resp = await client.get("/readyz")

    assert resp.status_code == 503
    assert resp.json()["status"] == "not_ready"
    assert resp.json()["rust_core"] is False


def test_unhandled_error_response_maps_shutdown_to_503():
    resp = _unhandled_error_response(
        RuntimeError("cannot schedule new futures after shutdown")
    )
    payload = json.loads(resp.body)
    assert resp.status_code == 503
    assert payload["error_code"] == "MODEL_UNAVAILABLE"
    assert payload["details"]["error_class"] == "RuntimeError"


def test_unhandled_error_response_maps_generic_to_500_without_stack():
    resp = _unhandled_error_response(ValueError("某个内部细节"))
    payload = json.loads(resp.body)
    assert resp.status_code == 500
    assert payload["error_code"] == "INTERNAL_ERROR"
    # 不回显原始异常信息，不漏栈。
    assert "某个内部细节" not in payload["message"]
    assert "Traceback" not in payload["message"]


@pytest.mark.anyio
async def test_chat_unexpected_exception_maps_to_internal_error(monkeypatch):
    import api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)

    def boom(doc_id, query, is_local, chat_history, forced_task):
        raise ValueError("非 ChatServiceError 的意外异常")

    app = create_app(chat_runner=boom, session_store=SessionStore())

    async with app.router.lifespan_context(app):
        async with await _client(app, raise_app_exceptions=False) as client:
            resp = await client.post("/v1/chat", json={"query": "问题", "doc_id": "kb"})

    assert resp.status_code == 500
    assert resp.json()["error_code"] == "INTERNAL_ERROR"
