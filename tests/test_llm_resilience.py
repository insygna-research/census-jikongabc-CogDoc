import pytest
from httpx import ASGITransport, AsyncClient
from cogdoc.api.app import create_app
from cogdoc.api.error_mapping import classify_error_code, status_for_code
from cogdoc.api.schemas import ErrorCode
from cogdoc.api.session_store import SessionStore
from cogdoc.service.chat_service import ChatServiceError


# 声明异步测试使用的后端。
@pytest.fixture
def anyio_backend():
    return "asyncio"


# 验证 stream stage stays interrupted even for timeout 场景。
def test_stream_stage_stays_interrupted_even_for_timeout():
    # 既有契约：流中断永远是 STREAM_INTERRUPTED，不被底层成因改写。
    code = classify_error_code("stream", "TimeoutError", "读取超时")
    assert code is ErrorCode.STREAM_INTERRUPTED
    assert status_for_code(code) == 502


# 验证 runtime stage classifies upstream cause 场景。
@pytest.mark.parametrize(
    "error_class, message, expected",
    [
        ("APITimeoutError", "request timed out", ErrorCode.LLM_TIMEOUT),
        ("ReadTimeout", "", ErrorCode.LLM_TIMEOUT),
        ("RuntimeError", "connection timeout after 90s", ErrorCode.LLM_TIMEOUT),
        ("RateLimitError", "rate limit exceeded", ErrorCode.RATE_LIMITED),
        ("APIError", "Error code: 429 Too Many Requests", ErrorCode.RATE_LIMITED),
        ("ValueError", "模型不可用", ErrorCode.MODEL_UNAVAILABLE),
        ("APIConnectionError", "connection refused", ErrorCode.MODEL_UNAVAILABLE),
    ],
)
def test_runtime_stage_classifies_upstream_cause(error_class, message, expected):
    assert classify_error_code("runtime", error_class, message) is expected


# 验证 status codes per error code 场景。
def test_status_codes_per_error_code():
    assert status_for_code(ErrorCode.LLM_TIMEOUT) == 504
    assert status_for_code(ErrorCode.RATE_LIMITED) == 429
    assert status_for_code(ErrorCode.MODEL_UNAVAILABLE) == 503


# 验证 get client reads timeout and retries from settings 场景。
def test_get_client_reads_timeout_and_retries_from_settings(monkeypatch):
    import cogdoc.agents.qa_generator as gen_module
    from types import SimpleNamespace

    gen_module.Generator._clients.clear()
    fake = SimpleNamespace(
        ollama_base_url="http://x/v1",
        ollama_api_key="k",
        ollama_model_name="m",
        ollama_timeout_seconds=222.0,
        ollama_max_retries=5,
    )
    monkeypatch.setattr(gen_module, "get_settings", lambda: fake)
    captured = {}

    # 构造chatopenai。
    def fake_chatopenai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(gen_module, "ChatOpenAI", fake_chatopenai)
    gen_module.Generator._get_client(is_local=True)
    assert captured["timeout"] == 222.0
    assert captured["max_retries"] == 5
    gen_module.Generator._clients.clear()


# 发送chat。
async def _post_chat(app, payload):
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post("/v1/chat", json=payload)


# 验证 chat endpoint maps timeout to 504 场景。
@pytest.mark.anyio
async def test_chat_endpoint_maps_timeout_to_504(monkeypatch):
    import cogdoc.api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)

    # 构造或驱动 失败路径运行器 测试场景。
    def failing_runner(doc_id, query, is_local, chat_history, forced_task):
        raise ChatServiceError(
            stage="runtime",
            error_class="APITimeoutError",
            message="request timed out",
            trace_id="t-timeout",
        )

    store = SessionStore()
    app = create_app(chat_runner=failing_runner, session_store=store)
    resp = await _post_chat(app, {"query": "问题", "doc_id": "kb", "session_id": "s"})

    assert resp.status_code == 504
    assert resp.json()["error_code"] == "LLM_TIMEOUT"
    # 失败不写会话。
    assert store.get_history("kb", "s") == []


# 验证 chat endpoint maps rate limit to 429 场景。
@pytest.mark.anyio
async def test_chat_endpoint_maps_rate_limit_to_429(monkeypatch):
    import cogdoc.api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)

    # 构造或驱动 失败路径运行器 测试场景。
    def failing_runner(doc_id, query, is_local, chat_history, forced_task):
        raise ChatServiceError(
            stage="runtime",
            error_class="RateLimitError",
            message="rate limit exceeded",
            trace_id="t-429",
        )

    app = create_app(chat_runner=failing_runner, session_store=SessionStore())
    resp = await _post_chat(app, {"query": "问题", "doc_id": "kb"})

    assert resp.status_code == 429
    assert resp.json()["error_code"] == "RATE_LIMITED"
