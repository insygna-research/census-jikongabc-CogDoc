import pytest
from httpx import ASGITransport, AsyncClient
from cogdoc.api.app import create_app
from cogdoc.api.metrics import Metrics
from cogdoc.api.session_store import SessionStore
from cogdoc.service.chat_service import ChatResult


# 声明异步测试使用的后端。
@pytest.fixture
def anyio_backend():
    return "asyncio"


# 运行成功路径。
def _runner_ok(doc_id, query, is_local, chat_history, forced_task):
    return ChatResult(
        answer="ok",
        task_type="qa",
        citations=[],
        evidence=[],
        critique="",
        is_valid=True,
        trace_id="t",
        request_id="t",
        steps=[],
        chat_messages=[],
        raw_output={"answer": "ok"},
    )


# 创建测试应用实例。
def _app(monkeypatch, **kwargs):
    import cogdoc.api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    return create_app(chat_runner=_runner_ok, session_store=SessionStore(), **kwargs)


# 创建测试客户端。
async def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


# 验证 metrics render is prometheus text 场景。
def test_metrics_render_is_prometheus_text():
    metrics = Metrics()
    metrics.requests.labels("GET", "/x", "200").inc()
    body = metrics.render().decode()
    assert "cogdoc_http_requests_total" in body
    assert "cogdoc_http_request_duration_seconds" in body


# 验证 middleware records 500 when call next raises 场景。
@pytest.mark.anyio
async def test_middleware_records_500_when_call_next_raises():
    # 下游 ASGI app 抛未兜底异常时：仍记一条 status=500、在途归零，且异常透传。
    from cogdoc.api.metrics import MetricsMiddleware

    metrics = Metrics()

    # 模拟失败结果。
    async def boom(_scope, _receive, _send):
        raise RuntimeError("kaboom")

    mw = MetricsMiddleware(app=boom, metrics=metrics)
    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "path": "/x",
        "raw_path": b"/x",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
    }

    async def receive():
        return {"type": "http.disconnect"}

    async def send(_message):
        return None

    with pytest.raises(RuntimeError):
        await mw(scope, receive, send)

    body = metrics.render().decode()
    assert 'status="500"' in body
    assert "cogdoc_http_requests_in_progress 0.0" in body


# 验证 metrics endpoint reachable and auth exempt 场景。
@pytest.mark.anyio
async def test_metrics_endpoint_reachable_and_auth_exempt(monkeypatch):
    # 开了鉴权，/metrics 仍应免鉴权可抓取。
    app = _app(monkeypatch, api_keys={"secret"})
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            resp = await c.get("/metrics")
    assert resp.status_code == 200
    assert "cogdoc_http_requests_total" in resp.text


# 验证 requests are counted 场景。
@pytest.mark.anyio
async def test_requests_are_counted(monkeypatch):
    app = _app(monkeypatch, api_keys=set())
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            await c.post("/v1/chat", json={"query": "q", "doc_id": "kb"})
            scraped = (await c.get("/metrics")).text
    # 计数按路由模板聚合，POST /v1/chat 200 应出现。
    assert (
        'cogdoc_http_requests_total{method="POST",route="/v1/chat",status="200"}'
        in scraped
    )
    assert "cogdoc_http_request_duration_seconds_count" in scraped


# 验证 path params collapse to route template 场景。
@pytest.mark.anyio
async def test_path_params_collapse_to_route_template(monkeypatch):
    # 不同 job_id 不能各成一条时间序列，必须聚到 /v1/index-jobs/{job_id}。
    app = _app(monkeypatch, api_keys=set())
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            await c.get("/v1/index-jobs/aaa")
            await c.get("/v1/index-jobs/bbb")
            scraped = (await c.get("/metrics")).text
    assert "/v1/index-jobs/{job_id}" in scraped
    assert "/v1/index-jobs/aaa" not in scraped


# 验证 chat result counter increments 场景。
@pytest.mark.anyio
async def test_chat_result_counter_increments(monkeypatch):
    app = _app(monkeypatch, api_keys=set())
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            await c.post("/v1/chat", json={"query": "q", "doc_id": "kb"})
            scraped = (await c.get("/metrics")).text
    assert 'cogdoc_chat_results_total{task_type="qa",valid="true"}' in scraped


# 验证 rejected requests are counted 场景。
@pytest.mark.anyio
async def test_rejected_requests_are_counted(monkeypatch):
    # 指标中间件在访问控制外层：401 也应计入。
    app = _app(monkeypatch, api_keys={"secret"})
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            await c.post("/v1/chat", json={"query": "q", "doc_id": "kb"})
            scraped = (await c.get("/metrics")).text
    assert 'status="401"' in scraped
