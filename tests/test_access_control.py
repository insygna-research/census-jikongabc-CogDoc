import pytest
from httpx import ASGITransport, AsyncClient
from api.access_control import TokenBucketRateLimiter, build_rate_limiter
from api.app import create_app
from api.session_store import SessionStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _app(monkeypatch, **kwargs):
    import api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)

    def runner(doc_id, query, is_local, chat_history, forced_task):
        from service.chat_service import ChatResult

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

    return create_app(chat_runner=runner, session_store=SessionStore(), **kwargs)


async def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


# ---- 限流器单元 ----


def test_token_bucket_allows_burst_then_throttles():
    limiter = TokenBucketRateLimiter(capacity=2, refill_per_second=0.0)
    assert limiter.allow("k") is True
    assert limiter.allow("k") is True
    assert limiter.allow("k") is False


def test_token_bucket_isolates_identities():
    limiter = TokenBucketRateLimiter(capacity=1, refill_per_second=0.0)
    assert limiter.allow("a") is True
    assert limiter.allow("a") is False
    # 另一身份不受影响。
    assert limiter.allow("b") is True


def test_token_bucket_disabled_when_capacity_zero():
    limiter = TokenBucketRateLimiter(capacity=0, refill_per_second=0.0)
    assert all(limiter.allow("k") for _ in range(100))


def test_identity_cap_is_enforced_under_distinct_flood():
    # 大量各做一次请求的不同身份（桶都未回满），仍必须把内存压回上限。
    limiter = TokenBucketRateLimiter(
        capacity=5, refill_per_second=0.0, max_identities=10
    )
    for i in range(1000):
        limiter.allow(f"id-{i}")
    assert len(limiter._buckets) <= 10


def test_eviction_is_lru_keeps_recently_active():
    # 访问会刷新活跃度：淘汰时丢最久未活跃，而非最早创建。
    limiter = TokenBucketRateLimiter(
        capacity=5, refill_per_second=0.0, max_identities=2
    )
    limiter.allow("a")
    limiter.allow("b")
    limiter.allow("a")  # a 重新变为最近活跃
    limiter.allow("c")  # 超额，淘汰最久未活跃的 b
    assert set(limiter._buckets) == {"a", "c"}


def test_build_rate_limiter_converts_per_minute():
    limiter = build_rate_limiter(per_minute=120, burst=60)
    assert limiter.capacity == 60
    assert limiter.refill_per_second == pytest.approx(2.0)


# ---- 鉴权 ----


@pytest.mark.anyio
async def test_auth_disabled_when_no_keys(monkeypatch):
    app = _app(monkeypatch, api_keys=set())
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            resp = await c.post("/v1/chat", json={"query": "q", "doc_id": "kb"})
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_startup_warns_when_auth_disabled(monkeypatch):
    import api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    events = []
    monkeypatch.setattr(
        app_module, "log_event", lambda *a, **k: events.append((a, k))
    )
    app = _app(monkeypatch, api_keys=set())
    async with app.router.lifespan_context(app):
        pass
    # 鉴权关闭时启动应发一条 auth_disabled 告警。
    assert any(a[:2] == ("startup", "auth_disabled") for a, _ in events)


@pytest.mark.anyio
async def test_no_startup_warning_when_auth_enabled(monkeypatch):
    import api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    events = []
    monkeypatch.setattr(
        app_module, "log_event", lambda *a, **k: events.append((a, k))
    )
    app = _app(monkeypatch, api_keys={"secret"})
    async with app.router.lifespan_context(app):
        pass
    assert not any(a[:2] == ("startup", "auth_disabled") for a, _ in events)


@pytest.mark.anyio
async def test_missing_key_rejected_401(monkeypatch):
    app = _app(monkeypatch, api_keys={"secret"})
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            resp = await c.post("/v1/chat", json={"query": "q", "doc_id": "kb"})
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "UNAUTHORIZED"


@pytest.mark.anyio
async def test_wrong_key_rejected_401(monkeypatch):
    app = _app(monkeypatch, api_keys={"secret"})
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            resp = await c.post(
                "/v1/chat",
                json={"query": "q", "doc_id": "kb"},
                headers={"Authorization": "Bearer nope"},
            )
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_bearer_and_x_api_key_accepted(monkeypatch):
    app = _app(monkeypatch, api_keys={"secret"})
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            via_bearer = await c.post(
                "/v1/chat",
                json={"query": "q", "doc_id": "kb"},
                headers={"Authorization": "Bearer secret"},
            )
            via_header = await c.post(
                "/v1/chat",
                json={"query": "q", "doc_id": "kb"},
                headers={"X-API-Key": "secret"},
            )
    assert via_bearer.status_code == 200
    assert via_header.status_code == 200


@pytest.mark.anyio
async def test_health_endpoints_exempt_from_auth(monkeypatch):
    app = _app(monkeypatch, api_keys={"secret"})
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            healthz = await c.get("/healthz")
            readyz = await c.get("/readyz")
    # 没带 key 也能过探针。
    assert healthz.status_code == 200
    assert readyz.status_code in (200, 503)


# ---- 限流（端到端）----


@pytest.mark.anyio
async def test_rate_limit_returns_429_after_capacity(monkeypatch):
    limiter = TokenBucketRateLimiter(capacity=2, refill_per_second=0.0)
    app = _app(monkeypatch, api_keys=set(), rate_limiter=limiter)
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            body = {"query": "q", "doc_id": "kb"}
            first = await c.post("/v1/chat", json=body)
            second = await c.post("/v1/chat", json=body)
            third = await c.post("/v1/chat", json=body)
    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert third.json()["error_code"] == "REQUEST_THROTTLED"


@pytest.mark.anyio
async def test_job_polling_exempt_from_rate_limit(monkeypatch):
    # 即便桶极小，入库 job 状态轮询也不该被限流（否则长任务轮询会误判失败）。
    limiter = TokenBucketRateLimiter(capacity=1, refill_per_second=0.0)
    app = _app(monkeypatch, api_keys=set(), rate_limiter=limiter)
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            statuses = [
                (await c.get("/v1/index-jobs/whatever")).status_code
                for _ in range(10)
            ]
    # 全部 404（job 不存在）而非 429，证明没走限流。
    assert all(code == 404 for code in statuses)


@pytest.mark.anyio
async def test_job_polling_still_requires_auth(monkeypatch):
    # 限流豁免不等于鉴权豁免：开了 key 还是要带。
    app = _app(monkeypatch, api_keys={"secret"})
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            unauth = await c.get("/v1/index-jobs/whatever")
            authed = await c.get(
                "/v1/index-jobs/whatever", headers={"X-API-Key": "secret"}
            )
    assert unauth.status_code == 401
    assert authed.status_code == 404


@pytest.mark.anyio
async def test_rate_limit_is_per_key(monkeypatch):
    limiter = TokenBucketRateLimiter(capacity=1, refill_per_second=0.0)
    app = _app(monkeypatch, api_keys={"k1", "k2"}, rate_limiter=limiter)
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            body = {"query": "q", "doc_id": "kb"}
            k1_first = await c.post(
                "/v1/chat", json=body, headers={"X-API-Key": "k1"}
            )
            k1_second = await c.post(
                "/v1/chat", json=body, headers={"X-API-Key": "k1"}
            )
            k2_first = await c.post(
                "/v1/chat", json=body, headers={"X-API-Key": "k2"}
            )
    assert k1_first.status_code == 200
    assert k1_second.status_code == 429
    # 另一个 key 的额度独立，不受 k1 耗尽影响。
    assert k2_first.status_code == 200
