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


# 返回带最终声明审计结果的成功回答。
def _runner_with_claim_audit(doc_id, query, is_local, chat_history, forced_task):
    return ChatResult(
        answer="修复后的可信答案",
        task_type="qa",
        citations=[],
        evidence=[],
        critique="",
        is_valid=True,
        trace_id="claim-trace",
        request_id="claim-trace",
        steps=[],
        chat_messages=[],
        raw_output={
            "answer": "修复后的可信答案",
            "claim_audit": {
                "status": "repaired",
                "claims": [],
                "counts": {
                    "claim_count": 4,
                    "supported": 2,
                    "unsupported": 1,
                    "insufficient": 1,
                    "cited": 3,
                    "skipped_statements": 1,
                },
                "metrics": {
                    "claim_support_rate": 0.5,
                    "citation_coverage": 0.75,
                    "unsupported_claim_rate": 0.25,
                },
                "repair": {
                    "attempted": True,
                    "attempt_count": 1,
                    "succeeded": True,
                },
                "verifier": {"duration_ms": 250.0},
            },
        },
    )


def _runner_with_malformed_claim_audit(
    doc_id, query, is_local, chat_history, forced_task
):
    result = _runner_ok(doc_id, query, is_local, chat_history, forced_task)
    result.raw_output["claim_audit"] = {
        "status": "failed",
        "reason_code": "malformed_fixture",
        "counts": {
            "claim_count": "not-an-int",
            "supported": float("inf"),
            "unsupported": -3,
            "insufficient": "broken",
            "cited": "999",
            "skipped_statements": None,
        },
        "metrics": {
            "claim_support_rate": "not-a-float",
            "citation_coverage": float("nan"),
            "unsupported_claim_rate": None,
        },
        "repair": {
            "attempted": True,
            "attempt_count": float("inf"),
            "succeeded": False,
        },
        "verifier": {"duration_ms": "not-a-duration"},
    }
    return result


def _runner_with_adaptive_retrieval(doc_id, query, is_local, chat_history, forced_task):
    result = _runner_ok(doc_id, query, is_local, chat_history, forced_task)
    result.raw_output.update(
        {
            "retrieval_abstained": False,
            "retrieval_retry_count": 1,
            "evidence_requirement_assessments": [
                {
                    "requirement_id": "r1",
                    "verdict": "supported",
                    "evidence_chunk_ids": ["c1"],
                    "reason": "证据完整",
                },
                {
                    "requirement_id": "r2",
                    "verdict": "supported",
                    "evidence_chunk_ids": ["c2"],
                    "reason": "补检索后证据完整",
                },
            ],
        }
    )
    return result


# 创建测试应用实例。
def _app(monkeypatch, **kwargs):
    import cogdoc.api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    chat_runner = kwargs.pop("chat_runner", _runner_ok)
    return create_app(chat_runner=chat_runner, session_store=SessionStore(), **kwargs)


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


# 验证最终回答只记录一次声明判定、修复结果和 verifier 耗时。
@pytest.mark.anyio
async def test_claim_audit_metrics_record_final_counts_repair_and_duration(monkeypatch):
    app = _app(
        monkeypatch,
        api_keys=set(),
        chat_runner=_runner_with_claim_audit,
    )
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            response = await c.post("/v1/chat", json={"query": "q", "doc_id": "kb"})
            scraped = (await c.get("/metrics")).text

    assert response.status_code == 200
    assert response.json()["claim_audit"]["status"] == "repaired"
    assert (
        'cogdoc_claim_audit_runs_total{status="repaired",task_type="qa"} 1.0' in scraped
    )
    assert (
        'cogdoc_claim_audit_claims_total{task_type="qa",verdict="supported"} 2.0'
        in scraped
    )
    assert (
        'cogdoc_claim_audit_claims_total{task_type="qa",verdict="unsupported"} 1.0'
        in scraped
    )
    assert (
        'cogdoc_claim_audit_claims_total{task_type="qa",verdict="insufficient"} 1.0'
        in scraped
    )
    assert (
        'cogdoc_claim_audit_claims_total{task_type="qa",verdict="not_factual"} 1.0'
        in scraped
    )
    assert (
        'cogdoc_claim_audit_citations_total{covered="true",task_type="qa"} 3.0'
        in scraped
    )
    assert (
        'cogdoc_claim_audit_citations_total{covered="false",task_type="qa"} 1.0'
        in scraped
    )
    assert (
        'cogdoc_claim_audit_repairs_total{outcome="succeeded",task_type="qa"} 1.0'
        in scraped
    )
    assert 'cogdoc_claim_audit_duration_seconds_count{task_type="qa"} 1.0' in scraped
    assert 'cogdoc_claim_audit_duration_seconds_sum{task_type="qa"} 0.25' in scraped


# 验证补检索救回和逐需求证据覆盖只按最终 QA 状态记录一次。
@pytest.mark.anyio
async def test_adaptive_retrieval_metrics_record_rescue_and_requirements(monkeypatch):
    app = _app(
        monkeypatch,
        api_keys=set(),
        chat_runner=_runner_with_adaptive_retrieval,
    )
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            response = await c.post("/v1/chat", json={"query": "q", "doc_id": "kb"})
            scraped = (await c.get("/metrics")).text

    assert response.status_code == 200
    assert 'cogdoc_retrieval_decisions_total{outcome="accepted"} 1.0' in scraped
    assert 'cogdoc_adaptive_retrieval_runs_total{outcome="rescued"} 1.0' in scraped
    assert "cogdoc_adaptive_retrieval_retry_count_count 1.0" in scraped
    assert "cogdoc_adaptive_retrieval_retry_count_sum 1.0" in scraped
    assert (
        'cogdoc_evidence_requirement_assessments_total{verdict="supported"} 2.0'
        in scraped
    )


# 验证指标旁路不会让注入 runner 的畸形 audit 把成功回答变成 500。
@pytest.mark.anyio
async def test_malformed_claim_audit_does_not_break_sync_delivery(monkeypatch):
    app = _app(
        monkeypatch,
        api_keys=set(),
        chat_runner=_runner_with_malformed_claim_audit,
    )
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            response = await c.post("/v1/chat", json={"query": "q", "doc_id": "kb"})
            scraped = (await c.get("/metrics")).text

    assert response.status_code == 200
    summary = response.json()["claim_audit"]
    assert summary["counts"] == {
        "claim_count": 0,
        "supported": 0,
        "unsupported": 0,
        "insufficient": 0,
        "cited": 999,
        "skipped_statements": 0,
    }
    assert summary["metrics"] == {
        "claim_support_rate": None,
        "citation_coverage": None,
        "unsupported_claim_rate": None,
    }
    assert summary["repair"]["attempt_count"] == 0
    assert summary["duration_ms"] is None
    assert (
        'cogdoc_claim_audit_runs_total{status="failed",task_type="qa"} 1.0' in scraped
    )


# 验证默认关闭门禁的 not_run 不会用 0 秒样本稀释 verifier 延迟。
def test_not_run_claim_audit_does_not_observe_verifier_duration():
    metrics = Metrics()

    metrics.observe_claim_audit(
        "qa",
        {
            "status": "not_run",
            "reason_code": "disabled",
            "verifier": {"duration_ms": 0.0, "call_count": 0},
        },
    )

    scraped = metrics.render().decode()
    assert (
        'cogdoc_claim_audit_runs_total{status="not_run",task_type="qa"} 1.0' in scraped
    )
    assert 'cogdoc_claim_audit_duration_seconds_count{task_type="qa"}' not in scraped


# 验证缺失显式终态的局部输出不会被误记为 accepted/rescued。
def test_retrieval_metrics_do_not_infer_terminal_decision_from_partial_output():
    metrics = Metrics()

    metrics.observe_retrieval(
        "qa",
        {
            "retrieval_retry_count": 1,
            "evidence_requirement_assessments": [],
        },
    )

    scraped = metrics.render().decode()
    assert "cogdoc_retrieval_decisions_total{outcome=" not in scraped
    assert "cogdoc_adaptive_retrieval_runs_total{outcome=" not in scraped
    assert "cogdoc_adaptive_retrieval_retry_count_count 0.0" in scraped
    assert "cogdoc_adaptive_retrieval_retry_count_sum 0.0" in scraped


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
