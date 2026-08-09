import httpx

from scripts import cogeval_cogdoc_wrapper as wrapper


def test_wrapper_health_uses_canonical_readyz_endpoint(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(200, json={"status": "ready"})

    monkeypatch.setattr(wrapper.httpx, "get", fake_get)
    monkeypatch.setattr(wrapper, "COGDOC_URL", "http://127.0.0.1:8002/v1/chat")

    result = wrapper.healthz()

    assert result == {"status": "ok", "cogdoc_status": 200}
    assert calls == [("http://127.0.0.1:8002/readyz", {"timeout": 10})]


def test_wrapper_health_fails_degraded_on_connection_error(monkeypatch):
    def fail_get(*_args, **_kwargs):
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(wrapper.httpx, "get", fail_get)

    result = wrapper.healthz()

    assert result == {
        "status": "degraded",
        "cogdoc_status": None,
        "error": "ConnectError",
    }


def test_wrapper_invoke_uses_declared_httpx_dependency(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={"answer": "服务可用", "trace_id": "cogdoc-trace", "task_type": "qa"},
        )

    monkeypatch.setattr(wrapper.httpx, "post", fake_post)
    monkeypatch.setattr(wrapper, "COGDOC_URL", "http://127.0.0.1:8002/v1/chat")

    result = wrapper.invoke(wrapper.InvokeRequest(query="状态如何？", trace_id="eval-trace"))

    assert result["answer"] == "服务可用"
    assert result["trace_id"] == "eval-trace"
    assert result["trace"]["metadata"]["cogdoc_trace_id"] == "cogdoc-trace"
    assert calls[0][0] == "http://127.0.0.1:8002/v1/chat"
    assert calls[0][1]["json"]["query"] == "状态如何？"
