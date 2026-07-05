import httpx
import pytest
from cogdoc.frontend.api_client import (
    CogDocAPIError,
    CogDocClient,
    format_api_error,
    iter_sse_events,
    response_payload,
)


# 验证 iter sse events parses token and final frames 场景。
def test_iter_sse_events_parses_token_and_final_frames():
    lines = [
        "event: start",
        'data: {"trace_id": "t1"}',
        "",
        "event: token",
        'data: {"content": "你好"}',
        "",
        "event: final",
        'data: {"answer": "完整答案", "citations": []}',
        "",
    ]

    events = list(iter_sse_events(lines))

    assert [name for name, _ in events] == ["start", "token", "final"]
    assert events[1][1]["content"] == "你好"
    assert events[2][1]["answer"] == "完整答案"


# 验证 iter sse events skips non json data 场景。
def test_iter_sse_events_skips_non_json_data():
    lines = [
        "event: token",
        "data: not-json",
        "",
        "event: token",
        'data: {"content": "ok"}',
    ]

    events = list(iter_sse_events(lines))

    assert events == [("token", {"content": "ok"})]


# 验证 format api error prefers structured error body 场景。
def test_format_api_error_prefers_structured_error_body():
    message = format_api_error(
        {"error_code": "REQUEST_THROTTLED", "message": "请求过于频繁，请稍后重试"},
        429,
    )

    assert message == "HTTP 429: [REQUEST_THROTTLED] 请求过于频繁，请稍后重试"


# 验证 list knowledge bases raises on structured error 场景。
def test_list_knowledge_bases_raises_on_structured_error(monkeypatch):
    def fake_get(*args, **kwargs):
        return httpx.Response(
            429,
            json={
                "schema_version": "v1",
                "error_code": "REQUEST_THROTTLED",
                "message": "请求过于频繁，请稍后重试",
                "request_id": None,
                "trace_id": None,
                "details": None,
            },
        )

    monkeypatch.setattr("cogdoc.frontend.api_client.httpx.get", fake_get)

    with pytest.raises(CogDocAPIError) as excinfo:
        CogDocClient("http://api").list_knowledge_bases()

    assert excinfo.value.status_code == 429
    assert "REQUEST_THROTTLED" in str(excinfo.value)


# 验证 list knowledge bases rejects non list success payload 场景。
def test_list_knowledge_bases_rejects_non_list_success_payload(monkeypatch):
    monkeypatch.setattr(
        "cogdoc.frontend.api_client.httpx.get",
        lambda *args, **kwargs: httpx.Response(200, json={"items": []}),
    )

    with pytest.raises(CogDocAPIError, match="知识库列表响应格式不符合预期"):
        CogDocClient("http://api").list_knowledge_bases()


# 验证 response payload falls back to text for non json response 场景。
def test_response_payload_falls_back_to_text_for_non_json_response():
    response = httpx.Response(502, text="bad gateway body")

    assert response_payload(response) == "bad gateway body"


# 验证 trace client methods call expected endpoints 场景。
def test_trace_client_methods_call_expected_endpoints(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr("cogdoc.frontend.api_client.httpx.get", fake_get)
    client = CogDocClient("http://api", api_key="secret")

    trace_resp = client.get_trace("trace-1")
    list_resp = client.list_traces(limit=7, kb_id="kb", session_id="s1")

    assert trace_resp.json() == {"ok": True}
    assert list_resp.json() == {"ok": True}
    assert calls[0][0] == "http://api/v1/traces/trace-1"
    assert calls[0][1]["headers"] == {"Authorization": "Bearer secret"}
    assert calls[1][0] == "http://api/v1/traces"
    assert calls[1][1]["params"] == {
        "limit": 7,
        "doc_id": "kb",
        "session_id": "s1",
    }
