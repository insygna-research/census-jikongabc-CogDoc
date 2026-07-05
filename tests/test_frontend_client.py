import httpx
import pytest
from cogdoc.frontend.api_client import iter_sse_events
from cogdoc.frontend.api_client import CogDocAPIError, CogDocClient, format_api_error


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
