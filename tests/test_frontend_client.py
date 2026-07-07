import httpx
import pytest
from cogdoc.frontend.api_client import (
    CogDocAPIError,
    CogDocClient,
    format_api_error,
    iter_sse_events,
    response_payload,
)


# 验证流式事件解析片段和最终帧场景。
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


# 验证流式事件跳过非法数据场景。
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


# 验证接口错误格式化优先使用结构化错误体场景。
def test_format_api_error_prefers_structured_error_body():
    message = format_api_error(
        {"error_code": "REQUEST_THROTTLED", "message": "请求过于频繁，请稍后重试"},
        429,
    )

    assert message == "HTTP 429: [REQUEST_THROTTLED] 请求过于频繁，请稍后重试"


# 验证知识库列表遇到结构化错误时抛出异常场景。
def test_list_knowledge_bases_raises_on_structured_error(monkeypatch):
    # 测试伪造读取。
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


# 验证知识库列表拒绝非列表成功载荷场景。
def test_list_knowledge_bases_rejects_non_list_success_payload(monkeypatch):
    monkeypatch.setattr(
        "cogdoc.frontend.api_client.httpx.get",
        lambda *args, **kwargs: httpx.Response(200, json={"items": []}),
    )

    with pytest.raises(CogDocAPIError, match="知识库列表响应格式不符合预期"):
        CogDocClient("http://api").list_knowledge_bases()


# 验证响应载荷在非对象响应时退回文本场景。
def test_response_payload_falls_back_to_text_for_non_json_response():
    response = httpx.Response(502, text="bad gateway body")

    assert response_payload(response) == "bad gateway body"


# 验证跟踪客户端方法调用预期端点场景。
def test_trace_client_methods_call_expected_endpoints(monkeypatch):
    calls = []

    # 测试伪造读取。
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


# 验证反馈客户端发送证据载荷场景。
def test_feedback_client_sends_citation_and_evidence_payload(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(201, json={"feedback_id": "f1", "is_bad_case": True})

    monkeypatch.setattr("cogdoc.frontend.api_client.httpx.post", fake_post)

    response = CogDocClient("http://api", api_key="secret").submit_feedback(
        trace_id="t1",
        feedback="thumbs_down",
        kb_id="kb",
        query="问题",
        answer="答案",
        citations=[{"chunk_id": "c1", "source": "a.pdf", "page": 1}],
        evidence=[{"chunk_id": "c1", "source": "a.pdf", "text_preview": "证据"}],
    )

    assert response.status_code == 201
    assert calls[0][0] == "http://api/v1/feedback"
    assert calls[0][1]["headers"] == {"Authorization": "Bearer secret"}
    assert calls[0][1]["json"]["citations"][0]["source"] == "a.pdf"
    assert calls[0][1]["json"]["evidence"][0]["text_preview"] == "证据"


# 验证反馈客户端发送保存知识字段场景。
def test_feedback_client_sends_save_as_knowledge_payload(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(
            201,
            json={
                "feedback_id": "f1",
                "is_bad_case": True,
                "knowledge_id": "K1",
                "knowledge_status": "pending",
            },
        )

    monkeypatch.setattr("cogdoc.frontend.api_client.httpx.post", fake_post)

    response = CogDocClient("http://api").submit_feedback(
        trace_id="t1",
        feedback="correction",
        kb_id="kb",
        correction_text="正确说法",
        save_as_knowledge=True,
        related_source="a.pdf",
        related_source_sha256="sha",
        related_chunk_ids=["c1"],
        certainty="high",
    )

    assert response.status_code == 201
    payload = calls[0][1]["json"]
    assert payload["save_as_knowledge"] is True
    assert payload["correction_text"] == "正确说法"
    assert payload["related_source"] == "a.pdf"
    assert payload["related_chunk_ids"] == ["c1"]
    assert payload["certainty"] == "high"


# 验证派生知识客户端方法调用稳定端点场景。
def test_knowledge_client_methods_call_expected_endpoints(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(("POST", url, kwargs))
        return httpx.Response(200, json={"ok": True})

    def fake_get(url, **kwargs):
        calls.append(("GET", url, kwargs))
        return httpx.Response(200, json={"knowledge": []})

    monkeypatch.setattr("cogdoc.frontend.api_client.httpx.post", fake_post)
    monkeypatch.setattr("cogdoc.frontend.api_client.httpx.get", fake_get)

    client = CogDocClient("http://api", api_key="secret")
    client.create_knowledge(
        kb_id="kb",
        text="知识",
        related_source="a.pdf",
        related_source_sha256="sha",
        related_chunk_ids=["c1"],
        source_note="人工确认",
        certainty="high",
        origin="saved_answer",
        created_from_trace_id="trace-1",
    )
    client.list_knowledge("kb", status="pending", origin="manual_entry")
    client.review_knowledge("K1", "approve", actor="admin")
    client.batch_review_knowledge(["K1", "K2"], "batch-reject", note="重复")

    assert calls[0][0:2] == ("POST", "http://api/v1/knowledge")
    assert calls[0][2]["headers"] == {"Authorization": "Bearer secret"}
    assert calls[0][2]["json"]["related_chunk_ids"] == ["c1"]
    assert calls[0][2]["json"]["origin"] == "saved_answer"
    assert calls[0][2]["json"]["created_from_trace_id"] == "trace-1"
    assert calls[1][0:2] == ("GET", "http://api/v1/knowledge")
    assert calls[1][2]["params"] == {
        "kb_id": "kb",
        "status": "pending",
        "origin": "manual_entry",
    }
    assert calls[2][0:2] == ("POST", "http://api/v1/knowledge/K1/approve")
    assert calls[2][2]["json"] == {"actor": "admin"}
    assert calls[3][0:2] == ("POST", "http://api/v1/knowledge/batch-reject")
    assert calls[3][2]["json"] == {"knowledge_ids": ["K1", "K2"], "note": "重复"}


# 验证检索调权客户端方法调用稳定端点场景。
def test_retrieval_feedback_client_methods_call_expected_endpoints(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(("POST", url, kwargs))
        return httpx.Response(200, json={"ok": True})

    def fake_get(url, **kwargs):
        calls.append(("GET", url, kwargs))
        return httpx.Response(200, json={"retrieval_feedback": []})

    monkeypatch.setattr("cogdoc.frontend.api_client.httpx.post", fake_post)
    monkeypatch.setattr("cogdoc.frontend.api_client.httpx.get", fake_get)

    client = CogDocClient("http://api", api_key="secret")
    client.list_retrieval_feedback("kb", enabled=False, limit=50)
    client.set_retrieval_feedback_enabled("rf1", False, actor="admin", reason="误点")
    client.set_retrieval_feedback_enabled("rf1", True)

    assert calls[0][0:2] == ("GET", "http://api/v1/retrieval-feedback")
    assert calls[0][2]["headers"] == {"Authorization": "Bearer secret"}
    assert calls[0][2]["params"] == {
        "kb_id": "kb",
        "enabled": False,
        "limit": 50,
    }
    assert calls[1][0:2] == (
        "POST",
        "http://api/v1/retrieval-feedback/rf1/disable",
    )
    assert calls[1][2]["json"] == {"actor": "admin", "reason": "误点"}
    assert calls[2][0:2] == (
        "POST",
        "http://api/v1/retrieval-feedback/rf1/enable",
    )
    assert "json" not in calls[2][2]


# 验证反馈分析客户端方法调用稳定端点场景。
def test_feedback_analysis_client_method_calls_expected_endpoint(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(200, json={"feedback_analysis": []})

    monkeypatch.setattr("cogdoc.frontend.api_client.httpx.get", fake_get)

    response = CogDocClient("http://api", api_key="secret").list_feedback_analysis(
        "kb", recommended_action="create_pending_knowledge", limit=25
    )

    assert response.status_code == 200
    assert calls[0][0] == "http://api/v1/feedback-analysis"
    assert calls[0][1]["headers"] == {"Authorization": "Bearer secret"}
    assert calls[0][1]["params"] == {
        "kb_id": "kb",
        "recommended_action": "create_pending_knowledge",
        "limit": 25,
    }
