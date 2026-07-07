import pytest
from pydantic import ValidationError
from cogdoc.api.schemas import (
    API_SCHEMA_VERSION,
    ChatRequest,
    ErrorCode,
    TraceResponse,
    build_error_response,
    chat_result_to_response,
)
from cogdoc.service.chat_service import ChatResult


# 验证对话请求默认值和强制任务。
def test_chat_request_defaults_and_forced_task():
    request = ChatRequest(query="  总结 a.pdf  ", mode="summary")

    assert request.schema_version == API_SCHEMA_VERSION
    assert request.query == "总结 a.pdf"
    assert request.doc_id
    assert request.forced_task == "summary"
    assert ChatRequest(query="问题").forced_task is None


# 验证对话请求拒绝空白和未知字段。
def test_chat_request_rejects_blank_and_unknown_fields():
    with pytest.raises(ValidationError):
        ChatRequest(query="  ")

    with pytest.raises(ValidationError):
        ChatRequest(query="问题", unexpected=True)


# 验证对话结果响应不泄漏原始正文。
def test_chat_result_to_response_maps_stable_fields_without_raw_text():
    result = ChatResult(
        answer="需要满足报名要求。[a.pdf:P1]",
        task_type="qa",
        citations=[
            {
                "chunk_id": "chunk:a:1",
                "source": "a.pdf",
                "page": 1,
                "page_start": 1,
                "page_end": 2,
                "text": "不应进入 API 响应的全文",
            }
        ],
        evidence=[
            {
                "chunk_id": "chunk:a:1",
                "chunk_index": 0,
                "source": "a.pdf",
                "page": 1,
                "page_start": 1,
                "page_end": 2,
                "rerank_score": "0.98",
                "rewrite_query": "报名要求",
                "text_preview": "报名要求摘要",
                "retrieval": {
                    "search_channel": "derived_knowledge",
                    "matched_terms": ["报名"],
                },
                "text": "不应进入 API 响应的全文",
            }
        ],
        critique="",
        is_valid=True,
        trace_id="trace-1",
        request_id="trace-1",
        steps=[{"node_name": "runtime.setup"}],
        chat_messages=[{"role": "user", "content": "报名要求是什么"}],
        raw_output={"answer": "raw"},
        trace_path="/tmp/trace-1.json",
    )

    response = chat_result_to_response(result, doc_id="kb", session_id="s1")
    payload = response.model_dump()

    assert payload["schema_version"] == "v1"
    assert payload["doc_id"] == "kb"
    assert payload["session_id"] == "s1"
    assert payload["task_type"] == "qa"
    assert payload["answer"] == "需要满足报名要求。[a.pdf:P1]"
    assert payload["citations"] == [
        {
            "chunk_id": "chunk:a:1",
            "source_type": "document",
            "knowledge_id": "",
            "source": "a.pdf",
            "page": 1,
            "page_start": 1,
            "page_end": 2,
        }
    ]
    assert payload["evidence"][0]["rerank_score"] == 0.98
    assert payload["evidence"][0]["source_type"] == "document"
    assert payload["evidence"][0]["text_preview"] == "报名要求摘要"
    assert payload["evidence"][0]["retrieval"]["search_channel"] == (
        "derived_knowledge"
    )
    assert payload["evidence"][0]["retrieval"]["matched_terms"] == ["报名"]
    assert "raw_output" not in payload
    assert "steps" not in payload
    assert "trace_path" not in payload
    assert "不应进入 API 响应的全文" not in str(payload)


# 验证未知任务类型会归一化。
def test_chat_result_to_response_normalizes_unknown_task():
    result = ChatResult(
        answer="无法识别",
        task_type="other",
        citations=[],
        evidence=[],
        critique="",
        is_valid=True,
        trace_id="trace-2",
        request_id="trace-2",
        steps=[],
        chat_messages=[],
        raw_output={},
    )

    response = chat_result_to_response(result, doc_id="kb")

    assert response.task_type == "unknown"


# 验证错误响应使用稳定错误码。
def test_error_response_uses_stable_error_code_values():
    response = build_error_response(
        ErrorCode.STREAM_INTERRUPTED,
        "stream closed",
        request_id="req-1",
        trace_id="trace-1",
        details={"stage": "stream"},
    )

    assert response.model_dump() == {
        "schema_version": "v1",
        "error_code": "STREAM_INTERRUPTED",
        "message": "stream closed",
        "request_id": "req-1",
        "trace_id": "trace-1",
        "details": {"stage": "stream"},
    }


# 验证跟踪响应使用稳定契约。
def test_trace_response_uses_stable_contract():
    response = TraceResponse(
        trace_id="trace-1",
        request_id="req-1",
        task_type="qa",
        status="ok",
        duration_ms=1.0,
        config={"doc_id": "kb"},
        summary={"step_count": 1, "node_names": ["intent_router"]},
        steps=[{"node_name": "intent_router"}],
    )
    payload = response.model_dump()

    assert payload["schema_version"] == "v1"
    assert payload["trace_id"] == "trace-1"
    assert payload["summary"]["step_count"] == 1
    assert payload["summary"]["error_count"] == 0
    assert payload["steps"][0]["node_name"] == "intent_router"
