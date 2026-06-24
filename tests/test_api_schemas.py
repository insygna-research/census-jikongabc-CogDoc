import pytest
from pydantic import ValidationError
from cogdoc.api.schemas import (
    API_SCHEMA_VERSION,
    ChatRequest,
    ErrorCode,
    build_error_response,
    chat_result_to_response,
)
from cogdoc.service.chat_service import ChatResult


def test_chat_request_defaults_and_forced_task():
    request = ChatRequest(query="  总结 a.pdf  ", mode="summary")

    assert request.schema_version == API_SCHEMA_VERSION
    assert request.query == "总结 a.pdf"
    assert request.doc_id
    assert request.forced_task == "summary"
    assert ChatRequest(query="问题").forced_task is None


def test_chat_request_rejects_blank_and_unknown_fields():
    with pytest.raises(ValidationError):
        ChatRequest(query="  ")

    with pytest.raises(ValidationError):
        ChatRequest(query="问题", unexpected=True)


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
            "source": "a.pdf",
            "page": 1,
            "page_start": 1,
            "page_end": 2,
        }
    ]
    assert payload["evidence"][0]["rerank_score"] == 0.98
    assert payload["evidence"][0]["text_preview"] == "报名要求摘要"
    assert "raw_output" not in payload
    assert "steps" not in payload
    assert "trace_path" not in payload
    assert "不应进入 API 响应的全文" not in str(payload)


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
