import pytest
from cogdoc.service import chat_service
from cogdoc.service.chat_service import ChatServiceError, run_chat_sync


# 构造测试用文档。
def _doc() -> dict:
    return {
        "text": "报名要求。",
        "meta": {
            "chunk_id": "chunk:a:1",
            "source": "a.pdf",
            "page": 1,
            "page_start": 1,
            "page_end": 1,
            "chunk_index": 0,
            "local_chunk_index": 0,
            "source_sha256": "sha",
            "origin": "file",
        },
        "retrieval": {"rrf_score": 0.1},
    }


# 定义假应用数据结构。
class FakeApp:
    # 流式返回结果。
    def stream(self, initial_state, config, stream_mode, subgraphs):
        assert initial_state["trace_id"]
        assert config["configurable"]["trace_id"] == initial_state["trace_id"]
        yield (
            (),
            "updates",
            {
                "intent_router": {
                    "query": "报名要求是什么",
                    "doc_id": "kb",
                    "is_local": False,
                    "task_type": "qa",
                    "router_reason": "用户询问信息",
                }
            },
        )
        yield (
            ("qa_subgraph",),
            "updates",
            {"rewrite_node": {"rewritten_queries": ["报名要求"]}},
        )
        yield (
            ("qa_subgraph",),
            "updates",
            {"retrieve_node": {"retrieved_docs": [_doc()]}},
        )
        yield (
            ("qa_subgraph",),
            "updates",
            {"citation_node": {"critique": "", "iteration_count": 1}},
        )
        yield (
            (),
            "updates",
            {
                "qa_subgraph": {
                    "answer": "需要满足报名要求。[a.pdf:P1]",
                    "critique": "",
                    "reranked_docs": [_doc()],
                    "sources": [_doc()["meta"]],
                    "evidence": [{"chunk_id": "chunk:a:1", "source": "a.pdf"}],
                }
            },
        )


# 验证同步对话返回结构化结果。
def test_run_chat_sync_returns_structured_result(monkeypatch):
    monkeypatch.setattr(chat_service, "app", FakeApp())
    monkeypatch.setattr(chat_service, "configure_logging", lambda: None)
    monkeypatch.setattr(chat_service, "log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_service, "export_trace", lambda **kwargs: None)

    result = run_chat_sync("kb", "报名要求是什么", is_local=False)

    assert result.task_type == "qa"
    assert result.answer == "需要满足报名要求。[a.pdf:P1]"
    assert result.is_valid is True
    assert result.citations[0]["source"] == "a.pdf"
    assert result.evidence[0]["chunk_id"] == "chunk:a:1"
    assert result.chat_messages
    assert [step["node_name"] for step in result.steps][:2] == [
        "runtime.setup",
        "intent_router",
    ]
    assert any(step["retrieval_top_k"] == 9 for step in result.steps)


# 验证对话会导出可审计跟踪。
def test_run_chat_exports_auditable_trace(monkeypatch):
    exported = []
    monkeypatch.setattr(chat_service, "app", FakeApp())
    monkeypatch.setattr(chat_service, "configure_logging", lambda: None)
    monkeypatch.setattr(chat_service, "log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        chat_service, "export_trace", lambda **kwargs: exported.append(kwargs)
    )

    result = run_chat_sync("kb", "报名要求是什么", is_local=False)

    assert result.trace_path is None
    assert exported[0]["status"] == "ok"
    assert exported[0]["task_type"] == "qa"
    assert exported[0]["duration_ms"] >= 0
    assert exported[0]["config"]["doc_id"] == "kb"
    assert exported[0]["config"]["query_preview"] == "报名要求是什么"
    assert exported[0]["config"]["query_length"] == len("报名要求是什么")
    assert exported[0]["error"] is None


# 验证流式对话事件顺序稳定。
def test_run_chat_emits_golden_event_sequence(monkeypatch):
    monkeypatch.setattr(chat_service, "app", FakeApp())
    monkeypatch.setattr(chat_service, "configure_logging", lambda: None)
    monkeypatch.setattr(chat_service, "log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_service, "export_trace", lambda **kwargs: None)

    events = list(chat_service.run_chat("kb", "报名要求是什么", is_local=False))

    assert [event.type for event in events] == [
        "request_started",
        "router_decided",
        "rewrite_queries",
        "citation_passed",
        "final",
    ]


# 路由后流式迭代中途崩溃，父子图始终未产出可信输出。
class StreamInterruptApp:
    # 路由后流式迭代中途崩溃，父子图始终未产出可信输出。
    def stream(self, initial_state, config, stream_mode, subgraphs):
        yield (
            (),
            "updates",
            {"intent_router": {"task_type": "qa", "router_reason": "x"}},
        )
        raise TimeoutError("流中断")


# 验证无可信输出时流式中断会抛错。
def test_run_chat_sync_raises_on_stream_interrupt_without_output(monkeypatch):
    monkeypatch.setattr(chat_service, "app", StreamInterruptApp())
    monkeypatch.setattr(chat_service, "configure_logging", lambda: None)
    monkeypatch.setattr(chat_service, "log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_service, "export_trace", lambda **kwargs: None)

    with pytest.raises(ChatServiceError) as excinfo:
        run_chat_sync("kb", "报名要求是什么", is_local=False)

    assert excinfo.value.stage == "stream"
    assert excinfo.value.error_class == "TimeoutError"


# 验证无可信输出时跟踪标记失败。
def test_run_chat_exports_failed_trace_on_stream_interrupt(monkeypatch):
    exported = []
    monkeypatch.setattr(chat_service, "app", StreamInterruptApp())
    monkeypatch.setattr(chat_service, "configure_logging", lambda: None)
    monkeypatch.setattr(chat_service, "log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        chat_service, "export_trace", lambda **kwargs: exported.append(kwargs)
    )

    with pytest.raises(ChatServiceError):
        run_chat_sync("kb", "报名要求是什么", is_local=False)

    assert exported[0]["status"] == "failed"
    assert exported[0]["error"]["stage"] == "stream"
    assert exported[0]["error"]["error_class"] == "TimeoutError"


# 父子图输出已落地后流才中断，属于可降级返回而非彻底失败。
class StreamInterruptWithPartialApp:
    # 父子图输出已落地后流才中断，属于可降级返回而非彻底失败。
    def stream(self, initial_state, config, stream_mode, subgraphs):
        yield (
            (),
            "updates",
            {"intent_router": {"task_type": "qa", "router_reason": "x"}},
        )
        yield (
            (),
            "updates",
            {
                "qa_subgraph": {
                    "answer": "部分答案",
                    "critique": "",
                    "reranked_docs": [],
                }
            },
        )
        raise TimeoutError("流中断")


# 验证部分输出已落地时可降级返回。
def test_run_chat_sync_returns_degraded_result_when_partial_output(monkeypatch):
    monkeypatch.setattr(chat_service, "app", StreamInterruptWithPartialApp())
    monkeypatch.setattr(chat_service, "configure_logging", lambda: None)
    monkeypatch.setattr(chat_service, "log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_service, "export_trace", lambda **kwargs: None)

    result = run_chat_sync("kb", "报名要求是什么", is_local=False)

    assert result.raw_output.get("answer") == "部分答案"


# 验证部分输出已落地时跟踪标记降级。
def test_run_chat_exports_degraded_trace_when_partial_output(monkeypatch):
    exported = []
    monkeypatch.setattr(chat_service, "app", StreamInterruptWithPartialApp())
    monkeypatch.setattr(chat_service, "configure_logging", lambda: None)
    monkeypatch.setattr(chat_service, "log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        chat_service, "export_trace", lambda **kwargs: exported.append(kwargs)
    )

    result = run_chat_sync("kb", "报名要求是什么", is_local=False)

    assert result.raw_output.get("answer") == "部分答案"
    assert exported[0]["status"] == "degraded"
    assert exported[0]["error"]["stage"] == "stream"
