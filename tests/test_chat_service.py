import pytest
from service import chat_service
from service.chat_service import ChatServiceError, run_chat_sync


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


class FakeApp:
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


class StreamInterruptApp:
    # 路由后流式迭代中途崩溃，父子图始终未产出可信输出。
    def stream(self, initial_state, config, stream_mode, subgraphs):
        yield (
            (),
            "updates",
            {"intent_router": {"task_type": "qa", "router_reason": "x"}},
        )
        raise TimeoutError("流中断")


def test_run_chat_sync_raises_on_stream_interrupt_without_output(monkeypatch):
    monkeypatch.setattr(chat_service, "app", StreamInterruptApp())
    monkeypatch.setattr(chat_service, "configure_logging", lambda: None)
    monkeypatch.setattr(chat_service, "log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_service, "export_trace", lambda **kwargs: None)

    with pytest.raises(ChatServiceError) as excinfo:
        run_chat_sync("kb", "报名要求是什么", is_local=False)

    assert excinfo.value.stage == "stream"
    assert excinfo.value.error_class == "TimeoutError"


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


def test_run_chat_sync_returns_degraded_result_when_partial_output(monkeypatch):
    monkeypatch.setattr(chat_service, "app", StreamInterruptWithPartialApp())
    monkeypatch.setattr(chat_service, "configure_logging", lambda: None)
    monkeypatch.setattr(chat_service, "log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_service, "export_trace", lambda **kwargs: None)

    result = run_chat_sync("kb", "报名要求是什么", is_local=False)

    assert result.raw_output.get("answer") == "部分答案"
