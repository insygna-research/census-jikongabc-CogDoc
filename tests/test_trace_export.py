import json
from cogdoc.config.settings import Settings
from cogdoc.observability.trace import build_trace_step, export_trace


def test_build_trace_step_keeps_only_safe_document_preview():
    output = {
        "retrieved_docs": [
            {
                "text": "非常长的正文" * 100,
                "meta": {
                    "chunk_id": "chunk-1",
                    "source": "a.pdf",
                    "page": 3,
                    "page_start": 3,
                    "page_end": 4,
                },
            }
        ],
        "answer": "不应进入 trace 的完整答案",
    }

    step = build_trace_step("retrieve_node", output, 12.345, model_name="model-a")

    assert step["node_name"] == "retrieve_node"
    assert step["duration_ms"] == 12.345
    assert step["model"] == "model-a"
    assert step["retrieval_top_k"] is None
    assert step["counts"]["retrieved_count"] == 1
    assert step["evidence"][0]["chunk_id"] == "chunk-1"
    assert len(step["evidence"][0]["text_preview"]) <= 120
    assert "answer" not in step


def test_build_trace_step_uses_explicit_retrieval_top_k():
    output = {"retrieved_docs": [{"text": "x", "meta": {"chunk_id": "c1"}}]}

    step = build_trace_step("retrieve_node", output, 1.0, retrieval_top_k=9)

    assert step["retrieval_top_k"] == 9
    assert step["counts"]["retrieved_count"] == 1


def test_export_trace_writes_json_file(tmp_path):
    settings = Settings(cogdoc_trace_dir=str(tmp_path), cogdoc_trace_enabled=True)
    step = build_trace_step("intent_router", {"task_type": "qa"}, 1.0)

    path = export_trace("trace-1", "req-1", "qa", [step], settings)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["trace_id"] == "trace-1"
    assert payload["request_id"] == "req-1"
    assert payload["task_type"] == "qa"
    assert payload["steps"][0]["node_name"] == "intent_router"


def test_export_trace_respects_disabled_flag(tmp_path):
    settings = Settings(cogdoc_trace_dir=str(tmp_path), cogdoc_trace_enabled=False)

    path = export_trace("trace-1", "req-1", "qa", [], settings)

    assert path is None
    assert not list(tmp_path.iterdir())
