import json
from cogdoc.config.settings import Settings
from cogdoc.observability.trace import (
    build_trace_payload,
    build_trace_step,
    export_trace,
    trace_dir,
    trace_path,
)


# 验证跟踪步骤只保留安全正文预览。
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
                    "source_type": "derived_knowledge",
                    "knowledge_id": "K1",
                },
                "retrieval": {
                    "search_channel": "derived_knowledge",
                    "matched_terms": ["报名"],
                    "match_coverage": 1.0,
                    "query_term_count": 1,
                    "unsafe": "不应保留",
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
    assert step["evidence"][0]["source_type"] == "derived_knowledge"
    assert step["evidence"][0]["knowledge_id"] == "K1"
    assert step["evidence"][0]["retrieval"]["search_channel"] == "derived_knowledge"
    assert step["evidence"][0]["retrieval"]["matched_terms"] == ["报名"]
    assert step["evidence"][0]["retrieval"]["query_term_count"] == 1
    assert "unsafe" not in step["evidence"][0]["retrieval"]
    assert len(step["evidence"][0]["text_preview"]) <= 120
    assert "answer" not in step


# 验证跟踪步骤使用显式检索截断值。
def test_build_trace_step_uses_explicit_retrieval_top_k():
    output = {
        "retrieved_docs": [{"text": "x", "meta": {"chunk_id": "c1"}}],
        "rewritten_queries": ["改写问题"],
    }

    step = build_trace_step("retrieve_node", output, 1.0, retrieval_top_k=9)

    assert step["retrieval_top_k"] == 9
    assert step["counts"]["retrieved_count"] == 1
    assert step["rewritten_queries"] == ["改写问题"]


# 验证跟踪载荷包含审计字段。
def test_build_trace_payload_includes_audit_fields():
    step = build_trace_step("intent_router", {"task_type": "qa"}, 1.0)

    payload = build_trace_payload(
        "trace-1",
        "req-1",
        "qa",
        [step],
        status="ok",
        duration_ms=12.3456,
        config={"doc_id": "kb", "query_length": 4},
    )

    assert payload["schema_version"] == "v1"
    assert payload["status"] == "ok"
    assert payload["duration_ms"] == 12.346
    assert payload["config"]["doc_id"] == "kb"
    assert payload["summary"]["step_count"] == 1
    assert payload["summary"]["node_names"] == ["intent_router"]
    assert payload["error"] is None


# 验证跟踪目录与文件路径使用同一根目录解析。
def test_trace_dir_and_path_share_resolved_base(tmp_path):
    settings = Settings(
        cogdoc_trace_dir="relative-traces", cogdoc_data_dir=str(tmp_path)
    )

    base = trace_dir(settings)
    path = trace_path("trace-1", settings)

    assert base == settings.project_root / "relative-traces"
    assert path == base / "trace-1.json"


# 验证跟踪导出会写入文件。
def test_export_trace_writes_json_file(tmp_path):
    settings = Settings(cogdoc_trace_dir=str(tmp_path), cogdoc_trace_enabled=True)
    step = build_trace_step("intent_router", {"task_type": "qa"}, 1.0)

    path = export_trace(
        "trace-1",
        "req-1",
        "qa",
        [step],
        settings,
        status="degraded",
        duration_ms=3.0,
        error={"stage": "stream", "error_class": "TimeoutError"},
        config={"doc_id": "kb"},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "v1"
    assert payload["trace_id"] == "trace-1"
    assert payload["request_id"] == "req-1"
    assert payload["task_type"] == "qa"
    assert payload["status"] == "degraded"
    assert payload["duration_ms"] == 3.0
    assert payload["config"]["doc_id"] == "kb"
    assert payload["summary"]["error_count"] == 0
    assert payload["error"]["error_class"] == "TimeoutError"
    assert payload["steps"][0]["node_name"] == "intent_router"


# 验证跟踪导出尊重关闭开关。
def test_export_trace_respects_disabled_flag(tmp_path):
    settings = Settings(cogdoc_trace_dir=str(tmp_path), cogdoc_trace_enabled=False)

    path = export_trace("trace-1", "req-1", "qa", [], settings)

    assert path is None
    assert not list(tmp_path.iterdir())
