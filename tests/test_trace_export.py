import json
from langchain_core.messages import AIMessage

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


# 验证 trace 保留拒答决策及其安全评分信号。
def test_build_trace_step_keeps_retrieval_abstention_decision():
    step = build_trace_step(
        "rerank_node",
        {
            "retrieval_abstained": True,
            "retrieval_confidence": 0.91,
            "retrieval_abstain_reason": "below_threshold",
            "retrieval_signals": {"distance": 0.95, "bm25_score": 5.0},
        },
        1.0,
    )

    assert step["retrieval_abstained"] is True
    assert step["retrieval_confidence"] == 0.91
    assert step["retrieval_abstain_reason"] == "below_threshold"
    assert step["retrieval_signals"] == {"distance": 0.95, "bm25_score": 5.0}


# 验证 trace 保留二阶段证据结论但不写入完整证据正文。
def test_build_trace_step_keeps_evidence_verification_decision():
    step = build_trace_step(
        "evidence_verify_node",
        {
            "evidence_verification_required": True,
            "evidence_supported": False,
            "evidence_verification_reason": "缺少明确的报销比例",
            "evidence_verified_chunk_ids": [],
            "evidence_requirement_assessments": [
                {
                    "requirement_id": "r1",
                    "verdict": "missing",
                    "evidence_chunk_ids": [],
                    "reason": "缺少比例",
                }
            ],
            "missing_evidence_requirement_ids": ["r1"],
            "adaptive_retrieval_retry_pending": True,
            "retrieval_abstained": True,
            "retrieval_abstain_reason": "evidence_not_supported",
        },
        1.0,
    )

    assert step["evidence_verification_required"] is True
    assert step["evidence_supported"] is False
    assert step["evidence_verification_reason"] == "缺少明确的报销比例"
    assert step["evidence_verified_chunk_ids"] == []
    assert step["evidence_requirement_assessments"][0]["requirement_id"] == "r1"
    assert step["missing_evidence_requirement_ids"] == ["r1"]
    assert step["adaptive_retrieval_retry_pending"] is True


# 验证 trace 保留补检索轮次、深度与已验证证据携带数。
def test_build_trace_step_keeps_adaptive_retrieval_round_metadata():
    step = build_trace_step(
        "retrieve_node",
        {
            "retrieval_retry_count": 1,
            "retrieval_retry_reason": "missing_requirements",
            "retrieval_round": 1,
            "retrieval_top_k_used": 18,
            "retrieval_query_count": 4,
            "retrieval_ranking_count": 6,
            "retrieval_channel_counts": {"hybrid": 12, "derived_knowledge": 2},
            "retrieval_carryover_count": 1,
        },
        1.0,
        retrieval_top_k=18,
    )

    assert step["retrieval_retry_count"] == 1
    assert step["retrieval_retry_reason"] == "missing_requirements"
    assert step["retrieval_round"] == 1
    assert step["retrieval_top_k_used"] == 18
    assert step["retrieval_query_count"] == 4
    assert step["retrieval_ranking_count"] == 6
    assert step["retrieval_channel_counts"] == {
        "hybrid": 12,
        "derived_knowledge": 2,
    }
    assert step["retrieval_carryover_count"] == 1


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
    assert "claim_audit" not in payload["summary"]
    assert payload["error"] is None


# 验证新 trace 的列表摘要包含审计统计，但不复制声明正文、逐条理由或证据。
def test_build_trace_payload_includes_safe_claim_audit_summary():
    sensitive_claim = "敏感声明正文：报名费为 999 元"
    sensitive_reason = "敏感逐条理由：证据中的金额不同"
    sensitive_evidence = "敏感证据全文"
    audit = {
        "status": "failed",
        "reason_code": "claims_not_supported",
        "claims": [
            {
                "claim_id": "c1",
                "text": sensitive_claim,
                "verdict": "unsupported",
                "reason": sensitive_reason,
                "cited_chunk_ids": ["chunk-1"],
                "supporting_chunk_ids": [],
                "evidence": sensitive_evidence,
            }
        ],
        "counts": {
            "claim_count": 1,
            "supported": 0,
            "unsupported": 1,
            "insufficient": 0,
            "cited": 1,
            "skipped_statements": 0,
        },
        "metrics": {
            "claim_support_rate": 0.0,
            "citation_coverage": 1.0,
            "unsupported_claim_rate": 1.0,
        },
        "repair": {
            "attempted": True,
            "attempt_count": 1,
            "succeeded": False,
            "private_note": sensitive_reason,
        },
        "verifier": {"duration_ms": 125.5, "call_count": 1, "version": "v1"},
    }

    payload = build_trace_payload(
        "trace-claim",
        "req-claim",
        "qa",
        [],
        output_payload={"answer": "已阻断", "claim_audit": audit},
    )

    summary = payload["summary"]["claim_audit"]
    assert summary == {
        "status": "failed",
        "reason_code": "claims_not_supported",
        "counts": audit["counts"],
        "metrics": audit["metrics"],
        "repair": {
            "attempted": True,
            "attempt_count": 1,
            "succeeded": False,
        },
        "duration_ms": 125.5,
    }
    serialized_summary = json.dumps(summary, ensure_ascii=False)
    assert sensitive_claim not in serialized_summary
    assert sensitive_reason not in serialized_summary
    assert sensitive_evidence not in serialized_summary
    assert "claims" not in summary


# 验证 trace 列表摘要对畸形数值容错，且仍只保留固定摘要键。
def test_build_trace_payload_sanitizes_malformed_claim_audit_summary():
    payload = build_trace_payload(
        "trace-malformed-claim",
        "req-malformed-claim",
        "qa",
        [],
        output_payload={
            "claim_audit": {
                "status": "failed",
                "claims": [{"text": "不应出现在摘要的声明"}],
                "counts": {
                    "claim_count": "bad",
                    "supported": float("inf"),
                    "unsupported": -1,
                },
                "metrics": {
                    "claim_support_rate": "bad",
                    "citation_coverage": float("nan"),
                },
                "repair": {"attempt_count": float("inf")},
                "verifier": {"duration_ms": "bad"},
            }
        },
    )

    summary = payload["summary"]["claim_audit"]
    assert summary["counts"] == {
        "claim_count": 0,
        "supported": 0,
        "unsupported": 0,
    }
    assert summary["metrics"] == {
        "claim_support_rate": None,
        "citation_coverage": None,
    }
    assert summary["repair"]["attempt_count"] == 0
    assert summary["duration_ms"] is None
    assert "claims" not in summary


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


# 验证 trace 导出能处理 LangChain Message 等非原生 JSON 对象。
def test_export_trace_serializes_runtime_message_objects(tmp_path):
    settings = Settings(cogdoc_trace_dir=str(tmp_path), cogdoc_trace_enabled=True)
    step = build_trace_step(
        "qa_node",
        {"messages": [AIMessage(content="回答内容")], "answer": "回答内容"},
        1.0,
    )

    path = export_trace(
        "trace-message",
        "req-message",
        "qa",
        [step],
        settings,
        input_payload={"chat_history": [AIMessage(content="历史回答")]},
        output_payload={"messages": [AIMessage(content="最终回答")]},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["input"]["chat_history"][0]["content"] == "历史回答"
    assert payload["output"]["messages"][0]["content"] == "最终回答"
    assert (
        payload["steps"][0]["output_snapshot"]["messages"][0]["content"] == "回答内容"
    )


# 验证跟踪导出尊重关闭开关。
def test_export_trace_respects_disabled_flag(tmp_path):
    settings = Settings(cogdoc_trace_dir=str(tmp_path), cogdoc_trace_enabled=False)

    path = export_trace("trace-1", "req-1", "qa", [], settings)

    assert path is None
    assert not list(tmp_path.iterdir())
