import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from cogdoc.config.settings import Settings, get_settings
from cogdoc.tools.retriever.metadata import safe_retrieval_metadata


TRACE_SCHEMA_VERSION = "v1"
TRACE_PREVIEW_CHARS = 120


# 返回单调毫秒时间。
def monotonic_ms() -> float:
    return time.monotonic() * 1000


# 构建短文本预览。
def _preview(text: Any, limit: int = TRACE_PREVIEW_CHARS) -> str:
    compact = " ".join(str(text or "").split())
    return compact[:limit]


# 将 LangChain / Pydantic 等运行期对象转成可写入 trace JSON 的结构。
def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _json_safe(value.dict())
        except Exception:
            pass
    if hasattr(value, "content"):
        payload = {
            "type": getattr(value, "type", value.__class__.__name__),
            "content": getattr(value, "content", ""),
        }
        return _json_safe(payload)
    return str(value)


# 构建文档引用摘要。
def _doc_ref(doc: Mapping[str, Any]) -> dict:
    meta = doc.get("meta", {})
    retrieval = doc.get("retrieval") or {}
    return {
        "chunk_id": meta.get("chunk_id", ""),
        "source_type": meta.get("source_type", "document"),
        "knowledge_id": meta.get("knowledge_id", ""),
        "source": meta.get("source", ""),
        "page": meta.get("page", 0),
        "page_start": meta.get("page_start", meta.get("page", 0)),
        "page_end": meta.get("page_end", meta.get("page", 0)),
        "rewrite_query": retrieval.get("rewrite_query", ""),
        "retrieval": safe_retrieval_metadata(retrieval),
        "text_preview": _preview(doc.get("text", "")),
    }


# 构建证据引用摘要。
def _evidence_ref(item: Mapping[str, Any]) -> dict:
    return {
        "chunk_id": item.get("chunk_id", ""),
        "source_type": item.get("source_type", "document"),
        "knowledge_id": item.get("knowledge_id", ""),
        "source": item.get("source", ""),
        "page": item.get("page", 0),
        "page_start": item.get("page_start", item.get("page", 0)),
        "page_end": item.get("page_end", item.get("page", 0)),
        "retrieval": safe_retrieval_metadata(item.get("retrieval") or {}),
        "text_preview": _preview(item.get("text_preview", "")),
    }


# 构建跟踪步骤。
def build_trace_step(
    node_name: str,
    output: Mapping[str, Any],
    duration_ms: float,
    model_name: str | None = None,
    retrieval_top_k: int | None = None,
) -> dict:
    step = {
        "node_name": node_name,
        "duration_ms": round(max(duration_ms, 0.0), 3),
        "model": model_name,
        "token": None,
        "retrieval_top_k": retrieval_top_k,
        "critique": None,
        "error_class": None,
        "counts": {},
        "evidence": [],
        # 评分和 Bad Case 回灌需要原始节点结果；展示层仍使用下方的摘要字段。
        "output_snapshot": _json_safe(dict(output)),
    }

    if "task_type" in output:
        step["task_type"] = output.get("task_type")
    if "router_reason" in output:
        step["router_reason"] = _preview(output.get("router_reason"), 240)
    if "error" in output:
        step["error_class"] = output.get("error") or "error"
    if "critique" in output:
        critique = str(output.get("critique") or "")
        step["critique"] = _preview(critique, 300) if critique else ""
    if "retrieval_abstained" in output:
        step["retrieval_abstained"] = bool(output.get("retrieval_abstained"))
        step["retrieval_confidence"] = output.get("retrieval_confidence")
        step["retrieval_abstain_reason"] = _preview(
            output.get("retrieval_abstain_reason"), 80
        )
        signals = output.get("retrieval_signals")
        if isinstance(signals, Mapping):
            step["retrieval_signals"] = {
                str(key): value for key, value in signals.items()
            }
    if "evidence_verification_required" in output:
        step["evidence_verification_required"] = bool(
            output.get("evidence_verification_required")
        )
        step["evidence_supported"] = bool(output.get("evidence_supported"))
        step["evidence_verification_reason"] = _preview(
            output.get("evidence_verification_reason"), 300
        )
        step["evidence_verified_chunk_ids"] = [
            _preview(chunk_id, 120)
            for chunk_id in list(output.get("evidence_verified_chunk_ids") or [])[:5]
        ]
        if output.get("evidence_verifier_error"):
            step["evidence_verifier_error"] = _preview(
                output.get("evidence_verifier_error"), 80
            )
    if "evidence_verification_pending" in output:
        step["evidence_verification_pending"] = bool(
            output.get("evidence_verification_pending")
        )
    if output.get("rewritten_queries"):
        step["rewritten_queries"] = [
            _preview(query, 120)
            for query in list(output.get("rewritten_queries") or [])[:5]
        ]
    if output.get("steps_trace"):
        step["steps_trace"] = [
            {
                "step_name": _preview(item.get("step_name", ""), 80),
                "input_summary": _preview(item.get("input_summary", ""), 400),
                "output_summary": _preview(item.get("output_summary", ""), 800),
            }
            for item in list(output.get("steps_trace") or [])[:5]
            if isinstance(item, Mapping)
        ]

    count_fields = {
        "rewritten_queries": "rewritten_query_count",
        "retrieved_docs": "retrieved_count",
        "reranked_docs": "reranked_count",
        "verification_docs": "verification_candidate_count",
        "summary_docs": "summary_doc_count",
        "summary_section_results": "summary_section_count",
        "compare_sources": "compare_source_count",
        "document_profiles": "document_profile_count",
        "evidence": "evidence_count",
    }
    for source_key, target_key in count_fields.items():
        value = output.get(source_key)
        if isinstance(value, list):
            step["counts"][target_key] = len(value)

    if output.get("retrieved_docs"):
        step["evidence"] = [_doc_ref(doc) for doc in output["retrieved_docs"][:5]]
    elif output.get("reranked_docs"):
        step["evidence"] = [_doc_ref(doc) for doc in output["reranked_docs"][:5]]
    elif output.get("evidence"):
        step["evidence"] = [_evidence_ref(item) for item in output["evidence"][:8]]

    return step


# 构建跟踪目录路径。
def trace_dir(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    base_dir = Path(settings.cogdoc_trace_dir)
    if not base_dir.is_absolute():
        base_dir = settings.project_root / base_dir
    return base_dir


# 构建跟踪文件路径。
def trace_path(trace_id: str, settings: Settings | None = None) -> Path:
    base_dir = trace_dir(settings)
    return base_dir / f"{trace_id}.json"


# 判断跟踪是否属于指定范围。
def _trace_matches_scope(
    payload: Mapping[str, Any], doc_id: str, session_id: str
) -> bool:
    config = payload.get("config") if isinstance(payload.get("config"), Mapping) else {}
    if doc_id and str(config.get("doc_id") or "") != doc_id:
        return False
    if session_id and str(config.get("session_id") or "") != session_id:
        return False
    return True


# 清理指定知识库或会话的跟踪文件。
def delete_trace_files(
    doc_id: str = "", session_id: str = "", settings: Settings | None = None
) -> int:
    if not doc_id and not session_id:
        return 0
    base_dir = trace_dir(settings)
    if not base_dir.exists() or not base_dir.is_dir():
        return 0
    deleted = 0
    for path in base_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping) and _trace_matches_scope(
                payload, doc_id, session_id
            ):
                path.unlink()
                deleted += 1
        except (OSError, json.JSONDecodeError):
            continue
    return deleted


# 汇总跟踪步骤。
def summarize_trace_steps(steps: list[dict]) -> dict:
    error_steps = [
        step for step in steps if step.get("error_class") or step.get("critique")
    ]
    evidence_count = sum(len(step.get("evidence", [])) for step in steps)
    return {
        "step_count": len(steps),
        "error_count": len(error_steps),
        "evidence_ref_count": evidence_count,
        "node_names": [step.get("node_name", "") for step in steps],
    }


# 构建跟踪导出载荷。
def build_trace_payload(
    trace_id: str,
    request_id: str,
    task_type: str,
    steps: list[dict],
    status: str = "ok",
    duration_ms: float | None = None,
    error: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    input_payload: Mapping[str, Any] | None = None,
    output_payload: Mapping[str, Any] | None = None,
    execution_status: str | None = None,
) -> dict:
    resolved_status = execution_status or (
        "SUCCESS" if status == "ok" else "TRACE_INCOMPLETE" if status == "degraded" else "TARGET_ERROR"
    )
    required_evidence = {"input", "output", "steps"}
    available_evidence = {
        name for name, value in {
            "input": input_payload,
            "output": output_payload,
            "steps": steps,
        }.items() if value is not None and (value or name == "steps")
    }
    evidence_completeness = len(required_evidence & available_evidence) / len(required_evidence)
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_id": trace_id,
        "request_id": request_id,
        "task_type": task_type,
        "status": status,
        "duration_ms": None if duration_ms is None else round(max(duration_ms, 0.0), 3),
        "execution_status": resolved_status,
        "input": _json_safe(dict(input_payload or {})),
        "output": _json_safe(dict(output_payload or {})),
        "evidence_completeness": evidence_completeness,
        "config": _json_safe(dict(config or {})),
        "summary": summarize_trace_steps(steps),
        "error": _json_safe(dict(error or {})) or None,
        "steps": _json_safe(steps),
    }


# 导出跟踪文件。
def export_trace(
    trace_id: str,
    request_id: str,
    task_type: str,
    steps: list[dict],
    settings: Settings | None = None,
    status: str = "ok",
    duration_ms: float | None = None,
    error: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    input_payload: Mapping[str, Any] | None = None,
    output_payload: Mapping[str, Any] | None = None,
    execution_status: str | None = None,
) -> Path | None:
    settings = settings or get_settings()
    if not settings.cogdoc_trace_enabled:
        return None

    path = trace_path(trace_id, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_trace_payload(
        trace_id=trace_id,
        request_id=request_id,
        task_type=task_type,
        steps=steps,
        status=status,
        duration_ms=duration_ms,
        error=error,
        config=config,
        input_payload=input_payload,
        output_payload=output_payload,
        execution_status=execution_status,
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
