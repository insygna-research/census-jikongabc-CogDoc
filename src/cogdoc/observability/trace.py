import json
import time
from pathlib import Path
from typing import Any, Mapping
from cogdoc.config.settings import Settings, get_settings


TRACE_SCHEMA_VERSION = "v1"
TRACE_PREVIEW_CHARS = 120


# 返回单调毫秒时间。
def monotonic_ms() -> float:
    return time.monotonic() * 1000


# 构建短文本预览。
def _preview(text: Any, limit: int = TRACE_PREVIEW_CHARS) -> str:
    compact = " ".join(str(text or "").split())
    return compact[:limit]


# 构建文档引用摘要。
def _doc_ref(doc: Mapping[str, Any]) -> dict:
    meta = doc.get("meta", {})
    return {
        "chunk_id": meta.get("chunk_id", ""),
        "source": meta.get("source", ""),
        "page": meta.get("page", 0),
        "page_start": meta.get("page_start", meta.get("page", 0)),
        "page_end": meta.get("page_end", meta.get("page", 0)),
        "text_preview": _preview(doc.get("text", "")),
    }


# 构建证据引用摘要。
def _evidence_ref(item: Mapping[str, Any]) -> dict:
    return {
        "chunk_id": item.get("chunk_id", ""),
        "source": item.get("source", ""),
        "page": item.get("page", 0),
        "page_start": item.get("page_start", item.get("page", 0)),
        "page_end": item.get("page_end", item.get("page", 0)),
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

    count_fields = {
        "rewritten_queries": "rewritten_query_count",
        "retrieved_docs": "retrieved_count",
        "reranked_docs": "reranked_count",
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


# 汇总跟踪步骤。
def summarize_trace_steps(steps: list[dict]) -> dict:
    error_steps = [
        step
        for step in steps
        if step.get("error_class") or step.get("critique")
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
) -> dict:
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_id": trace_id,
        "request_id": request_id,
        "task_type": task_type,
        "status": status,
        "duration_ms": None
        if duration_ms is None
        else round(max(duration_ms, 0.0), 3),
        "config": dict(config or {}),
        "summary": summarize_trace_steps(steps),
        "error": dict(error or {}) or None,
        "steps": steps,
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
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
