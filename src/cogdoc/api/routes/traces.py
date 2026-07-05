import json
import re
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from cogdoc.api.schemas import (
    ErrorCode,
    ErrorResponse,
    TraceListItem,
    TraceListResponse,
    TraceResponse,
    build_error_response,
)
from cogdoc.observability.trace import build_trace_payload, trace_dir, trace_path


router = APIRouter(prefix="/v1", tags=["traces"])
_TRACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


# 判断跟踪标识是否安全。
def _is_safe_trace_id(trace_id: str) -> bool:
    return bool(_TRACE_ID_PATTERN.fullmatch(trace_id))


# 构建跟踪查询错误响应。
def _trace_error(code: ErrorCode, message: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=build_error_response(code, message).model_dump(),
    )


# 兼容旧版跟踪载荷。
def _normalize_trace_payload(trace_id: str, payload: dict) -> dict:
    if payload.get("schema_version"):
        return payload
    return build_trace_payload(
        trace_id=str(payload.get("trace_id") or trace_id),
        request_id=str(
            payload.get("request_id") or payload.get("trace_id") or trace_id
        ),
        task_type=str(payload.get("task_type") or "unknown"),
        steps=list(payload.get("steps") or []),
        status="ok",
    )


# 处理modifiedAT。
def _modified_at(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


# 处理跟踪listitem。
def _trace_list_item(
    path: Path, doc_id: str = "", session_id: str = ""
) -> TraceListItem | None:
    trace_id = path.stem
    if not _is_safe_trace_id(trace_id):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        trace = TraceResponse.model_validate(
            _normalize_trace_payload(trace_id, payload)
        )
        if doc_id and str(trace.config.get("doc_id") or "") != doc_id:
            return None
        if session_id and str(trace.config.get("session_id") or "") != session_id:
            return None
        return TraceListItem(
            trace_id=trace.trace_id,
            request_id=trace.request_id,
            query_preview=str(trace.config.get("query_preview") or ""),
            task_type=trace.task_type,
            status=trace.status,
            duration_ms=trace.duration_ms,
            modified_at=_modified_at(path),
            summary=trace.summary,
        )
    except (OSError, JSONDecodeError, ValidationError):
        return None


# 列出最近跟踪文件。
@router.get("/traces", response_model=TraceListResponse)
async def list_traces(
    limit: int = Query(default=20, ge=1, le=100),
    doc_id: str = Query(default=""),
    session_id: str = Query(default=""),
):
    base_dir = trace_dir()
    if not base_dir.exists() or not base_dir.is_dir():
        return TraceListResponse()
    candidates = []
    for path in base_dir.glob("*.json"):
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    traces = []
    for _, path in sorted(candidates, reverse=True):
        item = _trace_list_item(path, doc_id=doc_id, session_id=session_id)
        if item is not None:
            traces.append(item)
        if len(traces) >= limit:
            break
    return TraceListResponse(traces=traces)


# 查询单次请求的跟踪文件。
@router.get(
    "/traces/{trace_id}",
    response_model=TraceResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_trace(trace_id: str):
    if not _is_safe_trace_id(trace_id):
        return _trace_error(ErrorCode.TRACE_NOT_FOUND, f"trace 不存在: {trace_id}", 404)
    path = trace_path(trace_id)
    if not path.exists() or not path.is_file():
        return _trace_error(ErrorCode.TRACE_NOT_FOUND, f"trace 不存在: {trace_id}", 404)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError:
        return _trace_error(
            ErrorCode.INTERNAL_ERROR, f"trace 文件损坏: {trace_id}", 500
        )
    return TraceResponse.model_validate(_normalize_trace_payload(trace_id, payload))
