import json
import re
from json import JSONDecodeError
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from cogdoc.api.schemas import (
    ErrorCode,
    ErrorResponse,
    TraceResponse,
    build_error_response,
)
from cogdoc.observability.trace import build_trace_payload, trace_path


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
        request_id=str(payload.get("request_id") or payload.get("trace_id") or trace_id),
        task_type=str(payload.get("task_type") or "unknown"),
        steps=list(payload.get("steps") or []),
        status="ok",
    )


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
        return _trace_error(ErrorCode.INTERNAL_ERROR, f"trace 文件损坏: {trace_id}", 500)
    return TraceResponse.model_validate(_normalize_trace_payload(trace_id, payload))
