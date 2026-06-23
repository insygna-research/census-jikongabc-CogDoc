import asyncio
from typing import Callable
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorCode,
    build_error_response,
    chat_result_to_response,
)
from service.chat_service import ChatResult, ChatServiceError, run_chat_sync


ChatRunner = Callable[..., ChatResult]

router = APIRouter(prefix="/v1", tags=["chat"])

# 服务层失败阶段 -> 稳定 error_code / HTTP 状态码。
_ERROR_CODE_BY_STAGE = {
    "stream": ErrorCode.STREAM_INTERRUPTED,
    "runtime": ErrorCode.MODEL_UNAVAILABLE,
}
_STATUS_BY_CODE = {
    ErrorCode.STREAM_INTERRUPTED: 502,
    ErrorCode.MODEL_UNAVAILABLE: 503,
}


@router.post("/chat", response_model=ChatResponse)
async def chat(request_body: ChatRequest, request: Request, response: Response):
    runner: ChatRunner = getattr(request.app.state, "chat_runner", run_chat_sync)
    session_store = request.app.state.session_store
    chat_history = session_store.get_history(
        request_body.doc_id,
        request_body.session_id,
    )

    try:
        # 用 app 级有界线程池 offload 同步图：不阻塞事件循环、不无界起线程、不走 anyio。
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            request.app.state.offload_executor,
            runner,
            request_body.doc_id,
            request_body.query,
            request_body.is_local,
            chat_history,
            request_body.forced_task,
        )
    except ChatServiceError as exc:
        error_code = _ERROR_CODE_BY_STAGE.get(exc.stage, ErrorCode.MODEL_UNAVAILABLE)
        error = build_error_response(
            error_code,
            exc.message,
            request_id=exc.trace_id,
            trace_id=exc.trace_id,
            details={"error_class": exc.error_class, "stage": exc.stage},
        )
        return JSONResponse(
            status_code=_STATUS_BY_CODE.get(error_code, 503),
            content=error.model_dump(),
            headers={"X-Trace-Id": exc.trace_id or ""},
        )

    session_store.append_messages(
        request_body.doc_id,
        request_body.session_id,
        result.chat_messages,
    )
    chat_response = chat_result_to_response(
        result,
        doc_id=request_body.doc_id,
        session_id=request_body.session_id,
    )
    response.headers["X-Trace-Id"] = chat_response.trace_id
    return chat_response
