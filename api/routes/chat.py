import asyncio
import json
from typing import Callable
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorCode,
    ErrorResponse,
    build_error_response,
    chat_result_to_response,
)
from service.chat_service import (
    ChatEvent,
    ChatResult,
    ChatServiceError,
    run_chat,
    run_chat_sync,
)


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
# OpenAPI 错误响应契约，让前端按稳定 schema 处理失败。
_ERROR_RESPONSES = {
    502: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}


@router.post("/chat", response_model=ChatResponse, responses=_ERROR_RESPONSES)
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


# 流式进度事件直接转发；token/final/error 单独成结构化帧。
_SSE_PROGRESS_TYPES = {
    "router_decided",
    "rewrite_queries",
    "citation_passed",
    "citation_rejected",
    "compare_citation_passed",
    "compare_citation_rejected",
}
_STREAM_DONE = object()


def _sse_frame(event_name: str, data: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _event_to_frame(
    event: ChatEvent, *, doc_id: str, session_id: str | None
) -> str | None:
    if event.type == "request_started":
        return _sse_frame(
            "start",
            {"trace_id": event.payload.get("trace_id"), "doc_id": doc_id},
        )
    if event.type == "token":
        return _sse_frame("token", {"content": event.payload.get("content", "")})
    if event.type in _SSE_PROGRESS_TYPES:
        # round_answer 是 CLI 展示用的整段模型回答，不进流式帧。
        data = {k: v for k, v in event.payload.items() if k != "round_answer"}
        data["stage"] = event.type
        return _sse_frame("node", data)
    if event.type == "final":
        chat_response = chat_result_to_response(
            event.payload["result"], doc_id=doc_id, session_id=session_id
        )
        return _sse_frame("final", chat_response.model_dump())
    if event.type == "error":
        error_code = _ERROR_CODE_BY_STAGE.get(
            event.payload.get("stage"), ErrorCode.MODEL_UNAVAILABLE
        )
        error = build_error_response(
            error_code,
            event.payload.get("message", ""),
            request_id=event.payload.get("trace_id"),
            trace_id=event.payload.get("trace_id"),
            details={
                "error_class": event.payload.get("error_class"),
                "stage": event.payload.get("stage"),
            },
        )
        return _sse_frame("error", error.model_dump())
    return None


@router.post("/chat/stream", responses=_ERROR_RESPONSES)
async def chat_stream(request_body: ChatRequest, request: Request):
    stream_runner = getattr(request.app.state, "chat_stream_runner", run_chat)
    session_store = request.app.state.session_store
    doc_id = request_body.doc_id
    session_id = request_body.session_id
    chat_history = session_store.get_history(doc_id, session_id)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def produce() -> None:
        # 同步事件流跑在有界线程池里，逐事件回投到事件循环的队列。
        try:
            for event in stream_runner(
                doc_id,
                request_body.query,
                request_body.is_local,
                chat_history,
                request_body.forced_task,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                ChatEvent(
                    "error",
                    {
                        "error_class": type(exc).__name__,
                        "message": str(exc),
                        "stage": "runtime",
                    },
                ),
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _STREAM_DONE)

    request.app.state.offload_executor.submit(produce)

    async def event_source():
        final_result: ChatResult | None = None
        while True:
            event = await queue.get()
            if event is _STREAM_DONE:
                break
            if event.type == "final":
                final_result = event.payload["result"]
            frame = _event_to_frame(event, doc_id=doc_id, session_id=session_id)
            if frame is not None:
                yield frame
        # 只有真正产出 final 才写会话；断连或失败不污染历史。
        if final_result is not None:
            session_store.append_messages(
                doc_id, session_id, final_result.chat_messages
            )

    return StreamingResponse(event_source(), media_type="text/event-stream")
