import asyncio
import json
from typing import Callable
from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from cogdoc.api.error_mapping import classify_error_code, status_for_code
from cogdoc.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    SessionHistoryResponse,
    SessionListResponse,
    build_error_response,
    chat_result_to_response,
)
from cogdoc.config.settings import get_settings
from cogdoc.service.chat_service import (
    ChatEvent,
    ChatResult,
    ChatServiceError,
    run_chat,
    run_chat_sync,
)


ChatRunner = Callable[..., ChatResult]

router = APIRouter(prefix="/v1", tags=["chat"])

# OpenAPI 错误响应契约，让前端按稳定 schema 处理失败。
_ERROR_RESPONSES = {
    429: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    504: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}


@router.post("/chat", response_model=ChatResponse, responses=_ERROR_RESPONSES)
async def chat(request_body: ChatRequest, request: Request, response: Response):
    # 同步问答：offload 跑图 → 写会话 → 映射结构化响应；服务层异常转稳定错误码。
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
        error_code = classify_error_code(exc.stage, exc.error_class, exc.message)
        error = build_error_response(
            error_code,
            exc.message,
            request_id=exc.trace_id,
            trace_id=exc.trace_id,
            details={"error_class": exc.error_class, "stage": exc.stage},
        )
        return JSONResponse(
            status_code=status_for_code(error_code),
            content=error.model_dump(),
            headers={"X-Trace-Id": exc.trace_id or ""},
        )

    # 记忆走门控后的 chat_messages；展示存「用户问题 + 实际答案」，切对话时能看到内容。
    session_store.record(
        request_body.doc_id,
        request_body.session_id,
        result.chat_messages,
        [
            {"role": "user", "content": request_body.query},
            {"role": "assistant", "content": result.answer},
        ],
    )
    request.app.state.metrics.chat_results.labels(
        result.task_type, str(result.is_valid).lower()
    ).inc()
    chat_response = chat_result_to_response(
        result,
        doc_id=request_body.doc_id,
        session_id=request_body.session_id,
    )
    response.headers["X-Trace-Id"] = chat_response.trace_id
    return chat_response


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(request: Request, doc_id: str = Query(default="")):
    # 列出某库下的全部对话，供前端多对话列表。
    kb_id = doc_id or get_settings().cogdoc_default_doc_id
    sessions = request.app.state.session_store.list_sessions(kb_id)
    return SessionListResponse(doc_id=kb_id, sessions=sessions)


@router.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
async def session_history(
    session_id: str, request: Request, doc_id: str = Query(default="")
):
    # 前端刷新后凭 URL 里的 session_id 拉回多轮历史；会话态仍在内存（服务存活期内）。
    kb_id = doc_id or get_settings().cogdoc_default_doc_id
    messages = request.app.state.session_store.get_display(kb_id, session_id)
    return SessionHistoryResponse(
        session_id=session_id, doc_id=kb_id, messages=messages
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str, request: Request, doc_id: str = Query(default="")
):
    # 删除一个对话的多轮历史（幂等，不存在也返回 204）。
    kb_id = doc_id or get_settings().cogdoc_default_doc_id
    request.app.state.session_store.clear(kb_id, session_id)
    return Response(status_code=204)


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
    # 把服务层 ChatEvent 转成 SSE 帧；final 发结构化响应、error 转稳定错误码、其余为进度。
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
        error_code = classify_error_code(
            event.payload.get("stage", ""),
            event.payload.get("error_class", ""),
            event.payload.get("message", ""),
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
    # SSE 流式问答：worker 线程跑事件流 → 队列桥到事件循环 → 逐帧输出。
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
        # 只有真正产出 final 才写会话；记忆走门控、展示存完整问答。
        if final_result is not None:
            request.app.state.metrics.chat_results.labels(
                final_result.task_type, str(final_result.is_valid).lower()
            ).inc()
            session_store.record(
                doc_id,
                session_id,
                final_result.chat_messages,
                [
                    {"role": "user", "content": request_body.query},
                    {"role": "assistant", "content": final_result.answer},
                ],
            )

    return StreamingResponse(event_source(), media_type="text/event-stream")
