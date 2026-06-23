import logging
from dataclasses import dataclass, field
from typing import Any, Iterator
from config.settings import get_settings
from agents.conversation_memory import extract_chat_turn, extract_final_answer
from agents.router import FORCED_TASK_TYPES
from graph.workflow import app
from observability.logger import configure_logging, log_event, new_trace_id
from observability.trace import build_trace_step, export_trace, monotonic_ms


class ChatServiceError(Exception):
    # 服务层灾难失败时携带稳定的错误归因，交付层据此映射 error_code，不漏栈。
    def __init__(
        self,
        stage: str,
        error_class: str,
        message: str,
        trace_id: str | None = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.error_class = error_class
        self.message = message
        self.trace_id = trace_id


@dataclass(frozen=True)
class ChatEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatResult:
    answer: str
    task_type: str
    citations: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    critique: str
    is_valid: bool
    trace_id: str
    request_id: str
    steps: list[dict[str, Any]]
    chat_messages: list[dict[str, Any]]
    raw_output: dict[str, Any]
    trace_path: str | None = None


def _extract_token(data: Any) -> str:
    message = data[0] if isinstance(data, tuple) and data else data
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else ""


def _build_result(
    task_type: str,
    task_output: dict[str, Any],
    query: str,
    trace_id: str,
    trace_steps: list[dict[str, Any]],
    trace_path: str | None,
) -> ChatResult:
    answer = extract_final_answer(task_type, task_output)
    critique = str(task_output.get("critique", "") or "")
    return ChatResult(
        answer=answer,
        task_type=task_type,
        citations=list(task_output.get("sources", []) or []),
        evidence=list(task_output.get("evidence", []) or []),
        critique=critique,
        is_valid=not bool(critique),
        trace_id=trace_id,
        request_id=trace_id,
        steps=trace_steps,
        chat_messages=extract_chat_turn(task_type, task_output, query),
        raw_output=task_output,
        trace_path=trace_path,
    )


def _runtime_error_step(node_name: str, exc: Exception) -> dict[str, Any]:
    return {
        "node_name": node_name,
        "duration_ms": 0.0,
        "model": None,
        "token": None,
        "retrieval_top_k": None,
        "critique": None,
        "error_class": type(exc).__name__,
        "counts": {},
        "evidence": [],
    }


def run_chat(
    doc_id: str,
    query: str,
    is_local: bool = False,
    chat_history: list | None = None,
    forced_task: str | None = None,
) -> Iterator[ChatEvent]:
    configure_logging()
    settings = get_settings()
    trace_id = new_trace_id()
    trace_steps: list[dict[str, Any]] = []
    request_start_ms = monotonic_ms()
    last_trace_ms = None
    initial_state = {
        "messages": [],
        "chat_history": list(chat_history or []),
        "iteration_count": 0,
        "max_iteration_count": 2,
        "request_id": trace_id,
        "trace_id": trace_id,
    }

    configurable = {
        "doc_id": doc_id,
        "query": query,
        "is_local": is_local,
        "request_id": trace_id,
        "trace_id": trace_id,
    }
    if forced_task in FORCED_TASK_TYPES:
        configurable["forced_task"] = forced_task
    runtime_config = {"configurable": configurable}

    current_task = "qa"
    saw_parent_output = False
    fallback_outputs = {"qa": {}, "summary": {}, "compare": {}, "unknown": {}}
    final_outputs: dict[str, dict[str, Any]] = {}

    log_event(
        "runtime",
        "request_start",
        initial_state,
        doc_id=doc_id,
        is_local=is_local,
        forced_task=forced_task,
        query_length=len(query),
    )
    yield ChatEvent(
        "request_started",
        {
            "trace_id": trace_id,
            "request_id": trace_id,
            "doc_id": doc_id,
            "is_local": is_local,
            "forced_task": forced_task,
        },
    )

    try:
        token_stream = app.stream(
            initial_state,
            config=runtime_config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        )

        try:
            for ns, mode, data in token_stream:
                in_subgraph = len(ns) > 0
                if mode == "messages":
                    token = _extract_token(data)
                    if token:
                        yield ChatEvent("token", {"content": token})
                    continue

                if mode == "updates":
                    now_ms = monotonic_ms()
                    if last_trace_ms is None:
                        trace_steps.append(
                            {
                                "node_name": "runtime.setup",
                                "duration_ms": round(
                                    max(now_ms - request_start_ms, 0.0), 3
                                ),
                                "model": None,
                                "token": None,
                                "retrieval_top_k": None,
                                "critique": None,
                                "error_class": None,
                                "counts": {},
                                "evidence": [],
                            }
                        )
                        duration_ms = 0.0
                    else:
                        duration_ms = now_ms - last_trace_ms
                    last_trace_ms = now_ms

                    namespace = ".".join(str(item) for item in ns)
                    model_name = (
                        settings.ollama_model_name
                        if is_local
                        else settings.llm_model_name
                    )
                    for node_name, node_output in data.items():
                        if not isinstance(node_output, dict):
                            continue
                        full_node_name = (
                            f"{namespace}.{node_name}" if namespace else node_name
                        )
                        retrieval_top_k = (
                            settings.qa_retrieval_top_k
                            if node_name == "retrieve_node"
                            else None
                        )
                        trace_steps.append(
                            build_trace_step(
                                full_node_name,
                                node_output,
                                duration_ms,
                                model_name=model_name,
                                retrieval_top_k=retrieval_top_k,
                            )
                        )

                if mode == "updates" and not in_subgraph and "intent_router" in data:
                    router_output = data["intent_router"]
                    current_task = router_output.get("task_type", "qa")
                    yield ChatEvent(
                        "router_decided",
                        {
                            "task_type": current_task,
                            "reason": router_output.get("router_reason", "无"),
                        },
                    )
                elif mode == "updates" and in_subgraph and "rewrite_node" in data:
                    rewrite_output = data["rewrite_node"]
                    yield ChatEvent(
                        "rewrite_queries",
                        {"queries": list(rewrite_output.get("rewritten_queries", []))},
                    )
                elif mode == "updates" and in_subgraph and "citation_node" in data:
                    citation_output = data["citation_node"]
                    fallback_outputs["qa"].update(citation_output)
                    critique = citation_output.get("critique", "")
                    iter_num = citation_output.get("iteration_count", 1)
                    max_iter = citation_output.get(
                        "max_iteration_count",
                        initial_state.get("max_iteration_count", 2),
                    )
                    if critique:
                        round_answer = fallback_outputs["qa"].get("answer", "")
                        yield ChatEvent(
                            "citation_rejected",
                            {
                                "critique": critique,
                                "iteration_count": iter_num,
                                "max_iteration_count": max_iter,
                                "round_answer": round_answer,
                                "will_retry": iter_num < max_iter,
                            },
                        )
                    else:
                        yield ChatEvent(
                            "citation_passed",
                            {
                                "iteration_count": iter_num,
                                "max_iteration_count": max_iter,
                            },
                        )
                elif (
                    mode == "updates"
                    and in_subgraph
                    and "compare_citation_node" in data
                ):
                    compare_citation_output = data["compare_citation_node"]
                    fallback_outputs["compare"].update(compare_citation_output)
                    critique = compare_citation_output.get("critique", "")
                    if critique:
                        yield ChatEvent(
                            "compare_citation_rejected",
                            {"critique": critique},
                        )
                    else:
                        yield ChatEvent("compare_citation_passed", {})
                elif mode == "updates" and in_subgraph:
                    task_output = fallback_outputs.setdefault(current_task, {})
                    for value in data.values():
                        if isinstance(value, dict):
                            task_output.update(value)
                elif mode == "updates" and not in_subgraph:
                    for parent_key, task_name in {
                        "qa_subgraph": "qa",
                        "summary_subgraph": "summary",
                        "compare_subgraph": "compare",
                        "unknown_node": "unknown",
                    }.items():
                        if parent_key in data:
                            saw_parent_output = True
                            final_outputs[task_name] = data[parent_key]
                            break

            if not saw_parent_output:
                task_output = fallback_outputs.get(current_task, {})
                if task_output:
                    final_outputs[current_task] = task_output

        except Exception as stream_err:
            trace_steps.append(_runtime_error_step("runtime.stream", stream_err))
            log_event(
                "runtime",
                "request_stream_error",
                initial_state,
                level=logging.ERROR,
                error_class=type(stream_err).__name__,
            )
            yield ChatEvent(
                "error",
                {
                    "error_class": type(stream_err).__name__,
                    "message": str(stream_err),
                    "stage": "stream",
                    "trace_id": trace_id,
                },
            )

        task_output = final_outputs.get(current_task, {})
        exported = export_trace(
            trace_id=trace_id,
            request_id=trace_id,
            task_type=current_task,
            steps=trace_steps,
            settings=settings,
        )
        trace_path = str(exported) if exported else None
        result = _build_result(
            current_task,
            task_output,
            query,
            trace_id,
            trace_steps,
            trace_path,
        )
        log_event(
            "runtime",
            "request_end",
            initial_state,
            task_type=current_task,
            has_output=bool(task_output),
            trace_path=trace_path,
        )
        yield ChatEvent("final", {"result": result, "output": task_output})

    except Exception as exc:
        trace_steps.append(_runtime_error_step("runtime.failed", exc))
        export_trace(
            trace_id=trace_id,
            request_id=trace_id,
            task_type="unknown",
            steps=trace_steps,
            settings=settings,
        )
        log_event(
            "runtime",
            "request_failed",
            initial_state,
            level=logging.ERROR,
            error_class=type(exc).__name__,
        )
        yield ChatEvent(
            "error",
            {
                "error_class": type(exc).__name__,
                "message": str(exc),
                "stage": "runtime",
                "trace_id": trace_id,
            },
        )


def run_chat_sync(*args: Any, **kwargs: Any) -> ChatResult:
    result = None
    last_error: dict[str, Any] | None = None
    for event in run_chat(*args, **kwargs):
        if event.type == "final":
            result = event.payload["result"]
        elif event.type == "error":
            last_error = event.payload
    # 出现 error 事件且最终无可信输出（raw_output 为空）即视为失败，不把空答案当成功返回。
    has_trustworthy_output = result is not None and bool(result.raw_output)
    if last_error is not None and not has_trustworthy_output:
        raise ChatServiceError(
            stage=last_error.get("stage", "runtime"),
            error_class=last_error.get("error_class", "RuntimeError"),
            message=last_error.get("message", "")
            or "chat service did not produce a usable result",
            trace_id=last_error.get("trace_id"),
        )
    if result is not None:
        return result
    raise ChatServiceError(
        stage="runtime",
        error_class="RuntimeError",
        message="chat service did not produce a final result",
    )
