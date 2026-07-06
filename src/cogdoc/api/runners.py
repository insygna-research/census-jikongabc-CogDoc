import inspect
from collections.abc import Callable
from typing import Any


# 判断 runner 是否接收会话编号。
def runner_accepts_session_id(runner: Callable[..., object]) -> bool:
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return False
    return "session_id" in signature.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


# 调用兼容新旧签名的对话 runner。
def run_with_optional_session(
    runner: Callable[..., Any],
    doc_id: str,
    query: str,
    is_local: bool,
    chat_history: list,
    forced_task: str | None,
    session_id: str | None,
) -> Any:
    args = (doc_id, query, is_local, chat_history, forced_task)
    if runner_accepts_session_id(runner):
        return runner(*args, session_id=session_id)
    return runner(*args)
