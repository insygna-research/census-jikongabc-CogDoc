from __future__ import annotations

import inspect
import math
import pickle
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from cogdoc.config.settings import get_settings
from cogdoc.research_control import (
    ResearchDeadlineExceeded,
    ResearchProviderError,
    ResearchProviderTimeout,
    current_research_control,
    model_input_char_count,
)


class ResearchProviderRemoteError(RuntimeError):
    """A hard-isolated provider process returned an ordinary call failure."""

    def __init__(self, error_class: str, message: str = "") -> None:
        self.error_class = str(error_class or "ProviderError")[:128]
        bounded = " ".join(str(message or "").split())[:500]
        super().__init__(f"{self.error_class}: {bounded}" if bounded else self.error_class)


@dataclass(frozen=True, slots=True)
class SerializedResearchProviderCall:
    """Pre-serialized operation so spawn never re-reduces prompt/schema objects."""

    payload: bytes = field(repr=False)
    research_process_isolated: bool = field(default=True, init=False, repr=False)

    def __call__(self) -> Any:
        operation = pickle.loads(self.payload)
        if not callable(operation):
            raise TypeError("serialized research provider operation is not callable")
        return operation()


def mark_research_process_isolation_compatible(client: Any) -> Any:
    """Mark a factory-built ChatOpenAI client whose recipe fields are known."""

    try:
        object.__setattr__(client, "_cogdoc_research_recipe_compatible", True)
    except Exception:
        # The later admission check remains fail-closed for immutable/custom
        # client implementations.
        pass
    return client


@dataclass(frozen=True, slots=True)
class IsolatedChatOpenAICall:
    """Pickle-safe recipe reconstructed inside a fresh spawned process."""

    model: str
    api_key: str = field(repr=False)
    base_url: str
    timeout_seconds: float
    messages: tuple[Any, ...] = field(repr=False)
    temperature: float | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)
    default_headers: dict[str, str] = field(default_factory=dict, repr=False)
    default_query: dict[str, str] = field(default_factory=dict, repr=False)
    organization: str = ""
    proxy: str = field(default="", repr=False)
    max_tokens: int | None = None
    reasoning_effort: str = ""
    schema: type[Any] | None = None
    structured_method: str = ""
    research_process_isolated: bool = field(default=True, init=False, repr=False)

    def __call__(self) -> Any:
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout_seconds,
            "max_retries": 0,
        }
        for name, value in (
            ("temperature", self.temperature),
            ("model_kwargs", self.model_kwargs or None),
            ("default_headers", self.default_headers or None),
            ("default_query", self.default_query or None),
            ("organization", self.organization or None),
            ("openai_proxy", self.proxy or None),
            ("max_tokens", self.max_tokens),
            ("reasoning_effort", self.reasoning_effort or None),
        ):
            if value is not None:
                kwargs[name] = value
        runnable: Any = ChatOpenAI(**kwargs)
        if self.schema is not None and self.structured_method:
            runnable = runnable.with_structured_output(
                self.schema,
                method=self.structured_method,
            )
        return runnable.invoke(list(self.messages), timeout=self.timeout_seconds)


def provider_process_entry(connection: Any, operation: Any) -> None:
    """Run one pickle-safe provider recipe and return a bounded IPC envelope."""

    try:
        result = operation()
        connection.send_bytes(pickle.dumps(("ok", result), protocol=5))
    except BaseException as exc:
        try:
            response = getattr(exc, "response", None)
            status_code = getattr(exc, "status_code", None) or getattr(
                response, "status_code", None
            )
            error_code = getattr(exc, "code", None)
            connection.send_bytes(
                pickle.dumps(
                    (
                        "error",
                        type(exc).__name__[:128],
                        " ".join(str(exc).split())[:500],
                        (
                            int(status_code)
                            if isinstance(status_code, int)
                            and not isinstance(status_code, bool)
                            else None
                        ),
                        str(error_code or "")[:128],
                    ),
                    protocol=5,
                )
            )
        except BaseException:
            pass
    finally:
        try:
            connection.close()
        except BaseException:
            pass


def is_process_isolated_provider_call(operation: Any) -> bool:
    return bool(getattr(operation, "research_process_isolated", False))


def _persist_control_deadline(control: Any, source: ResearchDeadlineExceeded) -> None:
    persist = getattr(control, "_persist_local_deadline", None)
    if callable(persist):
        persist(source)
    raise source


def run_standalone_research_provider(
    control: Any,
    provider: str,
    operation: Any,
    timeout_seconds: float,
    on_admitted: Any,
) -> Any:
    """Hard-isolate known production clients outside the durable job manager."""

    expires = time.monotonic() + float(timeout_seconds)
    if not is_process_isolated_provider_call(operation):
        # Explicit test doubles and non-standard adapters preserve their direct
        # compatibility path; recognized ChatOpenAI clients fail closed earlier.
        on_admitted()
        try:
            control.poll_local()
        except ResearchDeadlineExceeded as exc:
            _persist_control_deadline(control, exc)
        if time.monotonic() >= expires:
            control.poll()
            raise ResearchProviderTimeout(
                f"research {provider or 'unknown'} provider call timed out"
            )
        return operation()
    from cogdoc.research_isolation import run_spawn_isolated_provider

    settings = get_settings()
    on_admitted()
    try:
        control.poll_local()
    except ResearchDeadlineExceeded as exc:
        _persist_control_deadline(control, exc)
    timeout_seconds = min(float(timeout_seconds), expires - time.monotonic())
    remaining = control.remaining_seconds()
    if remaining is not None:
        timeout_seconds = min(float(timeout_seconds), remaining)
    if timeout_seconds <= 0:
        control._persist_local_deadline()
    return run_spawn_isolated_provider(
        operation,
        provider=provider,
        timeout_seconds=timeout_seconds,
        kill_grace_seconds=settings.cogdoc_research_provider_kill_grace_seconds,
        ipc_max_bytes=settings.cogdoc_research_provider_ipc_max_bytes,
        poll=control.poll_local,
    )


def _positive_timeout(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, Real):
        candidate = float(value)
        return candidate if math.isfinite(candidate) and candidate > 0 else None
    if isinstance(value, tuple):
        candidates = [
            candidate
            for item in value
            if (candidate := _positive_timeout(item)) is not None
        ]
        return max(candidates) if candidates else None
    # httpx.Timeout and compatible objects expose per-phase numeric values.
    timeout_fields = ("connect", "read", "write", "pool")
    if not any(hasattr(value, field) for field in timeout_fields):
        return None
    candidates = [
        candidate
        for field in timeout_fields
        if (candidate := _positive_timeout(getattr(value, field, None))) is not None
    ]
    return max(candidates) if candidates else None


def research_model_timeout_seconds(runnable: Any) -> float:
    """Return the strictest configured per-call timeout for a Research model."""

    configured = _positive_timeout(getattr(runnable, "request_timeout", None))
    envelope = float(get_settings().cogdoc_research_provider_call_timeout_seconds)
    return min(configured, envelope) if configured is not None else envelope


def _accepts_timeout(runnable: Any) -> bool:
    invoke = getattr(runnable, "invoke", None)
    if not callable(invoke):
        return False
    try:
        parameters = inspect.signature(invoke).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "timeout"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _secret_text(value: Any) -> str:
    reveal = getattr(value, "get_secret_value", None)
    if callable(reveal):
        try:
            return str(reveal())
        except Exception:
            return ""
    if callable(value):
        return ""
    return str(value or "")


def _is_chat_openai(value: Any) -> bool:
    module_name = str(getattr(value.__class__, "__module__", ""))
    class_name = str(getattr(value.__class__, "__name__", ""))
    return module_name.startswith("langchain_openai") and class_name == "ChatOpenAI"


def _isolated_chat_openai_call(
    source: Any,
    messages: list[Any],
    *,
    timeout_seconds: float,
    schema: type[Any] | None,
    structured_method: str,
) -> SerializedResearchProviderCall | None:
    if not _is_chat_openai(source):
        return None
    if getattr(source, "_cogdoc_research_recipe_compatible", False) is not True:
        return None
    model = str(
        getattr(source, "model_name", "") or getattr(source, "model", "") or ""
    )
    base_url = str(
        getattr(source, "openai_api_base", "")
        or getattr(source, "base_url", "")
        or ""
    )
    api_key = _secret_text(getattr(source, "openai_api_key", ""))
    if not model or not base_url or not api_key:
        return None
    temperature = getattr(source, "temperature", None)
    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        temperature = None
    max_tokens = getattr(source, "max_tokens", None)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        max_tokens = None
    operation = IsolatedChatOpenAICall(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=float(timeout_seconds),
        messages=tuple(messages),
        temperature=float(temperature) if temperature is not None else None,
        model_kwargs=dict(getattr(source, "model_kwargs", None) or {}),
        default_headers={
            str(key): str(value)
            for key, value in dict(getattr(source, "default_headers", None) or {}).items()
        },
        default_query={
            str(key): str(value)
            for key, value in dict(getattr(source, "default_query", None) or {}).items()
        },
        organization=str(getattr(source, "openai_organization", "") or ""),
        proxy=str(getattr(source, "openai_proxy", "") or ""),
        max_tokens=max_tokens,
        reasoning_effort=str(getattr(source, "reasoning_effort", "") or ""),
        schema=schema,
        structured_method=str(structured_method or ""),
    )
    try:
        payload = pickle.dumps(operation, protocol=5)
    except Exception:
        return None
    if len(payload) > get_settings().cogdoc_research_provider_ipc_max_bytes:
        return None
    return SerializedResearchProviderCall(payload=payload)


def invoke_research_model(
    runnable: Any,
    messages: Iterable[Any],
    *,
    reserve: bool = True,
    timeout_source: Any | None = None,
    schema: type[Any] | None = None,
    structured_method: str = "",
) -> Any:
    """Invoke a model under the active Research deadline and isolation policy.

    Outside a Research execution context this deliberately preserves the normal
    LangChain/test-double invocation surface. Inside one, standard ChatOpenAI
    calls run in a bounded spawned child; opaque test/integration adapters retain
    the compatibility executor.
    """

    rows = list(messages)
    control = current_research_control()
    if control is None:
        return runnable.invoke(rows)

    timeout_seconds = research_model_timeout_seconds(
        runnable if timeout_source is None else timeout_source
    )
    remaining = control.remaining_seconds()
    if remaining is not None and remaining > 0:
        timeout_seconds = min(timeout_seconds, remaining)
    supports_timeout = _accepts_timeout(runnable)

    source = runnable if timeout_source is None else timeout_source
    isolated = _isolated_chat_openai_call(
        source,
        rows,
        timeout_seconds=timeout_seconds,
        schema=schema,
        structured_method=structured_method,
    )
    if (
        isolated is None
        and _is_chat_openai(source)
        and get_settings().cogdoc_research_llm_process_isolation_enabled
    ):
        raise ResearchProviderError(
            "ChatOpenAI research call could not enter process isolation"
        )
    if isolated is not None:
        operation = isolated
    else:

        def operation() -> Any:
            if supports_timeout:
                return runnable.invoke(rows, timeout=timeout_seconds)
            return runnable.invoke(rows)

    return control.run_provider(
        operation,
        provider="llm",
        timeout_seconds=timeout_seconds,
        on_admitted=(
            lambda: control.checkpoint(
                {
                    "llm_calls": 1,
                    "model_input_chars": model_input_char_count(rows),
                }
            )
        )
        if reserve
        else None,
    )


__all__ = [
    "IsolatedChatOpenAICall",
    "SerializedResearchProviderCall",
    "ResearchProviderRemoteError",
    "invoke_research_model",
    "is_process_isolated_provider_call",
    "mark_research_process_isolation_compatible",
    "provider_process_entry",
    "research_model_timeout_seconds",
    "run_standalone_research_provider",
]
