from __future__ import annotations

import multiprocessing
import pickle
import time
from collections.abc import Callable
from typing import Any

from cogdoc.research_control import ResearchProviderError, ResearchProviderTimeout
from cogdoc.research_provider import (
    ResearchProviderRemoteError,
    SerializedResearchProviderCall,
    provider_process_entry,
)


def _terminate_process(
    process: multiprocessing.Process,
    *,
    kill_grace_seconds: float,
) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=kill_grace_seconds)
    if process.is_alive():
        process.kill()
        process.join(timeout=kill_grace_seconds)
    if process.is_alive():
        raise ResearchProviderError("research provider process could not be reaped")


def _decode_envelope(raw: bytes) -> Any:
    try:
        envelope = pickle.loads(raw)
    except (
        pickle.PickleError,
        AttributeError,
        ImportError,
        TypeError,
        ValueError,
    ) as exc:
        raise ResearchProviderError(
            "research provider returned an invalid IPC envelope"
        ) from exc
    if not isinstance(envelope, tuple) or not envelope:
        raise ResearchProviderError(
            "research provider returned an invalid IPC envelope"
        )
    if envelope[0] == "ok" and len(envelope) == 2:
        return envelope[1]
    if envelope[0] != "error" or len(envelope) != 5:
        raise ResearchProviderError(
            "research provider returned an invalid IPC envelope"
        )

    error_class = str(envelope[1] or "ProviderError")[:128]
    message = str(envelope[2] or "")[:500]
    status_code = envelope[3]
    error_code = str(envelope[4] or "")[:128]
    normalized = error_class.casefold()
    if "timeout" in normalized:
        raise ResearchProviderTimeout(message or error_class)
    if any(
        marker in normalized
        for marker in (
            "authentication",
            "permission",
            "ratelimit",
            "connection",
        )
    ):
        raise ResearchProviderError(
            f"{error_class}: {message}" if message else error_class
        )
    capability_detail = f"{error_code} {message}".casefold()
    if status_code in {400, 422} and any(
        marker in capability_detail
        for marker in (
            "response_format",
            "json_object",
            "json_schema",
            "tool_choice",
            "function_call",
            "unsupported parameter",
        )
    ):
        raise ResearchProviderRemoteError(error_class, message)
    raise ResearchProviderError(
        f"{error_class}: {message}" if message else error_class
    )


def _receive_envelope(receiver: Any, *, ipc_max_bytes: int) -> Any:
    try:
        raw = receiver.recv_bytes(ipc_max_bytes)
    except (EOFError, OSError) as exc:
        raise ResearchProviderError(
            "research provider returned an invalid IPC envelope"
        ) from exc
    return _decode_envelope(raw)


def run_spawn_isolated_provider(
    operation: Callable[[], Any],
    *,
    provider: str,
    timeout_seconds: float,
    kill_grace_seconds: float,
    ipc_max_bytes: int,
    poll: Callable[[], None] | None = None,
) -> Any:
    """Spawn, supervise, and unconditionally reap one provider process."""

    expires = time.monotonic() + float(timeout_seconds)
    if not isinstance(operation, SerializedResearchProviderCall):
        try:
            request_payload = pickle.dumps(operation, protocol=5)
        except Exception as exc:
            raise ResearchProviderError(
                "research provider operation is not process-serializable"
            ) from exc
        if len(request_payload) > ipc_max_bytes:
            raise ResearchProviderError(
                "research provider request exceeds the IPC envelope limit"
            )
        operation = SerializedResearchProviderCall(payload=request_payload)
    if time.monotonic() >= expires:
        raise ResearchProviderTimeout(
            f"research {provider or 'unknown'} provider call timed out"
        )
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=provider_process_entry,
        args=(sender, operation),
        name=f"cogdoc-research-{provider or 'provider'}",
        daemon=True,
    )
    started = False
    try:
        process.start()
        started = True
        sender.close()
        while True:
            remaining = expires - time.monotonic()
            if remaining <= 0:
                raise ResearchProviderTimeout(
                    f"research {provider or 'unknown'} provider call timed out"
                )
            if receiver.poll(min(0.1, remaining)):
                return _receive_envelope(receiver, ipc_max_bytes=ipc_max_bytes)
            if poll is not None:
                poll()
            if not process.is_alive():
                process.join(timeout=0)
                # The child may have sent its envelope and exited between the
                # timed poll and the liveness check.  Drain once more before
                # classifying the exit as a protocol failure.
                if receiver.poll(0):
                    return _receive_envelope(
                        receiver,
                        ipc_max_bytes=ipc_max_bytes,
                    )
                raise ResearchProviderError(
                    "research provider process exited without a result: "
                    f"exit_code={process.exitcode}"
                )
    finally:
        try:
            sender.close()
        except OSError:
            pass
        try:
            receiver.close()
        finally:
            if started:
                _terminate_process(
                    process,
                    kill_grace_seconds=kill_grace_seconds,
                )
            process.close()


__all__ = ["run_spawn_isolated_provider"]
