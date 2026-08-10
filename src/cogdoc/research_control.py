from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Event
from typing import Any


RESEARCH_RESOURCE_NAMES = (
    "retrieval_queries",
    "candidate_docs",
    "llm_calls",
    "model_input_chars",
)


class ResearchControlSignal(BaseException):
    """Cooperative stop signal that ordinary fail-closed catches must not swallow."""

    reason_code = "research_control_stopped"

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.reason_code)


class ResearchPaused(ResearchControlSignal):
    reason_code = "research_paused"


class ResearchCancelled(ResearchControlSignal):
    reason_code = "research_cancelled"


class ResearchDeadlineExceeded(ResearchControlSignal):
    reason_code = "research_deadline_exceeded"

    def __init__(self, message: str = "", *, durable: bool = False) -> None:
        self.durable = bool(durable)
        super().__init__(message)


class ResearchBudgetExceeded(ResearchControlSignal):
    reason_code = "research_budget_exceeded"


class ResearchProviderError(Exception):
    """Bounded external-provider execution failed before the run deadline."""


class ResearchProviderTimeout(ResearchProviderError):
    """One provider call exceeded its configured execution envelope."""


class ResearchProviderCapacityExceeded(ResearchProviderError):
    """The bounded provider-isolation pool has no remaining admission slot."""


ResearchReservation = Callable[[Mapping[str, int]], Mapping[str, Any] | None]
ResearchProviderOperation = Callable[[], Any]
ResearchProviderAdmission = Callable[[], None]
ResearchDeadlineCommit = Callable[[], str]
ResearchProviderRunner = Callable[
    [
        "ResearchRunController",
        str,
        ResearchProviderOperation,
        float,
        ResearchProviderAdmission,
    ],
    Any,
]


def normalize_resource_costs(costs: Mapping[str, int] | None) -> dict[str, int]:
    normalized = {name: 0 for name in RESEARCH_RESOURCE_NAMES}
    for name, value in dict(costs or {}).items():
        if name not in normalized:
            raise ValueError(f"unknown research resource: {name}")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"research resource cost {name} must be non-negative")
        normalized[name] = value
    return {name: value for name, value in normalized.items() if value}


@dataclass(slots=True)
class ResearchRunController:
    job_id: str
    phase: str
    attempt_id: str
    lease_id: str
    reserve_callback: ResearchReservation
    stop_event: Event
    deadline_at: str = ""
    provider_runner: ResearchProviderRunner | None = None
    deadline_callback: ResearchDeadlineCommit | None = None
    stop_reason: str = ""
    _last_durable_poll: float = field(default=0.0, init=False, repr=False)
    _monotonic_deadline: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.deadline_at:
            return
        try:
            deadline = datetime.fromisoformat(self.deadline_at.replace("Z", "+00:00"))
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            remaining = max(
                (deadline - datetime.now(timezone.utc)).total_seconds(),
                0.0,
            )
        except (TypeError, ValueError, OverflowError):
            remaining = 0.0
        # Wall time remains the durable cross-process authority.  This local
        # monotonic ceiling prevents an NTP/manual clock rollback from extending
        # an already-running attempt or planning request.
        self._monotonic_deadline = time.monotonic() + remaining

    def request_stop(self, reason: str) -> None:
        self.stop_reason = str(reason)
        self.stop_event.set()

    def _raise_if_stopped(self) -> None:
        if self.stop_event.is_set():
            if self.stop_reason == "paused":
                raise ResearchPaused()
            if self.stop_reason == "deadline":
                raise ResearchDeadlineExceeded()
            raise ResearchCancelled()

    def checkpoint(self, costs: Mapping[str, int] | None = None) -> None:
        # Always consult durable state.  The Event is a fast wake-up hint, while
        # the store remains authoritative across processes and restarts.
        try:
            self._raise_if_stopped()
        except ResearchDeadlineExceeded as exc:
            if not exc.durable:
                self._persist_local_deadline(exc)
            raise
        remaining = self.remaining_seconds()
        if remaining is not None and remaining <= 0:
            self._persist_local_deadline()
        self.reserve_callback(normalize_resource_costs(costs))
        self._last_durable_poll = time.monotonic()
        # Store lock/fsync latency is part of the phase envelope. A stop or
        # monotonic deadline that arrives during reservation must win before
        # the caller opens a retrieval/provider or commits output.
        try:
            self._raise_if_stopped()
        except ResearchDeadlineExceeded as exc:
            if not exc.durable:
                self._persist_local_deadline(exc)
            raise
        remaining = self.remaining_seconds()
        if remaining is not None and remaining <= 0:
            self._persist_local_deadline()

    def poll(self, *, durable_interval_seconds: float = 5.0) -> None:
        """Check local stop/deadline eagerly and durable state at a bounded rate."""

        try:
            self.poll_local()
        except ResearchDeadlineExceeded as exc:
            self._persist_local_deadline(exc)
        now = time.monotonic()
        if now - self._last_durable_poll >= max(float(durable_interval_seconds), 0.1):
            self.checkpoint()

    def poll_local(self) -> None:
        """Check only process-local control without performing store I/O.

        Provider supervisors use this path so a contended filesystem/SQLite
        heartbeat can never postpone terminating the isolated child.  The
        caller performs an authoritative durable checkpoint after the child is
        reaped and before observing a result or error.
        """

        self._raise_if_stopped()
        remaining = self.remaining_seconds()
        if remaining is not None and remaining <= 0:
            raise ResearchDeadlineExceeded()

    def _persist_local_deadline(
        self,
        source: ResearchDeadlineExceeded | None = None,
    ) -> None:
        if self.deadline_callback is not None:
            outcome = str(self.deadline_callback() or "")
            if outcome == "deadline":
                if source is not None:
                    source.durable = True
                    raise source
                raise ResearchDeadlineExceeded(durable=True)
            if outcome == "paused":
                raise ResearchPaused() from source
            raise ResearchCancelled(
                "research deadline lost to a newer control state"
            ) from source
        # Transient planning controls have no durable row; their callback is a
        # no-op but retaining it keeps the same pre-unwind checkpoint contract.
        # Call it directly: routing back through ``checkpoint`` would recurse
        # while the same local monotonic deadline remains elapsed.
        self.reserve_callback({})
        self._last_durable_poll = time.monotonic()
        if source is not None:
            source.durable = True
            raise source
        raise ResearchDeadlineExceeded(durable=True)

    def reserve(self, **costs: int) -> None:
        self.checkpoint(costs)

    def remaining_seconds(self, *, now: datetime | None = None) -> float | None:
        """Return the immutable durable deadline remainder for call contraction."""

        if not self.deadline_at:
            return None
        try:
            deadline = datetime.fromisoformat(self.deadline_at.replace("Z", "+00:00"))
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            current = now or datetime.now(timezone.utc)
            if current.tzinfo is None:
                current = current.replace(tzinfo=timezone.utc)
            wall_remaining = max((deadline - current).total_seconds(), 0.0)
            if now is not None or self._monotonic_deadline is None:
                return wall_remaining
            monotonic_remaining = max(
                self._monotonic_deadline - time.monotonic(),
                0.0,
            )
            return min(wall_remaining, monotonic_remaining)
        except (TypeError, ValueError, OverflowError):
            # Callers route zero through the same deadline termination path;
            # avoid recursively invoking checkpoint while parsing this value.
            return 0.0

    def run_provider(
        self,
        operation: ResearchProviderOperation,
        *,
        provider: str,
        timeout_seconds: float,
        on_admitted: ResearchProviderAdmission | None = None,
    ) -> Any:
        """Execute one provider call inside the manager-owned isolation pool."""

        if not callable(operation):
            raise TypeError("research provider operation must be callable")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0
        ):
            raise ValueError("research provider timeout must be positive and finite")
        # Admission performs the authoritative durable reservation immediately
        # before provider execution.  A local preflight avoids an extra empty
        # heartbeat/resource callback (and its JSON/SQLite write) here.
        try:
            self.poll_local()
        except ResearchDeadlineExceeded as exc:
            self._persist_local_deadline(exc)
        remaining = self.remaining_seconds()
        if remaining is not None:
            if remaining <= 0:
                self._persist_local_deadline()
            timeout_seconds = min(float(timeout_seconds), remaining)
        if timeout_seconds <= 0:
            self._persist_local_deadline()
        direct_expires = time.monotonic() + float(timeout_seconds)
        try:
            if self.provider_runner is None:
                if on_admitted is not None:
                    on_admitted()
                # Reservation can block on durable storage. Re-check the local
                # stop/deadline boundary before opening a transport afterwards.
                self.poll_local()
                if time.monotonic() >= direct_expires:
                    self.poll()
                    raise ResearchProviderTimeout(
                        f"research {provider or 'unknown'} provider call timed out"
                    )
                result = operation()
            else:
                result = self.provider_runner(
                    self,
                    str(provider or "unknown"),
                    operation,
                    float(timeout_seconds),
                    on_admitted or (lambda: None),
                )
        except ResearchDeadlineExceeded as exc:
            if not exc.durable:
                self._persist_local_deadline(exc)
            raise
        except ResearchControlSignal:
            # The durable reservation/checkpoint has already selected the
            # authoritative control outcome; never let a second empty
            # checkpoint translate budget/deadline/pause into another signal.
            raise
        except BaseException:
            # Durable pause/cancel/deadline wins a same-instant provider error.
            # For process-isolated calls the child has already been reaped by
            # the runner before this authoritative checkpoint can block.
            self.checkpoint()
            raise
        # A response that raced with pause/cancel/deadline is never observable
        # by the report pipeline, even if the transport completed successfully.
        self.checkpoint()
        return result


_CURRENT_RESEARCH_CONTROL: ContextVar[ResearchRunController | None] = ContextVar(
    "cogdoc_current_research_control",
    default=None,
)


@contextmanager
def bind_research_control(control: ResearchRunController):
    token = _CURRENT_RESEARCH_CONTROL.set(control)
    try:
        yield control
    finally:
        _CURRENT_RESEARCH_CONTROL.reset(token)


def current_research_control() -> ResearchRunController | None:
    return _CURRENT_RESEARCH_CONTROL.get()


def research_checkpoint(costs: Mapping[str, int] | None = None) -> None:
    control = current_research_control()
    if control is not None:
        control.checkpoint(costs)


def _message_content(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, Mapping):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def model_input_char_count(messages: Iterable[Any]) -> int:
    return sum(len(_message_content(message)) for message in messages)


def reserve_model_call(messages: Iterable[Any]) -> None:
    rows = list(messages)
    research_checkpoint(
        {
            "llm_calls": 1,
            "model_input_chars": model_input_char_count(rows),
        }
    )


__all__ = [
    "RESEARCH_RESOURCE_NAMES",
    "ResearchBudgetExceeded",
    "ResearchCancelled",
    "ResearchControlSignal",
    "ResearchDeadlineExceeded",
    "ResearchPaused",
    "ResearchProviderCapacityExceeded",
    "ResearchProviderError",
    "ResearchProviderAdmission",
    "ResearchDeadlineCommit",
    "ResearchProviderOperation",
    "ResearchProviderRunner",
    "ResearchProviderTimeout",
    "ResearchRunController",
    "bind_research_control",
    "current_research_control",
    "model_input_char_count",
    "normalize_resource_costs",
    "research_checkpoint",
    "reserve_model_call",
]
