from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
from threading import RLock, Thread
from typing import Any


_STOP = object()


class DaemonExecutorCapacityError(RuntimeError):
    """A bounded daemon executor has no free running/queued admission slot."""


@dataclass(frozen=True, slots=True)
class _WorkItem:
    future: Future
    function: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class DaemonFutureExecutor:
    """Small Future executor whose workers never block interpreter exit.

    The standard ThreadPoolExecutor registers every worker in CPython's global
    exit hook, which joins it even when ``shutdown(wait=False)`` was requested.
    Research calls can be opaque synchronous provider functions, so graceful
    application shutdown must not depend on such a call returning. This
    executor deliberately owns plain daemon threads and implements only the
    Future/submit/shutdown surface needed by ResearchExecutionManager.
    """

    def __init__(
        self,
        *,
        max_workers: int,
        thread_name_prefix: str,
        max_pending: int | None = None,
    ):
        if isinstance(max_workers, bool) or not isinstance(max_workers, int):
            raise TypeError("max_workers must be an integer")
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if max_pending is not None and (
            isinstance(max_pending, bool)
            or not isinstance(max_pending, int)
            or max_pending < 1
        ):
            raise ValueError("max_pending must be a positive integer")
        self._queue: Queue[_WorkItem | object] = Queue()
        self._lock = RLock()
        self._shutdown = False
        self._max_pending = max_pending
        self._pending = 0
        self._max_workers = max_workers
        self._thread_name_prefix = thread_name_prefix
        self._threads: tuple[Thread, ...] = ()

    def _ensure_started_locked(self) -> None:
        if self._threads:
            return
        self._threads = tuple(
            Thread(
                target=self._worker,
                name=f"{self._thread_name_prefix}-{position}",
                daemon=True,
            )
            for position in range(self._max_workers)
        )
        for thread in self._threads:
            thread.start()

    def submit(self, function: Callable[..., Any], /, *args, **kwargs) -> Future:
        if not callable(function):
            raise TypeError("submitted research work must be callable")
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            if self._max_pending is not None and self._pending >= self._max_pending:
                raise DaemonExecutorCapacityError("daemon executor queue is full")
            self._ensure_started_locked()
            future: Future = Future()
            self._pending += 1
            self._queue.put(_WorkItem(future, function, args, dict(kwargs)))
            return future

    def _release_pending(self) -> None:
        with self._lock:
            if self._pending:
                self._pending -= 1

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> bool:
        """Stop admission and return whether no submitted work remains."""

        with self._lock:
            first_shutdown = not self._shutdown
            self._shutdown = True
            if first_shutdown and cancel_futures:
                self._cancel_queued_locked()
            if first_shutdown:
                for _thread in self._threads:
                    self._queue.put(_STOP)
        if wait:
            for thread in self._threads:
                thread.join()
        return self.is_drained()

    def is_drained(self) -> bool:
        with self._lock:
            return self._pending == 0

    def pending_count(self) -> int:
        """Return physical running/queued work, including cancelled tombstones."""

        with self._lock:
            return self._pending

    def _cancel_queued_locked(self) -> None:
        retained_stops = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                break
            if item is _STOP:
                retained_stops += 1
            elif isinstance(item, _WorkItem):
                item.future.cancel()
                self._release_pending()
        for _position in range(retained_stops):
            self._queue.put(_STOP)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            if not isinstance(item, _WorkItem):
                continue
            future = item.future
            try:
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    result = item.function(*item.args, **item.kwargs)
                except BaseException as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(result)
            finally:
                self._release_pending()


__all__ = ["DaemonExecutorCapacityError", "DaemonFutureExecutor"]
