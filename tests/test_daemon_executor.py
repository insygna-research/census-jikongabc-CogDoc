import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from threading import Event

import pytest

from cogdoc.daemon_executor import (
    DaemonExecutorCapacityError,
    DaemonFutureExecutor,
)
from cogdoc.service.research_planning_runtime import ResearchPlanningRuntime


def test_daemon_future_executor_does_not_block_interpreter_exit():
    program = textwrap.dedent(
        """
        from threading import Event
        from cogdoc.daemon_executor import DaemonFutureExecutor

        blocker = Event()
        executor = DaemonFutureExecutor(max_workers=1, thread_name_prefix="probe")
        executor.submit(blocker.wait)
        executor.shutdown(wait=False)
        print("shutdown-returned", flush=True)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )

    assert completed.stdout.strip() == "shutdown-returned"


def test_daemon_future_executor_starts_workers_lazily():
    executor = DaemonFutureExecutor(
        max_workers=2,
        thread_name_prefix="lazy-probe",
    )
    assert executor._threads == ()
    executor.shutdown(wait=True, cancel_futures=True)
    assert executor._threads == ()


def test_daemon_future_executor_enforces_and_releases_pending_capacity():
    started = Event()
    release = Event()
    executor = DaemonFutureExecutor(
        max_workers=1,
        max_pending=1,
        thread_name_prefix="bounded-probe",
    )

    def block():
        started.set()
        release.wait()
        return "done"

    first = executor.submit(block)
    assert started.wait(timeout=2)
    with pytest.raises(DaemonExecutorCapacityError):
        executor.submit(lambda: None)

    release.set()
    assert first.result(timeout=2) == "done"
    assert executor.submit(lambda: "next").result(timeout=2) == "next"
    executor.shutdown(wait=True, cancel_futures=True)


def test_cancelled_queue_tombstones_continue_to_consume_bounded_capacity():
    started = Event()
    release = Event()
    executor = DaemonFutureExecutor(
        max_workers=1,
        max_pending=2,
        thread_name_prefix="cancel-probe",
    )

    def block():
        started.set()
        release.wait()

    active = executor.submit(block)
    assert started.wait(timeout=2)
    queued = executor.submit(lambda: None)
    assert queued.cancel() is True

    # Cancellation marks the Future, but the physical WorkItem remains queued.
    # It must retain its admission slot until the worker actually drains it.
    for _attempt in range(100):
        with pytest.raises(DaemonExecutorCapacityError):
            executor.submit(lambda: None)
    assert executor._queue.qsize() == 1

    release.set()
    active.result(timeout=2)
    deadline = time.monotonic() + 2
    while True:
        try:
            recovered = executor.submit(lambda: "recovered")
            break
        except DaemonExecutorCapacityError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)
    assert recovered.result(timeout=2) == "recovered"
    executor.shutdown(wait=True, cancel_futures=True)


def test_planning_runtime_shutdown_signals_active_control_and_reports_drain():
    runtime = ResearchPlanningRuntime(max_workers=1, max_pending=1)
    stop_event = Event()
    entered = Event()

    def wait_for_stop():
        entered.set()
        assert stop_event.wait(timeout=2)
        return "stopped"

    with runtime.register(stop_event):
        future = runtime.submit(wait_for_stop)
        assert entered.wait(timeout=2)
        assert runtime.shutdown(wait=False, cancel_futures=True) is False
        assert stop_event.is_set()
        assert future.result(timeout=2) == "stopped"
        assert runtime.is_drained() is False

    assert runtime.is_drained() is True


def test_planning_runtime_rejects_registration_after_shutdown():
    runtime = ResearchPlanningRuntime(max_workers=1, max_pending=1)
    assert runtime.shutdown(wait=False, cancel_futures=True) is True
    stop_event = Event()

    with pytest.raises(RuntimeError, match="closed"):
        with runtime.register(stop_event):
            raise AssertionError("closed runtime admitted planning work")

    assert stop_event.is_set()
