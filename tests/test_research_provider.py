import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Event, Thread

import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

import cogdoc.research_control as research_control_module
import cogdoc.api.research_job_store as research_store_module
from cogdoc.api.research_job_store import ResearchJobStore
from cogdoc.research_control import (
    ResearchBudgetExceeded,
    ResearchCancelled,
    ResearchDeadlineExceeded,
    ResearchPaused,
    ResearchProviderCapacityExceeded,
    ResearchProviderError,
    ResearchProviderTimeout,
    ResearchRunController,
    bind_research_control,
)
from cogdoc.research_provider import (
    IsolatedChatOpenAICall,
    ResearchProviderRemoteError,
    SerializedResearchProviderCall,
    invoke_research_model,
    mark_research_process_isolation_compatible,
    run_standalone_research_provider,
)
from cogdoc.research_isolation import (
    _decode_envelope,
    run_spawn_isolated_provider,
)
from cogdoc.service.research_execution import ResearchExecutionManager


@dataclass(frozen=True)
class _NeverReturningCall:
    pid_path: str
    research_process_isolated: bool = field(default=True, init=False, repr=False)

    def __call__(self):
        with open(self.pid_path, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        while True:
            time.sleep(1)


@dataclass(frozen=True)
class _SuccessfulCall:
    value: str
    research_process_isolated: bool = field(default=True, init=False, repr=False)

    def __call__(self):
        return {"value": self.value}


@dataclass(frozen=True)
class _TouchFileCall:
    path: str
    research_process_isolated: bool = field(default=True, init=False, repr=False)

    def __call__(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("started")
        return {"value": "started"}


@dataclass(frozen=True)
class _LargeResultCall:
    size: int
    research_process_isolated: bool = field(default=True, init=False, repr=False)

    def __call__(self):
        return "x" * self.size


@dataclass(frozen=True)
class _FileGateCall:
    started_path: str
    release_path: str
    research_process_isolated: bool = field(default=True, init=False, repr=False)

    def __call__(self):
        with open(self.started_path, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        while not os.path.exists(self.release_path):
            time.sleep(0.01)
        return {"value": "released"}


class _ProviderObserver:
    def __init__(self):
        self.calls = []

    def provider_call(self, **fields):
        self.calls.append(dict(fields))


def _controller(
    manager: ResearchExecutionManager,
    reservations: list[dict[str, int]] | None = None,
) -> ResearchRunController:
    def reserve(costs):
        if reservations is not None and costs:
            reservations.append(dict(costs))
        return {}

    return ResearchRunController(
        job_id="rj-provider",
        phase="report",
        attempt_id="attempt",
        lease_id="lease",
        reserve_callback=reserve,
        stop_event=Event(),
        deadline_at=(datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(),
        provider_runner=manager._run_provider_call,
    )


def test_controller_deadline_cannot_be_extended_by_wall_clock_rollback(monkeypatch):
    baseline = datetime.now(timezone.utc)

    class _RollbackClock(datetime):
        current = baseline

        @classmethod
        def now(cls, tz=None):
            value = cls.current
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(research_control_module, "datetime", _RollbackClock)
    control = ResearchRunController(
        job_id="rj-provider",
        phase="report",
        attempt_id="attempt",
        lease_id="lease",
        reserve_callback=lambda _costs: {},
        stop_event=Event(),
        deadline_at=(baseline + timedelta(seconds=30)).isoformat(),
    )
    initial = control.remaining_seconds()
    _RollbackClock.current = baseline - timedelta(hours=1)
    rolled_back = control.remaining_seconds()

    assert initial is not None
    assert rolled_back is not None
    assert 0 < rolled_back <= initial


def test_local_deadline_is_persisted_before_control_unwinds():
    durable_callbacks = []
    control = ResearchRunController(
        job_id="rj-provider",
        phase="report",
        attempt_id="attempt",
        lease_id="lease",
        reserve_callback=lambda costs: durable_callbacks.append(dict(costs)),
        stop_event=Event(),
        deadline_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
    )

    with pytest.raises(ResearchDeadlineExceeded):
        control.poll()

    assert durable_callbacks == [{}]


def test_transient_deadline_unwinds_as_durable_after_local_persist():
    durable_callbacks = []
    control = ResearchRunController(
        job_id="rj-provider",
        phase="report",
        attempt_id="attempt",
        lease_id="lease",
        reserve_callback=lambda costs: durable_callbacks.append(dict(costs)),
        stop_event=Event(),
        deadline_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
    )

    with pytest.raises(ResearchDeadlineExceeded) as exc_info:
        control.poll()

    assert exc_info.value.durable is True
    assert durable_callbacks == [{}]


def test_local_deadline_persist_preserves_source_exception():
    control = ResearchRunController(
        job_id="rj-provider",
        phase="report",
        attempt_id="attempt",
        lease_id="lease",
        reserve_callback=lambda _costs: {},
        stop_event=Event(),
        deadline_callback=lambda: "deadline",
    )
    source = ResearchDeadlineExceeded("custom deadline")

    with pytest.raises(ResearchDeadlineExceeded) as exc_info:
        control._persist_local_deadline(source)

    assert exc_info.value is source
    assert exc_info.value.durable is True
    assert str(exc_info.value) == "custom deadline"


def test_standalone_isolated_provider_expired_timeout_persists_deadline():
    commits = []

    class _Operation:
        research_process_isolated = True

        def __call__(self):
            raise AssertionError("provider should not run after deadline")

    control = ResearchRunController(
        job_id="rj-provider",
        phase="report",
        attempt_id="attempt",
        lease_id="lease",
        reserve_callback=lambda _costs: {},
        stop_event=Event(),
        deadline_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
        deadline_callback=lambda: commits.append("deadline") or "deadline",
    )

    with pytest.raises(ResearchDeadlineExceeded) as exc_info:
        run_standalone_research_provider(
            control,
            "openai",
            _Operation(),
            1,
            lambda: None,
        )

    assert exc_info.value.durable is True
    assert commits == ["deadline"]


def test_checkpoint_persists_deadline_crossed_during_reservation():
    commits = []

    def reserve(_costs):
        time.sleep(0.06)

    control = ResearchRunController(
        job_id="rj-provider",
        phase="report",
        attempt_id="attempt",
        lease_id="lease",
        reserve_callback=reserve,
        stop_event=Event(),
        deadline_at=(
            datetime.now(timezone.utc) + timedelta(milliseconds=30)
        ).isoformat(),
        deadline_callback=lambda: commits.append("deadline") or "deadline",
    )

    with pytest.raises(ResearchDeadlineExceeded) as exc_info:
        control.checkpoint({"llm_calls": 1})

    assert exc_info.value.durable is True
    assert commits == ["deadline"]


def test_provider_error_cannot_override_a_racing_pause():
    def runner(control, _provider, _operation, _timeout, on_admitted):
        on_admitted()
        control.request_stop("paused")
        raise ResearchProviderError("late provider error")

    control = ResearchRunController(
        job_id="rj-provider",
        phase="report",
        attempt_id="attempt",
        lease_id="lease",
        reserve_callback=lambda _costs: {},
        stop_event=Event(),
        deadline_at=(datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(),
        provider_runner=runner,
    )

    with pytest.raises(ResearchPaused):
        control.run_provider(
            lambda: None,
            provider="llm",
            timeout_seconds=10,
        )


def test_budget_signal_is_not_masked_by_a_second_checkpoint():
    reservations = []

    def reserve(costs):
        reservations.append(dict(costs))
        if costs:
            raise ResearchBudgetExceeded("budget")
        raise ResearchCancelled("stale lease")

    control = ResearchRunController(
        job_id="rj-provider",
        phase="report",
        attempt_id="attempt",
        lease_id="lease",
        reserve_callback=reserve,
        stop_event=Event(),
    )

    with pytest.raises(ResearchBudgetExceeded):
        control.run_provider(
            lambda: None,
            provider="llm",
            timeout_seconds=10,
            on_admitted=lambda: control.reserve(llm_calls=1),
        )

    assert reservations == [{"llm_calls": 1}]


def test_process_provider_rechecks_stop_after_budget_admission(tmp_path):
    manager = ResearchExecutionManager(
        ResearchJobStore(str(tmp_path / "jobs.json")),
        retrieve=lambda _kb, _query: [],
        kb_exists=lambda _kb: True,
        max_workers=1,
        provider_workers=1,
        provider_max_pending=1,
    )
    control = _controller(manager)
    started_path = tmp_path / "provider-started"

    def stop_after_reservation():
        control.reserve(llm_calls=1)
        control.request_stop("paused")

    try:
        with pytest.raises(ResearchPaused):
            control.run_provider(
                _TouchFileCall(str(started_path)),
                provider="llm",
                timeout_seconds=10,
                on_admitted=stop_after_reservation,
            )
    finally:
        manager.shutdown(wait=False)

    assert started_path.exists() is False


def test_spawn_provider_timeout_reaps_child_and_capacity_recovers(tmp_path):
    observer = _ProviderObserver()
    manager = ResearchExecutionManager(
        ResearchJobStore(str(tmp_path / "jobs.json")),
        retrieve=lambda _kb, _query: [],
        kb_exists=lambda _kb: True,
        max_workers=1,
        provider_workers=1,
        provider_max_pending=1,
        observer=observer,
    )
    control = _controller(manager)
    pid_path = tmp_path / "provider.pid"
    try:
        with pytest.raises(ResearchProviderTimeout):
            control.run_provider(
                _NeverReturningCall(str(pid_path)),
                provider="llm",
                timeout_seconds=1.0,
            )

        assert manager._provider_calls_in_use == 0
        if pid_path.exists():
            pid = int(pid_path.read_text(encoding="utf-8"))
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)

        assert control.run_provider(
            _SuccessfulCall("recovered"),
            provider="llm",
            timeout_seconds=10.0,
        ) == {"value": "recovered"}
        assert manager._provider_calls_in_use == 0
        assert [row["outcome"] for row in observer.calls] == [
            "timeout",
            "succeeded",
        ]
        assert {row["isolation"] for row in observer.calls} == {"process"}
    finally:
        manager.shutdown(wait=False)


def test_provider_phase_deadline_is_durably_expired_after_child_reap(tmp_path):
    store = ResearchJobStore(str(tmp_path / "jobs.json"))
    created = store.create(kb_id="kb", objective="deadline", section_titles=["s"])
    deadline_at = (
        datetime.now(timezone.utc) + timedelta(milliseconds=250)
    ).isoformat()
    running = store.start(created["job_id"], deadline_at=deadline_at)
    control_row = running["sections"][0]["execution_metrics"][
        "_research_control"
    ]["evidence"]
    manager = ResearchExecutionManager(
        store,
        retrieve=lambda _kb, _query: [],
        kb_exists=lambda _kb: True,
        max_workers=1,
        provider_workers=1,
        provider_max_pending=1,
    )
    control = manager._controller(
        job_id=created["job_id"],
        phase="evidence",
        attempt_id=control_row["attempt_id"],
        lease_id=control_row["lease_id"],
        deadline_at=deadline_at,
    )
    pid_path = tmp_path / "expired-provider.pid"
    try:
        with pytest.raises(ResearchDeadlineExceeded):
            control.run_provider(
                _NeverReturningCall(str(pid_path)),
                provider="llm",
                timeout_seconds=10,
            )
        expired = store.get(created["job_id"])
        durable = expired["sections"][0]["execution_metrics"][
            "_research_control"
        ]["evidence"]
        assert expired["status"] == "failed"
        assert expired["error"] == "ResearchDeadlineExceeded"
        assert durable["control_state"] == "expired"
        assert durable["lease_id"] == ""
    finally:
        manager.shutdown(wait=False)


def test_monotonic_deadline_commit_survives_durable_wall_clock_rollback(
    tmp_path,
    monkeypatch,
):
    baseline = datetime.now(timezone.utc)
    deadline_at = (baseline + timedelta(milliseconds=100)).isoformat()
    store = ResearchJobStore(str(tmp_path / "jobs.json"))
    created = store.create(kb_id="kb", objective="rollback", section_titles=["s"])
    running = store.start(created["job_id"], deadline_at=deadline_at)
    control_row = running["sections"][0]["execution_metrics"][
        "_research_control"
    ]["evidence"]
    manager = ResearchExecutionManager(
        store,
        retrieve=lambda _kb, _query: [],
        kb_exists=lambda _kb: True,
        max_workers=1,
    )
    control = manager._controller(
        job_id=created["job_id"],
        phase="evidence",
        attempt_id=control_row["attempt_id"],
        lease_id=control_row["lease_id"],
        deadline_at=deadline_at,
    )

    class _RollbackStoreClock(datetime):
        @classmethod
        def now(cls, tz=None):
            value = baseline - timedelta(hours=1)
            return value if tz is None else value.astimezone(tz)

    time.sleep(0.15)
    monkeypatch.setattr(research_store_module, "datetime", _RollbackStoreClock)
    try:
        with pytest.raises(ResearchDeadlineExceeded) as exc_info:
            control.poll()
        expired = store.get(created["job_id"])
        durable = expired["sections"][0]["execution_metrics"][
            "_research_control"
        ]["evidence"]
        assert exc_info.value.durable is True
        assert expired["status"] == "failed"
        assert durable["control_state"] == "expired"
        assert durable["lease_id"] == ""
    finally:
        manager.shutdown(wait=False)


def test_evidence_checkpoint_expires_after_retrieval_under_wall_clock_rollback(
    tmp_path,
    monkeypatch,
):
    baseline = datetime.now(timezone.utc)
    deadline_at = (baseline + timedelta(milliseconds=80)).isoformat()
    store = ResearchJobStore(str(tmp_path / "jobs.json"))
    created = store.create(kb_id="kb", objective="rollback", section_titles=["s"])
    running = store.start(created["job_id"], deadline_at=deadline_at)
    control_row = running["sections"][0]["execution_metrics"][
        "_research_control"
    ]["evidence"]

    def slow_retrieve(_kb, _query):
        time.sleep(0.12)
        return []

    manager = ResearchExecutionManager(
        store,
        retrieve=slow_retrieve,
        kb_exists=lambda _kb: True,
        max_workers=1,
    )
    control = manager._controller(
        job_id=created["job_id"],
        phase="evidence",
        attempt_id=control_row["attempt_id"],
        lease_id=control_row["lease_id"],
        deadline_at=deadline_at,
    )

    class _RollbackStoreClock(datetime):
        @classmethod
        def now(cls, tz=None):
            value = baseline - timedelta(hours=1)
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(research_store_module, "datetime", _RollbackStoreClock)
    try:
        manager._run_job(
            created["job_id"],
            control_row["attempt_id"],
            control_row["lease_id"],
            control,
        )
        expired = store.get(created["job_id"])
        durable = expired["sections"][0]["execution_metrics"][
            "_research_control"
        ]["evidence"]
        assert expired["status"] == "failed"
        assert expired["error"] == "ResearchDeadlineExceeded"
        assert durable["control_state"] == "expired"
        assert durable["lease_id"] == ""
    finally:
        manager.shutdown(wait=False)


def test_chat_openai_research_call_uses_pickle_safe_process_recipe():
    captured = {}

    def runner(_control, provider, operation, timeout_seconds, on_admitted):
        on_admitted()
        captured.update(
            {
                "provider": provider,
                "operation": operation,
                "timeout_seconds": timeout_seconds,
            }
        )
        return AIMessage(content="ok")

    control = ResearchRunController(
        job_id="rj-provider",
        phase="report",
        attempt_id="attempt",
        lease_id="lease",
        reserve_callback=lambda _costs: {},
        stop_event=Event(),
        deadline_at=(datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(),
        provider_runner=runner,
    )
    llm = ChatOpenAI(
        model="demo",
        api_key="secret-key",
        base_url="http://127.0.0.1:9/v1",
        timeout=90,
        max_retries=2,
    )
    mark_research_process_isolation_compatible(llm)

    with bind_research_control(control):
        response = invoke_research_model(
            llm,
            [{"role": "user", "content": "hello"}],
        )

    assert response.content == "ok"
    assert captured["provider"] == "llm"
    assert isinstance(captured["operation"], SerializedResearchProviderCall)
    recipe = pickle.loads(captured["operation"].payload)
    assert isinstance(recipe, IsolatedChatOpenAICall)
    assert recipe.api_key == "secret-key"
    assert "secret-key" not in repr(captured["operation"])
    assert "secret-key" not in repr(recipe)
    assert 0 < captured["timeout_seconds"] <= 30


def test_spawn_provider_worker_limit_bounds_active_children(tmp_path):
    manager = ResearchExecutionManager(
        ResearchJobStore(str(tmp_path / "jobs.json")),
        retrieve=lambda _kb, _query: [],
        kb_exists=lambda _kb: True,
        max_workers=1,
        provider_workers=1,
        provider_max_pending=2,
    )
    started = tmp_path / "started"
    release = tmp_path / "release"
    result = []
    error = []

    def run_first():
        try:
            result.append(
                _controller(manager).run_provider(
                    _FileGateCall(str(started), str(release)),
                    provider="llm",
                    # A cold ``spawn`` child imports the application graph and
                    # can take several seconds on a loaded CI host.  Keep this
                    # comfortably above the startup bound; the second call is
                    # the one that exercises the short slot-wait timeout.
                    timeout_seconds=30,
                )
            )
        except BaseException as exc:
            error.append(exc)

    thread = Thread(target=run_first)
    thread.start()
    deadline = time.monotonic() + 20
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        assert started.exists()
        assert manager._provider_processes_in_use == 1
        reservations = []
        waiting_control = _controller(manager, reservations)
        with pytest.raises(ResearchProviderTimeout):
            waiting_control.run_provider(
                _SuccessfulCall("must-wait"),
                provider="llm",
                timeout_seconds=0.2,
                on_admitted=lambda: waiting_control.reserve(llm_calls=1),
            )
        # A waiter that never acquired the slot must not release another call's
        # active-child accounting.
        assert manager._provider_processes_in_use == 1
        assert reservations == []
    finally:
        release.touch()
        thread.join(timeout=10)
        manager.shutdown(wait=False)

    assert not thread.is_alive()
    assert not error
    assert result == [{"value": "released"}]
    assert manager._provider_processes_in_use == 0


def test_compatibility_provider_cancel_churn_cannot_grow_physical_queue(tmp_path):
    manager = ResearchExecutionManager(
        ResearchJobStore(str(tmp_path / "jobs.json")),
        retrieve=lambda _kb, _query: [],
        kb_exists=lambda _kb: True,
        max_workers=1,
        provider_workers=1,
        provider_max_pending=2,
    )
    entered = Event()
    release = Event()
    first_errors = []

    def block():
        entered.set()
        release.wait()
        return "released"

    def run_first():
        try:
            _controller(manager).run_provider(
                block,
                provider="llm",
                timeout_seconds=30,
            )
        except BaseException as exc:
            first_errors.append(exc)

    thread = Thread(target=run_first)
    thread.start()
    assert entered.wait(timeout=2)
    with pytest.raises(ResearchProviderTimeout):
        _controller(manager).run_provider(
            lambda: "queued",
            provider="llm",
            timeout_seconds=0.05,
        )

    for _attempt in range(50):
        with pytest.raises(ResearchProviderCapacityExceeded):
            _controller(manager).run_provider(
                lambda: "must-not-queue",
                provider="llm",
                timeout_seconds=0.05,
            )
    assert manager._provider_executor._queue.qsize() == 1
    assert manager._provider_executor._pending == 2

    release.set()
    thread.join(timeout=5)
    manager.shutdown(wait=False)
    assert not thread.is_alive()
    assert first_errors == []


def test_recognized_chat_openai_fails_closed_when_recipe_is_not_pickle_safe():
    class LocalSchema(BaseModel):
        value: str

    runner_called = False

    def runner(*_args):
        nonlocal runner_called
        runner_called = True
        return None

    control = ResearchRunController(
        job_id="rj-provider",
        phase="report",
        attempt_id="attempt",
        lease_id="lease",
        reserve_callback=lambda _costs: {},
        stop_event=Event(),
        deadline_at=(datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(),
        provider_runner=runner,
    )
    llm = ChatOpenAI(
        model="demo",
        api_key="secret-key",
        base_url="http://127.0.0.1:9/v1",
        timeout=90,
        max_retries=0,
    )
    mark_research_process_isolation_compatible(llm)

    with bind_research_control(control), pytest.raises(
        ResearchProviderError,
        match="could not enter process isolation",
    ):
        invoke_research_model(
            llm,
            [{"role": "user", "content": "hello"}],
            schema=LocalSchema,
            structured_method="json_mode",
        )

    assert runner_called is False


@pytest.mark.parametrize(
    "envelope",
    [
        ("error", "InternalServerError", "upstream failed", 500, "server_error"),
        (
            "error",
            "AuthenticationError",
            "response_format credentials rejected",
            401,
            "invalid_api_key",
        ),
        (
            "error",
            "BadRequestError",
            "maximum context length exceeded",
            400,
            "context_length_exceeded",
        ),
        ("error", "ValueError", "ordinary adapter failure", None, ""),
    ],
)
def test_provider_error_envelope_is_terminal_except_capability_errors(envelope):
    with pytest.raises(ResearchProviderError) as exc_info:
        _decode_envelope(pickle.dumps(envelope, protocol=5))

    assert not isinstance(exc_info.value, ResearchProviderRemoteError)


@pytest.mark.parametrize(
    "envelope",
    [
        (
            "error",
            "BadRequestError",
            "unsupported parameter: response_format",
            400,
            "invalid_request_error",
        ),
        (
            "error",
            "UnprocessableEntityError",
            "json_schema is not supported",
            422,
            "unsupported_parameter",
        ),
    ],
)
def test_provider_capability_envelope_allows_structured_fallback(envelope):
    with pytest.raises(ResearchProviderRemoteError) as exc_info:
        _decode_envelope(pickle.dumps(envelope, protocol=5))

    assert exc_info.value.error_class == envelope[1]


@pytest.mark.parametrize(
    "raw",
    [
        b"not-a-pickle-envelope",
        pickle.dumps({"status": "ok"}, protocol=5),
        pickle.dumps(("ok",), protocol=5),
        pickle.dumps(("error", "BadRequestError", "broken"), protocol=5),
    ],
)
def test_provider_rejects_malformed_ipc_envelope(raw):
    with pytest.raises(ResearchProviderError, match="invalid IPC envelope"):
        _decode_envelope(raw)


def test_spawn_provider_rejects_oversized_ipc_envelope_and_reaps_child():
    with pytest.raises(ResearchProviderError, match="invalid IPC envelope"):
        run_spawn_isolated_provider(
            _LargeResultCall(16_384),
            provider="llm",
            timeout_seconds=10.0,
            kill_grace_seconds=0.5,
            ipc_max_bytes=1024,
        )
