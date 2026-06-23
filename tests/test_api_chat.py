import pytest
import threading
from httpx import ASGITransport, AsyncClient
from api.app import create_app
from api.session_store import SessionStore
from service.chat_service import ChatResult, ChatServiceError


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def _post_chat(app, payload: dict):
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/v1/chat", json=payload)


def _result(answer: str, trace_id: str, messages=None) -> ChatResult:
    return ChatResult(
        answer=answer,
        task_type="qa",
        citations=[{"chunk_id": "chunk-1", "source": "a.pdf", "page": 1}],
        evidence=[
            {
                "chunk_id": "chunk-1",
                "chunk_index": 0,
                "source": "a.pdf",
                "page": 1,
                "text_preview": "证据预览",
            }
        ],
        critique="",
        is_valid=True,
        trace_id=trace_id,
        request_id=trace_id,
        steps=[],
        chat_messages=messages
        or [
            {"role": "user", "content": "问题", "timestamp": None},
            {"role": "assistant", "content": answer, "timestamp": None},
        ],
        raw_output={"answer": answer},
    )


@pytest.mark.anyio
async def test_chat_endpoint_maps_response_and_trace_header(monkeypatch):
    import api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    calls = []

    def fake_runner(doc_id, query, is_local, chat_history, forced_task):
        calls.append(
            {
                "doc_id": doc_id,
                "query": query,
                "is_local": is_local,
                "chat_history": chat_history,
                "forced_task": forced_task,
            }
        )
        return _result("答案", "trace-sync")

    app = create_app(chat_runner=fake_runner, session_store=SessionStore())

    response = await _post_chat(
        app,
        {
            "query": "  问题  ",
            "doc_id": "kb",
            "session_id": "s1",
            "mode": "summary",
            "is_local": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Trace-Id"] == "trace-sync"
    payload = response.json()
    assert payload["schema_version"] == "v1"
    assert payload["doc_id"] == "kb"
    assert payload["session_id"] == "s1"
    assert payload["answer"] == "答案"
    assert payload["citations"][0]["source"] == "a.pdf"
    assert calls == [
        {
            "doc_id": "kb",
            "query": "问题",
            "is_local": True,
            "chat_history": [],
            "forced_task": "summary",
        }
    ]


@pytest.mark.anyio
async def test_chat_endpoint_reuses_session_history(monkeypatch):
    import api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    history_lengths = []

    def fake_runner(doc_id, query, is_local, chat_history, forced_task):
        history_lengths.append(len(chat_history))
        return _result(f"history={len(chat_history)}", f"trace-{len(history_lengths)}")

    app = create_app(chat_runner=fake_runner, session_store=SessionStore())

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            first = await client.post(
                "/v1/chat",
                json={"query": "第一问", "doc_id": "kb", "session_id": "s1"},
            )
            second = await client.post(
                "/v1/chat",
                json={"query": "第二问", "doc_id": "kb", "session_id": "s1"},
            )

    assert first.status_code == 200
    assert second.status_code == 200
    assert history_lengths == [0, 2]
    assert second.json()["answer"] == "history=2"


@pytest.mark.anyio
async def test_chat_endpoint_offloads_runner(monkeypatch):
    import api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    caller_thread_id = threading.get_ident()
    runner_thread_ids = []

    def fake_runner(doc_id, query, is_local, chat_history, forced_task):
        runner_thread_ids.append(threading.get_ident())
        return _result("答案", "trace-offload")

    app = create_app(chat_runner=fake_runner, session_store=SessionStore())

    response = await _post_chat(app, {"query": "问题", "doc_id": "kb"})

    assert response.status_code == 200
    assert runner_thread_ids
    assert runner_thread_ids[0] != caller_thread_id


def test_session_store_uses_doc_id_in_key_and_evicts_oldest():
    store = SessionStore(max_sessions=1, ttl_seconds=3600)
    store.append_messages("kb-a", "s1", [{"role": "user", "content": "a"}])
    store.append_messages("kb-b", "s1", [{"role": "user", "content": "b"}])

    assert store.get_history("kb-a", "s1") == []
    assert store.get_history("kb-b", "s1") == [{"role": "user", "content": "b"}]


def test_session_store_purges_expired_history():
    store = SessionStore(max_sessions=10, ttl_seconds=1)
    store.append_messages("kb", "s1", [{"role": "user", "content": "a"}])
    store._entries[("kb", "s1")].updated_at -= 2

    assert store.get_history("kb", "s1") == []


@pytest.mark.anyio
async def test_chat_endpoint_maps_runtime_error_to_stable_error_code(monkeypatch):
    import api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)

    def failing_runner(doc_id, query, is_local, chat_history, forced_task):
        raise ChatServiceError(
            stage="runtime",
            error_class="ValueError",
            message="模型不可用",
            trace_id="trace-fail",
        )

    store = SessionStore()
    app = create_app(chat_runner=failing_runner, session_store=store)

    response = await _post_chat(app, {"query": "问题", "doc_id": "kb", "session_id": "s1"})

    assert response.status_code == 503
    assert response.headers["X-Trace-Id"] == "trace-fail"
    payload = response.json()
    assert payload["error_code"] == "MODEL_UNAVAILABLE"
    assert payload["trace_id"] == "trace-fail"
    assert payload["details"]["error_class"] == "ValueError"
    # 失败不写会话，不漏栈。
    assert store.get_history("kb", "s1") == []
    assert "Traceback" not in payload["message"]


@pytest.mark.anyio
async def test_chat_endpoint_maps_stream_stage_to_interrupted(monkeypatch):
    import api.app as app_module

    monkeypatch.setattr(app_module, "configure_logging", lambda: None)

    def failing_runner(doc_id, query, is_local, chat_history, forced_task):
        raise ChatServiceError(
            stage="stream",
            error_class="TimeoutError",
            message="流中断",
            trace_id="trace-stream",
        )

    app = create_app(chat_runner=failing_runner, session_store=SessionStore())

    response = await _post_chat(app, {"query": "问题", "doc_id": "kb"})

    assert response.status_code == 502
    assert response.json()["error_code"] == "STREAM_INTERRUPTED"


def test_run_chat_sync_raises_typed_error_when_no_final(monkeypatch):
    from service import chat_service

    class CrashingApp:
        def stream(self, *args, **kwargs):
            raise RuntimeError("graph 调度崩溃")

    monkeypatch.setattr(chat_service, "app", CrashingApp())
    monkeypatch.setattr(chat_service, "configure_logging", lambda: None)
    monkeypatch.setattr(chat_service, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(chat_service, "export_trace", lambda **k: None)

    with pytest.raises(chat_service.ChatServiceError) as excinfo:
        chat_service.run_chat_sync("kb", "问题", is_local=False)

    assert excinfo.value.stage == "runtime"
    assert excinfo.value.error_class == "RuntimeError"
    assert excinfo.value.message == "graph 调度崩溃"
