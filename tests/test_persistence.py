import time
from types import SimpleNamespace
from api.ingest import IndexJobManager
from api.persistence import SqliteJobStore, SqliteSessionStore


def test_session_survives_new_store_instance(tmp_path):
    db = str(tmp_path / "state.db")
    store = SqliteSessionStore(db)
    store.record(
        "kb",
        "s1",
        [{"role": "user", "content": "门控记忆"}],
        [
            {"role": "user", "content": "今年营收多少"},
            {"role": "assistant", "content": "一千万"},
        ],
    )
    # 重开一个实例模拟进程重启：历史、展示、会话列表都应还在。
    reopened = SqliteSessionStore(db)
    assert reopened.get_history("kb", "s1") == [{"role": "user", "content": "门控记忆"}]
    assert len(reopened.get_display("kb", "s1")) == 2
    sessions = reopened.list_sessions("kb")
    assert sessions[0]["session_id"] == "s1"
    assert sessions[0]["title"] == "今年营收多少"
    assert sessions[0]["message_count"] == 2


def test_session_record_appends_across_turns(tmp_path):
    store = SqliteSessionStore(str(tmp_path / "state.db"))
    store.record(
        "kb",
        "s1",
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "a"}],
    )
    store.record(
        "kb",
        "s1",
        [{"role": "user", "content": "b"}],
        [{"role": "user", "content": "b"}],
    )
    assert len(store.get_history("kb", "s1")) == 2


def test_session_doc_id_isolates_and_clear(tmp_path):
    store = SqliteSessionStore(str(tmp_path / "state.db"))
    store.record("kb1", "s", [], [{"role": "user", "content": "x"}])
    store.record("kb2", "s", [], [{"role": "user", "content": "y"}])
    assert [s["session_id"] for s in store.list_sessions("kb1")] == ["s"]
    store.clear("kb1", "s")
    assert store.list_sessions("kb1") == []
    assert len(store.list_sessions("kb2")) == 1


def test_session_purges_expired(tmp_path):
    store = SqliteSessionStore(str(tmp_path / "state.db"), ttl_seconds=1)
    store.record("kb", "s", [], [{"role": "user", "content": "x"}])
    time.sleep(1.1)
    assert store.get_display("kb", "s") == []
    assert store.list_sessions("kb") == []


def test_session_evicts_oldest_over_capacity(tmp_path):
    store = SqliteSessionStore(str(tmp_path / "state.db"), max_sessions=1)
    store.record("kb", "old", [], [{"role": "user", "content": "old"}])
    time.sleep(0.01)
    store.record("kb", "new", [], [{"role": "user", "content": "new"}])
    ids = [s["session_id"] for s in store.list_sessions("kb")]
    assert ids == ["new"]


def _ok_ingest(kb_id, source_dir):
    return SimpleNamespace(document_count=2, chunk_count=7)


def _wait(manager, job_id, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline and manager.get(job_id)["status"] in (
        "pending",
        "running",
    ):
        time.sleep(0.01)
    return manager.get(job_id)


def test_job_record_survives_new_manager(tmp_path):
    db = str(tmp_path / "state.db")
    manager = IndexJobManager(
        ingest_fn=_ok_ingest,
        source_dir_for=lambda kb: str(tmp_path / kb),
        job_store=SqliteJobStore(db),
    )
    job_id = manager.submit("kb")["job_id"]
    done = _wait(manager, job_id)
    assert done["status"] == "succeeded" and done["document_count"] == 2
    manager.shutdown()

    # 新进程：换一个 manager + 新的 store 实例，已完成任务仍可查询。
    reopened = IndexJobManager(ingest_fn=_ok_ingest, job_store=SqliteJobStore(db))
    assert reopened.get(job_id)["status"] == "succeeded"


def test_reconcile_marks_orphaned_running_job_failed(tmp_path):
    db = str(tmp_path / "state.db")
    store = SqliteJobStore(db)
    # 模拟上次进程崩在 running：直接写一条非终态记录。
    store.create(
        {"job_id": "orphan", "kb_id": "kb", "status": "running", "message": None}
    )
    # 重开 store 触发启动协调，非终态被判失败。
    reopened = SqliteJobStore(db)
    rec = reopened.get("orphan")
    assert rec["status"] == "failed"
    assert rec["error_code"] == "INGEST_FAILED"
    assert rec["message"] == "服务重启，任务中断"


def test_get_missing_job_returns_none(tmp_path):
    store = SqliteJobStore(str(tmp_path / "state.db"))
    assert store.get("nope") is None
