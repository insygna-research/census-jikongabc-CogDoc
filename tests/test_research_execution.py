import time
from threading import Event

from cogdoc.api.research_job_store import ResearchJobStore
from cogdoc.service.research_execution import (
    ResearchExecutionManager,
    public_research_evidence,
)


def _wait_for(store, job_id: str, expected: str, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = store.get(job_id)
        if row["status"] == expected:
            return row
        time.sleep(0.01)
    raise AssertionError(f"job did not reach {expected}: {store.get(job_id)}")


def _doc(chunk_id: str, text: str = "完整证据正文"):
    return {
        "text": text,
        "meta": {
            "chunk_id": chunk_id,
            "source": "rules.pdf",
            "page": 2,
            "section_title": "报名规则",
        },
        "retrieval": {"rerank_score": 0.9, "search_channel": "hybrid"},
    }


def test_public_research_evidence_is_bounded_and_deduplicated():
    invalid_score = _doc("c2")
    invalid_score["retrieval"]["rerank_score"] = float("nan")
    evidence = public_research_evidence(
        [_doc("c1", "a" * 800), _doc("c1", "duplicate"), invalid_score],
        limit=2,
        preview_chars=20,
    )

    assert [item["chunk_id"] for item in evidence] == ["c1", "c2"]
    assert len(evidence[0]["text_preview"]) == 20
    assert "text" not in evidence[0]
    assert evidence[1]["rerank_score"] is None


def test_research_execution_manager_collects_section_evidence(tmp_path):
    store = ResearchJobStore(str(tmp_path / "research.json"))
    job = store.create(
        kb_id="kb", objective="研究", section_titles=["有证据", "无证据"]
    )

    def retrieve(kb_id, query):
        assert kb_id == "kb"
        return [_doc("c1")] if "有证据" in query else []

    manager = ResearchExecutionManager(
        store, retrieve=retrieve, kb_exists=lambda kb_id: kb_id == "kb"
    )
    try:
        started = manager.start(job["job_id"])
        assert started["status"] == "running"
        completed = _wait_for(store, job["job_id"], "evidence_ready")
    finally:
        manager.shutdown()

    assert [section["evidence_status"] for section in completed["sections"]] == [
        "partial",
        "missing",
    ]
    assert completed["sections"][0]["evidence"][0]["chunk_id"] == "c1"
    assert completed["sections"][0]["execution_metrics"]["candidate_count"] == 1


def test_research_execution_manager_pauses_between_sections(tmp_path):
    store = ResearchJobStore(str(tmp_path / "research.json"))
    job = store.create(
        kb_id="kb", objective="研究", section_titles=["第一章", "第二章"]
    )
    entered = Event()
    release = Event()

    def retrieve(_kb_id, query):
        if "第一章" in query:
            entered.set()
            assert release.wait(2)
        return [_doc(query)]

    manager = ResearchExecutionManager(
        store, retrieve=retrieve, kb_exists=lambda _kb_id: True
    )
    try:
        manager.start(job["job_id"])
        assert entered.wait(2)
        paused = manager.pause(job["job_id"])
        assert paused["status"] == "paused"
        release.set()
        paused_after_section = _wait_for(store, job["job_id"], "paused")
        deadline = time.monotonic() + 2
        while (
            paused_after_section["sections"][0]["status"] != "completed"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
            paused_after_section = store.get(job["job_id"])
        assert paused_after_section["sections"][0]["status"] == "completed"
        assert paused_after_section["sections"][1]["status"] == "pending"

        manager.resume(job["job_id"])
        completed = _wait_for(store, job["job_id"], "evidence_ready")
    finally:
        release.set()
        manager.shutdown()

    assert all(section["status"] == "completed" for section in completed["sections"])


def test_research_execution_manager_records_failure_and_allows_retry(tmp_path):
    store = ResearchJobStore(str(tmp_path / "research.json"))
    job = store.create(kb_id="kb", objective="研究", section_titles=["章节"])
    attempts = 0

    def retrieve(_kb_id, _query):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("temporary")
        return [_doc("c1")]

    manager = ResearchExecutionManager(
        store, retrieve=retrieve, kb_exists=lambda _kb_id: True
    )
    try:
        manager.start(job["job_id"])
        failed = _wait_for(store, job["job_id"], "failed")
        assert failed["error"] == "TimeoutError"
        manager.start(job["job_id"])
        completed = _wait_for(store, job["job_id"], "evidence_ready")
    finally:
        manager.shutdown()

    assert attempts == 2
    assert completed["sections"][0]["evidence_status"] == "partial"


def test_research_execution_manager_builds_and_persists_report(tmp_path):
    store = ResearchJobStore(str(tmp_path / "research.json"))
    job = store.create(kb_id="kb", objective="研究", section_titles=["章节"])

    def report_builder(current):
        assert current["status"] == "generating"
        return {
            "status": "ready",
            "markdown": "# 研究\n\n## 章节\n\n结论。[rules.pdf:P2]\n",
            "citation_ledger": [],
            "verification_metrics": {"supported_count": 1},
            "sections": [
                {
                    "section_id": "s1",
                    "status": "generated",
                    "verification_status": "supported",
                    "verification_reason_code": "supported",
                    "content": "结论。[rules.pdf:P2]",
                    "evidence": [{"chunk_id": "c1"}],
                    "error": "",
                }
            ],
        }

    manager = ResearchExecutionManager(
        store,
        retrieve=lambda _kb_id, _query: [_doc("c1")],
        kb_exists=lambda _kb_id: True,
        report_builder=report_builder,
    )
    try:
        manager.start(job["job_id"])
        _wait_for(store, job["job_id"], "evidence_ready")
        generating = manager.compile(job["job_id"])
        assert generating["status"] == "generating"
        completed = _wait_for(store, job["job_id"], "completed")
    finally:
        manager.shutdown()

    assert completed["report_status"] == "ready"
    assert completed["report"]["content"].startswith("# 研究")
    assert completed["sections"][0]["generation_status"] == "generated"


def test_research_execution_manager_fails_report_closed_and_allows_retry(tmp_path):
    store = ResearchJobStore(str(tmp_path / "research.json"))
    job = store.create(kb_id="kb", objective="研究", section_titles=["章节"])
    attempts = 0

    def report_builder(_current):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("temporary")
        return {
            "status": "ready_with_gaps",
            "markdown": "# 研究\n\n章节被阻止。\n",
            "citation_ledger": [],
            "verification_metrics": {"verification_error_count": 1},
            "sections": [],
        }

    manager = ResearchExecutionManager(
        store,
        retrieve=lambda _kb_id, _query: [],
        kb_exists=lambda _kb_id: True,
        report_builder=report_builder,
    )
    try:
        manager.start(job["job_id"])
        _wait_for(store, job["job_id"], "evidence_ready")
        manager.compile(job["job_id"])
        failed = _wait_for(store, job["job_id"], "failed")
        assert failed["report_status"] == "failed"
        assert failed["error"] == "TimeoutError"
        manager.compile(job["job_id"])
        completed = _wait_for(store, job["job_id"], "completed")
    finally:
        manager.shutdown()

    assert attempts == 2
    assert completed["report_status"] == "ready_with_gaps"
