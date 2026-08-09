import pytest

from cogdoc.api.research_job_store import (
    ResearchJobRevisionConflictError,
    ResearchJobStateConflictError,
    ResearchJobStore,
    SqliteResearchJobStore,
)


@pytest.fixture(params=["json", "sqlite"])
def store(request, tmp_path):
    if request.param == "sqlite":
        value = SqliteResearchJobStore(str(tmp_path / "state.db"))
    else:
        value = ResearchJobStore(str(tmp_path / "research_jobs.json"))
    yield value
    close = getattr(value, "close", None)
    if callable(close):
        close()


def test_research_job_store_creates_editable_grounded_plan(store):
    row = store.create(
        kb_id="kb",
        objective="比较三份规程并提出参赛建议",
        section_titles=["报名要求", "评分规则"],
    )

    assert row["job_id"].startswith("rj_")
    assert row["status"] == "planned"
    assert row["revision"] == 1
    assert [section["section_id"] for section in row["sections"]] == ["s1", "s2"]
    assert [section["title"] for section in row["sections"]] == [
        "报名要求",
        "评分规则",
    ]
    assert all(section["evidence_status"] == "unsearched" for section in row["sections"])
    assert store.get(row["job_id"]) == row


def test_research_job_store_updates_plan_with_optimistic_revision(store):
    created = store.create(kb_id="kb", objective="研究目标")

    updated = store.update_plan(
        created["job_id"],
        expected_revision=1,
        sections=[
            {"title": "证据", "research_question": "有哪些直接证据？"},
            {"title": "结论", "research_question": "证据支持什么结论？"},
        ],
    )

    assert updated["revision"] == 2
    assert [section["position"] for section in updated["sections"]] == [1, 2]
    with pytest.raises(ResearchJobRevisionConflictError):
        store.update_plan(
            created["job_id"],
            expected_revision=1,
            sections=[{"title": "过期编辑", "research_question": "不应覆盖什么？"}],
        )
    assert store.get(created["job_id"])["revision"] == 2


def test_research_job_store_filters_and_clears_by_kb(store):
    first = store.create(kb_id="a", objective="A")
    store.create(kb_id="b", objective="B")

    assert [row["job_id"] for row in store.list(kb_id="a")] == [first["job_id"]]
    store.clear_kb("a")
    assert store.get(first["job_id"]) is None
    assert len(store.list(kb_id="b")) == 1


def test_research_job_store_runs_sections_to_evidence_ready(store):
    created = store.create(
        kb_id="kb", objective="研究", section_titles=["第一章", "第二章"]
    )
    running = store.start(created["job_id"])
    execution_id = running["execution_id"]
    with pytest.raises(ResearchJobStateConflictError):
        store.update_plan(
            created["job_id"],
            expected_revision=running["revision"],
            sections=[{"title": "执行中编辑", "research_question": "允许吗？"}],
        )

    _, first = store.claim_next_section(created["job_id"], execution_id)
    assert first["section_id"] == "s1"
    store.complete_section(
        created["job_id"],
        "s1",
        execution_id=execution_id,
        evidence_status="partial",
        evidence=[{"chunk_id": "c1", "text_preview": "证据"}],
        execution_metrics={"candidate_count": 1},
    )
    _, second = store.claim_next_section(created["job_id"], execution_id)
    assert second["section_id"] == "s2"
    completed = store.complete_section(
        created["job_id"],
        "s2",
        execution_id=execution_id,
        evidence_status="missing",
        evidence=[],
        execution_metrics={"candidate_count": 0},
    )

    assert completed["status"] == "evidence_ready"
    assert [section["evidence_status"] for section in completed["sections"]] == [
        "partial",
        "missing",
    ]
    with pytest.raises(ResearchJobStateConflictError):
        store.start(created["job_id"])


def test_research_job_store_pause_resume_cancel_and_reconcile(store):
    first = store.create(kb_id="kb", objective="暂停研究")
    running = store.start(first["job_id"])
    store.claim_next_section(first["job_id"], running["execution_id"])
    paused = store.pause(first["job_id"])
    assert paused["status"] == "paused"
    resumed = store.resume(first["job_id"])
    assert resumed["execution_id"] == running["execution_id"]
    cancelled = store.cancel(first["job_id"])
    assert cancelled["status"] == "cancelled"
    assert cancelled["sections"][0]["status"] == "pending"

    orphan = store.create(kb_id="kb", objective="孤儿研究")
    store.start(orphan["job_id"])
    assert store.reconcile_running() == 1
    reconciled = store.get(orphan["job_id"])
    assert reconciled["status"] == "paused"
    assert reconciled["error"] == "service_restarted"


def test_research_job_store_resume_requeues_orphaned_running_section(store):
    created = store.create(kb_id="kb", objective="恢复遗留章节", section_titles=["章节"])
    running = store.start(created["job_id"])
    _, claimed = store.claim_next_section(created["job_id"], running["execution_id"])
    assert claimed["status"] == "running"
    store.pause(created["job_id"])

    resumed = store.resume(created["job_id"])
    assert resumed["execution_id"] == running["execution_id"]
    assert resumed["sections"][0]["status"] == "pending"
    _, reclaimed = store.claim_next_section(
        created["job_id"], resumed["execution_id"]
    )
    assert reclaimed["section_id"] == "s1"
    completed = store.complete_section(
        created["job_id"],
        "s1",
        execution_id=resumed["execution_id"],
        evidence_status="missing",
        evidence=[],
        execution_metrics={"candidate_count": 0},
    )
    assert completed["status"] == "evidence_ready"


def test_research_job_store_resume_requires_paused_state(store):
    created = store.create(kb_id="kb", objective="仅允许恢复暂停任务")

    with pytest.raises(ResearchJobStateConflictError, match="cannot resume"):
        store.resume(created["job_id"])


def _advance_to_evidence_ready(store, job_id):
    running = store.start(job_id)
    execution_id = running["execution_id"]
    while True:
        _, section = store.claim_next_section(job_id, execution_id)
        if section is None:
            break
        store.complete_section(
            job_id,
            section["section_id"],
            execution_id=execution_id,
            evidence_status="partial",
            evidence=[{"chunk_id": section["section_id"], "text_preview": "证据"}],
            execution_metrics={"candidate_count": 1},
        )
    return store.get(job_id)


def test_research_job_store_persists_report_and_section_verdicts(store):
    created = store.create(kb_id="kb", objective="形成报告", section_titles=["证据"])
    _advance_to_evidence_ready(store, created["job_id"])

    generating = store.begin_report(created["job_id"])
    assert generating["status"] == "generating"
    with pytest.raises(ResearchJobStateConflictError):
        store.update_plan(
            created["job_id"],
            expected_revision=generating["revision"],
            sections=[{"title": "覆盖", "research_question": "允许吗？"}],
        )
    completed = store.complete_report(
        created["job_id"],
        report_execution_id=generating["report_execution_id"],
        result={
            "status": "ready",
            "markdown": "# 报告\n\n结论。[rules.pdf:P1]\n",
            "citation_ledger": [],
            "verification_metrics": {"supported_count": 1},
            "sections": [
                {
                    "section_id": "s1",
                    "status": "generated",
                    "verification_status": "supported",
                    "verification_reason_code": "supported",
                    "content": "结论。[rules.pdf:P1]",
                    "evidence": [{"chunk_id": "c1"}],
                    "error": "",
                }
            ],
        },
    )

    assert completed["status"] == "completed"
    assert completed["report_status"] == "ready"
    assert completed["report"]["format"] == "markdown"
    assert completed["sections"][0]["verification_status"] == "supported"
    assert completed["sections"][0]["generation_status"] == "generated"


def test_research_job_store_recovers_orphaned_report_generation(store):
    created = store.create(kb_id="kb", objective="恢复报告", section_titles=["证据"])
    _advance_to_evidence_ready(store, created["job_id"])
    store.begin_report(created["job_id"])

    assert store.reconcile_running() == 1
    recovered = store.get(created["job_id"])
    assert recovered["status"] == "evidence_ready"
    assert recovered["report_status"] == "failed"
    assert recovered["error"] == "service_restarted"


def _complete_test_report(store, job_id, sections, *, report_status="ready"):
    generating = store.begin_report(job_id)
    return store.complete_report(
        job_id,
        report_execution_id=generating["report_execution_id"],
        result={
            "status": report_status,
            "markdown": "# 报告\n",
            "citation_ledger": [],
            "verification_metrics": {},
            "sections": sections,
        },
    )


def test_research_job_store_requires_complete_review_before_publish(store):
    created = store.create(
        kb_id="kb", objective="审阅报告", section_titles=["正文", "缺口"]
    )
    _advance_to_evidence_ready(store, created["job_id"])
    completed = _complete_test_report(
        store,
        created["job_id"],
        [
            {
                "section_id": "s1",
                "status": "generated",
                "verification_status": "supported",
                "content": "正文",
                "evidence": [],
            },
            {
                "section_id": "s2",
                "status": "no_evidence",
                "verification_status": "no_evidence",
                "content": "缺口",
                "evidence": [],
            },
        ],
        report_status="ready_with_gaps",
    )
    with pytest.raises(ResearchJobStateConflictError):
        store.publish_report(
            created["job_id"], expected_revision=completed["revision"]
        )

    partially_reviewed = store.review_report(
        created["job_id"],
        expected_revision=completed["revision"],
        decisions=[{"section_id": "s1", "decision": "approved", "note": ""}],
    )
    assert partially_reviewed["review_status"] == "pending"
    with pytest.raises(ResearchJobRevisionConflictError):
        store.review_report(
            created["job_id"],
            expected_revision=completed["revision"],
            decisions=[
                {"section_id": "s2", "decision": "accepted_gap", "note": ""}
            ],
        )
    approved = store.review_report(
        created["job_id"],
        expected_revision=partially_reviewed["revision"],
        decisions=[
            {"section_id": "s2", "decision": "accepted_gap", "note": "已知限制"}
        ],
    )
    assert approved["review_status"] == "approved"

    published = store.publish_report(
        created["job_id"], expected_revision=approved["revision"]
    )
    assert published["report_status"] == "published"
    assert published["review_status"] == "published"
    assert published["published_report"]["version"] == 1
    with pytest.raises(ResearchJobStateConflictError):
        store.update_plan(
            created["job_id"],
            expected_revision=published["revision"],
            sections=[{"title": "篡改", "research_question": "允许吗？"}],
        )


def test_research_job_store_archives_rejected_report_before_regeneration(store):
    created = store.create(kb_id="kb", objective="修订报告", section_titles=["正文"])
    _advance_to_evidence_ready(store, created["job_id"])
    first = _complete_test_report(
        store,
        created["job_id"],
        [
            {
                "section_id": "s1",
                "status": "generated",
                "verification_status": "supported",
                "content": "第一版",
                "evidence": [],
            }
        ],
    )
    rejected = store.review_report(
        created["job_id"],
        expected_revision=first["revision"],
        decisions=[
            {
                "section_id": "s1",
                "decision": "changes_requested",
                "note": "补充时间范围并重新核验证据",
            }
        ],
    )
    assert rejected["review_status"] == "changes_requested"
    assert rejected["sections"][0]["revision_instruction"].startswith("补充时间")
    with pytest.raises(ResearchJobStateConflictError):
        store.review_report(
            created["job_id"],
            expected_revision=rejected["revision"],
            decisions=[
                {"section_id": "s1", "decision": "approved", "note": "撤销退回"}
            ],
        )

    generating = store.begin_report(created["job_id"])
    assert generating["status"] == "generating"
    assert generating["report_history"][0]["version"] == 1
    assert generating["regeneration_section_ids"] == ["s1"]
    second = store.complete_report(
        created["job_id"],
        report_execution_id=generating["report_execution_id"],
        result={
            "status": "ready",
            "markdown": "# 第二版\n",
            "citation_ledger": [],
            "verification_metrics": {},
            "sections": [
                {
                    "section_id": "s1",
                    "status": "generated",
                    "verification_status": "supported",
                    "content": "第二版",
                    "evidence": [],
                }
            ],
        },
    )
    assert second["report_version"] == 2
    assert second["review_status"] == "pending"
    assert second["regeneration_section_ids"] == []
    assert second["last_regenerated_section_ids"] == ["s1"]
    assert second["report_history"][0]["report"]["content"] == "# 报告\n"
