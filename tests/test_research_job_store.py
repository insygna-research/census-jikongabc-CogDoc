import copy
import json
import sqlite3
import threading

import pytest

from cogdoc.api.research_job_store import (
    ResearchJobRevisionConflictError,
    ResearchJobStateConflictError,
    ResearchJobStore,
    SqliteResearchJobStore,
    research_run_control,
)
from cogdoc.research_control import (
    ResearchBudgetExceeded,
    ResearchDeadlineExceeded,
)
from cogdoc.service.research_artifact_composer import (
    canonical_research_gap_content,
    compose_research_markdown,
)
from cogdoc.service.research_provenance import (
    RESEARCH_CONTRACT_VERSION,
    RESEARCH_PROVENANCE_VERSION,
    research_artifact_integrity_status,
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


def _provenance(generation: str = "generation-1"):
    return {
        "schema_version": RESEARCH_PROVENANCE_VERSION,
        "kb_id": "kb",
        "index_generation": generation,
        "index_build_version": "index-build-v1",
        "chunk_identity_version": "chunk-identity-v1",
        "source_versions": [{"source": "rules.pdf", "sha256": "source-sha-1"}],
        "derived_knowledge_revision": "derived-1",
        "retrieval_tuning_revision": "tuning-1",
        "research_contract_version": RESEARCH_CONTRACT_VERSION,
        "research_contract_revision": "contract-1",
        "captured_at": "2026-08-09T00:00:00+00:00",
    }


def _passed_claim_audit():
    return {
        "status": "passed",
        "counts": {
            "claim_count": 1,
            "supported": 1,
            "unsupported": 0,
            "insufficient": 0,
            "cited": 1,
        },
    }


def _passed_coverage_audit(requirement_count: int = 1):
    return {
        "status": "passed",
        "requirement_count": requirement_count,
        "covered_count": requirement_count,
        "missing_requirement_ids": [],
    }


def _grounded_section_fields(section_id: str, content: str) -> dict:
    source = f"{section_id}.pdf"
    citation = f"[{source}:P1]"
    answer = f"{content}{citation}"
    start = len(content)
    return {
        "content": answer,
        "citation_ledger": [
            {
                "evidence_id": "E001",
                "chunk_id": f"chunk:{section_id}",
                "source_type": "document",
                "source": source,
                "page": 1,
                "page_start": 1,
                "page_end": 1,
                "span_start": 0,
                "span_end": max(len(content), 1),
                "occurrences": [
                    {
                        "index": 0,
                        "answer_start": start,
                        "answer_end": start + len(citation),
                    }
                ],
            }
        ],
        "evidence": [
            {
                "chunk_id": f"chunk:{section_id}",
                "source_type": "document",
                "source": source,
                "page": 1,
                "page_start": 1,
                "page_end": 1,
                "span_start": 0,
                "span_end": max(len(content), 1),
            }
        ],
    }


def _complete_from_generation(
    store,
    generating,
    sections,
    *,
    report_status="ready",
    verification_metrics=None,
):
    plan_by_id = {section["section_id"]: section for section in generating["sections"]}
    normalized_sections = []
    for raw in sections:
        section = copy.deepcopy(raw)
        section_id = section["section_id"]
        if section.get("status") == "generated":
            plan = plan_by_id[section_id]
            requirement_ids = list(plan["evidence_requirement_ids"])
            section.update(
                _grounded_section_fields(
                    section_id, str(section.get("content") or "正文")
                )
            )
            section["verification_status"] = "supported"
            section["verification_reason_code"] = "supported"
            section["evidence_requirement_results"] = [
                {
                    "requirement_id": requirement_id,
                    "status": "supported",
                    "reason_code": "supported",
                    "evidence_count": 1,
                }
                for requirement_id in requirement_ids
            ]
            section["claim_audit"] = section.get("claim_audit") or _passed_claim_audit()
            section["coverage_audit"] = section.get(
                "coverage_audit"
            ) or _passed_coverage_audit(len(requirement_ids))
        else:
            section["content"] = canonical_research_gap_content(
                str(section.get("status") or ""),
                str(section.get("verification_status") or ""),
            )
            section["citation_ledger"] = []
        normalized_sections.append(section)

    candidate_sections = copy.deepcopy(generating["sections"])
    candidate_by_id = {section["section_id"]: section for section in candidate_sections}
    for section in normalized_sections:
        candidate_by_id[section["section_id"]].update(
            {
                **section,
                "generation_status": section.get("status", ""),
            }
        )
    markdown, ledger = compose_research_markdown(generating, candidate_sections)
    return store.complete_report(
        generating["job_id"],
        report_execution_id=generating["report_execution_id"],
        result={
            "status": report_status,
            "markdown": markdown,
            "citation_ledger": list(ledger),
            "verification_metrics": verification_metrics or {},
            "sections": normalized_sections,
        },
    )


def test_research_job_store_creates_editable_grounded_plan(store):
    row = store.create(
        kb_id="kb",
        objective="比较三份规程并提出参赛建议",
        section_titles=["报名要求", "评分规则"],
        is_local=True,
    )

    assert row["job_id"].startswith("rj_")
    assert row["status"] == "planned"
    assert row["revision"] == 1
    assert row["is_local"] is True
    assert [section["section_id"] for section in row["sections"]] == ["s1", "s2"]
    assert [section["title"] for section in row["sections"]] == [
        "报名要求",
        "评分规则",
    ]
    assert all(
        section["evidence_status"] == "unsearched" for section in row["sections"]
    )
    for section in row["sections"]:
        requirement = section["evidence_requirements"][0]
        assert (
            requirement["retrieval_query"].casefold()
            != requirement["recovery_query"].casefold()
        )
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
    assert updated["is_local"] is False
    with pytest.raises(ResearchJobRevisionConflictError):
        store.update_plan(
            created["job_id"],
            expected_revision=1,
            sections=[{"title": "过期编辑", "research_question": "不应覆盖什么？"}],
        )
    assert store.get(created["job_id"])["revision"] == 2


def test_research_job_store_enforces_atomic_plan_contract(store):
    created = store.create(kb_id="kb", objective="严格计划")

    with pytest.raises(ValueError, match="must be distinct"):
        store.update_plan(
            created["job_id"],
            expected_revision=created["revision"],
            sections=[
                {
                    "title": "资格",
                    "research_question": "资格是什么？",
                    "evidence_requirements": [
                        {
                            "question": "资格是什么？",
                            "retrieval_query": "ＡＢＣ 资格",
                            "recovery_query": "abc 资格",
                        }
                    ],
                }
            ],
        )

    with pytest.raises(ValueError, match="questions must be unique"):
        store.update_plan(
            created["job_id"],
            expected_revision=created["revision"],
            sections=[
                {
                    "title": "资格",
                    "research_question": "资格是什么？",
                    "evidence_requirements": [
                        {
                            "question": "报名 对象",
                            "retrieval_query": "报名对象",
                            "recovery_query": "参赛人员资格",
                        },
                        {
                            "question": "报名　对象",
                            "retrieval_query": "对象限制",
                            "recovery_query": "允许人员范围",
                        },
                    ],
                }
            ],
        )

    assert store.get(created["job_id"])["revision"] == created["revision"]


def test_research_job_store_filters_and_clears_by_kb(store):
    first = store.create(kb_id="a", objective="A")
    store.create(kb_id="b", objective="B")

    assert [row["job_id"] for row in store.list(kb_id="a")] == [first["job_id"]]
    store.clear_kb("a")
    assert store.get(first["job_id"]) is None
    assert len(store.list(kb_id="b")) == 1


def test_research_job_summary_page_is_keyset_stable_and_body_free(store):
    created_ids = []
    for position in range(5):
        row = store.create(kb_id="kb", objective=f"研究目标 {position}")
        created_ids.append(row["job_id"])

    first = store.list_summary_rows(kb_id="kb", limit=2)
    assert len(first) == 2
    before = first[-1]
    second = store.list_summary_rows(
        kb_id="kb",
        limit=2,
        before_updated_at=before["updated_at"],
        before_job_id=before["job_id"],
    )
    tail = second[-1]
    third = store.list_summary_rows(
        kb_id="kb",
        limit=2,
        before_updated_at=tail["updated_at"],
        before_job_id=tail["job_id"],
    )

    rows = [*first, *second, *third]
    assert {row["job_id"] for row in rows} == set(created_ids)
    assert len(rows) == len({row["job_id"] for row in rows})
    for row in rows:
        assert "sections" not in row
        assert "report" not in row
        assert "report_history" not in row
        assert "published_report" not in row
        assert "content" not in row

    started = store.start(created_ids[0], evidence_provenance=_provenance())
    refreshed = store.list_summary_rows(kb_id="kb", status="running", limit=10)
    assert [row["job_id"] for row in refreshed] == [created_ids[0]]
    assert refreshed[0]["revision"] == started["revision"]

    if isinstance(store, SqliteResearchJobStore):
        statements = []
        store._conn.set_trace_callback(statements.append)
        try:
            store.list_summary_rows(kb_id="kb", limit=2)
        finally:
            store._conn.set_trace_callback(None)
        selects = [
            statement.lower()
            for statement in statements
            if statement.lstrip().lower().startswith("select")
        ]
        assert any("select summary from research_jobs" in item for item in selects)
        assert all("select data" not in item for item in selects)


def test_sqlite_research_summary_index_backfills_legacy_table(tmp_path):
    source = ResearchJobStore(str(tmp_path / "legacy.json"))
    row = source.create(kb_id="kb", objective="旧版任务")
    database = str(tmp_path / "legacy.db")
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "CREATE TABLE research_jobs ("
            "job_id TEXT PRIMARY KEY, kb_id TEXT NOT NULL, "
            "status TEXT NOT NULL, updated_at TEXT NOT NULL, data TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO research_jobs(job_id, kb_id, status, updated_at, data) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                row["job_id"],
                row["kb_id"],
                row["status"],
                row["updated_at"],
                json.dumps(row, ensure_ascii=False),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    migrated = SqliteResearchJobStore(database)
    try:
        summaries = migrated.list_summary_rows(kb_id="kb")
        assert [item["job_id"] for item in summaries] == [row["job_id"]]
        columns = {
            item[1]
            for item in migrated._conn.execute(
                "PRAGMA table_info(research_jobs)"
            ).fetchall()
        }
        assert "summary" in columns
    finally:
        migrated.close()


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


def test_research_job_store_no_work_finishes_evidence_control(store):
    created = store.create(
        kb_id="kb",
        objective="报告失败后误启动",
        section_titles=["章节"],
    )
    running = store.start(created["job_id"])
    control = research_run_control(running, "evidence")
    _, section = store.claim_next_section(
        created["job_id"],
        running["execution_id"],
        lease_id=control["lease_id"],
    )
    store.complete_section(
        created["job_id"],
        section["section_id"],
        execution_id=running["execution_id"],
        evidence_status="missing",
        evidence=[],
        execution_metrics={},
        lease_id=control["lease_id"],
    )
    generating = store.begin_report(created["job_id"])
    report_control = research_run_control(generating, "report")
    store.fail_report(
        created["job_id"],
        report_execution_id=generating["report_execution_id"],
        error_class="RuntimeError",
        lease_id=report_control["lease_id"],
    )

    restarted = store.start(created["job_id"])
    restarted_control = research_run_control(restarted, "evidence")
    finished, claimed = store.claim_next_section(
        created["job_id"],
        restarted["execution_id"],
        lease_id=restarted_control["lease_id"],
    )
    final_control = research_run_control(finished, "evidence")

    assert claimed is None
    assert finished["status"] == "evidence_ready"
    assert final_control["control_state"] == "completed"
    assert final_control["lease_id"] == ""


def test_research_job_store_pause_wins_late_worker_failure(store):
    created = store.create(kb_id="kb", objective="暂停优先")
    running = store.start(created["job_id"])
    control = research_run_control(running, "evidence")
    _, section = store.claim_next_section(
        created["job_id"],
        running["execution_id"],
        lease_id=control["lease_id"],
    )
    paused = store.pause(created["job_id"])
    late = store.fail_section(
        created["job_id"],
        section["section_id"],
        execution_id=running["execution_id"],
        error_class="RuntimeError",
        lease_id=control["lease_id"],
    )

    assert paused["status"] == "paused"
    assert late["status"] == "paused"
    assert late["sections"][0]["status"] == "pending"
    assert research_run_control(late, "evidence")["control_state"] == "paused"


def test_research_job_store_pause_wins_late_run_failure(store):
    created = store.create(
        kb_id="kb", objective="暂停回调优先", section_titles=["章节"]
    )
    running = store.start(created["job_id"])
    control = research_run_control(running, "evidence")
    store.claim_next_section(
        created["job_id"],
        running["execution_id"],
        lease_id=control["lease_id"],
    )
    paused = store.pause(created["job_id"])

    late = store.fail_run(
        created["job_id"],
        phase="evidence",
        attempt_id=running["execution_id"],
        lease_id=control["lease_id"],
        error_class="LateCallbackError",
    )

    assert late == paused
    assert late["status"] == "paused"
    assert late["sections"][0]["status"] == "running"
    assert research_run_control(late, "evidence")["control_state"] == "paused"


def test_research_job_store_fail_run_rejects_empty_lease_after_completion(store):
    created = store.create(
        kb_id="kb", objective="完成态租约围栏", section_titles=["章节"]
    )
    running = store.start(created["job_id"])
    control = research_run_control(running, "evidence")
    section = store.claim_next_section(
        created["job_id"],
        running["execution_id"],
        lease_id=control["lease_id"],
    )[1]
    completed = store.complete_section(
        created["job_id"],
        section["section_id"],
        execution_id=running["execution_id"],
        evidence_status="missing",
        evidence=[],
        execution_metrics={},
        lease_id=control["lease_id"],
    )

    late = store.fail_run(
        created["job_id"],
        phase="evidence",
        attempt_id=running["execution_id"],
        lease_id="",
        error_class="InjectedFailure",
    )

    assert late == completed
    assert late["status"] == "evidence_ready"
    assert research_run_control(late, "evidence")["control_state"] == "completed"


def test_research_job_store_reconcile_reports_exact_termination_counts(store):
    normal = store.create(kb_id="kb", objective="普通孤儿")
    expired = store.create(kb_id="kb", objective="过期孤儿")
    store.start(normal["job_id"])
    store.start(expired["job_id"], deadline_at="2000-01-01T00:00:00+00:00")

    outcomes = store.reconcile_running_outcomes()

    assert outcomes == {
        "service_restarted": 1,
        "deadline_exceeded": 1,
        "shutdown": 0,
    }
    assert store.get(normal["job_id"])["status"] == "paused"
    assert store.get(expired["job_id"])["error"] == "ResearchDeadlineExceeded"


def test_research_job_store_resume_requeues_orphaned_running_section(store):
    created = store.create(
        kb_id="kb", objective="恢复遗留章节", section_titles=["章节"]
    )
    running = store.start(created["job_id"])
    _, claimed = store.claim_next_section(created["job_id"], running["execution_id"])
    assert claimed["status"] == "running"
    store.pause(created["job_id"])

    resumed = store.resume(created["job_id"])
    assert resumed["execution_id"] == running["execution_id"]
    assert resumed["sections"][0]["status"] == "pending"
    _, reclaimed = store.claim_next_section(created["job_id"], resumed["execution_id"])
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


def test_research_job_store_resume_rotates_worker_lease_and_rejects_old_commit(store):
    created = store.create(kb_id="kb", objective="恢复租约", section_titles=["章节"])
    running = store.start(created["job_id"])
    first_control = research_run_control(running, "evidence")
    _, claimed = store.claim_next_section(
        created["job_id"],
        running["execution_id"],
        lease_id=first_control["lease_id"],
    )
    assert claimed["section_id"] == "s1"

    store.pause(created["job_id"])
    resumed = store.resume(created["job_id"])
    resumed_control = research_run_control(resumed, "evidence")

    assert resumed["execution_id"] == running["execution_id"]
    assert resumed_control["lease_id"] != first_control["lease_id"]
    stale = store.complete_section(
        created["job_id"],
        "s1",
        execution_id=running["execution_id"],
        lease_id=first_control["lease_id"],
        evidence_status="partial",
        evidence=[{"chunk_id": "stale"}],
        execution_metrics={},
    )
    assert stale["sections"][0]["status"] == "pending"

    _, reclaimed = store.claim_next_section(
        created["job_id"],
        resumed["execution_id"],
        lease_id=resumed_control["lease_id"],
    )
    assert reclaimed["section_id"] == "s1"


def test_research_job_store_deadline_and_budget_are_durable_fail_closed(store):
    deadline_job = store.create(kb_id="kb", objective="截止时间")
    running = store.start(
        deadline_job["job_id"],
        deadline_at="2026-08-10T00:01:00+00:00",
        now="2026-08-10T00:00:00+00:00",
    )
    control = research_run_control(running, "evidence")
    with pytest.raises(ResearchDeadlineExceeded):
        store.reserve_research_resources(
            deadline_job["job_id"],
            phase="evidence",
            attempt_id=control["attempt_id"],
            lease_id=control["lease_id"],
            now="2026-08-10T00:02:00+00:00",
        )
    expired = store.get(deadline_job["job_id"])
    assert expired["status"] == "failed"
    assert expired["error"] == "ResearchDeadlineExceeded"

    budget_job = store.create(kb_id="kb", objective="资源预算")
    running = store.start(
        budget_job["job_id"],
        resource_limits={"retrieval_queries": 1},
    )
    control = research_run_control(running, "evidence")
    store.reserve_research_resources(
        budget_job["job_id"],
        phase="evidence",
        attempt_id=control["attempt_id"],
        lease_id=control["lease_id"],
        costs={"retrieval_queries": 1},
    )
    with pytest.raises(ResearchBudgetExceeded):
        store.reserve_research_resources(
            budget_job["job_id"],
            phase="evidence",
            attempt_id=control["attempt_id"],
            lease_id=control["lease_id"],
            costs={"retrieval_queries": 1},
        )
    exhausted = store.get(budget_job["job_id"])
    assert exhausted["status"] == "failed"
    assert exhausted["error"] == "ResearchBudgetExceeded"


def _advance_to_evidence_ready(store, job_id, *, evidence_provenance=None):
    running = store.start(
        job_id,
        evidence_provenance=(
            _provenance() if evidence_provenance is None else evidence_provenance
        ),
    )
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
    completed = _complete_from_generation(
        store,
        generating,
        [
            {
                "section_id": "s1",
                "status": "generated",
                "verification_status": "supported",
                "verification_reason_code": "supported",
                "content": "结论。",
                "error": "",
            }
        ],
        verification_metrics={"supported_count": 1},
    )

    assert completed["status"] == "completed"
    assert completed["report_status"] == "ready"
    assert completed["report"]["format"] == "markdown"
    assert completed["sections"][0]["verification_status"] == "supported"
    assert completed["sections"][0]["generation_status"] == "generated"


def test_research_job_store_rejects_generated_section_without_coverage_audit(store):
    created = store.create(kb_id="kb", objective="覆盖门禁", section_titles=["证据"])
    _advance_to_evidence_ready(store, created["job_id"])
    generating = store.begin_report(created["job_id"])

    with pytest.raises(
        ResearchJobStateConflictError,
        match="requires a passed requirement coverage audit",
    ):
        store.complete_report(
            created["job_id"],
            report_execution_id=generating["report_execution_id"],
            result={
                "status": "ready",
                "markdown": "# 报告\n",
                "citation_ledger": [],
                "verification_metrics": {},
                "sections": [
                    {
                        "section_id": "s1",
                        "status": "generated",
                        "verification_status": "supported",
                        "content": "正文",
                        "claim_audit": _passed_claim_audit(),
                        "coverage_audit": {"status": "failed"},
                    }
                ],
            },
        )

    assert store.get(created["job_id"])["status"] == "generating"


def test_research_job_store_recovers_orphaned_report_generation(store):
    created = store.create(kb_id="kb", objective="恢复报告", section_titles=["证据"])
    _advance_to_evidence_ready(store, created["job_id"])
    store.begin_report(created["job_id"])

    assert store.reconcile_running() == 1
    recovered = store.get(created["job_id"])
    assert recovered["status"] == "evidence_ready"
    assert recovered["report_status"] == "failed"
    assert recovered["error"] == "service_restarted"


def test_research_job_store_cancel_invalidates_report_lease_and_stale_commit(store):
    created = store.create(kb_id="kb", objective="取消报告", section_titles=["证据"])
    _advance_to_evidence_ready(store, created["job_id"])
    generating = store.begin_report(created["job_id"])
    control = research_run_control(generating, "report")

    cancelled = store.cancel(created["job_id"])

    assert cancelled["status"] == "cancelled"
    assert cancelled["execution_id"] == ""
    assert cancelled["report_execution_id"] == ""
    assert cancelled["report_status"] == "failed"
    assert research_run_control(cancelled, "report")["control_state"] == "cancelled"
    stale = store.fail_report(
        created["job_id"],
        report_execution_id=generating["report_execution_id"],
        lease_id=control["lease_id"],
        error_class="LateFailure",
    )
    assert stale == cancelled


def _complete_test_report(store, job_id, sections, *, report_status="ready"):
    generating = store.begin_report(job_id)
    return _complete_from_generation(
        store,
        generating,
        sections,
        report_status=report_status,
    )


@pytest.mark.parametrize(
    ("generation_status", "declared_status", "expected_status"),
    [
        ("generated", "published", "ready"),
        ("no_evidence", "ready", "ready_with_gaps"),
    ],
)
def test_research_job_store_derives_report_status_from_sections(
    store,
    generation_status,
    declared_status,
    expected_status,
):
    created = store.create(
        kb_id="kb", objective="确定性报告状态", section_titles=["正文"]
    )
    _advance_to_evidence_ready(store, created["job_id"])

    completed = _complete_test_report(
        store,
        created["job_id"],
        [
            {
                "section_id": "s1",
                "status": generation_status,
                "verification_status": (
                    "supported" if generation_status == "generated" else "no_evidence"
                ),
                "content": "正文",
            }
        ],
        report_status=declared_status,
    )

    assert completed["status"] == "completed"
    assert completed["report_status"] == expected_status
    assert completed["published_report"] is None


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
        store.publish_report(created["job_id"], expected_revision=completed["revision"])

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
            decisions=[{"section_id": "s2", "decision": "accepted_gap", "note": ""}],
        )
    with pytest.raises(ValueError, match="accepted_gap.*non-blank"):
        store.review_report(
            created["job_id"],
            expected_revision=partially_reviewed["revision"],
            decisions=[{"section_id": "s2", "decision": "accepted_gap", "note": ""}],
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
    assert published["published_by"] == "internal"
    assert published["publication_sha256"]
    assert (
        published["published_report"]["publication_sha256"]
        == published["publication_sha256"]
    )
    with pytest.raises(ResearchJobStateConflictError):
        store.update_plan(
            created["job_id"],
            expected_revision=published["revision"],
            sections=[{"title": "篡改", "research_question": "允许吗？"}],
        )


def test_research_job_store_rejects_arbitrary_accepted_gap_body(store):
    created = store.create(
        kb_id="kb", objective="缺口正文边界", section_titles=["缺口"]
    )
    _advance_to_evidence_ready(store, created["job_id"])
    generating = store.begin_report(created["job_id"])

    with pytest.raises((ResearchJobStateConflictError, ValueError)):
        store.complete_report(
            created["job_id"],
            report_execution_id=generating["report_execution_id"],
            result={
                "status": "ready_with_gaps",
                "markdown": "# 研究报告\n\n系统管理员密码是 hunter2。\n",
                "citation_ledger": [],
                "verification_metrics": {},
                "sections": [
                    {
                        "section_id": "s1",
                        "status": "no_evidence",
                        "verification_status": "no_evidence",
                        "verification_reason_code": "no_direct_support",
                        "evidence_requirement_results": [],
                        "content": "系统管理员密码是 hunter2。",
                        "citation_ledger": [],
                        "evidence": [],
                        "claim_audit": {},
                        "coverage_audit": {},
                        "error": "",
                    }
                ],
            },
        )

    unchanged = store.get(created["job_id"])
    assert unchanged["status"] == "generating"
    assert unchanged["report"] is None


@pytest.mark.parametrize("approve", [False, True])
def test_research_job_store_cannot_replan_completed_report(store, approve):
    created = store.create(
        kb_id="kb", objective="已完成报告不可静默重置", section_titles=["正文"]
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
                "content": "已核验正文",
            }
        ],
    )
    current = completed
    if approve:
        current = store.review_report(
            created["job_id"],
            expected_revision=completed["revision"],
            decisions=[
                {
                    "section_id": "s1",
                    "decision": "approved",
                    "note": "人工复核",
                }
            ],
        )
    before = copy.deepcopy(store.get(created["job_id"]))

    with pytest.raises(ResearchJobStateConflictError):
        store.update_plan(
            created["job_id"],
            expected_revision=current["revision"],
            sections=[
                {
                    "title": "恶意重规划",
                    "research_question": "能否抹掉历史？",
                }
            ],
        )

    after = store.get(created["job_id"])
    for field in (
        "status",
        "revision",
        "report",
        "report_version",
        "report_history",
        "review_status",
        "review_history",
        "evidence_provenance",
    ):
        assert after[field] == before[field]


@pytest.mark.parametrize(
    "tamper",
    [
        "section_reviewed_at",
        "section_note",
        "history_reviewed_at",
        "history_note",
        "history_decision",
        "history_missing",
        "semantic_gate",
    ],
)
def test_research_job_store_publish_requires_exact_current_review_trace(store, tamper):
    created = store.create(
        kb_id="kb", objective="审阅轨迹绑定", section_titles=["正文"]
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
                "content": "可信正文",
            }
        ],
    )
    reviewed = store.review_report(
        created["job_id"],
        expected_revision=completed["revision"],
        decisions=[{"section_id": "s1", "decision": "approved", "note": "人工核验"}],
    )
    corrupted = store.get(created["job_id"])
    if tamper == "section_reviewed_at":
        corrupted["sections"][0]["reviewed_at"] = "2099-01-01T00:00:00Z"
    elif tamper == "section_note":
        corrupted["sections"][0]["review_note"] = "另一条备注"
    elif tamper == "history_reviewed_at":
        corrupted["review_history"][-1]["reviewed_at"] = "2099-01-01T00:00:00Z"
    elif tamper == "history_note":
        corrupted["review_history"][-1]["decisions"][0]["note"] = "另一条备注"
    elif tamper == "history_decision":
        corrupted["review_history"][-1]["decisions"][0]["decision"] = (
            "changes_requested"
        )
    elif tamper == "history_missing":
        corrupted["review_history"] = []
    else:
        corrupted["sections"][0]["claim_audit"]["counts"]["supported"] = 0
    store.import_records([corrupted])

    with pytest.raises(ResearchJobStateConflictError):
        store.publish_report(created["job_id"], expected_revision=reviewed["revision"])

    persisted = store.get(created["job_id"])
    assert persisted["review_status"] == "approved"
    assert persisted["published_report"] is None


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
    second = _complete_from_generation(
        store,
        generating,
        [
            {
                "section_id": "s1",
                "status": "generated",
                "verification_status": "supported",
                "content": "第二版",
            }
        ],
    )
    assert second["report_version"] == 2
    assert second["review_status"] == "pending"
    assert second["regeneration_section_ids"] == []
    assert second["last_regenerated_section_ids"] == ["s1"]
    assert "第一版" in second["report_history"][0]["report"]["content"]


def test_research_job_store_refresh_resets_evidence_and_archives_report(store):
    created = store.create(kb_id="kb", objective="刷新证据", section_titles=["正文"])
    evidence_ready = _advance_to_evidence_ready(
        store,
        created["job_id"],
        evidence_provenance=_provenance("generation-1"),
    )
    completed = _complete_test_report(
        store,
        created["job_id"],
        [
            {
                "section_id": "s1",
                "status": "generated",
                "verification_status": "supported",
                "verification_reason_code": "supported",
                "content": "第一版正文",
                "evidence_requirement_results": [
                    {"requirement_id": "s1:r1", "status": "supported"}
                ],
                "citation_ledger": [{"evidence_id": "E001"}],
                "claim_audit": _passed_claim_audit(),
                "coverage_audit": _passed_coverage_audit(),
                "evidence": [{"chunk_id": "c1"}],
            }
        ],
    )
    assert completed["sections"][0]["evidence"]
    assert completed["report"] is not None
    reviewed = store.review_report(
        created["job_id"],
        expected_revision=completed["revision"],
        decisions=[{"section_id": "s1", "decision": "approved", "note": "已核验"}],
    )
    assert len(reviewed["review_history"]) == 1

    refreshed = store.refresh_evidence(
        created["job_id"],
        evidence_provenance=_provenance("generation-2"),
    )

    assert refreshed["status"] == "running"
    assert refreshed["execution_id"] != evidence_ready["execution_id"]
    assert refreshed["evidence_provenance"] == _provenance("generation-2")
    assert refreshed["report"] is None
    assert refreshed["report_status"] == "not_started"
    assert refreshed["report_version"] == 1
    assert len(refreshed["report_history"]) == 1
    assert [event["result"] for event in refreshed["review_history"]] == [
        "approved",
        "evidence_refreshed",
    ]
    assert refreshed["review_history"][0]["decisions"][0]["note"] == "已核验"
    assert "第一版正文" in refreshed["report_history"][0]["report"]["content"]
    section = refreshed["sections"][0]
    assert section["status"] == "pending"
    assert section["evidence_status"] == "unsearched"
    assert section["evidence_requirement_results"] == []
    assert section["evidence"] == []
    assert set(section["execution_metrics"]) == {"_research_control"}
    assert section["citation_ledger"] == []
    assert section["claim_audit"] == {}
    assert section["coverage_audit"] == {}
    assert section["verification_status"] == ""
    assert section["generation_status"] == ""
    assert section["content"] == ""


def test_research_job_store_rejects_tampered_report_artifact(store):
    created = store.create(kb_id="kb", objective="发布完整性", section_titles=["正文"])
    _advance_to_evidence_ready(
        store,
        created["job_id"],
        evidence_provenance=_provenance(),
    )
    completed = _complete_test_report(
        store,
        created["job_id"],
        [
            {
                "section_id": "s1",
                "status": "generated",
                "verification_status": "supported",
                "content": "可信正文",
                "evidence": [],
            }
        ],
    )
    approved = store.review_report(
        created["job_id"],
        expected_revision=completed["revision"],
        decisions=[{"section_id": "s1", "decision": "approved", "note": ""}],
    )
    tampered = store.get(created["job_id"])
    tampered["report"]["content"] += "\n未经审计的篡改"
    assert store.import_records([tampered]) == {"imported": 1, "skipped": 0}

    with pytest.raises(
        ResearchJobStateConflictError,
        match="artifact integrity check failed",
    ):
        store.publish_report(created["job_id"], expected_revision=approved["revision"])

    persisted = store.get(created["job_id"])
    assert persisted["review_status"] == "approved"
    assert persisted["published_report"] is None


@pytest.mark.parametrize("transition", ["review", "publish"])
def test_research_job_store_rejects_current_provenance_projection_drift(
    store, transition
):
    created = store.create(
        kb_id="kb", objective="来源投影完整性", section_titles=["正文"]
    )
    _advance_to_evidence_ready(
        store,
        created["job_id"],
        evidence_provenance=_provenance(),
    )
    completed = _complete_test_report(
        store,
        created["job_id"],
        [
            {
                "section_id": "s1",
                "status": "generated",
                "verification_status": "supported",
                "content": "可信正文",
            }
        ],
    )
    current = completed
    if transition == "publish":
        current = store.review_report(
            created["job_id"],
            expected_revision=completed["revision"],
            decisions=[{"section_id": "s1", "decision": "approved", "note": ""}],
        )

    tampered = store.get(created["job_id"])
    tampered["evidence_provenance"]["index_generation"] = "tampered-generation"
    assert research_artifact_integrity_status(tampered["report"]) == "verified"
    assert tampered["report"]["provenance"] != tampered["evidence_provenance"]
    assert store.import_records([tampered]) == {"imported": 1, "skipped": 0}

    with pytest.raises(ResearchJobStateConflictError):
        if transition == "review":
            store.review_report(
                created["job_id"],
                expected_revision=current["revision"],
                decisions=[{"section_id": "s1", "decision": "approved", "note": ""}],
            )
        else:
            store.publish_report(
                created["job_id"], expected_revision=current["revision"]
            )

    persisted = store.get(created["job_id"])
    assert persisted["published_report"] is None
    assert persisted["review_status"] == (
        "approved" if transition == "publish" else "pending"
    )


def test_research_job_store_rejects_section_audit_drift_from_artifact(store):
    created = store.create(kb_id="kb", objective="审计承诺", section_titles=["正文"])
    _advance_to_evidence_ready(store, created["job_id"])
    completed = _complete_test_report(
        store,
        created["job_id"],
        [
            {
                "section_id": "s1",
                "status": "generated",
                "verification_status": "supported",
                "content": "可信正文",
                "evidence": [{"chunk_id": "c1", "text_hash": "text-sha-1"}],
            }
        ],
    )
    approved = store.review_report(
        created["job_id"],
        expected_revision=completed["revision"],
        decisions=[{"section_id": "s1", "decision": "approved", "note": ""}],
    )
    tampered = store.get(created["job_id"])
    tampered["sections"][0]["coverage_audit"]["reason_code"] = "tampered"
    tampered["sections"][0]["evidence"][0]["text_hash"] = "text-sha-2"
    assert store.import_records([tampered]) == {"imported": 1, "skipped": 0}

    with pytest.raises(
        ResearchJobStateConflictError,
        match="verification state does not match",
    ):
        store.publish_report(created["job_id"], expected_revision=approved["revision"])

    assert store.get(created["job_id"])["published_report"] is None


def test_research_job_store_preserves_selective_regeneration_scope_after_crash(
    store,
):
    created = store.create(
        kb_id="kb",
        objective="选择性重生成",
        section_titles=["保留章节", "重写章节"],
    )
    _advance_to_evidence_ready(store, created["job_id"])
    first = _complete_test_report(
        store,
        created["job_id"],
        [
            {
                "section_id": "s1",
                "status": "generated",
                "verification_status": "supported",
                "content": "保留",
                "evidence": [],
            },
            {
                "section_id": "s2",
                "status": "generated",
                "verification_status": "supported",
                "content": "待重写",
                "evidence": [],
            },
        ],
    )
    reviewed = store.review_report(
        created["job_id"],
        expected_revision=first["revision"],
        decisions=[
            {"section_id": "s1", "decision": "approved", "note": ""},
            {
                "section_id": "s2",
                "decision": "changes_requested",
                "note": "补充边界条件",
            },
        ],
    )
    assert reviewed["review_status"] == "changes_requested"

    first_attempt = store.begin_report(created["job_id"])
    assert first_attempt["regeneration_section_ids"] == ["s2"]
    assert store.reconcile_running() == 1
    recovered = store.get(created["job_id"])
    assert recovered["status"] == "evidence_ready"
    assert recovered["regeneration_section_ids"] == ["s2"]

    retry = store.begin_report(created["job_id"])
    assert retry["regeneration_section_ids"] == ["s2"]
    assert len(retry["report_history"]) == 1
    second = _complete_from_generation(
        store,
        retry,
        [
            {
                "section_id": "s2",
                "status": "generated",
                "verification_status": "supported",
                "content": "已重写",
            }
        ],
    )
    assert second["last_regenerated_section_ids"] == ["s2"]


def test_research_job_store_rejects_out_of_scope_selective_mutation(store):
    created = store.create(
        kb_id="kb",
        objective="选择性重生成权限边界",
        section_titles=["保留章节", "重写章节"],
    )
    _advance_to_evidence_ready(store, created["job_id"])
    first = _complete_test_report(
        store,
        created["job_id"],
        [
            {
                "section_id": "s1",
                "status": "generated",
                "verification_status": "supported",
                "content": "可信保留正文",
            },
            {
                "section_id": "s2",
                "status": "generated",
                "verification_status": "supported",
                "content": "待重写正文",
            },
        ],
    )
    store.review_report(
        created["job_id"],
        expected_revision=first["revision"],
        decisions=[
            {"section_id": "s1", "decision": "approved", "note": "人工确认"},
            {
                "section_id": "s2",
                "decision": "changes_requested",
                "note": "补充证据",
            },
        ],
    )
    generating = store.begin_report(created["job_id"])
    assert generating["regeneration_section_ids"] == ["s2"]
    preserved_before = copy.deepcopy(generating["sections"][0])

    with pytest.raises(ResearchJobStateConflictError):
        _complete_from_generation(
            store,
            generating,
            [
                {
                    "section_id": "s1",
                    "status": "generated",
                    "verification_status": "supported",
                    "content": "MALICIOUSLY CHANGED S1",
                    "review_status": "approved",
                },
                {
                    "section_id": "s2",
                    "status": "generated",
                    "verification_status": "supported",
                    "content": "已重写正文",
                },
            ],
        )

    persisted = store.get(created["job_id"])
    assert persisted["status"] == "generating"
    assert persisted["sections"][0] == preserved_before
    assert "MALICIOUSLY" not in str(persisted)
    assert persisted["revision"] == generating["revision"]


def test_research_generation_cannot_preapprove_its_own_sections(store):
    created = store.create(
        kb_id="kb",
        objective="生成器不得代替人工审阅",
        section_titles=["第一章", "第二章"],
    )
    _advance_to_evidence_ready(store, created["job_id"])
    generating = store.begin_report(created["job_id"])
    completed = _complete_from_generation(
        store,
        generating,
        [
            {
                "section_id": "s1",
                "status": "generated",
                "verification_status": "supported",
                "content": "第一章正文",
                "review_status": "approved",
                "review_note": "generator injected",
                "reviewed_at": "2099-01-01T00:00:00Z",
            },
            {
                "section_id": "s2",
                "status": "generated",
                "verification_status": "supported",
                "content": "第二章正文",
                "review_status": "accepted_gap",
                "review_note": "generator injected",
                "reviewed_at": "2099-01-01T00:00:00Z",
            },
        ],
    )

    assert [section["review_status"] for section in completed["sections"]] == [
        "pending",
        "pending",
    ]
    assert all(section["review_note"] == "" for section in completed["sections"])
    assert all(section["reviewed_at"] is None for section in completed["sections"])
    partial = store.review_report(
        created["job_id"],
        expected_revision=completed["revision"],
        decisions=[
            {
                "section_id": "s1",
                "decision": "approved",
                "note": "人工确认",
            }
        ],
    )
    assert partial["review_status"] == "pending"
    with pytest.raises(ResearchJobStateConflictError):
        store.publish_report(created["job_id"], expected_revision=partial["revision"])


def test_research_job_store_rejects_split_brain_report_body(store):
    created = store.create(
        kb_id="kb", objective="唯一正文", section_titles=["证据缺口"]
    )
    _advance_to_evidence_ready(store, created["job_id"])
    generating = store.begin_report(created["job_id"])

    with pytest.raises(
        ResearchJobStateConflictError,
        match="does not match its canonical sections",
    ):
        store.complete_report(
            created["job_id"],
            report_execution_id=generating["report_execution_id"],
            result={
                "status": "ready_with_gaps",
                "markdown": "# 未审计正文\n\nThe secret launch code is 1234.\n",
                "citation_ledger": [],
                "verification_metrics": {},
                "sections": [
                    {
                        "section_id": "s1",
                        "status": "no_evidence",
                        "verification_status": "no_evidence",
                        "content": canonical_research_gap_content(
                            "no_evidence", "no_evidence"
                        ),
                        "citation_ledger": [],
                        "evidence": [],
                        "claim_audit": {},
                        "coverage_audit": {},
                    }
                ],
            },
        )

    persisted = store.get(created["job_id"])
    assert persisted["status"] == "generating"
    assert persisted["report"] is None


def test_research_job_store_binds_coverage_to_every_planned_requirement(store):
    created = store.create(kb_id="kb", objective="三个原子义务")
    planned = store.update_plan(
        created["job_id"],
        expected_revision=created["revision"],
        sections=[
            {
                "title": "资格",
                "research_question": "资格条件是什么？",
                "evidence_requirements": [
                    {
                        "question": f"条件 {position} 是什么？",
                        "retrieval_query": f"条件 {position} 主查询",
                        "recovery_query": f"条件 {position} 恢复查询",
                    }
                    for position in range(1, 4)
                ],
            }
        ],
    )
    _advance_to_evidence_ready(store, created["job_id"])
    generating = store.begin_report(created["job_id"])
    grounded = _grounded_section_fields("s1", "只覆盖一个条件。")

    with pytest.raises(
        ResearchJobStateConflictError,
        match="complete atomic requirement plan",
    ):
        store.complete_report(
            created["job_id"],
            report_execution_id=generating["report_execution_id"],
            result={
                "status": "ready",
                "markdown": "# 不应发布\n",
                "citation_ledger": [],
                "verification_metrics": {},
                "sections": [
                    {
                        "section_id": "s1",
                        "status": "generated",
                        "verification_status": "supported",
                        "verification_reason_code": "supported",
                        **grounded,
                        "evidence_requirement_results": [
                            {
                                "requirement_id": planned["sections"][0][
                                    "evidence_requirement_ids"
                                ][0],
                                "status": "supported",
                                "reason_code": "supported",
                                "evidence_count": 1,
                            }
                        ],
                        "claim_audit": _passed_claim_audit(),
                        "coverage_audit": _passed_coverage_audit(1),
                    }
                ],
            },
        )

    assert store.get(created["job_id"])["status"] == "generating"


def test_research_job_store_rechecks_atomic_coverage_before_publish(store):
    created = store.create(kb_id="kb", objective="发布前重审")
    planned = store.update_plan(
        created["job_id"],
        expected_revision=created["revision"],
        sections=[
            {
                "title": "资格",
                "research_question": "全部条件是什么？",
                "evidence_requirements": [
                    {
                        "question": f"发布条件 {position}？",
                        "retrieval_query": f"发布条件 {position} 主查询",
                        "recovery_query": f"发布条件 {position} 恢复查询",
                    }
                    for position in range(1, 4)
                ],
            }
        ],
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
                "content": "三个条件均有证据。",
            }
        ],
    )
    approved = store.review_report(
        created["job_id"],
        expected_revision=completed["revision"],
        decisions=[{"section_id": "s1", "decision": "approved", "note": ""}],
    )
    tampered = store.get(created["job_id"])
    assert len(planned["sections"][0]["evidence_requirement_ids"]) == 3
    tampered["sections"][0]["evidence_requirement_results"] = tampered["sections"][0][
        "evidence_requirement_results"
    ][:1]
    tampered["sections"][0]["coverage_audit"].update(
        {"requirement_count": 1, "covered_count": 1}
    )
    store.import_records([tampered])

    with pytest.raises(
        ResearchJobStateConflictError,
        match="atomic requirement plan",
    ):
        store.publish_report(created["job_id"], expected_revision=approved["revision"])


def test_sqlite_reconcile_reads_after_acquiring_write_transaction(tmp_path):
    database = str(tmp_path / "reconcile-race.db")
    reconciler = SqliteResearchJobStore(database)
    worker = SqliteResearchJobStore(database)
    try:
        created = reconciler.create(kb_id="kb", objective="恢复竞态")
        running = reconciler.start(created["job_id"])
        reconciler.claim_next_section(created["job_id"], running["execution_id"])
        select_reached = threading.Event()
        release_select = threading.Event()

        def trace(statement: str) -> None:
            if statement.startswith("SELECT job_id, data FROM research_jobs"):
                select_reached.set()
                assert release_select.wait(timeout=5)

        reconciler._conn.set_trace_callback(trace)
        reconcile_thread = threading.Thread(target=reconciler.reconcile_running)
        reconcile_thread.start()
        assert select_reached.wait(timeout=5)

        result: list[dict] = []
        worker_entered = threading.Event()
        worker_finished = threading.Event()

        def complete() -> None:
            worker_entered.set()
            result.append(
                worker.complete_section(
                    created["job_id"],
                    "s1",
                    execution_id=running["execution_id"],
                    evidence_status="partial",
                    evidence=[{"chunk_id": "c1", "text_preview": "证据"}],
                    execution_metrics={"candidate_count": 1},
                )
            )
            worker_finished.set()

        worker_thread = threading.Thread(target=complete)
        worker_thread.start()
        assert worker_entered.wait(timeout=5)
        # The recovery transaction already owns the SQLite write reservation.
        # A second connection therefore cannot commit a newer state between
        # recovery's SELECT and UPSERT.
        assert not worker_finished.wait(timeout=0.2)
        release_select.set()
        reconcile_thread.join(timeout=5)
        worker_thread.join(timeout=5)
        assert not reconcile_thread.is_alive()
        assert not worker_thread.is_alive()
        assert result[0]["status"] == "paused"
        persisted = reconciler.get(created["job_id"])
        assert persisted["status"] == "paused"
        assert persisted["sections"][0]["status"] == "pending"
    finally:
        reconciler.close()
        worker.close()
