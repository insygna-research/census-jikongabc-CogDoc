from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from threading import RLock
from typing import Any
from uuid import uuid4

from cogdoc.api.persistence import connect_sqlite
from cogdoc.api.time_utils import now_iso
from cogdoc.config.settings import get_settings


class ResearchJobRevisionConflictError(ValueError):
    """The caller edited an obsolete research-job revision."""


class ResearchJobStateConflictError(ValueError):
    """The requested execution transition is invalid for the current state."""


def build_research_plan(
    objective: str,
    section_titles: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build a safe, editable initial plan without inventing domain facts."""

    titles = [" ".join(str(title).split()) for title in (section_titles or [])]
    titles = [title for title in titles if title]
    if not titles:
        titles = [
            "目标与范围",
            "关键事实与证据",
            "综合分析",
            "风险与局限",
            "结论与建议",
        ]
    normalized_objective = " ".join(objective.split())
    return [
        {
            "section_id": f"s{position}",
            "position": position,
            "title": title,
            "research_question": (
                f"围绕“{normalized_objective}”，需要查明哪些与“{title}”"
                "直接相关且可由知识库验证的信息？"
            ),
            "status": "pending",
            "evidence_status": "unsearched",
            "evidence_requirement_ids": [],
            "evidence": [],
            "execution_metrics": {},
            "citation_ledger": [],
            "revision_instruction": "",
            "review_status": "not_started",
            "review_note": "",
            "reviewed_at": None,
            "error": "",
        }
        for position, title in enumerate(titles, start=1)
    ]


def _clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _normalize_sections(sections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for position, section in enumerate(sections, start=1):
        title = " ".join(str(section.get("title") or "").split())
        question = " ".join(str(section.get("research_question") or "").split())
        if not title or not question:
            raise ValueError("research sections require non-blank title and research_question")
        title_key = title.casefold()
        if title_key in seen_titles:
            raise ValueError(f"duplicate research section title: {title}")
        seen_titles.add(title_key)
        normalized.append(
            {
                "section_id": f"s{position}",
                "position": position,
                "title": title,
                "research_question": question,
                "status": "pending",
                "evidence_status": "unsearched",
                "evidence_requirement_ids": [],
                "evidence": [],
                "execution_metrics": {},
                "citation_ledger": [],
                "revision_instruction": "",
                "review_status": "not_started",
                "review_note": "",
                "reviewed_at": None,
                "error": "",
            }
        )
    if not normalized:
        raise ValueError("research plan requires at least one section")
    return normalized


def _touch(row: dict[str, Any]) -> dict[str, Any]:
    row["revision"] = int(row.get("revision") or 0) + 1
    row["updated_at"] = now_iso()
    return row


def _start_job(row: dict[str, Any]) -> dict[str, Any]:
    status = str(row.get("status") or "planned")
    if status == "running":
        return row
    if status not in {"planned", "paused", "failed"}:
        raise ResearchJobStateConflictError(
            f"research job cannot start from status {status}"
        )
    if status in {"planned", "failed"}:
        row["execution_id"] = uuid4().hex
        row["report_execution_id"] = ""
        row["report_status"] = "not_started"
        row["report"] = None
        row["published_report"] = None
        row["review_status"] = "not_started"
    # A paused process may have lost its worker after a section was claimed.
    # Resume must make every non-terminal claim eligible again or the queue can
    # retain a permanent ``running`` section that no worker will ever claim.
    for section in row.get("sections") or []:
        if section.get("status") in {"running", "failed"}:
            section["status"] = "pending"
            section["error"] = ""
    row["status"] = "running"
    row["started_at"] = row.get("started_at") or now_iso()
    row["evidence_completed_at"] = None
    row["error"] = ""
    return _touch(row)


def _resume_job(row: dict[str, Any]) -> dict[str, Any]:
    status = str(row.get("status") or "")
    if status == "running":
        return row
    if status != "paused":
        raise ResearchJobStateConflictError(
            f"research job cannot resume from status {status}"
        )
    return _start_job(row)


def _begin_report(row: dict[str, Any]) -> dict[str, Any]:
    status = str(row.get("status") or "")
    retrying_failed_report = (
        status == "failed" and row.get("report_status") == "failed"
    )
    regenerating_reviewed_report = (
        status == "completed" and row.get("review_status") == "changes_requested"
    )
    if status == "generating":
        return row
    if (
        status != "evidence_ready"
        and not retrying_failed_report
        and not regenerating_reviewed_report
    ):
        raise ResearchJobStateConflictError(
            f"research report cannot generate from status {status}"
        )
    current_report = row.get("report")
    if regenerating_reviewed_report and isinstance(current_report, Mapping):
        history = list(row.get("report_history") or [])
        history.append(
            {
                "version": int(row.get("report_version") or 1),
                "report_status": str(row.get("report_status") or "ready"),
                "review_status": str(row.get("review_status") or "not_started"),
                "archived_at": now_iso(),
                "report": dict(_clone(current_report)),
            }
        )
        row["report_history"] = history[-10:]
        row["regeneration_section_ids"] = [
            str(section.get("section_id") or "")
            for section in row.get("sections") or []
            if section.get("review_status") == "changes_requested"
        ]
    elif status == "evidence_ready":
        row["regeneration_section_ids"] = []
    row["status"] = "generating"
    row["report_status"] = "generating"
    row["report_execution_id"] = uuid4().hex
    row["report"] = None
    row["published_report"] = None
    row["review_status"] = "not_started"
    row["error"] = ""
    return _touch(row)


def _complete_report(
    row: dict[str, Any],
    *,
    report_execution_id: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        row.get("status") != "generating"
        or row.get("report_execution_id") != report_execution_id
    ):
        return row
    section_results = {
        str(section.get("section_id") or ""): section
        for section in result.get("sections") or []
        if isinstance(section, Mapping)
    }
    for section in row.get("sections") or []:
        generated = section_results.get(str(section.get("section_id") or ""))
        if generated is None:
            continue
        verification_status = str(
            generated.get("verification_status") or "verification_error"
        )
        section.update(
            {
                "status": "completed",
                "evidence_status": (
                    verification_status
                    if verification_status in {"supported", "contradictory"}
                    else "missing"
                ),
                "verification_status": verification_status,
                "verification_reason_code": str(
                    generated.get("verification_reason_code") or ""
                ),
                "generation_status": str(generated.get("status") or ""),
                "content": str(generated.get("content") or ""),
                "citation_ledger": [
                    dict(_clone(item))
                    for item in generated.get("citation_ledger") or []
                    if isinstance(item, Mapping)
                ],
                "review_status": str(
                    generated.get("review_status") or "pending"
                ),
                "review_note": str(generated.get("review_note") or ""),
                "reviewed_at": generated.get("reviewed_at"),
                "evidence": [
                    dict(_clone(item))
                    for item in generated.get("evidence") or []
                    if isinstance(item, Mapping)
                ],
                "error": str(generated.get("error") or ""),
            }
        )
    timestamp = now_iso()
    report_version = int(row.get("report_version") or 0) + 1
    row["status"] = "completed"
    row["report_status"] = str(result.get("status") or "ready_with_gaps")
    row["report"] = {
        "format": "markdown",
        "content": str(result.get("markdown") or ""),
        "citation_ledger": [
            dict(_clone(item))
            for item in result.get("citation_ledger") or []
            if isinstance(item, Mapping)
        ],
        "verification_metrics": dict(
            _clone(result.get("verification_metrics") or {})
        ),
        "version": report_version,
        "generated_at": timestamp,
    }
    row["report_version"] = report_version
    row["report_completed_at"] = timestamp
    row["last_regenerated_section_ids"] = list(
        row.get("regeneration_section_ids") or []
    )
    row["regeneration_section_ids"] = []
    row["review_status"] = "pending"
    row["error"] = ""
    return _touch(row)


def _fail_report(
    row: dict[str, Any],
    *,
    report_execution_id: str,
    error_class: str,
) -> dict[str, Any]:
    if (
        row.get("status") != "generating"
        or row.get("report_execution_id") != report_execution_id
    ):
        return row
    row["status"] = "failed"
    row["report_status"] = "failed"
    row["error"] = error_class
    return _touch(row)


def _review_report(
    row: dict[str, Any],
    *,
    decisions: Sequence[Mapping[str, Any]],
    expected_revision: int,
) -> dict[str, Any]:
    if not decisions:
        raise ValueError("research report review requires at least one decision")
    actual = int(row.get("revision") or 0)
    if actual != expected_revision:
        raise ResearchJobRevisionConflictError(
            "research job revision conflict: "
            f"expected {expected_revision}, found {actual}"
        )
    if row.get("status") != "completed" or not isinstance(
        row.get("report"), Mapping
    ):
        raise ResearchJobStateConflictError(
            "research report can only be reviewed after generation"
        )
    if row.get("review_status") == "published":
        raise ResearchJobStateConflictError("published research report is immutable")
    section_by_id = {
        str(section.get("section_id") or ""): section
        for section in row.get("sections") or []
        if isinstance(section, dict)
    }
    seen: set[str] = set()
    review_event: list[dict[str, Any]] = []
    timestamp = now_iso()
    for raw in decisions:
        section_id = str(raw.get("section_id") or "")
        decision = str(raw.get("decision") or "")
        note = " ".join(str(raw.get("note") or "").split())
        if len(note) > 2000:
            raise ValueError("research report review note exceeds 2000 characters")
        if section_id in seen:
            raise ValueError(f"duplicate review section_id: {section_id}")
        seen.add(section_id)
        section = section_by_id.get(section_id)
        if section is None:
            raise ValueError(f"unknown review section_id: {section_id}")
        generated = section.get("generation_status") == "generated"
        allowed = (
            {"approved", "changes_requested"}
            if generated
            else {"accepted_gap", "changes_requested"}
        )
        if decision not in allowed:
            raise ValueError(
                f"review decision {decision} is invalid for section {section_id}"
            )
        if (
            section.get("review_status") == "changes_requested"
            and decision != "changes_requested"
        ):
            raise ResearchJobStateConflictError(
                f"section {section_id} must be regenerated after changes are requested"
            )
        if decision == "changes_requested" and not note:
            raise ValueError("changes_requested review requires a non-blank note")
        section["review_status"] = decision
        section["review_note"] = note
        section["reviewed_at"] = timestamp
        if decision == "changes_requested":
            section["revision_instruction"] = note
        review_event.append(
            {"section_id": section_id, "decision": decision, "note": note}
        )

    section_reviews = [
        str(section.get("review_status") or "pending")
        for section in row.get("sections") or []
    ]
    if "changes_requested" in section_reviews:
        review_status = "changes_requested"
    elif all(
        status in {"approved", "accepted_gap"} for status in section_reviews
    ):
        review_status = "approved"
    else:
        review_status = "pending"
    row["review_status"] = review_status
    history = list(row.get("review_history") or [])
    history.append(
        {
            "report_version": int(row.get("report_version") or 1),
            "reviewed_at": timestamp,
            "decisions": review_event,
            "result": review_status,
        }
    )
    row["review_history"] = history[-100:]
    return _touch(row)


def _publish_report(
    row: dict[str, Any], *, expected_revision: int
) -> dict[str, Any]:
    actual = int(row.get("revision") or 0)
    if actual != expected_revision:
        raise ResearchJobRevisionConflictError(
            "research job revision conflict: "
            f"expected {expected_revision}, found {actual}"
        )
    if row.get("status") != "completed" or row.get("review_status") != "approved":
        raise ResearchJobStateConflictError(
            "research report requires complete section review before publication"
        )
    report = row.get("report")
    if not isinstance(report, Mapping):
        raise ResearchJobStateConflictError("research report is unavailable")
    timestamp = now_iso()
    published = dict(_clone(report))
    published["published_at"] = timestamp
    row["published_report"] = published
    row["report_status"] = "published"
    row["review_status"] = "published"
    row["published_at"] = timestamp
    return _touch(row)


def _pause_job(row: dict[str, Any]) -> dict[str, Any]:
    status = str(row.get("status") or "")
    if status == "paused":
        return row
    if status != "running":
        raise ResearchJobStateConflictError(
            f"research job cannot pause from status {status}"
        )
    row["status"] = "paused"
    return _touch(row)


def _cancel_job(row: dict[str, Any]) -> dict[str, Any]:
    status = str(row.get("status") or "")
    if status == "cancelled":
        return row
    if status in {"evidence_ready", "completed"}:
        raise ResearchJobStateConflictError(
            f"research job cannot cancel from status {status}"
        )
    row["status"] = "cancelled"
    row["execution_id"] = ""
    for section in row.get("sections") or []:
        if section.get("status") == "running":
            section["status"] = "pending"
    return _touch(row)


def _claim_next_section(
    row: dict[str, Any], execution_id: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if row.get("status") != "running" or row.get("execution_id") != execution_id:
        return row, None
    for section in row.get("sections") or []:
        if section.get("status") != "pending":
            continue
        section["status"] = "running"
        section["error"] = ""
        _touch(row)
        return row, section
    if not any(
        section.get("status") == "running" for section in row.get("sections") or []
    ):
        row["status"] = "evidence_ready"
        row["evidence_completed_at"] = now_iso()
        _touch(row)
    return row, None


def _complete_section(
    row: dict[str, Any],
    section_id: str,
    *,
    execution_id: str,
    evidence_status: str,
    evidence: Sequence[Mapping[str, Any]],
    execution_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if evidence_status not in {"partial", "missing"}:
        raise ValueError("retrieval execution may only produce partial or missing evidence")
    if (
        row.get("status") not in {"running", "paused"}
        or row.get("execution_id") != execution_id
    ):
        return row
    sections = row.get("sections") or []
    target = next(
        (section for section in sections if section.get("section_id") == section_id),
        None,
    )
    if target is None:
        raise KeyError(section_id)
    if target.get("status") != "running":
        return row
    target.update(
        {
            "status": "completed",
            "evidence_status": evidence_status,
            "evidence": [dict(_clone(item)) for item in evidence],
            "execution_metrics": dict(_clone(execution_metrics)),
            "error": "",
        }
    )
    if row.get("status") == "running" and not any(
        section.get("status") in {"pending", "running"} for section in sections
    ):
        row["status"] = "evidence_ready"
        row["evidence_completed_at"] = now_iso()
    return _touch(row)


def _fail_section(
    row: dict[str, Any],
    section_id: str,
    *,
    execution_id: str,
    error_class: str,
) -> dict[str, Any]:
    if row.get("execution_id") != execution_id:
        return row
    target = next(
        (
            section
            for section in row.get("sections") or []
            if section.get("section_id") == section_id
        ),
        None,
    )
    if target is None:
        raise KeyError(section_id)
    if target.get("status") != "running":
        return row
    target["status"] = "failed"
    target["error"] = error_class
    row["status"] = "failed"
    row["error"] = error_class
    return _touch(row)


def _reconcile_running_job(row: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(_clone(row))
    for section in updated.get("sections") or []:
        if section.get("status") == "running":
            section["status"] = "pending"
            section["error"] = ""
    updated["status"] = "paused"
    updated["error"] = "service_restarted"
    return _touch(updated)


def _reconcile_generating_job(row: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(_clone(row))
    updated["status"] = "evidence_ready"
    updated["report_status"] = "failed"
    updated["report_execution_id"] = ""
    updated["error"] = "service_restarted"
    return _touch(updated)


class ResearchJobStore:
    """Atomic JSON store for durable, editable research plans."""

    def __init__(self, path: str | None = None):
        self._path = path or get_settings().research_jobs_path
        self._lock = RLock()
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)

    def create(
        self,
        *,
        kb_id: str,
        objective: str,
        title: str = "",
        section_titles: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        timestamp = now_iso()
        clean_objective = " ".join(objective.split())
        record = {
            "job_id": f"rj_{uuid4().hex}",
            "kb_id": kb_id,
            "title": " ".join(title.split()) or clean_objective[:80],
            "objective": clean_objective,
            "status": "planned",
            "revision": 1,
            "created_at": timestamp,
            "updated_at": timestamp,
            "sections": build_research_plan(clean_objective, section_titles),
            "execution_id": "",
            "started_at": None,
            "evidence_completed_at": None,
            "report_status": "not_started",
            "report_execution_id": "",
            "report_completed_at": None,
            "report": None,
            "report_version": 0,
            "report_history": [],
            "review_status": "not_started",
            "review_history": [],
            "published_report": None,
            "published_at": None,
            "regeneration_section_ids": [],
            "last_regenerated_section_ids": [],
            "error": "",
        }
        with self._lock:
            rows = self._read_all_locked()
            rows.append(record)
            self._write_all_locked(rows)
        return _clone(record)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            for row in self._read_all_locked():
                if row.get("job_id") == job_id:
                    return _clone(row)
        return None

    def list(
        self,
        *,
        kb_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._read_all_locked()
        if kb_id is not None:
            rows = [row for row in rows if row.get("kb_id") == kb_id]
        if status is not None:
            rows = [row for row in rows if row.get("status") == status]
        rows.sort(
            key=lambda row: (str(row.get("updated_at") or ""), str(row.get("job_id") or "")),
            reverse=True,
        )
        return _clone(rows[: max(0, limit)])

    def update_plan(
        self,
        job_id: str,
        *,
        sections: Sequence[Mapping[str, Any]],
        expected_revision: int,
    ) -> dict[str, Any]:
        normalized = _normalize_sections(sections)
        with self._lock:
            rows = self._read_all_locked()
            for position, row in enumerate(rows):
                if row.get("job_id") != job_id:
                    continue
                if row.get("status") in {"running", "generating"}:
                    raise ResearchJobStateConflictError(
                        "research plan cannot be edited while execution is running"
                    )
                if row.get("review_status") == "published":
                    raise ResearchJobStateConflictError(
                        "published research report is immutable"
                    )
                actual = int(row.get("revision") or 0)
                if actual != expected_revision:
                    raise ResearchJobRevisionConflictError(
                        "research job revision conflict: "
                        f"expected {expected_revision}, found {actual}"
                    )
                updated = {
                    **row,
                    "sections": normalized,
                    "status": "planned",
                    "revision": actual + 1,
                    "updated_at": now_iso(),
                    "execution_id": "",
                    "started_at": None,
                    "evidence_completed_at": None,
                    "report_status": "not_started",
                    "report_execution_id": "",
                    "report_completed_at": None,
                    "report": None,
                    "report_version": 0,
                    "report_history": [],
                    "review_status": "not_started",
                    "review_history": [],
                    "published_report": None,
                    "published_at": None,
                    "regeneration_section_ids": [],
                    "last_regenerated_section_ids": [],
                    "error": "",
                }
                rows[position] = updated
                self._write_all_locked(rows)
                return _clone(updated)
        raise KeyError(job_id)

    def start(self, job_id: str) -> dict[str, Any]:
        return self._mutate(job_id, _start_job)

    def resume(self, job_id: str) -> dict[str, Any]:
        return self._mutate(job_id, _resume_job)

    def pause(self, job_id: str) -> dict[str, Any]:
        return self._mutate(job_id, _pause_job)

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self._mutate(job_id, _cancel_job)

    def begin_report(self, job_id: str) -> dict[str, Any]:
        return self._mutate(job_id, _begin_report)

    def complete_report(
        self,
        job_id: str,
        *,
        report_execution_id: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._mutate(
            job_id,
            lambda row: _complete_report(
                row,
                report_execution_id=report_execution_id,
                result=result,
            ),
            write_if_unchanged=False,
        )

    def fail_report(
        self,
        job_id: str,
        *,
        report_execution_id: str,
        error_class: str,
    ) -> dict[str, Any]:
        return self._mutate(
            job_id,
            lambda row: _fail_report(
                row,
                report_execution_id=report_execution_id,
                error_class=error_class,
            ),
            write_if_unchanged=False,
        )

    def review_report(
        self,
        job_id: str,
        *,
        decisions: Sequence[Mapping[str, Any]],
        expected_revision: int,
    ) -> dict[str, Any]:
        return self._mutate(
            job_id,
            lambda row: _review_report(
                row,
                decisions=decisions,
                expected_revision=expected_revision,
            ),
        )

    def publish_report(
        self, job_id: str, *, expected_revision: int
    ) -> dict[str, Any]:
        return self._mutate(
            job_id,
            lambda row: _publish_report(
                row,
                expected_revision=expected_revision,
            ),
        )

    def claim_next_section(
        self, job_id: str, execution_id: str
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        claimed: dict[str, Any] | None = None

        def transition(row: dict[str, Any]) -> dict[str, Any]:
            nonlocal claimed
            updated, claimed = _claim_next_section(row, execution_id)
            return updated

        row = self._mutate(job_id, transition, write_if_unchanged=False)
        return row, _clone(claimed) if claimed is not None else None

    def complete_section(
        self,
        job_id: str,
        section_id: str,
        *,
        execution_id: str,
        evidence_status: str,
        evidence: Sequence[Mapping[str, Any]],
        execution_metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._mutate(
            job_id,
            lambda row: _complete_section(
                row,
                section_id,
                execution_id=execution_id,
                evidence_status=evidence_status,
                evidence=evidence,
                execution_metrics=execution_metrics,
            ),
            write_if_unchanged=False,
        )

    def fail_section(
        self,
        job_id: str,
        section_id: str,
        *,
        execution_id: str,
        error_class: str,
    ) -> dict[str, Any]:
        return self._mutate(
            job_id,
            lambda row: _fail_section(
                row,
                section_id,
                execution_id=execution_id,
                error_class=error_class,
            ),
            write_if_unchanged=False,
        )

    def reconcile_running(self) -> int:
        with self._lock:
            rows = self._read_all_locked()
            changed = 0
            for position, row in enumerate(rows):
                if row.get("status") == "running":
                    rows[position] = _reconcile_running_job(row)
                elif row.get("status") == "generating":
                    rows[position] = _reconcile_generating_job(row)
                else:
                    continue
                changed += 1
            if changed:
                self._write_all_locked(rows)
        return changed

    def clear_kb(self, kb_id: str) -> None:
        with self._lock:
            rows = [
                row for row in self._read_all_locked() if row.get("kb_id") != kb_id
            ]
            self._write_all_locked(rows)

    def export_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return _clone(self._read_all_locked())

    def import_records(self, records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        incoming = [dict(_clone(record)) for record in records]
        with self._lock:
            rows = self._read_all_locked()
            positions = {str(row.get("job_id") or ""): idx for idx, row in enumerate(rows)}
            changed = 0
            for record in incoming:
                job_id = str(record.get("job_id") or "")
                if not job_id:
                    raise ValueError("research job import requires job_id")
                position = positions.get(job_id)
                if position is not None and rows[position] == record:
                    continue
                if position is None:
                    positions[job_id] = len(rows)
                    rows.append(record)
                else:
                    rows[position] = record
                changed += 1
            if changed:
                self._write_all_locked(rows)
        return {"imported": changed, "skipped": len(incoming) - changed}

    def _read_all_locked(self) -> list[dict[str, Any]]:
        if not os.path.exists(self._path):
            return []
        with open(self._path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("research job store must contain a JSON list")
        return [dict(row) for row in payload if isinstance(row, Mapping)]

    def _mutate(
        self,
        job_id: str,
        transition,
        *,
        write_if_unchanged: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            rows = self._read_all_locked()
            for position, row in enumerate(rows):
                if row.get("job_id") != job_id:
                    continue
                updated = transition(_clone(row))
                if write_if_unchanged or updated != row:
                    rows[position] = updated
                    self._write_all_locked(rows)
                return _clone(updated)
        raise KeyError(job_id)

    def _write_all_locked(self, rows: list[dict[str, Any]]) -> None:
        temporary_path = f"{self._path}.tmp"
        try:
            with open(temporary_path, "w", encoding="utf-8") as handle:
                json.dump(rows, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)


class SqliteResearchJobStore(ResearchJobStore):
    """SQLite adapter with the same contract as ``ResearchJobStore``."""

    def __init__(self, db_path: str):
        self._lock = RLock()
        self._closed = False
        self._conn = connect_sqlite(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS research_jobs ("
            "job_id TEXT PRIMARY KEY, kb_id TEXT NOT NULL, status TEXT NOT NULL, "
            "updated_at TEXT NOT NULL, data TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_research_jobs_queue "
            "ON research_jobs(kb_id, status, updated_at DESC)"
        )

    def create(
        self,
        *,
        kb_id: str,
        objective: str,
        title: str = "",
        section_titles: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        timestamp = now_iso()
        clean_objective = " ".join(objective.split())
        record = {
            "job_id": f"rj_{uuid4().hex}",
            "kb_id": kb_id,
            "title": " ".join(title.split()) or clean_objective[:80],
            "objective": clean_objective,
            "status": "planned",
            "revision": 1,
            "created_at": timestamp,
            "updated_at": timestamp,
            "sections": build_research_plan(clean_objective, section_titles),
            "execution_id": "",
            "started_at": None,
            "evidence_completed_at": None,
            "report_status": "not_started",
            "report_execution_id": "",
            "report_completed_at": None,
            "report": None,
            "report_version": 0,
            "report_history": [],
            "review_status": "not_started",
            "review_history": [],
            "published_report": None,
            "published_at": None,
            "regeneration_section_ids": [],
            "last_regenerated_section_ids": [],
            "error": "",
        }
        with self._lock:
            self._ensure_open()
            self._upsert_locked(record)
        return _clone(record)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._ensure_open()
            row = self._conn.execute(
                "SELECT data FROM research_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return _clone(json.loads(row[0])) if row is not None else None

    def list(
        self,
        *,
        kb_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if kb_id is not None:
            clauses.append("kb_id=?")
            params.append(kb_id)
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        query = "SELECT data FROM research_jobs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, job_id DESC LIMIT ?"
        params.append(max(0, limit))
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(query, tuple(params)).fetchall()
        return [_clone(json.loads(row[0])) for row in rows]

    def update_plan(
        self,
        job_id: str,
        *,
        sections: Sequence[Mapping[str, Any]],
        expected_revision: int,
    ) -> dict[str, Any]:
        normalized = _normalize_sections(sections)
        with self._lock:
            self._ensure_open()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT data FROM research_jobs WHERE job_id=?", (job_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(job_id)
                current = dict(json.loads(row[0]))
                if current.get("status") in {"running", "generating"}:
                    raise ResearchJobStateConflictError(
                        "research plan cannot be edited while execution is running"
                    )
                if current.get("review_status") == "published":
                    raise ResearchJobStateConflictError(
                        "published research report is immutable"
                    )
                actual = int(current.get("revision") or 0)
                if actual != expected_revision:
                    raise ResearchJobRevisionConflictError(
                        "research job revision conflict: "
                        f"expected {expected_revision}, found {actual}"
                    )
                updated = {
                    **current,
                    "sections": normalized,
                    "status": "planned",
                    "revision": actual + 1,
                    "updated_at": now_iso(),
                    "execution_id": "",
                    "started_at": None,
                    "evidence_completed_at": None,
                    "report_status": "not_started",
                    "report_execution_id": "",
                    "report_completed_at": None,
                    "report": None,
                    "report_version": 0,
                    "report_history": [],
                    "review_status": "not_started",
                    "review_history": [],
                    "published_report": None,
                    "published_at": None,
                    "regeneration_section_ids": [],
                    "last_regenerated_section_ids": [],
                    "error": "",
                }
                self._upsert_locked(updated)
                self._conn.execute("COMMIT")
                return _clone(updated)
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def start(self, job_id: str) -> dict[str, Any]:
        return self._mutate_sqlite(job_id, _start_job)

    def resume(self, job_id: str) -> dict[str, Any]:
        return self._mutate_sqlite(job_id, _resume_job)

    def pause(self, job_id: str) -> dict[str, Any]:
        return self._mutate_sqlite(job_id, _pause_job)

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self._mutate_sqlite(job_id, _cancel_job)

    def begin_report(self, job_id: str) -> dict[str, Any]:
        return self._mutate_sqlite(job_id, _begin_report)

    def complete_report(
        self,
        job_id: str,
        *,
        report_execution_id: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._mutate_sqlite(
            job_id,
            lambda row: _complete_report(
                row,
                report_execution_id=report_execution_id,
                result=result,
            ),
        )

    def fail_report(
        self,
        job_id: str,
        *,
        report_execution_id: str,
        error_class: str,
    ) -> dict[str, Any]:
        return self._mutate_sqlite(
            job_id,
            lambda row: _fail_report(
                row,
                report_execution_id=report_execution_id,
                error_class=error_class,
            ),
        )

    def review_report(
        self,
        job_id: str,
        *,
        decisions: Sequence[Mapping[str, Any]],
        expected_revision: int,
    ) -> dict[str, Any]:
        return self._mutate_sqlite(
            job_id,
            lambda row: _review_report(
                row,
                decisions=decisions,
                expected_revision=expected_revision,
            ),
        )

    def publish_report(
        self, job_id: str, *, expected_revision: int
    ) -> dict[str, Any]:
        return self._mutate_sqlite(
            job_id,
            lambda row: _publish_report(
                row,
                expected_revision=expected_revision,
            ),
        )

    def claim_next_section(
        self, job_id: str, execution_id: str
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        claimed: dict[str, Any] | None = None

        def transition(row: dict[str, Any]) -> dict[str, Any]:
            nonlocal claimed
            updated, claimed = _claim_next_section(row, execution_id)
            return updated

        row = self._mutate_sqlite(job_id, transition)
        return row, _clone(claimed) if claimed is not None else None

    def complete_section(
        self,
        job_id: str,
        section_id: str,
        *,
        execution_id: str,
        evidence_status: str,
        evidence: Sequence[Mapping[str, Any]],
        execution_metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._mutate_sqlite(
            job_id,
            lambda row: _complete_section(
                row,
                section_id,
                execution_id=execution_id,
                evidence_status=evidence_status,
                evidence=evidence,
                execution_metrics=execution_metrics,
            ),
        )

    def fail_section(
        self,
        job_id: str,
        section_id: str,
        *,
        execution_id: str,
        error_class: str,
    ) -> dict[str, Any]:
        return self._mutate_sqlite(
            job_id,
            lambda row: _fail_section(
                row,
                section_id,
                execution_id=execution_id,
                error_class=error_class,
            ),
        )

    def reconcile_running(self) -> int:
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(
                "SELECT job_id, data FROM research_jobs "
                "WHERE status IN ('running', 'generating')"
            ).fetchall()
            if not rows:
                return 0
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for _, raw in rows:
                    current = json.loads(raw)
                    transition = (
                        _reconcile_running_job
                        if current.get("status") == "running"
                        else _reconcile_generating_job
                    )
                    self._upsert_locked(transition(current))
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return len(rows)

    def clear_kb(self, kb_id: str) -> None:
        with self._lock:
            self._ensure_open()
            self._conn.execute("DELETE FROM research_jobs WHERE kb_id=?", (kb_id,))

    def export_records(self) -> list[dict[str, Any]]:
        return self.list(limit=2**31 - 1)

    def import_records(self, records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        incoming = [dict(_clone(record)) for record in records]
        with self._lock:
            self._ensure_open()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                changed = 0
                for record in incoming:
                    job_id = str(record.get("job_id") or "")
                    if not job_id:
                        raise ValueError("research job import requires job_id")
                    existing = self._conn.execute(
                        "SELECT data FROM research_jobs WHERE job_id=?", (job_id,)
                    ).fetchone()
                    if existing is not None and json.loads(existing[0]) == record:
                        continue
                    self._upsert_locked(record)
                    changed += 1
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return {"imported": changed, "skipped": len(incoming) - changed}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SqliteResearchJobStore is closed")

    def _upsert_locked(self, record: Mapping[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO research_jobs(job_id, kb_id, status, updated_at, data) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(job_id) DO UPDATE SET "
            "kb_id=excluded.kb_id, status=excluded.status, "
            "updated_at=excluded.updated_at, data=excluded.data",
            (
                record["job_id"],
                record["kb_id"],
                record["status"],
                record["updated_at"],
                json.dumps(record, ensure_ascii=False),
            ),
        )

    def _mutate_sqlite(self, job_id: str, transition) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT data FROM research_jobs WHERE job_id=?", (job_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(job_id)
                current = dict(json.loads(row[0]))
                updated = transition(_clone(current))
                if updated != current:
                    self._upsert_locked(updated)
                self._conn.execute("COMMIT")
                return _clone(updated)
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
