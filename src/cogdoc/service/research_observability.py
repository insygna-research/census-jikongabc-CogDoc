from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from cogdoc.observability.logger import log_event


# These are intentionally closed sets. Identifiers and provider/model/error detail
# belong in structured logs or traces, never in Prometheus labels.
RESEARCH_LIFECYCLE_ACTIONS = frozenset(
    {
        "create",
        "update_plan",
        "auto_plan",
        "start",
        "resume",
        "pause",
        "cancel",
        "refresh",
        "generate",
        "review",
        "publish",
        "reconcile",
    }
)
RESEARCH_LIFECYCLE_OUTCOMES = frozenset(
    {
        "accepted",
        "succeeded",
        "failed",
        "conflict",
        "stale",
        "not_found",
        "invalid",
        "unavailable",
        "noop",
    }
)
RESEARCH_BACKGROUND_STAGES = frozenset({"planning", "evidence", "report"})
RESEARCH_BACKGROUND_OUTCOMES = frozenset(
    {
        "succeeded",
        "failed",
        "cancelled",
        "superseded",
        "schedule_failed",
        "orphaned",
    }
)
RESEARCH_TERMINATION_REASONS = frozenset(
    {
        "budget_exhausted",
        "deadline_exceeded",
        "paused",
        "cancelled",
        "stale_evidence",
        "superseded",
        "service_restarted",
        "shutdown",
        "kb_missing",
        "worker_error",
    }
)
RESEARCH_COVERAGE_STATUSES = frozenset(
    {"not_run", "passed", "repaired", "failed", "rejected", "error"}
)
RESEARCH_PROVIDER_KINDS = frozenset({"llm"})
RESEARCH_PROVIDER_ISOLATIONS = frozenset({"process", "compatibility"})
RESEARCH_PROVIDER_OUTCOMES = frozenset(
    {"succeeded", "timeout", "capacity", "cancelled", "failed"}
)

_RESEARCH_LOG_STATUSES = frozenset(
    {
        "planned",
        "pending",
        "running",
        "paused",
        "cancelled",
        "completed",
        "failed",
        "evidence_ready",
        "generating",
        "not_started",
        "ready",
        "ready_with_gaps",
        "published",
        "unsearched",
        "missing",
        "partial",
        "supported",
        "contradictory",
        "no_evidence",
        "budget_exhausted",
        "retrieval_error",
        "verification_error",
        "generated",
        "claim_rejected",
        "generation_error",
        "approved",
        "accepted_gap",
        "changes_requested",
    }
)
_ERROR_CLASS_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_MAX_COUNT = 1_000_000_000
_MAX_DURATION_MS = 86_400_000.0


def _enum_label(value: Any, allowed: frozenset[str]) -> str:
    try:
        normalized = str(value or "").strip().casefold()
    except Exception:
        return "unknown"
    return normalized if normalized in allowed else "unknown"


def normalize_research_action(value: Any) -> str:
    return _enum_label(value, RESEARCH_LIFECYCLE_ACTIONS)


def normalize_research_lifecycle_outcome(value: Any) -> str:
    return _enum_label(value, RESEARCH_LIFECYCLE_OUTCOMES)


def normalize_research_stage(value: Any) -> str:
    return _enum_label(value, RESEARCH_BACKGROUND_STAGES)


def normalize_research_background_outcome(value: Any) -> str:
    return _enum_label(value, RESEARCH_BACKGROUND_OUTCOMES)


def normalize_research_termination_reason(value: Any) -> str:
    return _enum_label(value, RESEARCH_TERMINATION_REASONS)


def normalize_research_coverage_status(value: Any) -> str:
    return _enum_label(value, RESEARCH_COVERAGE_STATUSES)


def normalize_research_provider(value: Any) -> str:
    return _enum_label(value, RESEARCH_PROVIDER_KINDS)


def normalize_research_provider_isolation(value: Any) -> str:
    return _enum_label(value, RESEARCH_PROVIDER_ISOLATIONS)


def normalize_research_provider_outcome(value: Any) -> str:
    return _enum_label(value, RESEARCH_PROVIDER_OUTCOMES)


def _safe_status(value: Any) -> str:
    return _enum_label(value, _RESEARCH_LOG_STATUSES)


def _safe_identifier(value: Any, *, limit: int = 128) -> str:
    try:
        raw = str(value or "")
    except Exception:
        return ""
    # Identifiers may be Unicode, but control characters never carry identity.
    return "".join(character for character in raw if character.isprintable())[:limit]


def _safe_error_class(value: Any) -> str:
    try:
        raw = str(value or "").strip()
    except Exception:
        return "unknown"
    if not raw:
        return ""
    return raw if _ERROR_CLASS_PATTERN.fullmatch(raw) else "unknown"


def _safe_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return min(max(int(value or 0), 0), _MAX_COUNT)
    except (TypeError, ValueError, OverflowError):
        return 0


def _safe_duration_ms(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(duration):
        return None
    return round(min(max(duration, 0.0), _MAX_DURATION_MS), 3)


def _safe_missing_count(value: Any) -> int:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return 0
    return min(len(value), _MAX_COUNT)


class ResearchMetricsSink(Protocol):
    def observe_research_lifecycle(self, action: Any, outcome: Any) -> None: ...

    def research_background_started(self, stage: Any) -> None: ...

    def research_background_finished(
        self, stage: Any, outcome: Any, *, duration_ms: Any = None
    ) -> None: ...

    def observe_research_termination(
        self, reason: Any, *, count: Any = 1
    ) -> None: ...

    def observe_research_section(
        self, *, candidate_count: Any = None, evidence_count: Any = None
    ) -> None: ...

    def observe_research_coverage_audit(self, audit_or_status: Any) -> None: ...

    def observe_research_provider_call(
        self,
        provider: Any,
        isolation: Any,
        outcome: Any,
        *,
        duration_ms: Any = None,
    ) -> None: ...

    def observe_claim_audit(self, task_type: str, audit: Any) -> None: ...


class ResearchObserver:
    """Best-effort, content-free telemetry facade for research orchestration.

    Public methods deliberately accept only correlation identifiers, closed-set
    state, counts, durations, and an exception class name. Objective/query,
    evidence, claims, generated prose, reviewer notes, and report bodies have no
    parameter through which they can accidentally reach logs.
    """

    def __init__(self, metrics: ResearchMetricsSink | None = None):
        self._metrics = metrics

    def lifecycle(
        self,
        *,
        action: Any,
        outcome: Any,
        job_id: Any = "",
        kb_id: Any = "",
        execution_id: Any = "",
        status: Any = "",
        error_class: Any = "",
    ) -> None:
        normalized_action = normalize_research_action(action)
        normalized_outcome = normalize_research_lifecycle_outcome(outcome)
        self._metric(
            "observe_research_lifecycle", normalized_action, normalized_outcome
        )
        level = (
            logging.ERROR
            if normalized_outcome == "failed"
            else logging.WARNING
            if normalized_outcome
            in {"conflict", "stale", "invalid", "unavailable", "unknown"}
            else logging.INFO
        )
        self._emit(
            "research_lifecycle",
            job_id=job_id,
            kb_id=kb_id,
            execution_id=execution_id,
            level=level,
            action=normalized_action,
            outcome=normalized_outcome,
            status=_safe_status(status),
            error_class=_safe_error_class(error_class),
        )

    def background_started(
        self,
        *,
        stage: Any,
        job_id: Any,
        kb_id: Any = "",
        execution_id: Any = "",
        status: Any = "running",
    ) -> None:
        normalized_stage = normalize_research_stage(stage)
        self._metric("research_background_started", normalized_stage)
        self._emit(
            "research_background_started",
            job_id=job_id,
            kb_id=kb_id,
            execution_id=execution_id,
            stage=normalized_stage,
            status=_safe_status(status),
        )

    def background_finished(
        self,
        *,
        stage: Any,
        outcome: Any,
        job_id: Any,
        kb_id: Any = "",
        execution_id: Any = "",
        status: Any = "",
        duration_ms: Any = None,
        error_class: Any = "",
    ) -> None:
        normalized_stage = normalize_research_stage(stage)
        normalized_outcome = normalize_research_background_outcome(outcome)
        safe_duration = _safe_duration_ms(duration_ms)
        self._metric(
            "research_background_finished",
            normalized_stage,
            normalized_outcome,
            duration_ms=safe_duration,
        )
        self._emit(
            "research_background_finished",
            job_id=job_id,
            kb_id=kb_id,
            execution_id=execution_id,
            level=(
                logging.ERROR
                if normalized_outcome in {"failed", "schedule_failed"}
                else logging.WARNING
                if normalized_outcome in {"cancelled", "superseded", "orphaned"}
                else logging.INFO
            ),
            stage=normalized_stage,
            outcome=normalized_outcome,
            status=_safe_status(status),
            duration_ms=safe_duration,
            error_class=_safe_error_class(error_class),
        )

    def terminated(
        self,
        *,
        reason: Any,
        job_id: Any,
        kb_id: Any = "",
        execution_id: Any = "",
        section_id: Any = "",
        stage: Any = "",
        status: Any = "",
        error_class: Any = "",
    ) -> None:
        normalized_reason = normalize_research_termination_reason(reason)
        self._metric("observe_research_termination", normalized_reason)
        self._emit(
            "research_terminated",
            job_id=job_id,
            kb_id=kb_id,
            execution_id=execution_id,
            section_id=section_id,
            level=(
                logging.INFO
                if normalized_reason in {"paused", "cancelled"}
                else logging.WARNING
            ),
            stage=normalize_research_stage(stage),
            reason=normalized_reason,
            status=_safe_status(status),
            error_class=_safe_error_class(error_class),
        )

    def control_terminated(
        self,
        *,
        reason: Any,
        job_id: Any,
        kb_id: Any = "",
        execution_id: Any = "",
        section_id: Any = "",
        stage: Any = "",
        status: Any = "",
        error_class: Any = "",
    ) -> None:
        """Compatibility name for manager control/budget termination sites."""

        self.terminated(
            reason=reason,
            job_id=job_id,
            kb_id=kb_id,
            execution_id=execution_id,
            section_id=section_id,
            stage=stage,
            status=status,
            error_class=error_class,
        )

    def provider_call(
        self,
        *,
        provider: Any,
        isolation: Any,
        outcome: Any,
        job_id: Any = "",
        kb_id: Any = "",
        execution_id: Any = "",
        section_id: Any = "",
        stage: Any = "",
        duration_ms: Any = None,
        error_class: Any = "",
    ) -> None:
        """Record one bounded provider attempt without logging request content."""

        normalized_provider = normalize_research_provider(provider)
        normalized_isolation = normalize_research_provider_isolation(isolation)
        normalized_outcome = normalize_research_provider_outcome(outcome)
        safe_duration = _safe_duration_ms(duration_ms)
        self._metric(
            "observe_research_provider_call",
            normalized_provider,
            normalized_isolation,
            normalized_outcome,
            duration_ms=safe_duration,
        )
        # Provider/isolation/outcome are bounded Prometheus dimensions. Keep the
        # structured log even narrower: correlation IDs, stage, duration, and the
        # exception class are sufficient for joining it to the surrounding run.
        self._emit(
            "research_provider_call",
            job_id=job_id,
            kb_id=kb_id,
            execution_id=execution_id,
            section_id=section_id,
            level=(
                logging.ERROR
                if normalized_outcome in {"failed", "unknown"}
                else logging.WARNING
                if normalized_outcome in {"timeout", "capacity"}
                else logging.INFO
            ),
            stage=normalize_research_stage(stage),
            duration_ms=safe_duration,
            error_class=_safe_error_class(error_class),
        )

    def section_completed(
        self,
        *,
        job_id: Any,
        section_id: Any,
        kb_id: Any = "",
        execution_id: Any = "",
        status: Any = "completed",
        candidate_count: Any = 0,
        evidence_count: Any = 0,
        query_count: Any = 0,
        duration_ms: Any = None,
        error_class: Any = "",
    ) -> None:
        candidates = _safe_count(candidate_count)
        evidence = _safe_count(evidence_count)
        queries = _safe_count(query_count)
        safe_duration = _safe_duration_ms(duration_ms)
        self._metric(
            "observe_research_section",
            candidate_count=candidates,
            evidence_count=evidence,
        )
        self._emit(
            "research_section_completed",
            job_id=job_id,
            kb_id=kb_id,
            execution_id=execution_id,
            section_id=section_id,
            level=logging.ERROR if _safe_error_class(error_class) else logging.INFO,
            status=_safe_status(status),
            candidate_count=candidates,
            evidence_count=evidence,
            query_count=queries,
            duration_ms=safe_duration,
            error_class=_safe_error_class(error_class),
        )

    def coverage_audit(
        self,
        *,
        audit: Any,
        job_id: Any,
        section_id: Any,
        kb_id: Any = "",
        execution_id: Any = "",
        error_class: Any = "",
    ) -> None:
        bounded = audit if isinstance(audit, Mapping) else {}
        try:
            raw_status = bounded.get("status")
            requirement_count = _safe_count(bounded.get("requirement_count"))
            covered_count = min(
                _safe_count(bounded.get("covered_count")), requirement_count
            )
            missing_count = _safe_missing_count(bounded.get("missing_requirement_ids"))
            repair = bounded.get("repair")
            repair_attempt_count = (
                _safe_count(repair.get("attempt_count"))
                if isinstance(repair, Mapping)
                else 0
            )
        except Exception:
            raw_status = None
            requirement_count = covered_count = missing_count = 0
            repair_attempt_count = 0
        status = normalize_research_coverage_status(raw_status)
        self._metric("observe_research_coverage_audit", status)
        self._emit(
            "research_coverage_audit",
            job_id=job_id,
            kb_id=kb_id,
            execution_id=execution_id,
            section_id=section_id,
            level=(
                logging.WARNING
                if status in {"failed", "rejected", "error", "unknown"}
                else logging.INFO
            ),
            status=status,
            requirement_count=requirement_count,
            covered_count=covered_count,
            missing_count=missing_count,
            repair_attempt_count=repair_attempt_count,
            error_class=_safe_error_class(error_class),
        )

    def orphan_reconciled(
        self,
        *,
        count: Any,
        termination_counts: Mapping[str, Any] | None = None,
    ) -> None:
        reconciled = _safe_count(count)
        if reconciled <= 0:
            return
        self._metric("observe_research_lifecycle", "reconcile", "succeeded")
        raw_counts = (
            termination_counts if isinstance(termination_counts, Mapping) else {}
        )
        restarted = _safe_count(raw_counts.get("service_restarted"))
        expired = _safe_count(raw_counts.get("deadline_exceeded"))
        if restarted + expired != reconciled:
            # Compatibility fallback for stores that only return a total.
            restarted = reconciled
            expired = 0
        if restarted:
            self._metric(
                "observe_research_termination",
                "service_restarted",
                count=restarted,
            )
        if expired:
            self._metric(
                "observe_research_termination",
                "deadline_exceeded",
                count=expired,
            )
        self._emit(
            "research_orphans_reconciled",
            level=logging.WARNING,
            status="failed" if expired and not restarted else "paused",
            count=reconciled,
            service_restarted_count=restarted,
            deadline_exceeded_count=expired,
        )

    def _metric(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        if self._metrics is None:
            return
        try:
            method = getattr(self._metrics, method_name, None)
            if callable(method):
                method(*args, **kwargs)
        except Exception:
            # Instrumentation is never allowed to mutate control-flow outcomes.
            return

    @staticmethod
    def _emit(
        event: str,
        *,
        job_id: Any = "",
        kb_id: Any = "",
        execution_id: Any = "",
        section_id: Any = "",
        level: int = logging.INFO,
        **safe_fields: Any,
    ) -> None:
        safe_job_id = _safe_identifier(job_id)
        safe_execution_id = _safe_identifier(execution_id)
        fields = {
            "job_id": safe_job_id,
            "kb_id": _safe_identifier(kb_id, limit=64),
            "execution_id": safe_execution_id,
            "section_id": _safe_identifier(section_id, limit=64),
            **safe_fields,
        }
        # Keep logs compact and make the whitelist visible in emitted JSON.
        fields = {
            key: value for key, value in fields.items() if value not in (None, "")
        }
        try:
            log_event(
                "research",
                event,
                {
                    "request_id": safe_job_id or None,
                    "trace_id": safe_execution_id or safe_job_id or None,
                },
                level=level,
                **fields,
            )
        except Exception:
            return

    def claim_audit(
        self,
        *,
        audit: Any,
        job_id: Any,
        section_id: Any,
        kb_id: Any = "",
        execution_id: Any = "",
        error_class: Any = "",
    ) -> None:
        bounded = audit if isinstance(audit, Mapping) else {}
        counts = bounded.get("counts")
        counts = counts if isinstance(counts, Mapping) else {}
        status = normalize_research_coverage_status(bounded.get("status"))
        try:
            self._metric("observe_claim_audit", "research", bounded)
            self._emit(
                "research_claim_audit",
                job_id=job_id,
                kb_id=kb_id,
                execution_id=execution_id,
                section_id=section_id,
                level=(
                    logging.ERROR
                    if status in {"failed", "rejected", "error", "unknown"}
                    else logging.INFO
                ),
                status=status,
                claim_count=_safe_count(counts.get("claim_count")),
                supported_count=_safe_count(counts.get("supported")),
                unsupported_count=_safe_count(counts.get("unsupported")),
                insufficient_count=_safe_count(counts.get("insufficient")),
                cited_count=_safe_count(counts.get("cited")),
                error_class=_safe_error_class(error_class),
            )
        except Exception:
            return


__all__ = [
    "RESEARCH_BACKGROUND_OUTCOMES",
    "RESEARCH_BACKGROUND_STAGES",
    "RESEARCH_COVERAGE_STATUSES",
    "RESEARCH_LIFECYCLE_ACTIONS",
    "RESEARCH_LIFECYCLE_OUTCOMES",
    "RESEARCH_PROVIDER_ISOLATIONS",
    "RESEARCH_PROVIDER_KINDS",
    "RESEARCH_PROVIDER_OUTCOMES",
    "RESEARCH_TERMINATION_REASONS",
    "ResearchMetricsSink",
    "ResearchObserver",
    "normalize_research_action",
    "normalize_research_background_outcome",
    "normalize_research_coverage_status",
    "normalize_research_lifecycle_outcome",
    "normalize_research_provider",
    "normalize_research_provider_isolation",
    "normalize_research_provider_outcome",
    "normalize_research_stage",
    "normalize_research_termination_reason",
]
