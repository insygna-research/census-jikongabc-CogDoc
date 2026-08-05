from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from cogdoc.service.evidence_unit_pipeline import (
    EvidenceUnitBatchResult,
    EvidenceUnitExecutionResult,
    EvidenceUnitExecutionStatus,
)
from cogdoc.service.evidence_units import (
    EvidenceClosureStatus,
    EvidenceUnit,
)
from cogdoc.tools.citation_ledger import build_evidence_ledger, is_valid_evidence_id

if TYPE_CHECKING:
    from cogdoc.agents.evidence_unit_verifier import (
        EvidenceUnitVerificationBatchResult,
        EvidenceUnitVerificationResult,
    )


class EvidenceUnitGateAction(str, Enum):
    """Task-independent downstream action for one evidence obligation."""

    GENERATE = "generate"
    RETRY = "retry"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class EvidenceUnitGatePolicy:
    """Admission policy shared by QA, Summary, Compare, and future agents."""

    allow_unverified_ready: bool = False
    contradictory_action: EvidenceUnitGateAction = EvidenceUnitGateAction.TERMINAL
    require_all_required_units: bool = True

    def __post_init__(self) -> None:
        for name in ("allow_unverified_ready", "require_all_required_units"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if not isinstance(self.contradictory_action, EvidenceUnitGateAction):
            raise ValueError("contradictory_action must be an EvidenceUnitGateAction")


@dataclass(frozen=True, slots=True)
class EvidenceUnitGateDecision:
    """One mergeable gate outcome, keyed by the opaque Evidence Unit ID."""

    unit: EvidenceUnit
    action: EvidenceUnitGateAction
    execution_status: EvidenceUnitExecutionStatus
    verification_status: EvidenceClosureStatus | None
    retrieval_round: int
    retries_remaining: int
    reason_code: str
    source_reason_code: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.unit, EvidenceUnit):
            raise ValueError("unit must be an EvidenceUnit")
        if not isinstance(self.action, EvidenceUnitGateAction):
            raise ValueError("action must be an EvidenceUnitGateAction")
        if not isinstance(self.execution_status, EvidenceUnitExecutionStatus):
            raise ValueError("execution_status must be an EvidenceUnitExecutionStatus")
        if self.verification_status is not None and not isinstance(
            self.verification_status, EvidenceClosureStatus
        ):
            raise ValueError("verification_status must be an EvidenceClosureStatus")
        for name in ("retrieval_round", "retries_remaining"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.reason_code, str) or not self.reason_code.strip():
            raise ValueError("reason_code is required")
        if not isinstance(self.source_reason_code, str):
            raise ValueError("source_reason_code must be a string")

    @property
    def unit_id(self) -> str:
        return self.unit.unit_id

    @property
    def required(self) -> bool:
        return self.unit.policy.required

    def to_state(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "required": self.required,
            "action": self.action.value,
            "execution_status": self.execution_status.value,
            "verification_status": (
                self.verification_status.value if self.verification_status else ""
            ),
            "retrieval_round": self.retrieval_round,
            "retries_remaining": self.retries_remaining,
            "reason_code": self.reason_code,
            "source_reason_code": self.source_reason_code,
        }


@dataclass(frozen=True, slots=True)
class EvidenceUnitGateBatchResult:
    """Ordered gate decisions plus a batch-level generation admission signal."""

    decisions: tuple[EvidenceUnitGateDecision, ...]
    policy: EvidenceUnitGatePolicy

    def __post_init__(self) -> None:
        if not self.decisions:
            raise ValueError("gate result requires at least one decision")
        if not all(
            isinstance(decision, EvidenceUnitGateDecision)
            for decision in self.decisions
        ):
            raise ValueError("decisions must contain EvidenceUnitGateDecision values")
        unit_ids = [decision.unit_id for decision in self.decisions]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("gate decisions must not contain duplicate unit_id values")
        if not isinstance(self.policy, EvidenceUnitGatePolicy):
            raise ValueError("policy must be an EvidenceUnitGatePolicy")

    @property
    def generate_unit_ids(self) -> tuple[str, ...]:
        return tuple(
            decision.unit_id
            for decision in self.decisions
            if decision.action is EvidenceUnitGateAction.GENERATE
        )

    @property
    def retry_unit_ids(self) -> tuple[str, ...]:
        return tuple(
            decision.unit_id
            for decision in self.decisions
            if decision.action is EvidenceUnitGateAction.RETRY
        )

    @property
    def terminal_unit_ids(self) -> tuple[str, ...]:
        return tuple(
            decision.unit_id
            for decision in self.decisions
            if decision.action is EvidenceUnitGateAction.TERMINAL
        )

    @property
    def batch_can_generate(self) -> bool:
        if not self.generate_unit_ids:
            return False
        if not self.policy.require_all_required_units:
            return True
        return all(
            not decision.required or decision.action is EvidenceUnitGateAction.GENERATE
            for decision in self.decisions
        )

    @property
    def metrics(self) -> dict[str, int | bool]:
        counts = Counter(decision.action.value for decision in self.decisions)
        return {
            "planned_count": len(self.decisions),
            "generate_count": counts[EvidenceUnitGateAction.GENERATE.value],
            "retry_count": counts[EvidenceUnitGateAction.RETRY.value],
            "terminal_count": counts[EvidenceUnitGateAction.TERMINAL.value],
            "batch_can_generate": self.batch_can_generate,
        }

    def to_state(self) -> dict[str, Any]:
        return {
            "evidence_unit_gate_decisions": [
                decision.to_state() for decision in self.decisions
            ],
            "evidence_unit_generate_ids": list(self.generate_unit_ids),
            "evidence_unit_retry_ids": list(self.retry_unit_ids),
            "evidence_unit_terminal_ids": list(self.terminal_unit_ids),
            "evidence_unit_batch_can_generate": self.batch_can_generate,
            "evidence_unit_gate_metrics": self.metrics,
        }


def _execution_results(
    batch: EvidenceUnitBatchResult,
) -> tuple[EvidenceUnitExecutionResult, ...]:
    if not isinstance(batch, EvidenceUnitBatchResult):
        raise ValueError("execution_batch must be an EvidenceUnitBatchResult")
    results = tuple(batch.results)
    if not results:
        raise ValueError("execution_batch must contain at least one unit")
    if not all(isinstance(item, EvidenceUnitExecutionResult) for item in results):
        raise ValueError("execution_batch contains an invalid result")
    unit_ids = [result.unit.unit_id for result in results]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("execution_batch contains duplicate unit_id values")
    for result in results:
        if not isinstance(result.unit, EvidenceUnit):
            raise ValueError("execution result contains an invalid unit")
        if not isinstance(result.status, EvidenceUnitExecutionStatus):
            raise ValueError("execution result contains an invalid status")
        if (
            isinstance(result.retrieval_round, bool)
            or not isinstance(result.retrieval_round, int)
            or result.retrieval_round < 0
        ):
            raise ValueError("execution result contains an invalid retrieval_round")
    return results


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _ledger_identity(entry: Mapping[str, Any]) -> tuple[str, int, int]:
    chunk_id = str(entry.get("chunk_id") or "").strip()
    span_start = entry.get("span_start")
    span_end = entry.get("span_end")
    if (
        not chunk_id
        or isinstance(span_start, bool)
        or not isinstance(span_start, int)
        or span_start < 0
        or isinstance(span_end, bool)
        or not isinstance(span_end, int)
        or span_end <= span_start
    ):
        raise ValueError("evidence ledger contains an invalid view identity")
    return chunk_id, span_start, span_end


def _validate_execution_ledger(
    batch: EvidenceUnitBatchResult,
    executions: tuple[EvidenceUnitExecutionResult, ...],
) -> dict[str, Mapping[str, Any]]:
    ready_docs = [
        doc
        for execution in executions
        if execution.status is EvidenceUnitExecutionStatus.READY
        for doc in execution.selected_docs
    ]
    expected = build_evidence_ledger(ready_docs) if ready_docs else []
    declared_by_id: dict[str, Mapping[str, Any]] = {}
    identity_to_id: dict[tuple[str, int, int], str] = {}
    for entry in batch.evidence_ledger:
        evidence_id = str(entry.get("evidence_id") or "")
        if not is_valid_evidence_id(evidence_id):
            raise ValueError("evidence ledger contains an invalid evidence_id")
        if evidence_id in declared_by_id:
            raise ValueError("evidence ledger contains duplicate evidence_id values")
        identity = _ledger_identity(entry)
        alias = identity_to_id.get(identity)
        if alias is not None and alias != evidence_id:
            raise ValueError("evidence view is mapped by multiple evidence_id values")
        identity_to_id[identity] = evidence_id
        declared_by_id[evidence_id] = entry

    for expected_entry in expected:
        evidence_id = str(expected_entry["evidence_id"])
        declared = declared_by_id.get(evidence_id)
        if declared is None:
            raise ValueError(
                f"ready evidence {evidence_id} is absent from the execution ledger"
            )
        if any(declared.get(key) != value for key, value in expected_entry.items()):
            raise ValueError(
                f"ready evidence {evidence_id} does not exactly match the execution ledger"
            )
    return declared_by_id


def _candidate_evidence_ids(
    execution: EvidenceUnitExecutionResult,
) -> tuple[str, ...]:
    return tuple(
        str(_mapping(doc.get("retrieval")).get("evidence_id") or "").strip()
        for doc in execution.selected_docs
    )


def _validate_grounded_closure(
    execution: EvidenceUnitExecutionResult,
    verification: EvidenceUnitVerificationResult,
    ledger_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    candidate_ids = _candidate_evidence_ids(execution)
    if verification.candidate_evidence_ids != candidate_ids:
        raise ValueError(
            "verification candidate_evidence_ids do not match final execution "
            f"documents for unit_id {execution.unit.unit_id}"
        )

    grounded_statuses = {
        EvidenceClosureStatus.SUPPORTED,
        EvidenceClosureStatus.CONTRADICTORY,
    }
    if verification.status not in grounded_statuses:
        return

    if len(verification.closure.evidence) != len(candidate_ids):
        raise ValueError(
            "verification closure does not contain the exact final candidate set "
            f"for unit_id {execution.unit.unit_id}"
        )
    docs_by_id = {
        evidence_id: doc
        for evidence_id, doc in zip(
            candidate_ids, execution.selected_docs, strict=True
        )
    }
    for evidence_id, view in zip(
        candidate_ids, verification.closure.evidence, strict=True
    ):
        entry = ledger_by_id[evidence_id]
        chunk_id, span_start, span_end = _ledger_identity(entry)
        meta = _mapping(docs_by_id[evidence_id].get("meta"))
        expected_source = str(
            meta.get("source") or meta.get("knowledge_id") or ""
        ).strip()
        expected_source_type = str(meta.get("source_type") or "document").strip()
        if (
            view.chunk_id != chunk_id
            or view.source != expected_source
            or view.source_type.value != expected_source_type
            or view.related_source != str(meta.get("related_source") or "").strip()
            or view.parent_chunk_id
            != str(meta.get("parent_chunk_id") or "").strip()
            or (view.span_start, view.span_end) != (span_start, span_end)
        ):
            raise ValueError(
                "verification closure evidence view does not exactly match "
                f"{evidence_id} for unit_id {execution.unit.unit_id}"
            )

    grounded_chunk_ids = tuple(
        dict.fromkeys(
            str(ledger_by_id[evidence_id].get("chunk_id") or "")
            for evidence_id in verification.evidence_ids
        )
    )
    if (
        verification.evidence_chunk_ids != grounded_chunk_ids
        or verification.closure.grounding_chunk_ids != grounded_chunk_ids
    ):
        raise ValueError(
            "verification grounding IDs do not match cited final evidence "
            f"for unit_id {execution.unit.unit_id}"
        )


def _verification_results(
    verification_batch: EvidenceUnitVerificationBatchResult,
) -> tuple[EvidenceUnitVerificationResult, ...]:
    # Local import keeps the service contract free from an import-time dependency
    # on a particular verifier implementation while retaining exact runtime checks.
    from cogdoc.agents.evidence_unit_verifier import EvidenceUnitVerificationResult

    results = tuple(verification_batch.results)
    if not all(isinstance(item, EvidenceUnitVerificationResult) for item in results):
        raise ValueError("verification_batch contains an invalid result")
    unit_ids = [result.unit.unit_id for result in results]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("verification_batch contains duplicate unit_id values")
    return results


_UNIT_SCOPED_PROTOCOL_ERRORS = frozenset(
    {
        "missing_unit",
        "duplicate_unit",
        "duplicate_evidence_id",
        "unknown_evidence_id",
        "supported_without_evidence",
        "contradictory_without_evidence",
        "no_evidence_with_evidence",
    }
)


def _protocol_error_unit_ids(
    protocol_errors: Sequence[str], execution_ids: Sequence[str]
) -> frozenset[str]:
    known_ids = set(execution_ids)
    affected: set[str] = set()
    for error in protocol_errors:
        parts = error.split(":", 2)
        if (
            len(parts) >= 2
            and parts[0] in _UNIT_SCOPED_PROTOCOL_ERRORS
            and parts[1] in known_ids
        ):
            affected.add(parts[1])
    return frozenset(affected)


def _validate_batch_alignment(
    executions: tuple[EvidenceUnitExecutionResult, ...],
    verification_batch: EvidenceUnitVerificationBatchResult,
    *,
    ledger_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[EvidenceUnitVerificationResult, ...]:
    verifications = _verification_results(verification_batch)
    execution_ids = tuple(result.unit.unit_id for result in executions)
    verification_ids = tuple(result.unit.unit_id for result in verifications)
    if verification_ids != execution_ids:
        if set(verification_ids) == set(execution_ids) and len(verification_ids) == len(
            execution_ids
        ):
            raise ValueError(
                "verification_batch unit order does not match execution_batch"
            )
        missing = sorted(set(execution_ids) - set(verification_ids))
        unknown = sorted(set(verification_ids) - set(execution_ids))
        raise ValueError(
            "verification_batch must contain exactly one ordered result per execution "
            f"unit (missing={missing}, unknown={unknown})"
        )

    ready_verifications: list[EvidenceUnitVerificationResult] = []
    for execution, verification in zip(executions, verifications, strict=True):
        if verification.unit != execution.unit:
            raise ValueError(
                f"verification unit plan differs for unit_id {execution.unit.unit_id}"
            )
        if verification.closure.retrieval_round != execution.retrieval_round:
            raise ValueError(
                f"verification retrieval_round differs for unit_id {execution.unit.unit_id}"
            )
        if execution.status is EvidenceUnitExecutionStatus.READY:
            _validate_grounded_closure(execution, verification, ledger_by_id)
        if execution.status is EvidenceUnitExecutionStatus.READY:
            ready_verifications.append(verification)
            allowed = {
                EvidenceClosureStatus.SUPPORTED,
                EvidenceClosureStatus.NO_EVIDENCE,
                EvidenceClosureStatus.CONTRADICTORY,
                EvidenceClosureStatus.VERIFICATION_ERROR,
            }
            if verification.status not in allowed:
                raise ValueError(
                    "READY execution has incompatible verification status "
                    f"{verification.status.value} for unit_id {execution.unit.unit_id}"
                )
        else:
            expected = {
                EvidenceUnitExecutionStatus.NO_EVIDENCE: (
                    EvidenceClosureStatus.NO_EVIDENCE
                ),
                EvidenceUnitExecutionStatus.RETRIEVAL_ERROR: (
                    EvidenceClosureStatus.RETRIEVAL_ERROR
                ),
                EvidenceUnitExecutionStatus.BUDGET_EXHAUSTED: (
                    EvidenceClosureStatus.BUDGET_EXHAUSTED
                ),
            }[execution.status]
            if verification.status is not expected:
                raise ValueError(
                    "operational execution status was reclassified for unit_id "
                    f"{execution.unit.unit_id}"
                )

        if (
            execution.status is EvidenceUnitExecutionStatus.READY
            and verification.status is EvidenceClosureStatus.NO_EVIDENCE
            and (
                verification.error_class
                or verification.reason_code.startswith("verification_")
            )
        ):
            raise ValueError(
                "verification failure cannot be represented as semantic no_evidence "
                f"for unit_id {execution.unit.unit_id}"
            )

    verification_by_id = {
        result.unit.unit_id: result for result in ready_verifications
    }
    affected_unit_ids = _protocol_error_unit_ids(
        verification_batch.protocol_errors, tuple(verification_by_id)
    )
    for unit_id in affected_unit_ids:
        if (
            verification_by_id[unit_id].status
            is not EvidenceClosureStatus.VERIFICATION_ERROR
        ):
            raise ValueError(
                "unit-scoped verification protocol failure must fail its READY "
                f"unit closed: {unit_id}"
            )
    if verification_batch.error_class and ready_verifications and not any(
        result.status is EvidenceClosureStatus.VERIFICATION_ERROR
        for result in ready_verifications
    ):
        raise ValueError(
            "verification model failure must fail at least one READY unit closed"
        )
    return verifications


def _retries_remaining(execution: EvidenceUnitExecutionResult) -> int:
    return max(
        0,
        execution.unit.policy.max_retrieval_retries - execution.retrieval_round,
    )


def _decision(
    execution: EvidenceUnitExecutionResult,
    *,
    action: EvidenceUnitGateAction,
    verification_status: EvidenceClosureStatus | None,
    reason_code: str,
    source_reason_code: str,
) -> EvidenceUnitGateDecision:
    return EvidenceUnitGateDecision(
        unit=execution.unit,
        action=action,
        execution_status=execution.status,
        verification_status=verification_status,
        retrieval_round=execution.retrieval_round,
        retries_remaining=_retries_remaining(execution),
        reason_code=reason_code,
        source_reason_code=source_reason_code,
    )


def _no_evidence_decision(
    execution: EvidenceUnitExecutionResult,
    *,
    verification_status: EvidenceClosureStatus | None,
    source_reason_code: str,
) -> EvidenceUnitGateDecision:
    remaining = _retries_remaining(execution)
    return _decision(
        execution,
        action=(
            EvidenceUnitGateAction.RETRY
            if remaining > 0
            else EvidenceUnitGateAction.TERMINAL
        ),
        verification_status=verification_status,
        reason_code=(
            "no_evidence_retry_available"
            if remaining > 0
            else "no_evidence_retries_exhausted"
        ),
        source_reason_code=source_reason_code,
    )


def _contradictory_decision(
    execution: EvidenceUnitExecutionResult,
    *,
    policy: EvidenceUnitGatePolicy,
    source_reason_code: str,
) -> EvidenceUnitGateDecision:
    action = policy.contradictory_action
    reason_code = {
        EvidenceUnitGateAction.GENERATE: "contradictory_generate_allowed",
        EvidenceUnitGateAction.RETRY: "contradictory_retry_available",
        EvidenceUnitGateAction.TERMINAL: "contradictory_terminal",
    }[action]
    if action is EvidenceUnitGateAction.RETRY and _retries_remaining(execution) == 0:
        action = EvidenceUnitGateAction.TERMINAL
        reason_code = "contradictory_retries_exhausted"
    return _decision(
        execution,
        action=action,
        verification_status=EvidenceClosureStatus.CONTRADICTORY,
        reason_code=reason_code,
        source_reason_code=source_reason_code,
    )


def _gate_without_verification(
    execution: EvidenceUnitExecutionResult,
    *,
    policy: EvidenceUnitGatePolicy,
) -> EvidenceUnitGateDecision:
    if execution.status is EvidenceUnitExecutionStatus.READY:
        return _decision(
            execution,
            action=(
                EvidenceUnitGateAction.GENERATE
                if policy.allow_unverified_ready
                else EvidenceUnitGateAction.TERMINAL
            ),
            verification_status=None,
            reason_code=(
                "unverified_ready_allowed"
                if policy.allow_unverified_ready
                else "verification_required"
            ),
            source_reason_code=execution.reason_code,
        )
    if execution.status is EvidenceUnitExecutionStatus.NO_EVIDENCE:
        return _no_evidence_decision(
            execution,
            verification_status=None,
            source_reason_code=execution.reason_code,
        )
    reason_code = {
        EvidenceUnitExecutionStatus.RETRIEVAL_ERROR: "retrieval_error_terminal",
        EvidenceUnitExecutionStatus.BUDGET_EXHAUSTED: "budget_exhausted_terminal",
    }[execution.status]
    return _decision(
        execution,
        action=EvidenceUnitGateAction.TERMINAL,
        verification_status=None,
        reason_code=reason_code,
        source_reason_code=execution.reason_code,
    )


def _gate_with_verification(
    execution: EvidenceUnitExecutionResult,
    verification: EvidenceUnitVerificationResult,
    *,
    policy: EvidenceUnitGatePolicy,
) -> EvidenceUnitGateDecision:
    status = verification.status
    if status is EvidenceClosureStatus.SUPPORTED:
        return _decision(
            execution,
            action=EvidenceUnitGateAction.GENERATE,
            verification_status=status,
            reason_code="verified_supported",
            source_reason_code=verification.reason_code,
        )
    if status is EvidenceClosureStatus.NO_EVIDENCE:
        return _no_evidence_decision(
            execution,
            verification_status=status,
            source_reason_code=verification.reason_code,
        )
    if status is EvidenceClosureStatus.CONTRADICTORY:
        return _contradictory_decision(
            execution,
            policy=policy,
            source_reason_code=verification.reason_code,
        )
    reason_code = {
        EvidenceClosureStatus.RETRIEVAL_ERROR: "retrieval_error_terminal",
        EvidenceClosureStatus.VERIFICATION_ERROR: "verification_error_terminal",
        EvidenceClosureStatus.BUDGET_EXHAUSTED: "budget_exhausted_terminal",
    }[status]
    return _decision(
        execution,
        action=EvidenceUnitGateAction.TERMINAL,
        verification_status=status,
        reason_code=reason_code,
        source_reason_code=verification.reason_code,
    )


def evaluate_evidence_unit_gate(
    execution_batch: EvidenceUnitBatchResult,
    verification_batch: EvidenceUnitVerificationBatchResult | None = None,
    *,
    policy: EvidenceUnitGatePolicy | None = None,
) -> EvidenceUnitGateBatchResult:
    """Classify a complete Evidence Unit batch without task- or agent-specific logic."""

    normalized_policy = policy or EvidenceUnitGatePolicy()
    if not isinstance(normalized_policy, EvidenceUnitGatePolicy):
        raise ValueError("policy must be an EvidenceUnitGatePolicy")
    executions = _execution_results(execution_batch)
    ledger_by_id = _validate_execution_ledger(execution_batch, executions)
    if verification_batch is None:
        decisions = tuple(
            _gate_without_verification(execution, policy=normalized_policy)
            for execution in executions
        )
    else:
        verifications = _validate_batch_alignment(
            executions,
            verification_batch,
            ledger_by_id=ledger_by_id,
        )
        decisions = tuple(
            _gate_with_verification(
                execution,
                verification,
                policy=normalized_policy,
            )
            for execution, verification in zip(executions, verifications, strict=True)
        )
    return EvidenceUnitGateBatchResult(
        decisions=decisions,
        policy=normalized_policy,
    )


__all__ = [
    "EvidenceUnitGateAction",
    "EvidenceUnitGateBatchResult",
    "EvidenceUnitGateDecision",
    "EvidenceUnitGatePolicy",
    "evaluate_evidence_unit_gate",
]
