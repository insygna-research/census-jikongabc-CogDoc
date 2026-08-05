from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, cast

from cogdoc.agents.evidence_unit_verifier import (
    EvidenceUnitVerificationBatchResult,
    EvidenceUnitVerificationResult,
    verify_evidence_unit_batch,
)
from cogdoc.config.settings import get_settings
from cogdoc.graph.state import EvidenceLedgerEntry, RetrievedDoc
from cogdoc.service.evidence_unit_pipeline import (
    EvidenceUnitBatchResult,
    EvidenceUnitExecutionResult,
    EvidenceUnitExecutionStatus,
)
from cogdoc.service.evidence_units import (
    EvidenceClosureStatus,
    EvidenceUnit,
    build_qa_evidence_units,
)
from cogdoc.tools.citation_ledger import (
    CitationLedgerError,
    build_evidence_ledger,
    validate_evidence_citations,
)
from cogdoc.tools.evidence_rendering import evidence_block_char_count


class QAEvidenceUnitAdapterOutcome(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    VERIFIED = "verified"
    VERIFICATION_ERROR = "verification_error"


class QAEvidenceUnitBatchVerifier(Protocol):
    def __call__(
        self,
        batch: EvidenceUnitBatchResult,
        *,
        is_local: bool = False,
        max_chars_per_doc: int = 1600,
        max_units_per_batch: int = 8,
    ) -> EvidenceUnitVerificationBatchResult: ...


@dataclass(frozen=True, slots=True)
class QAEvidenceUnitAdapterResult:
    outcome: QAEvidenceUnitAdapterOutcome
    state_update: Mapping[str, Any] = field(default_factory=dict)
    batch: EvidenceUnitBatchResult | None = None
    verification: EvidenceUnitVerificationBatchResult | None = None
    reason_code: str = ""
    error_class: str = ""


def _mapping_rows(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence of mappings")
    rows = tuple(value)
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{name} must contain only mappings")
    return cast(tuple[Mapping[str, Any], ...], rows)


def _verification_docs(state: Mapping[str, Any]) -> tuple[RetrievedDoc, ...]:
    rows = _mapping_rows(state.get("verification_docs"), "verification_docs")
    return tuple(cast(RetrievedDoc, copy.deepcopy(dict(row))) for row in rows)


def _validated_ledger(
    state: Mapping[str, Any], docs: Sequence[RetrievedDoc]
) -> tuple[EvidenceLedgerEntry, ...]:
    rows = _mapping_rows(state.get("evidence_ledger"), "evidence_ledger")
    validation = validate_evidence_citations(
        "", rows, require_citation=False
    )
    if not validation.get("is_valid"):
        raise CitationLedgerError(str(validation.get("critique") or "invalid ledger"))

    declared_by_id: dict[str, Mapping[str, Any]] = {
        str(row.get("evidence_id") or ""): row for row in rows
    }
    expected = build_evidence_ledger(docs) if docs else []
    for entry in expected:
        evidence_id = str(entry["evidence_id"])
        declared = declared_by_id.get(evidence_id)
        if declared is None:
            raise CitationLedgerError(
                f"verification evidence {evidence_id} is absent from the ledger"
            )
        if any(declared.get(key) != value for key, value in entry.items()):
            raise CitationLedgerError(
                f"verification evidence {evidence_id} does not match the ledger"
            )
    return tuple(
        cast(EvidenceLedgerEntry, copy.deepcopy(dict(row))) for row in rows
    )


def _retrieval_round(state: Mapping[str, Any]) -> int:
    value = state.get("retrieval_round", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("retrieval_round must be a non-negative integer")
    return value


def build_qa_verification_batch(
    state: Mapping[str, Any],
    *,
    max_retrieval_retries: int | None = None,
) -> EvidenceUnitBatchResult | None:
    """Rebuild the current QA verifier view without changing its frozen pack.

    QA requirements intentionally share the same verifier-visible document set in
    this compatibility stage.  The response ledger may contain additional views
    from the larger generation pack, but every verifier-visible view must already
    have an exact matching frozen EID entry.
    """

    requirements = _mapping_rows(
        state.get("evidence_requirements"), "evidence_requirements"
    )
    if not requirements:
        return None
    settings = get_settings()
    retry_limit = (
        settings.qa_adaptive_retrieval_max_retries
        if max_retrieval_retries is None
        else max_retrieval_retries
    )
    units = build_qa_evidence_units(
        str(state.get("query") or ""),
        requirements,
        max_retrieval_retries=retry_limit,
    )
    docs = _verification_docs(state)
    ledger = _validated_ledger(state, docs)
    retrieval_round = _retrieval_round(state)
    selected_chars = sum(
        evidence_block_char_count(doc, str(doc.get("text") or "")) for doc in docs
    )
    results: list[EvidenceUnitExecutionResult] = []
    for unit in units:
        if docs:
            results.append(
                EvidenceUnitExecutionResult(
                    unit=unit,
                    status=EvidenceUnitExecutionStatus.READY,
                    selected_docs=tuple(copy.deepcopy(docs)),
                    retrieval_round=retrieval_round,
                    executed_queries=(unit.retrieval_query,),
                    candidate_count=len(docs),
                    selected_chars=selected_chars,
                )
            )
        else:
            results.append(
                EvidenceUnitExecutionResult(
                    unit=unit,
                    status=EvidenceUnitExecutionStatus.NO_EVIDENCE,
                    retrieval_round=retrieval_round,
                    executed_queries=(unit.retrieval_query,),
                    reason_code="no_verification_docs",
                )
            )
    return EvidenceUnitBatchResult(results=tuple(results), evidence_ledger=ledger)


def _ordered_verification_results(
    batch: EvidenceUnitBatchResult,
    verification: EvidenceUnitVerificationBatchResult,
) -> tuple[EvidenceUnitVerificationResult, ...]:
    expected = {result.unit.unit_id: result.unit for result in batch.results}
    by_id: dict[str, EvidenceUnitVerificationResult] = {}
    for result in verification.results:
        unit_id = result.unit.unit_id
        if unit_id in by_id:
            raise ValueError("generic verifier returned a duplicate unit result")
        if unit_id not in expected or result.unit != expected[unit_id]:
            raise ValueError("generic verifier returned an unknown or changed unit")
        by_id[unit_id] = result
    if set(by_id) != set(expected):
        raise ValueError("generic verifier omitted a QA evidence unit")
    return tuple(by_id[result.unit.unit_id] for result in batch.results)


def _requirement_ids(units: Sequence[EvidenceUnit]) -> list[str]:
    return [unit.binding.requirement_id for unit in units]


def _failure_state(
    units: Sequence[EvidenceUnit],
    *,
    reason: str,
    error_class: str,
) -> dict[str, Any]:
    requirement_ids = _requirement_ids(units)
    return {
        "evidence_verification_required": True,
        "evidence_supported": False,
        "evidence_verification_reason": reason,
        "evidence_verified_chunk_ids": [],
        "evidence_requirement_assessments": [
            {
                "requirement_id": requirement_id,
                "verdict": "missing",
                "evidence_chunk_ids": [],
                "reason": reason,
            }
            for requirement_id in requirement_ids
        ],
        "missing_evidence_requirement_ids": requirement_ids,
        "evidence_verifier_error": error_class,
        "retrieval_abstained": True,
        "retrieval_abstain_reason": "evidence_verifier_error",
    }


def _legacy_state(
    ordered: Sequence[EvidenceUnitVerificationResult],
) -> dict[str, Any]:
    assessments: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    verified_chunk_ids: list[str] = []
    reasons: list[str] = []
    for result in ordered:
        requirement_id = result.unit.binding.requirement_id
        reason = result.reason or result.reason_code
        reasons.append(f"{requirement_id}: {reason}")
        if result.status is EvidenceClosureStatus.SUPPORTED:
            verdict = "supported"
            chunk_ids = list(result.evidence_chunk_ids)
            for chunk_id in chunk_ids:
                if chunk_id not in verified_chunk_ids:
                    verified_chunk_ids.append(chunk_id)
        elif result.status is EvidenceClosureStatus.CONTRADICTORY:
            verdict = "contradictory"
            chunk_ids = list(result.evidence_chunk_ids)
            missing_ids.append(requirement_id)
        elif result.status is EvidenceClosureStatus.NO_EVIDENCE:
            verdict = "missing"
            chunk_ids = []
            missing_ids.append(requirement_id)
        else:  # Operational statuses are handled before legacy mapping.
            raise ValueError("operational verification status cannot be mapped as semantic")
        assessments.append(
            {
                "requirement_id": requirement_id,
                "verdict": verdict,
                "evidence_chunk_ids": chunk_ids,
                "reason": reason,
            }
        )

    supported = not missing_ids and bool(ordered)
    return {
        "evidence_verification_required": True,
        "evidence_supported": supported,
        "evidence_verification_reason": "；".join(reasons),
        "evidence_verified_chunk_ids": verified_chunk_ids,
        "evidence_requirement_assessments": assessments,
        "missing_evidence_requirement_ids": missing_ids,
        "evidence_verifier_error": "",
        "retrieval_abstained": not supported,
        "retrieval_abstain_reason": (
            "evidence_supported" if supported else "evidence_not_supported"
        ),
    }


def _verification_error_class(
    verification: EvidenceUnitVerificationBatchResult,
) -> str:
    if verification.error_class:
        return verification.error_class
    if verification.protocol_errors:
        return "EvidenceUnitVerificationProtocolError"
    for result in verification.results:
        if result.error_class:
            return result.error_class
    return "EvidenceUnitVerificationError"


def adapt_qa_evidence_verification(
    state: Mapping[str, Any],
    *,
    verifier: QAEvidenceUnitBatchVerifier = verify_evidence_unit_batch,
) -> QAEvidenceUnitAdapterResult:
    """Run the generic closed-set verifier behind the legacy QA state contract."""

    settings = get_settings()
    batch: EvidenceUnitBatchResult | None = None
    try:
        batch = build_qa_verification_batch(
            state,
            max_retrieval_retries=settings.qa_adaptive_retrieval_max_retries,
        )
    except Exception as exc:
        units: tuple[EvidenceUnit, ...] = ()
        try:
            raw_requirements = _mapping_rows(
                state.get("evidence_requirements"), "evidence_requirements"
            )
            if raw_requirements:
                units = build_qa_evidence_units(
                    str(state.get("query") or ""),
                    raw_requirements,
                    max_retrieval_retries=(
                        settings.qa_adaptive_retrieval_max_retries
                    ),
                )
        except Exception:
            units = ()
        error_class = type(exc).__name__
        reason = "证据单元闭集无效，已安全拒答"
        return QAEvidenceUnitAdapterResult(
            outcome=QAEvidenceUnitAdapterOutcome.VERIFICATION_ERROR,
            state_update=_failure_state(
                units, reason=reason, error_class=error_class
            ),
            reason_code="invalid_qa_evidence_batch",
            error_class=error_class,
        )

    if batch is None:
        return QAEvidenceUnitAdapterResult(
            outcome=QAEvidenceUnitAdapterOutcome.NOT_APPLICABLE,
            reason_code="no_evidence_requirements",
        )

    units = tuple(result.unit for result in batch.results)
    try:
        verification = verifier(
            batch,
            is_local=bool(state.get("is_local", False)),
            max_chars_per_doc=settings.qa_evidence_verify_max_chars_per_doc,
            max_units_per_batch=settings.evidence_unit_verify_max_units_per_batch,
        )
        ordered = _ordered_verification_results(batch, verification)
    except Exception as exc:
        error_class = type(exc).__name__
        reason = "证据需求校验器异常，已安全拒答"
        return QAEvidenceUnitAdapterResult(
            outcome=QAEvidenceUnitAdapterOutcome.VERIFICATION_ERROR,
            state_update=_failure_state(
                units, reason=reason, error_class=error_class
            ),
            batch=batch,
            reason_code="verification_model_error",
            error_class=error_class,
        )

    operational = {
        EvidenceClosureStatus.RETRIEVAL_ERROR,
        EvidenceClosureStatus.VERIFICATION_ERROR,
        EvidenceClosureStatus.BUDGET_EXHAUSTED,
    }
    if verification.error_class or any(
        result.status in operational for result in ordered
    ):
        error_class = _verification_error_class(verification)
        reason = "证据需求校验未能安全完成，已安全拒答"
        return QAEvidenceUnitAdapterResult(
            outcome=QAEvidenceUnitAdapterOutcome.VERIFICATION_ERROR,
            state_update=_failure_state(
                units, reason=reason, error_class=error_class
            ),
            batch=batch,
            verification=verification,
            reason_code="verification_error",
            error_class=error_class,
        )

    return QAEvidenceUnitAdapterResult(
        outcome=QAEvidenceUnitAdapterOutcome.VERIFIED,
        state_update=_legacy_state(ordered),
        batch=batch,
        verification=verification,
    )


__all__ = [
    "QAEvidenceUnitAdapterOutcome",
    "QAEvidenceUnitAdapterResult",
    "QAEvidenceUnitBatchVerifier",
    "adapt_qa_evidence_verification",
    "build_qa_verification_batch",
]
