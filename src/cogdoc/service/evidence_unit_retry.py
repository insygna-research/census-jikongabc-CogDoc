"""Agent-independent targeted retry and stable evidence-ledger merging."""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

from cogdoc.graph.state import EvidenceLedgerEntry, RetrievalMetrics, RetrievedDoc
from cogdoc.service.evidence_unit_pipeline import (
    EvidenceUnitBatchResult,
    EvidenceUnitExecutionResult,
    EvidenceUnitExecutionStatus,
)
from cogdoc.service.evidence_units import (
    EvidenceSourceType,
    EvidenceUnit,
    EvidenceUnitBudget,
)
from cogdoc.tools.citation_ledger import (
    assign_evidence_ids,
    build_evidence_ledger,
    format_evidence_id,
    is_valid_evidence_id,
)
from cogdoc.tools.evidence_rendering import evidence_block_char_count


EvidenceUnitRetryRunner = Callable[[tuple[EvidenceUnit, ...]], EvidenceUnitBatchResult]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _doc_cost(doc: Mapping[str, Any]) -> int:
    return evidence_block_char_count(doc, str(doc.get("text") or ""))


def _ledger_identity(entry: Mapping[str, Any]) -> tuple[str, int, int]:
    chunk_id = str(entry.get("chunk_id") or "").strip()
    start = entry.get("span_start")
    end = entry.get("span_end")
    if (
        not chunk_id
        or isinstance(start, bool)
        or not isinstance(start, int)
        or start < 0
        or isinstance(end, bool)
        or not isinstance(end, int)
        or end <= start
    ):
        raise ValueError("evidence ledger contains an invalid view identity")
    return chunk_id, start, end


def _result_docs(
    results: Sequence[EvidenceUnitExecutionResult],
) -> list[RetrievedDoc]:
    return [
        doc
        for result in results
        if result.status is EvidenceUnitExecutionStatus.READY
        for doc in result.selected_docs
    ]


def _validate_batch_registry(batch: EvidenceUnitBatchResult, *, name: str) -> None:
    ids = [result.unit.unit_id for result in batch.results]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} contains duplicate unit results")
    docs = _result_docs(batch.results)
    rebuilt = build_evidence_ledger(docs) if docs else []
    rebuilt_by_id = {
        str(entry.get("evidence_id") or ""): dict(entry) for entry in rebuilt
    }
    declared_by_id: dict[str, dict[str, Any]] = {}
    for raw_entry in batch.evidence_ledger:
        entry = dict(raw_entry)
        evidence_id = str(entry.get("evidence_id") or "")
        if not is_valid_evidence_id(evidence_id):
            raise ValueError(f"{name} contains an invalid evidence_id")
        if evidence_id in declared_by_id:
            raise ValueError(f"{name} contains duplicate evidence_id values")
        _ledger_identity(entry)
        declared_by_id[evidence_id] = entry
    if rebuilt_by_id != declared_by_id:
        raise ValueError(f"{name} ledger does not exactly match its ready documents")


def _source_type(doc: Mapping[str, Any]) -> EvidenceSourceType | None:
    value = str(_mapping(doc.get("meta")).get("source_type") or "document")
    try:
        return EvidenceSourceType(value)
    except ValueError:
        return None


def _scope_violations(result: EvidenceUnitExecutionResult) -> int:
    violations = 0
    for doc in result.selected_docs:
        meta = _mapping(doc.get("meta"))
        source_type = _source_type(doc)
        if source_type is None or not result.unit.scope.contains(
            source=str(meta.get("source") or ""),
            source_type=source_type or EvidenceSourceType.DOCUMENT,
            related_source=str(meta.get("related_source") or ""),
        ):
            violations += 1
    return violations


def _doc_identity(doc: RetrievedDoc) -> tuple[str, int, int]:
    entries = build_evidence_ledger([doc])
    if len(entries) != 1:  # pragma: no cover - one validated doc has one view.
        raise ValueError("evidence document does not resolve to exactly one view")
    return _ledger_identity(entries[0])


def _retry_first_union(
    initial_docs: Sequence[RetrievedDoc],
    retried_docs: Sequence[RetrievedDoc],
) -> tuple[RetrievedDoc, ...]:
    old_by_identity: dict[tuple[str, int, int], RetrievedDoc] = {}
    for doc in initial_docs:
        old_by_identity.setdefault(_doc_identity(doc), doc)
    unique: "OrderedDict[tuple[str, int, int], RetrievedDoc]" = OrderedDict()
    for doc in retried_docs:
        identity = _doc_identity(doc)
        # An exact retry duplicate keeps the original frozen snapshot and EID,
        # while occupying the retry-first position in the candidate order.
        unique.setdefault(identity, old_by_identity.get(identity, doc))
    for doc in initial_docs:
        unique.setdefault(_doc_identity(doc), doc)
    return tuple(unique.values())


def _operational_failure(
    initial: EvidenceUnitExecutionResult,
    *,
    error_class: str,
    reason_code: str = "targeted_retry_failed",
) -> EvidenceUnitExecutionResult:
    return EvidenceUnitExecutionResult(
        unit=initial.unit,
        status=EvidenceUnitExecutionStatus.RETRIEVAL_ERROR,
        retrieval_round=initial.retrieval_round + 1,
        executed_queries=initial.executed_queries,
        candidate_count=initial.candidate_count,
        parent_context_count=initial.parent_context_count,
        neighbor_context_count=initial.neighbor_context_count,
        scope_violation_count=initial.scope_violation_count,
        span_input_chars=initial.span_input_chars,
        span_selected_chars=initial.span_selected_chars,
        reason_code=reason_code,
        error_class=error_class,
        fallback_used=initial.fallback_used,
    )


def _merge_retry_result(
    initial: EvidenceUnitExecutionResult,
    retried: EvidenceUnitExecutionResult,
) -> EvidenceUnitExecutionResult:
    if retried.unit != initial.unit:
        raise ValueError("retry result changed the immutable evidence unit plan")
    if retried.status is EvidenceUnitExecutionStatus.NO_EVIDENCE and (
        retried.error_class or "error" in retried.reason_code.casefold()
    ):
        retried = replace(
            retried,
            status=EvidenceUnitExecutionStatus.RETRIEVAL_ERROR,
            selected_docs=(),
            selected_chars=0,
            error_class=retried.error_class or "EvidenceUnitRetryError",
            reason_code="targeted_retry_operational_error",
        )
    violations = _scope_violations(retried)
    if violations:
        retried = replace(
            retried,
            status=EvidenceUnitExecutionStatus.RETRIEVAL_ERROR,
            selected_docs=(),
            selected_chars=0,
            reason_code="targeted_retry_scope_violation",
            error_class="EvidenceUnitRetryScopeViolation",
            scope_violation_count=retried.scope_violation_count + violations,
        )

    merged_status = retried.status
    merged_docs = retried.selected_docs
    merged_reason = retried.reason_code
    merged_error = retried.error_class
    if initial.status is EvidenceUnitExecutionStatus.READY:
        if retried.status is EvidenceUnitExecutionStatus.READY:
            merged_docs = _retry_first_union(
                initial.selected_docs, retried.selected_docs
            )
        elif retried.status is EvidenceUnitExecutionStatus.NO_EVIDENCE:
            # Semantic verification may have found the original evidence
            # insufficient, but an empty retry does not erase an already valid
            # execution result.  The second verifier decides semantic closure.
            merged_status = EvidenceUnitExecutionStatus.READY
            merged_docs = initial.selected_docs
            merged_reason = ""
            merged_error = ""
    elif retried.status is EvidenceUnitExecutionStatus.READY:
        merged_docs = _retry_first_union((), retried.selected_docs)

    return replace(
        retried,
        status=merged_status,
        selected_docs=merged_docs,
        retrieval_round=max(initial.retrieval_round + 1, retried.retrieval_round),
        executed_queries=initial.executed_queries + retried.executed_queries,
        candidate_count=initial.candidate_count + retried.candidate_count,
        parent_context_count=(
            initial.parent_context_count + retried.parent_context_count
        ),
        neighbor_context_count=(
            initial.neighbor_context_count + retried.neighbor_context_count
        ),
        scope_violation_count=(
            initial.scope_violation_count + retried.scope_violation_count
        ),
        span_input_chars=initial.span_input_chars + retried.span_input_chars,
        span_selected_chars=(initial.span_selected_chars + retried.span_selected_chars),
        selected_chars=sum(_doc_cost(doc) for doc in merged_docs),
        reason_code=merged_reason,
        error_class=merged_error,
        fallback_used=initial.fallback_used or retried.fallback_used,
    )


def _normalize_retry_ids(
    batch: EvidenceUnitBatchResult, retry_unit_ids: Sequence[str]
) -> tuple[str, ...]:
    if isinstance(retry_unit_ids, (str, bytes, bytearray)):
        raise ValueError("retry_unit_ids must be a sequence")
    requested = tuple(str(value).strip() for value in retry_unit_ids)
    if not requested or any(not value for value in requested):
        raise ValueError("at least one non-empty retry unit_id is required")
    if len(requested) != len(set(requested)):
        raise ValueError("retry_unit_ids must not contain duplicates")
    by_id = {result.unit.unit_id: result for result in batch.results}
    unknown = [unit_id for unit_id in requested if unit_id not in by_id]
    if unknown:
        raise ValueError("retry_unit_ids contains an unknown unit")
    for unit_id in requested:
        result = by_id[unit_id]
        if result.status not in {
            EvidenceUnitExecutionStatus.READY,
            EvidenceUnitExecutionStatus.NO_EVIDENCE,
        }:
            raise ValueError(
                "only ready or no_evidence units are eligible for targeted retry"
            )
        if result.retrieval_round >= result.unit.policy.max_retrieval_retries:
            raise ValueError("targeted retry exceeds the unit retry policy")
    requested_set = set(requested)
    return tuple(
        result.unit.unit_id
        for result in batch.results
        if result.unit.unit_id in requested_set
    )


def _validate_result_budget(
    result: EvidenceUnitExecutionResult, budget: EvidenceUnitBudget
) -> bool:
    return (
        len(result.selected_docs) <= budget.max_docs_per_unit
        and sum(_doc_cost(doc) for doc in result.selected_docs)
        <= budget.max_chars_per_unit
    )


def _bounded_unit_candidates(
    docs: Sequence[RetrievedDoc], budget: EvidenceUnitBudget
) -> tuple[RetrievedDoc, ...]:
    selected: list[RetrievedDoc] = []
    selected_chars = 0
    for doc in docs:
        if len(selected) >= budget.max_docs_per_unit:
            break
        cost = _doc_cost(doc)
        if selected_chars + cost > budget.max_chars_per_unit:
            continue
        selected.append(doc)
        selected_chars += cost
    return tuple(selected)


def _apply_retry_budget(
    initial: EvidenceUnitBatchResult,
    results: Sequence[EvidenceUnitExecutionResult],
    *,
    retry_ids: set[str],
    budget: EvidenceUnitBudget,
) -> list[EvidenceUnitExecutionResult]:
    budget.validate_plan_capacity([result.unit for result in results])
    preserved_ready = [
        result
        for result in initial.results
        if result.unit.unit_id not in retry_ids
        if result.status is EvidenceUnitExecutionStatus.READY
    ]
    if any(not _validate_result_budget(result, budget) for result in preserved_ready):
        raise ValueError("non-target results already exceed a per-unit budget")
    used_docs = sum(len(result.selected_docs) for result in preserved_ready)
    used_chars = sum(
        _doc_cost(doc) for result in preserved_ready for doc in result.selected_docs
    )
    if used_docs > budget.max_total_docs or used_chars > budget.max_total_chars:
        raise ValueError("non-target results already exceed the global budget")

    candidates_by_id: dict[str, tuple[RetrievedDoc, ...]] = {}
    for result in results:
        unit_id = result.unit.unit_id
        if unit_id in retry_ids and result.status is EvidenceUnitExecutionStatus.READY:
            candidates_by_id[unit_id] = _bounded_unit_candidates(
                result.selected_docs, budget
            )

    groups: "OrderedDict[str, list[EvidenceUnitExecutionResult]]" = OrderedDict()
    for result in results:
        unit_id = result.unit.unit_id
        if unit_id in candidates_by_id:
            groups.setdefault(result.unit.policy.admission_group, []).append(result)

    allocated: dict[str, list[RetrievedDoc]] = {}
    exhausted: set[str] = set()
    admitted: list[EvidenceUnitExecutionResult] = []
    for group_results in groups.values():
        required = [result for result in group_results if result.unit.policy.required]
        reservation = {
            result.unit.unit_id: list(
                candidates_by_id[result.unit.unit_id][
                    : budget.min_docs_per_required_unit
                ]
            )
            for result in required
        }
        reservation_docs = sum(len(docs) for docs in reservation.values())
        reservation_chars = sum(
            _doc_cost(doc) for docs in reservation.values() for doc in docs
        )
        if (
            any(
                len(docs) < budget.min_docs_per_required_unit
                for docs in reservation.values()
            )
            or used_docs + reservation_docs > budget.max_total_docs
            or used_chars + reservation_chars > budget.max_total_chars
        ):
            exhausted.update(result.unit.unit_id for result in group_results)
            continue
        for result in group_results:
            unit_id = result.unit.unit_id
            docs = reservation.get(unit_id, [])
            allocated[unit_id] = docs
            used_docs += len(docs)
            used_chars += sum(_doc_cost(doc) for doc in docs)
            admitted.append(result)

    cursor = {
        result.unit.unit_id: len(allocated.get(result.unit.unit_id, ()))
        for result in admitted
    }
    while admitted and used_docs < budget.max_total_docs:
        progressed = False
        for result in admitted:
            unit_id = result.unit.unit_id
            index = cursor[unit_id]
            candidates = candidates_by_id[unit_id]
            if index >= len(candidates):
                continue
            doc = candidates[index]
            cursor[unit_id] += 1
            cost = _doc_cost(doc)
            if used_chars + cost > budget.max_total_chars:
                continue
            allocated[unit_id].append(doc)
            used_docs += 1
            used_chars += cost
            progressed = True
            if used_docs >= budget.max_total_docs:
                break
        if not progressed:
            break

    output: list[EvidenceUnitExecutionResult] = []
    for result in results:
        unit_id = result.unit.unit_id
        if unit_id not in retry_ids:
            output.append(result)
            continue
        if result.status is not EvidenceUnitExecutionStatus.READY:
            output.append(result)
            continue
        selected_docs = tuple(allocated.get(unit_id, ()))
        if unit_id in exhausted or not selected_docs:
            output.append(
                replace(
                    result,
                    status=EvidenceUnitExecutionStatus.BUDGET_EXHAUSTED,
                    selected_docs=(),
                    selected_chars=0,
                    reason_code="targeted_retry_budget_exhausted",
                    error_class="",
                )
            )
            continue
        output.append(
            replace(
                result,
                selected_docs=selected_docs,
                selected_chars=sum(_doc_cost(doc) for doc in selected_docs),
            )
        )
    return output


def _prior_identity_ids(
    batch: EvidenceUnitBatchResult,
) -> dict[tuple[str, int, int], str]:
    identities: dict[tuple[str, int, int], str] = {}
    seen_ids: set[str] = set()
    for entry in batch.evidence_ledger:
        evidence_id = str(entry.get("evidence_id") or "")
        identity = _ledger_identity(entry)
        if evidence_id in seen_ids:
            raise ValueError("initial ledger contains duplicate evidence_id values")
        alias = identities.get(identity)
        if alias is not None and alias != evidence_id:
            raise ValueError("initial evidence view has multiple evidence_id values")
        seen_ids.add(evidence_id)
        identities[identity] = evidence_id
    return identities


def _sanitize_unit_doc(doc: RetrievedDoc, unit_id: str) -> RetrievedDoc:
    snapshot = copy.deepcopy(doc)
    retrieval = cast(RetrievalMetrics, dict(_mapping(snapshot.get("retrieval"))))
    retrieval["matched_requirement_ids"] = [unit_id]
    retrieval["matched_unit_ids"] = [unit_id]
    snapshot["retrieval"] = retrieval
    return snapshot


def _freeze_stable_evidence_ids(
    initial: EvidenceUnitBatchResult,
    results: Sequence[EvidenceUnitExecutionResult],
    *,
    retry_ids: set[str],
) -> tuple[list[EvidenceUnitExecutionResult], list[EvidenceLedgerEntry]]:
    prior = _prior_identity_ids(initial)
    target_counts: dict[str, int] = {}
    flattened_targets: list[RetrievedDoc] = []
    for result in results:
        unit_id = result.unit.unit_id
        if (
            unit_id not in retry_ids
            or result.status is not EvidenceUnitExecutionStatus.READY
        ):
            continue
        target_counts[unit_id] = len(result.selected_docs)
        flattened_targets.extend(
            _sanitize_unit_doc(doc, unit_id) for doc in result.selected_docs
        )

    annotated: list[RetrievedDoc] = []
    temporary_ledger: list[EvidenceLedgerEntry] = []
    if flattened_targets:
        annotated, temporary_ledger = assign_evidence_ids(flattened_targets)

    used_ids = set(prior.values())
    next_index = max((int(value[1:]) for value in used_ids), default=0) + 1
    temporary_to_stable: dict[str, str] = {}
    for raw_entry in temporary_ledger:
        entry = dict(raw_entry)
        temporary_id = str(entry["evidence_id"])
        identity = _ledger_identity(entry)
        stable_id = prior.get(identity)
        if stable_id is None:
            while format_evidence_id(next_index) in used_ids:
                next_index += 1
            stable_id = format_evidence_id(next_index)
            next_index += 1
        used_ids.add(stable_id)
        temporary_to_stable[temporary_id] = stable_id

    remapped_docs: list[RetrievedDoc] = []
    for doc in annotated:
        snapshot = copy.deepcopy(doc)
        retrieval = cast(RetrievalMetrics, dict(_mapping(snapshot.get("retrieval"))))
        temporary_id = str(retrieval.get("evidence_id") or "")
        retrieval["evidence_id"] = temporary_to_stable[temporary_id]
        snapshot["retrieval"] = retrieval
        remapped_docs.append(snapshot)

    cursor = 0
    frozen: list[EvidenceUnitExecutionResult] = []
    for result in results:
        unit_id = result.unit.unit_id
        if (
            unit_id not in retry_ids
            or result.status is not EvidenceUnitExecutionStatus.READY
        ):
            frozen.append(result)
            continue
        count = target_counts.get(unit_id, 0)
        docs = tuple(remapped_docs[cursor : cursor + count])
        cursor += count
        frozen.append(replace(result, selected_docs=docs))
    ledger = build_evidence_ledger(_result_docs(frozen)) if frozen else []
    ledger.sort(key=lambda entry: int(str(entry["evidence_id"])[1:]))
    return frozen, ledger


def merge_targeted_evidence_retry(
    initial: EvidenceUnitBatchResult,
    retried: EvidenceUnitBatchResult,
    retry_unit_ids: Sequence[str],
    *,
    budget: EvidenceUnitBudget,
) -> EvidenceUnitBatchResult:
    """Merge explicit semantic-gap retries without perturbing other units.

    Retried views are budgeted before the target's old candidates, because the
    verifier has already judged that old closure insufficient.  Exact old views
    retain their response-scoped EIDs; genuinely new views receive IDs above
    the initial ledger's maximum, so removal may intentionally leave ID gaps.
    """

    _validate_batch_registry(initial, name="initial batch")
    _validate_batch_registry(retried, name="retry batch")
    ordered_retry_ids = _normalize_retry_ids(initial, retry_unit_ids)
    requested = set(ordered_retry_ids)
    retried_by_id = {result.unit.unit_id: result for result in retried.results}
    extra = set(retried_by_id) - requested
    if extra:
        raise ValueError("retry batch contains results for non-requested units")

    merged: list[EvidenceUnitExecutionResult] = []
    for initial_result in initial.results:
        unit_id = initial_result.unit.unit_id
        if unit_id not in requested:
            merged.append(initial_result)
            continue
        retry_result = retried_by_id.get(unit_id)
        if retry_result is None:
            merged.append(
                _operational_failure(
                    initial_result,
                    error_class="MissingEvidenceUnitRetryResult",
                )
            )
            continue
        merged.append(_merge_retry_result(initial_result, retry_result))

    budgeted = _apply_retry_budget(initial, merged, retry_ids=requested, budget=budget)
    frozen, ledger = _freeze_stable_evidence_ids(initial, budgeted, retry_ids=requested)
    channel_counts = dict(initial.channel_counts)
    for channel, count in retried.channel_counts.items():
        channel_counts[channel] = channel_counts.get(channel, 0) + int(count)
    return EvidenceUnitBatchResult(
        results=tuple(frozen),
        evidence_ledger=tuple(ledger),
        channel_counts=channel_counts,
        ranking_count=initial.ranking_count + retried.ranking_count,
        feedback_errors=tuple(
            dict.fromkeys((*initial.feedback_errors, *retried.feedback_errors))
        ),
    )


def retry_evidence_units(
    initial: EvidenceUnitBatchResult,
    retry_unit_ids: Sequence[str],
    *,
    budget: EvidenceUnitBudget,
    runner: EvidenceUnitRetryRunner,
) -> EvidenceUnitBatchResult:
    """Execute exactly the selected semantic gaps, then merge them fail-closed."""

    _validate_batch_registry(initial, name="initial batch")
    ordered_retry_ids = _normalize_retry_ids(initial, retry_unit_ids)
    by_id = {result.unit.unit_id: result for result in initial.results}
    retry_units = tuple(by_id[unit_id].unit for unit_id in ordered_retry_ids)
    try:
        retried = runner(retry_units)
    except Exception as exc:
        failures = tuple(
            EvidenceUnitExecutionResult(
                unit=unit,
                status=EvidenceUnitExecutionStatus.RETRIEVAL_ERROR,
                reason_code="targeted_retry_failed",
                error_class=type(exc).__name__,
            )
            for unit in retry_units
        )
        retried = EvidenceUnitBatchResult(results=failures)
    return merge_targeted_evidence_retry(
        initial,
        retried,
        ordered_retry_ids,
        budget=budget,
    )
