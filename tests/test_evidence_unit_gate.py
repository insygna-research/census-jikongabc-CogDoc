from __future__ import annotations

from dataclasses import replace

import pytest

from cogdoc.agents.evidence_unit_verifier import (
    EvidenceUnitVerificationBatchResult,
    EvidenceUnitVerificationResult,
)
from cogdoc.service.evidence_unit_gate import (
    EvidenceUnitGateAction,
    EvidenceUnitGatePolicy,
    evaluate_evidence_unit_gate,
)
from cogdoc.service.evidence_unit_pipeline import (
    EvidenceUnitBatchResult,
    EvidenceUnitExecutionResult,
    EvidenceUnitExecutionStatus,
)
from cogdoc.service.evidence_units import (
    EvidenceClosureStatus,
    EvidenceUnit,
    EvidenceUnitClosure,
    EvidenceView,
    build_compare_evidence_units,
    build_qa_evidence_units,
    build_summary_evidence_units,
)
from cogdoc.tools.citation_ledger import build_evidence_ledger


def _summary_units(
    count: int = 3,
    *,
    max_retrieval_retries: int = 1,
) -> tuple[EvidenceUnit, ...]:
    return build_summary_evidence_units(
        "总结 handbook.pdf",
        "handbook.pdf",
        [
            {
                "section_id": f"section-{index}",
                "title": f"章节 {index}",
                "instruction": f"提炼章节 {index} 的事实",
            }
            for index in range(1, count + 1)
        ],
        max_retrieval_retries=max_retrieval_retries,
    )


def _doc(unit: EvidenceUnit, index: int = 1) -> dict:
    text = "直接证据"
    return {
        "text": text,
        "meta": {
            "chunk_id": f"chunk-{index}",
            "source": unit.scope.allowed_sources[0]
            if unit.scope.allowed_sources
            else "qa.pdf",
            "page": 1,
        },
        "retrieval": {
            "evidence_id": f"E{index:03d}",
            "evidence_text_start": 0,
            "evidence_text_end": len(text),
        },
    }


def _execution(
    unit: EvidenceUnit,
    status: EvidenceUnitExecutionStatus,
    *,
    index: int = 1,
    retrieval_round: int = 0,
) -> EvidenceUnitExecutionResult:
    return EvidenceUnitExecutionResult(
        unit=unit,
        status=status,
        selected_docs=(_doc(unit, index),)
        if status is EvidenceUnitExecutionStatus.READY
        else (),
        retrieval_round=retrieval_round,
        reason_code="" if status is EvidenceUnitExecutionStatus.READY else status.value,
        error_class=(
            "RuntimeError"
            if status is EvidenceUnitExecutionStatus.RETRIEVAL_ERROR
            else ""
        ),
    )


def _verification(
    execution: EvidenceUnitExecutionResult,
    status: EvidenceClosureStatus,
    *,
    reason_code: str | None = None,
    error_class: str = "",
    unit: EvidenceUnit | None = None,
) -> EvidenceUnitVerificationResult:
    result_unit = unit or execution.unit
    retrieval = (
        execution.selected_docs[0].get("retrieval", {})
        if execution.selected_docs
        else {}
    )
    evidence_id = str(retrieval.get("evidence_id") or "E001")
    chunk_id = (
        str(execution.selected_docs[0].get("meta", {}).get("chunk_id") or "chunk-1")
        if execution.selected_docs
        else "chunk-1"
    )
    grounded = status in {
        EvidenceClosureStatus.SUPPORTED,
        EvidenceClosureStatus.CONTRADICTORY,
    }
    resolved_reason = (
        reason_code
        or {
            EvidenceClosureStatus.SUPPORTED: "evidence_supported",
            EvidenceClosureStatus.NO_EVIDENCE: "evidence_no_evidence",
            EvidenceClosureStatus.CONTRADICTORY: "evidence_contradictory",
            EvidenceClosureStatus.RETRIEVAL_ERROR: "retrieval_error",
            EvidenceClosureStatus.VERIFICATION_ERROR: "verification_model_error",
            EvidenceClosureStatus.BUDGET_EXHAUSTED: "budget_exhausted",
        }[status]
    )
    view = EvidenceView(
        chunk_id=chunk_id,
        source=(
            result_unit.scope.allowed_sources[0]
            if result_unit.scope.allowed_sources
            else "qa.pdf"
        ),
        estimated_chars=20,
        span_start=(
            int(retrieval["evidence_text_start"]) if execution.selected_docs else None
        ),
        span_end=(
            int(retrieval["evidence_text_end"]) if execution.selected_docs else None
        ),
    )
    closure = EvidenceUnitClosure(
        unit_id=result_unit.unit_id,
        status=status,
        evidence=(view,) if grounded else (),
        grounding_chunk_ids=(chunk_id,) if grounded else (),
        retrieval_round=execution.retrieval_round,
        reason_code="" if grounded else resolved_reason,
    )
    return EvidenceUnitVerificationResult(
        unit=result_unit,
        status=status,
        closure=closure,
        candidate_evidence_ids=(evidence_id,)
        if execution.status is EvidenceUnitExecutionStatus.READY
        else (),
        evidence_ids=(evidence_id,) if grounded else (),
        evidence_chunk_ids=(chunk_id,) if grounded else (),
        reason_code=resolved_reason,
        error_class=error_class,
    )


def _batches(
    executions: tuple[EvidenceUnitExecutionResult, ...],
    statuses: tuple[EvidenceClosureStatus, ...],
    **verification_batch_fields,
) -> tuple[EvidenceUnitBatchResult, EvidenceUnitVerificationBatchResult]:
    return (
        _execution_batch(executions),
        EvidenceUnitVerificationBatchResult(
            results=tuple(
                _verification(execution, status)
                for execution, status in zip(executions, statuses, strict=True)
            ),
            **verification_batch_fields,
        ),
    )


def _execution_batch(
    executions: tuple[EvidenceUnitExecutionResult, ...],
) -> EvidenceUnitBatchResult:
    docs = [
        doc
        for execution in executions
        if execution.status is EvidenceUnitExecutionStatus.READY
        for doc in execution.selected_docs
    ]
    ledger = build_evidence_ledger(docs) if docs else []
    return EvidenceUnitBatchResult(
        results=executions,
        evidence_ledger=tuple(ledger),
    )


def test_unverified_ready_is_fail_closed_unless_policy_explicitly_allows_it():
    unit = _summary_units(1)[0]
    execution_batch = _execution_batch(
        (_execution(unit, EvidenceUnitExecutionStatus.READY),)
    )

    closed = evaluate_evidence_unit_gate(execution_batch)
    allowed = evaluate_evidence_unit_gate(
        execution_batch,
        policy=EvidenceUnitGatePolicy(allow_unverified_ready=True),
    )

    assert closed.terminal_unit_ids == (unit.unit_id,)
    assert not closed.batch_can_generate
    assert closed.decisions[0].reason_code == "verification_required"
    assert allowed.generate_unit_ids == (unit.unit_id,)
    assert allowed.batch_can_generate
    assert allowed.decisions[0].verification_status is None


def test_no_evidence_retries_only_while_unit_retry_budget_remains():
    units = _summary_units(3, max_retrieval_retries=1)
    executions = (
        _execution(units[0], EvidenceUnitExecutionStatus.NO_EVIDENCE),
        _execution(
            units[1],
            EvidenceUnitExecutionStatus.NO_EVIDENCE,
            retrieval_round=1,
        ),
        _execution(
            replace(
                units[2],
                policy=replace(units[2].policy, max_retrieval_retries=0),
            ),
            EvidenceUnitExecutionStatus.NO_EVIDENCE,
        ),
    )

    result = evaluate_evidence_unit_gate(_execution_batch(executions))

    assert result.retry_unit_ids == (units[0].unit_id,)
    assert result.terminal_unit_ids == (units[1].unit_id, units[2].unit_id)
    assert [decision.retries_remaining for decision in result.decisions] == [1, 0, 0]


def test_retrieval_and_budget_errors_are_terminal_and_never_retry():
    first, second = _summary_units(2, max_retrieval_retries=5)
    executions = (
        _execution(first, EvidenceUnitExecutionStatus.RETRIEVAL_ERROR),
        _execution(second, EvidenceUnitExecutionStatus.BUDGET_EXHAUSTED),
    )

    result = evaluate_evidence_unit_gate(_execution_batch(executions))

    assert result.retry_unit_ids == ()
    assert result.terminal_unit_ids == (first.unit_id, second.unit_id)
    assert [decision.reason_code for decision in result.decisions] == [
        "retrieval_error_terminal",
        "budget_exhausted_terminal",
    ]


def test_verified_results_produce_ordered_generate_retry_and_terminal_sets():
    units = _summary_units(3, max_retrieval_retries=2)
    executions = tuple(
        _execution(unit, EvidenceUnitExecutionStatus.READY, index=index)
        for index, unit in enumerate(units, start=1)
    )
    execution_batch, verification_batch = _batches(
        executions,
        (
            EvidenceClosureStatus.SUPPORTED,
            EvidenceClosureStatus.NO_EVIDENCE,
            EvidenceClosureStatus.VERIFICATION_ERROR,
        ),
    )

    result = evaluate_evidence_unit_gate(execution_batch, verification_batch)

    assert [decision.unit_id for decision in result.decisions] == [
        unit.unit_id for unit in units
    ]
    assert result.generate_unit_ids == (units[0].unit_id,)
    assert result.retry_unit_ids == (units[1].unit_id,)
    assert result.terminal_unit_ids == (units[2].unit_id,)
    assert not result.batch_can_generate
    assert result.metrics == {
        "planned_count": 3,
        "generate_count": 1,
        "retry_count": 1,
        "terminal_count": 1,
        "batch_can_generate": False,
    }
    assert result.to_state()["evidence_unit_generate_ids"] == [units[0].unit_id]


@pytest.mark.parametrize(
    ("configured", "retrieval_round", "expected", "reason_code"),
    [
        (
            EvidenceUnitGateAction.GENERATE,
            0,
            EvidenceUnitGateAction.GENERATE,
            "contradictory_generate_allowed",
        ),
        (
            EvidenceUnitGateAction.RETRY,
            0,
            EvidenceUnitGateAction.RETRY,
            "contradictory_retry_available",
        ),
        (
            EvidenceUnitGateAction.TERMINAL,
            0,
            EvidenceUnitGateAction.TERMINAL,
            "contradictory_terminal",
        ),
        (
            EvidenceUnitGateAction.RETRY,
            1,
            EvidenceUnitGateAction.TERMINAL,
            "contradictory_retries_exhausted",
        ),
    ],
)
def test_contradictory_action_is_configurable_but_retry_respects_budget(
    configured,
    retrieval_round,
    expected,
    reason_code,
):
    unit = _summary_units(1, max_retrieval_retries=1)[0]
    execution = _execution(
        unit,
        EvidenceUnitExecutionStatus.READY,
        retrieval_round=retrieval_round,
    )
    execution_batch, verification_batch = _batches(
        (execution,), (EvidenceClosureStatus.CONTRADICTORY,)
    )

    result = evaluate_evidence_unit_gate(
        execution_batch,
        verification_batch,
        policy=EvidenceUnitGatePolicy(contradictory_action=configured),
    )

    assert result.decisions[0].action is expected
    assert result.decisions[0].reason_code == reason_code


def test_required_units_control_batch_admission_without_hiding_unit_decisions():
    required, optional_base = _summary_units(2)
    optional = replace(
        optional_base,
        policy=replace(optional_base.policy, required=False),
    )
    executions = (
        _execution(required, EvidenceUnitExecutionStatus.READY, index=1),
        _execution(optional, EvidenceUnitExecutionStatus.READY, index=2),
    )
    execution_batch, verification_batch = _batches(
        executions,
        (
            EvidenceClosureStatus.NO_EVIDENCE,
            EvidenceClosureStatus.SUPPORTED,
        ),
    )

    strict = evaluate_evidence_unit_gate(execution_batch, verification_batch)
    partial = evaluate_evidence_unit_gate(
        execution_batch,
        verification_batch,
        policy=EvidenceUnitGatePolicy(require_all_required_units=False),
    )

    assert strict.generate_unit_ids == (optional.unit_id,)
    assert strict.retry_unit_ids == (required.unit_id,)
    assert not strict.batch_can_generate
    assert partial.generate_unit_ids == strict.generate_unit_ids
    assert partial.retry_unit_ids == strict.retry_unit_ids
    assert partial.batch_can_generate


def test_optional_failure_does_not_block_a_successful_required_unit():
    required, optional_base = _summary_units(2)
    optional = replace(
        optional_base,
        policy=replace(optional_base.policy, required=False),
    )
    executions = (
        _execution(required, EvidenceUnitExecutionStatus.READY, index=1),
        _execution(optional, EvidenceUnitExecutionStatus.RETRIEVAL_ERROR, index=2),
    )
    execution_batch, verification_batch = _batches(
        executions,
        (
            EvidenceClosureStatus.SUPPORTED,
            EvidenceClosureStatus.RETRIEVAL_ERROR,
        ),
    )

    result = evaluate_evidence_unit_gate(execution_batch, verification_batch)

    assert result.generate_unit_ids == (required.unit_id,)
    assert result.terminal_unit_ids == (optional.unit_id,)
    assert result.batch_can_generate


@pytest.mark.parametrize("case", ["missing", "unknown", "duplicate", "order"])
def test_verification_batch_requires_the_exact_execution_unit_closed_set_and_order(
    case,
):
    units = _summary_units(3)
    executions = tuple(
        _execution(unit, EvidenceUnitExecutionStatus.READY, index=index)
        for index, unit in enumerate(units, start=1)
    )
    execution_batch = _execution_batch(executions[:2])
    verified = tuple(
        _verification(execution, EvidenceClosureStatus.SUPPORTED)
        for execution in executions
    )
    variants = {
        "missing": (verified[0],),
        "unknown": (verified[0], verified[2]),
        "duplicate": (verified[0], verified[0]),
        "order": (verified[1], verified[0]),
    }

    with pytest.raises(ValueError, match="verification_batch"):
        evaluate_evidence_unit_gate(
            execution_batch,
            EvidenceUnitVerificationBatchResult(results=variants[case]),
        )


def test_verification_plan_and_retrieval_round_must_match_execution_exactly():
    first, second = _summary_units(2)
    execution = _execution(first, EvidenceUnitExecutionStatus.READY)
    changed_plan = replace(first, policy=replace(first.policy, required=False))
    mismatched_plan = _verification(
        execution,
        EvidenceClosureStatus.SUPPORTED,
        unit=changed_plan,
    )
    wrong_round = replace(
        _verification(execution, EvidenceClosureStatus.SUPPORTED),
        closure=replace(
            _verification(execution, EvidenceClosureStatus.SUPPORTED).closure,
            retrieval_round=1,
        ),
    )
    execution_batch = _execution_batch((execution,))

    with pytest.raises(ValueError, match="plan differs"):
        evaluate_evidence_unit_gate(
            execution_batch,
            EvidenceUnitVerificationBatchResult(results=(mismatched_plan,)),
        )
    with pytest.raises(ValueError, match="retrieval_round differs"):
        evaluate_evidence_unit_gate(
            execution_batch,
            EvidenceUnitVerificationBatchResult(results=(wrong_round,)),
        )
    assert second.unit_id != first.unit_id


def test_operational_errors_cannot_be_reclassified_as_semantic_no_evidence():
    unit = _summary_units(1, max_retrieval_retries=5)[0]
    execution = _execution(unit, EvidenceUnitExecutionStatus.RETRIEVAL_ERROR)
    reclassified = _verification(execution, EvidenceClosureStatus.NO_EVIDENCE)

    with pytest.raises(ValueError, match="reclassified"):
        evaluate_evidence_unit_gate(
            _execution_batch((execution,)),
            EvidenceUnitVerificationBatchResult(results=(reclassified,)),
        )


def test_unit_scoped_protocol_failure_cannot_degrade_to_no_evidence():
    unit = _summary_units(1, max_retrieval_retries=5)[0]
    execution = _execution(unit, EvidenceUnitExecutionStatus.READY)
    semantic_no_evidence = _verification(execution, EvidenceClosureStatus.NO_EVIDENCE)

    with pytest.raises(ValueError, match="must fail its READY unit closed"):
        evaluate_evidence_unit_gate(
            _execution_batch((execution,)),
            EvidenceUnitVerificationBatchResult(
                results=(semantic_no_evidence,),
                protocol_errors=(f"missing_unit:{unit.unit_id}",),
            ),
        )


def test_unscoped_unknown_unit_output_does_not_poison_known_unit_result():
    unit = _summary_units(1, max_retrieval_retries=5)[0]
    execution = _execution(unit, EvidenceUnitExecutionStatus.READY)
    semantic_no_evidence = _verification(execution, EvidenceClosureStatus.NO_EVIDENCE)

    result = evaluate_evidence_unit_gate(
        _execution_batch((execution,)),
        EvidenceUnitVerificationBatchResult(
            results=(semantic_no_evidence,),
            protocol_errors=("unknown_unit:eu_unknown",),
        ),
    )

    assert result.retry_unit_ids == (unit.unit_id,)


def test_model_failure_cannot_degrade_to_no_evidence():
    unit = _summary_units(1, max_retrieval_retries=5)[0]
    execution = _execution(unit, EvidenceUnitExecutionStatus.READY)
    semantic_no_evidence = _verification(execution, EvidenceClosureStatus.NO_EVIDENCE)

    with pytest.raises(ValueError, match="must fail at least one READY unit closed"):
        evaluate_evidence_unit_gate(
            _execution_batch((execution,)),
            EvidenceUnitVerificationBatchResult(
                results=(semantic_no_evidence,),
                error_class="TimeoutError",
            ),
        )


def test_per_unit_model_error_cannot_masquerade_as_no_evidence():
    unit = _summary_units(1)[0]
    execution = _execution(unit, EvidenceUnitExecutionStatus.READY)
    invalid = _verification(
        execution,
        EvidenceClosureStatus.NO_EVIDENCE,
        reason_code="verification_model_error",
        error_class="TimeoutError",
    )

    with pytest.raises(ValueError, match="cannot be represented"):
        evaluate_evidence_unit_gate(
            _execution_batch((execution,)),
            EvidenceUnitVerificationBatchResult(results=(invalid,)),
        )


def test_verification_error_is_terminal_even_when_retries_remain():
    unit = _summary_units(1, max_retrieval_retries=5)[0]
    execution = _execution(unit, EvidenceUnitExecutionStatus.READY)
    verification = _verification(
        execution,
        EvidenceClosureStatus.VERIFICATION_ERROR,
        error_class="TimeoutError",
    )

    result = evaluate_evidence_unit_gate(
        _execution_batch((execution,)),
        EvidenceUnitVerificationBatchResult(
            results=(verification,),
            error_class="TimeoutError",
        ),
    )

    assert result.retry_unit_ids == ()
    assert result.terminal_unit_ids == (unit.unit_id,)
    assert result.decisions[0].reason_code == "verification_error_terminal"


def test_bounded_batch_failure_does_not_poison_supported_sibling_batch():
    first, second = _summary_units(2)
    executions = (
        _execution(first, EvidenceUnitExecutionStatus.READY, index=1),
        _execution(second, EvidenceUnitExecutionStatus.READY, index=2),
    )
    failed = _verification(
        executions[0],
        EvidenceClosureStatus.VERIFICATION_ERROR,
        reason_code="verification_protocol_error",
    )
    supported = _verification(executions[1], EvidenceClosureStatus.SUPPORTED)

    result = evaluate_evidence_unit_gate(
        _execution_batch(executions),
        EvidenceUnitVerificationBatchResult(
            results=(failed, supported),
            protocol_errors=(f"missing_unit:{first.unit_id}",),
        ),
        policy=EvidenceUnitGatePolicy(require_all_required_units=False),
    )

    assert result.terminal_unit_ids == (first.unit_id,)
    assert result.generate_unit_ids == (second.unit_id,)
    assert result.batch_can_generate


def test_gate_rejects_forged_candidate_eid_even_if_verifier_calls_it_supported():
    unit = _summary_units(1)[0]
    execution = _execution(unit, EvidenceUnitExecutionStatus.READY)
    supported = _verification(execution, EvidenceClosureStatus.SUPPORTED)
    forged = replace(
        supported,
        candidate_evidence_ids=("E999",),
        evidence_ids=("E999",),
    )

    with pytest.raises(ValueError, match="candidate_evidence_ids"):
        evaluate_evidence_unit_gate(
            _execution_batch((execution,)),
            EvidenceUnitVerificationBatchResult(results=(forged,)),
        )


def test_gate_rejects_wrong_span_and_tampered_ledger_identity():
    unit = _summary_units(1)[0]
    execution = _execution(unit, EvidenceUnitExecutionStatus.READY)
    supported = _verification(execution, EvidenceClosureStatus.SUPPORTED)
    wrong_view = replace(
        supported.closure.evidence[0],
        span_start=1,
        span_end=4,
    )
    wrong_span = replace(
        supported,
        closure=replace(supported.closure, evidence=(wrong_view,)),
    )
    batch = _execution_batch((execution,))

    with pytest.raises(ValueError, match="evidence view does not exactly match"):
        evaluate_evidence_unit_gate(
            batch,
            EvidenceUnitVerificationBatchResult(results=(wrong_span,)),
        )

    tampered_entry = dict(batch.evidence_ledger[0])
    tampered_entry["span_end"] = int(tampered_entry["span_end"]) + 1
    with pytest.raises(ValueError, match="does not exactly match the execution ledger"):
        evaluate_evidence_unit_gate(
            replace(batch, evidence_ledger=(tampered_entry,)),
            EvidenceUnitVerificationBatchResult(results=(supported,)),
        )


def test_gate_contract_is_identical_for_qa_summary_and_compare_units():
    qa = build_qa_evidence_units(
        "什么是闭集校验？",
        [{"requirement_id": "definition", "question": "定义闭集校验"}],
    )[0]
    summary = _summary_units(1)[0]
    compare = build_compare_evidence_units(
        "对比 a.pdf 与 b.pdf 的方法",
        ("a.pdf", "b.pdf"),
        [{"dimension_id": "method", "title": "方法", "instruction": "提炼方法"}],
    )[0]
    units = (qa, summary, compare)
    executions = tuple(
        _execution(unit, EvidenceUnitExecutionStatus.READY, index=index)
        for index, unit in enumerate(units, start=1)
    )
    execution_batch, verification_batch = _batches(
        executions,
        (EvidenceClosureStatus.SUPPORTED,) * len(executions),
    )

    result = evaluate_evidence_unit_gate(execution_batch, verification_batch)

    assert result.generate_unit_ids == tuple(unit.unit_id for unit in units)
    assert result.retry_unit_ids == ()
    assert result.terminal_unit_ids == ()
    assert result.batch_can_generate


def test_duplicate_execution_unit_ids_are_rejected_before_gating():
    unit = _summary_units(1)[0]
    execution = _execution(unit, EvidenceUnitExecutionStatus.READY)

    with pytest.raises(ValueError, match="duplicate unit_id"):
        evaluate_evidence_unit_gate(
            _execution_batch((execution, execution))
        )
