from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from cogdoc.agents.evidence_unit_verifier import verify_evidence_unit_batch
from cogdoc.service.evidence_unit_pipeline import (
    EvidenceUnitBatchResult,
    EvidenceUnitExecutionResult,
    EvidenceUnitExecutionStatus,
)
from cogdoc.service.evidence_unit_gate import EvidenceUnitGatePolicy
from cogdoc.service.evidence_unit_workflow import verify_and_retry_evidence_units
from cogdoc.service.evidence_units import (
    EvidenceClosureStatus,
    EvidenceUnit,
    EvidenceUnitBudget,
    build_summary_evidence_units,
)
from cogdoc.tools.citation_ledger import assign_evidence_ids


def _doc(chunk_id: str, text: str) -> dict:
    return {
        "text": text,
        "meta": {
            "chunk_id": chunk_id,
            "source_sha256": "sha:a.pdf",
            "local_chunk_index": 0,
            "chunk_index": 0,
            "source": "a.pdf",
            "page": 1,
            "page_start": 1,
            "page_end": 1,
            "origin": "file",
        },
    }


def _units() -> tuple[EvidenceUnit, ...]:
    return build_summary_evidence_units(
        "总结 a.pdf",
        "a.pdf",
        [
            {"section_id": "method", "title": "方法", "instruction": "概括方法"},
            {"section_id": "limits", "title": "限制", "instruction": "概括限制"},
        ],
    )


def _batch(
    units: Sequence[EvidenceUnit],
    docs_by_unit: Sequence[Sequence[dict]],
    *,
    retrieval_round: int = 0,
) -> EvidenceUnitBatchResult:
    flattened = [doc for docs in docs_by_unit for doc in docs]
    annotated, ledger = assign_evidence_ids(flattened)
    cursor = 0
    results = []
    for unit, docs in zip(units, docs_by_unit, strict=True):
        selected = tuple(annotated[cursor : cursor + len(docs)])
        cursor += len(docs)
        results.append(
            EvidenceUnitExecutionResult(
                unit=unit,
                status=EvidenceUnitExecutionStatus.READY,
                selected_docs=selected,
                retrieval_round=retrieval_round,
                executed_queries=(
                    unit.recovery_query if retrieval_round else unit.retrieval_query,
                ),
                candidate_count=len(selected),
                selected_chars=sum(len(doc["text"]) for doc in selected),
            )
        )
    return EvidenceUnitBatchResult(
        results=tuple(results), evidence_ledger=tuple(ledger)
    )


def _evidence_id(result: EvidenceUnitExecutionResult, chunk_id: str) -> str:
    return next(
        doc["retrieval"]["evidence_id"]
        for doc in result.selected_docs
        if doc["meta"]["chunk_id"] == chunk_id
    )


def test_verifier_retry_targets_only_missing_unit_and_preserves_supported_outcome():
    units = _units()
    initial = _batch(
        units,
        [
            [_doc("a-supported", "文档明确说明了分层检索方法。")],
            [_doc("b-old", "这里只是无关背景。")],
        ],
    )
    calls: list[tuple[str, ...]] = []

    def verifier(batch: EvidenceUnitBatchResult):
        calls.append(tuple(result.unit.unit_id for result in batch.results))
        assessments = []
        for result in batch.results:
            if result.unit.unit_id == units[0].unit_id:
                assessments.append(
                    {
                        "unit_id": result.unit.unit_id,
                        "status": "supported",
                        "evidence_ids": [
                            _evidence_id(result, "a-supported")
                        ],
                        "reason": "方法证据明确",
                    }
                )
            elif any(
                doc["meta"]["chunk_id"] == "b-new"
                for doc in result.selected_docs
            ):
                assessments.append(
                    {
                        "unit_id": result.unit.unit_id,
                        "status": "supported",
                        "evidence_ids": [_evidence_id(result, "b-new")],
                        "reason": "重试证据明确",
                    }
                )
            else:
                assessments.append(
                    {
                        "unit_id": result.unit.unit_id,
                        "status": "no_evidence",
                        "evidence_ids": [],
                        "reason": "候选未说明限制",
                    }
                )
        return verify_evidence_unit_batch(
            batch,
            structured_client=lambda _schema, _messages: {
                "assessments": assessments
            },
        )

    def retry_runner(retry_units: tuple[EvidenceUnit, ...]):
        assert retry_units == (units[1],)
        return _batch(
            retry_units,
            [[_doc("b-new", "文档明确说明必须离线运行。")]],
            retrieval_round=1,
        )

    result = verify_and_retry_evidence_units(
        initial,
        budget=EvidenceUnitBudget(
            max_total_docs=4,
            max_total_chars=4000,
            max_docs_per_unit=2,
            max_chars_per_unit=2000,
        ),
        verifier=verifier,
        retry_runner=retry_runner,
    )

    assert calls == [
        (units[0].unit_id, units[1].unit_id),
        (units[1].unit_id,),
    ]
    assert result.retry_unit_ids == (units[1].unit_id,)
    assert result.gate.generate_unit_ids == (
        units[0].unit_id,
        units[1].unit_id,
    )
    assert result.gate.retry_unit_ids == ()
    assert [item.status for item in result.verification.results] == [
        EvidenceClosureStatus.SUPPORTED,
        EvidenceClosureStatus.SUPPORTED,
    ]
    assert result.execution.results[0] is initial.results[0]
    assert [doc["meta"]["chunk_id"] for doc in result.grounded_docs] == [
        "a-supported",
        "b-new",
    ]
    state = result.to_state()
    assert [row["status"] for row in state["evidence_unit_results"]] == [
        "supported",
        "supported",
    ]
    assert state["evidence_unit_metrics"]["targeted_retry_count"] == 1


def test_verifier_orchestration_error_never_becomes_no_evidence():
    units = _units()[:1]
    initial = _batch(units, [[_doc("candidate", "候选证据")]])

    def broken_verifier(_batch):
        raise RuntimeError("verifier unavailable")

    result = verify_and_retry_evidence_units(
        initial,
        budget=EvidenceUnitBudget(
            max_total_docs=1,
            max_total_chars=1000,
            max_docs_per_unit=1,
            max_chars_per_unit=1000,
        ),
        verifier=broken_verifier,
        retry_runner=lambda _units: (_ for _ in ()).throw(
            AssertionError("verification errors are not retryable")
        ),
    )

    assert result.verification.results[0].status is (
        EvidenceClosureStatus.VERIFICATION_ERROR
    )
    assert result.retry_unit_ids == ()
    assert result.gate.terminal_unit_ids == (units[0].unit_id,)
    state = result.to_state()
    assert state["evidence_unit_results"][0]["status"] == "verification_error"
    assert state["evidence_unit_results"][0]["selected_docs"] == []
    assert state["evidence_unit_metrics"]["no_evidence_count"] == 0


def test_disabled_verifier_preserves_generation_ready_execution_contract():
    units = _units()[:1]
    initial = _batch(units, [[_doc("candidate", "候选证据")]])

    result = verify_and_retry_evidence_units(
        initial,
        budget=EvidenceUnitBudget(
            max_total_docs=1,
            max_total_chars=1000,
            max_docs_per_unit=1,
            max_chars_per_unit=1000,
        ),
        verifier=None,
    )

    assert result.verification is None
    assert result.gate.generate_unit_ids == (units[0].unit_id,)
    assert result.grounded_docs == initial.ready_docs
    assert result.to_state()["evidence_unit_results"][0]["status"] == "ready"


def test_terminal_gate_removes_unverified_ready_docs_from_generation_state():
    units = _units()[:1]
    initial = _batch(units, [[_doc("candidate", "候选证据")]])

    result = verify_and_retry_evidence_units(
        initial,
        budget=EvidenceUnitBudget(
            max_total_docs=1,
            max_total_chars=1000,
            max_docs_per_unit=1,
            max_chars_per_unit=1000,
        ),
        verifier=None,
        gate_policy=EvidenceUnitGatePolicy(
            allow_unverified_ready=False,
            require_all_required_units=False,
        ),
    )

    assert result.gate.terminal_unit_ids == (units[0].unit_id,)
    assert result.grounded_docs == ()
    assert result.evidence_ledger == ()
    row = result.to_state()["evidence_unit_results"][0]
    assert row["gate_action"] == "terminal"
    assert row["selected_docs"] == []
    assert result.metrics["ready_count"] == 0
    assert result.metrics["coverage_rate"] == 0.0


def test_workflow_uses_every_configured_retry_before_no_evidence_is_terminal():
    base = _units()[0]
    unit = replace(
        base,
        policy=replace(base.policy, max_retrieval_retries=2),
    )
    initial = _batch((unit,), [[_doc("old", "这里只是背景。")]])
    runner_calls: list[int] = []

    def no_evidence_verifier(batch: EvidenceUnitBatchResult):
        return verify_evidence_unit_batch(
            batch,
            structured_client=lambda _schema, _messages: {
                "assessments": [
                    {
                        "unit_id": result.unit.unit_id,
                        "status": "no_evidence",
                        "evidence_ids": [],
                        "reason": "仍缺少直接证据",
                    }
                    for result in batch.results
                ]
            },
        )

    def retry_runner(retry_units: tuple[EvidenceUnit, ...]):
        runner_calls.append(len(runner_calls) + 1)
        return _batch(
            retry_units,
            [[_doc(f"retry-{runner_calls[-1]}", "仍是无关背景。")]],
            retrieval_round=runner_calls[-1],
        )

    result = verify_and_retry_evidence_units(
        initial,
        budget=EvidenceUnitBudget(
            max_total_docs=2,
            max_total_chars=4000,
            max_docs_per_unit=2,
            max_chars_per_unit=4000,
        ),
        verifier=no_evidence_verifier,
        retry_runner=retry_runner,
    )

    assert runner_calls == [1, 2]
    assert result.execution.results[0].retrieval_round == 2
    assert result.retry_history == ((unit.unit_id,), (unit.unit_id,))
    assert result.retry_unit_ids == (unit.unit_id,)
    assert result.verification_rounds == 3
    assert result.gate.retry_unit_ids == ()
    assert result.gate.terminal_unit_ids == (unit.unit_id,)
    assert result.metrics["targeted_retry_count"] == 2
    assert result.metrics["targeted_retry_round_count"] == 2
    assert result.grounded_docs == ()


def test_missing_retry_runner_becomes_operational_terminal_not_pending_retry():
    unit = _units()[0]
    initial = _batch((unit,), [[_doc("candidate", "候选背景。")]])

    result = verify_and_retry_evidence_units(
        initial,
        budget=EvidenceUnitBudget(
            max_total_docs=1,
            max_total_chars=1000,
            max_docs_per_unit=1,
            max_chars_per_unit=1000,
        ),
        verifier=lambda batch: verify_evidence_unit_batch(
            batch,
            structured_client=lambda _schema, _messages: {
                "assessments": [
                    {
                        "unit_id": unit.unit_id,
                        "status": "no_evidence",
                        "evidence_ids": [],
                        "reason": "缺少直接证据",
                    }
                ]
            },
        ),
        retry_runner=None,
    )

    assert result.execution.results[0].status is (
        EvidenceUnitExecutionStatus.RETRIEVAL_ERROR
    )
    assert result.gate.retry_unit_ids == ()
    assert result.gate.terminal_unit_ids == (unit.unit_id,)
    assert result.grounded_docs == ()
    assert result.to_state()["evidence_unit_results"][0]["status"] == "retrieval_error"


def test_forged_verifier_eid_fails_closed_before_generation():
    unit = _units()[0]
    initial = _batch((unit,), [[_doc("candidate", "候选证据。")]])

    def forged_verifier(batch: EvidenceUnitBatchResult):
        legitimate = verify_evidence_unit_batch(
            batch,
            structured_client=lambda _schema, _messages: {
                "assessments": [
                    {
                        "unit_id": unit.unit_id,
                        "status": "supported",
                        "evidence_ids": [
                            batch.results[0].selected_docs[0]["retrieval"]["evidence_id"]
                        ],
                        "reason": "支持",
                    }
                ]
            },
        )
        forged = replace(
            legitimate.results[0],
            candidate_evidence_ids=("E999",),
            evidence_ids=("E999",),
        )
        return replace(legitimate, results=(forged,))

    result = verify_and_retry_evidence_units(
        initial,
        budget=EvidenceUnitBudget(
            max_total_docs=1,
            max_total_chars=1000,
            max_docs_per_unit=1,
            max_chars_per_unit=1000,
        ),
        verifier=forged_verifier,
    )

    assert result.verification.results[0].status is (
        EvidenceClosureStatus.VERIFICATION_ERROR
    )
    assert result.verification.results[0].reason_code == "verification_contract_error"
    assert result.gate.terminal_unit_ids == (unit.unit_id,)
    assert result.grounded_docs == ()
    assert result.evidence_ledger == ()


def test_disabled_verifier_still_exhausts_operational_retry_budget():
    base = _units()[0]
    unit = replace(
        base,
        policy=replace(base.policy, max_retrieval_retries=2),
    )
    initial = EvidenceUnitBatchResult(
        results=(
            EvidenceUnitExecutionResult(
                unit=unit,
                status=EvidenceUnitExecutionStatus.NO_EVIDENCE,
                reason_code="source_scope_exhausted",
            ),
        )
    )
    rounds: list[int] = []

    def retry_runner(retry_units: tuple[EvidenceUnit, ...]):
        rounds.append(len(rounds) + 1)
        return EvidenceUnitBatchResult(
            results=(
                EvidenceUnitExecutionResult(
                    unit=retry_units[0],
                    status=EvidenceUnitExecutionStatus.NO_EVIDENCE,
                    retrieval_round=rounds[-1],
                    reason_code="source_scope_exhausted",
                ),
            )
        )

    result = verify_and_retry_evidence_units(
        initial,
        budget=EvidenceUnitBudget(
            max_total_docs=1,
            max_total_chars=1000,
            max_docs_per_unit=1,
            max_chars_per_unit=1000,
        ),
        verifier=None,
        retry_runner=retry_runner,
    )

    assert rounds == [1, 2]
    assert result.verification is None
    assert result.execution.results[0].retrieval_round == 2
    assert result.gate.retry_unit_ids == ()
    assert result.gate.terminal_unit_ids == (unit.unit_id,)
    assert result.metrics["targeted_retry_count"] == 2
    assert result.to_state()["evidence_unit_results"][0]["retry_attempted"] is True


def test_disabled_verifier_without_retry_runner_fails_operationally_closed():
    unit = _units()[0]
    initial = EvidenceUnitBatchResult(
        results=(
            EvidenceUnitExecutionResult(
                unit=unit,
                status=EvidenceUnitExecutionStatus.NO_EVIDENCE,
                reason_code="source_scope_exhausted",
            ),
        )
    )

    result = verify_and_retry_evidence_units(
        initial,
        budget=EvidenceUnitBudget(
            max_total_docs=1,
            max_total_chars=1000,
            max_docs_per_unit=1,
            max_chars_per_unit=1000,
        ),
        verifier=None,
    )

    assert result.execution.results[0].status is (
        EvidenceUnitExecutionStatus.RETRIEVAL_ERROR
    )
    assert result.gate.retry_unit_ids == ()
    assert result.gate.terminal_unit_ids == (unit.unit_id,)
    assert result.grounded_docs == ()
