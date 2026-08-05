from __future__ import annotations

from dataclasses import replace

import pytest

from cogdoc.agents.evidence_unit_verifier import verify_evidence_unit_batch
from cogdoc.service.evidence_unit_pipeline import (
    EvidenceUnitBatchResult,
    EvidenceUnitExecutionResult,
    EvidenceUnitExecutionStatus,
)
from cogdoc.service.evidence_unit_retry import (
    merge_targeted_evidence_retry,
    retry_evidence_units,
)
from cogdoc.service.evidence_units import (
    EvidenceClosureStatus,
    EvidenceUnit,
    EvidenceUnitBudget,
    build_summary_evidence_units,
)
from cogdoc.tools.citation_ledger import assign_evidence_ids
from cogdoc.tools.evidence_rendering import evidence_block_char_count


def _units(*, max_retrieval_retries: int = 1) -> tuple[EvidenceUnit, ...]:
    return build_summary_evidence_units(
        "总结 handbook.pdf 的规则和限制",
        "handbook.pdf",
        [
            {"section_id": "rules", "title": "规则", "instruction": "概括规则"},
            {"section_id": "limits", "title": "限制", "instruction": "概括限制"},
        ],
        max_retrieval_retries=max_retrieval_retries,
    )


def _budget(**overrides: int) -> EvidenceUnitBudget:
    values = {
        "max_total_docs": 8,
        "max_total_chars": 20_000,
        "max_docs_per_unit": 4,
        "max_chars_per_unit": 8_000,
    }
    values.update(overrides)
    return EvidenceUnitBudget(**values)


def _doc(
    unit: EvidenceUnit,
    chunk_id: str,
    text: str,
    *,
    source: str = "handbook.pdf",
    start: int = 0,
) -> dict:
    return {
        "text": text,
        "meta": {
            "chunk_id": chunk_id,
            "source_sha256": f"sha:{source}",
            "local_chunk_index": 0,
            "chunk_index": 0,
            "source": source,
            "page": 1,
            "page_start": 1,
            "page_end": 1,
            "origin": "file",
        },
        "retrieval": {
            "evidence_text_start": start,
            "matched_requirement_ids": [unit.unit_id],
            "matched_unit_ids": [unit.unit_id],
            "search_channel": "vector",
        },
    }


def _ready(unit: EvidenceUnit, *docs: dict) -> EvidenceUnitExecutionResult:
    return EvidenceUnitExecutionResult(
        unit=unit,
        status=EvidenceUnitExecutionStatus.READY,
        selected_docs=docs,
        executed_queries=(unit.retrieval_query,),
        candidate_count=len(docs),
    )


def _missing(unit: EvidenceUnit) -> EvidenceUnitExecutionResult:
    return EvidenceUnitExecutionResult(
        unit=unit,
        status=EvidenceUnitExecutionStatus.NO_EVIDENCE,
        executed_queries=(unit.retrieval_query,),
        reason_code="source_scope_exhausted",
    )


def _batch(
    *results: EvidenceUnitExecutionResult,
    channel_counts: dict[str, int] | None = None,
    ranking_count: int = 0,
) -> EvidenceUnitBatchResult:
    flattened = [
        doc
        for result in results
        if result.status is EvidenceUnitExecutionStatus.READY
        for doc in result.selected_docs
    ]
    annotated, ledger = assign_evidence_ids(flattened) if flattened else ([], [])
    cursor = 0
    frozen: list[EvidenceUnitExecutionResult] = []
    for result in results:
        if result.status is not EvidenceUnitExecutionStatus.READY:
            frozen.append(result)
            continue
        count = len(result.selected_docs)
        docs = tuple(annotated[cursor : cursor + count])
        cursor += count
        frozen.append(
            replace(
                result,
                selected_docs=docs,
                selected_chars=sum(
                    evidence_block_char_count(doc, str(doc.get("text") or ""))
                    for doc in docs
                ),
            )
        )
    return EvidenceUnitBatchResult(
        results=tuple(frozen),
        evidence_ledger=tuple(ledger),
        channel_counts=channel_counts or {},
        ranking_count=ranking_count,
    )


def _eids(result: EvidenceUnitExecutionResult) -> list[str]:
    return [
        str(doc.get("retrieval", {}).get("evidence_id") or "")
        for doc in result.selected_docs
    ]


def test_retry_ready_target_only_unions_its_old_and_new_evidence() -> None:
    unit_a, unit_b = _units()
    initial = _batch(
        _ready(unit_a, _doc(unit_a, "chunk-a", "规则证据。")),
        _ready(unit_b, _doc(unit_b, "chunk-b-old", "旧的限制候选。")),
        channel_counts={"vector": 2},
        ranking_count=2,
    )
    seen: list[tuple[EvidenceUnit, ...]] = []

    def runner(units: tuple[EvidenceUnit, ...]) -> EvidenceUnitBatchResult:
        seen.append(units)
        return _batch(
            _ready(unit_b, _doc(unit_b, "chunk-b-new", "补充限制证据。")),
            channel_counts={"bm25": 1},
            ranking_count=1,
        )

    merged = retry_evidence_units(
        initial,
        [unit_b.unit_id],
        budget=_budget(),
        runner=runner,
    )

    assert seen == [(unit_b,)]
    assert merged.results[0] is initial.results[0]
    assert merged.results[0].selected_docs[0] is initial.results[0].selected_docs[0]
    assert [doc["meta"]["chunk_id"] for doc in merged.results[1].selected_docs] == [
        "chunk-b-new",
        "chunk-b-old",
    ]
    assert _eids(merged.results[0]) == ["E001"]
    assert _eids(merged.results[1]) == ["E003", "E002"]
    assert [entry["evidence_id"] for entry in merged.evidence_ledger] == [
        "E001",
        "E002",
        "E003",
    ]
    assert merged.channel_counts == {"vector": 2, "bm25": 1}
    assert merged.ranking_count == 3


def test_retry_no_evidence_target_adds_only_its_new_view() -> None:
    unit_a, unit_b = _units()
    initial = _batch(
        _ready(unit_a, _doc(unit_a, "chunk-a", "规则证据。")),
        _missing(unit_b),
    )
    retry = _batch(_ready(unit_b, _doc(unit_b, "chunk-b", "找到限制。")))

    merged = merge_targeted_evidence_retry(
        initial,
        retry,
        [unit_b.unit_id],
        budget=_budget(),
    )

    assert merged.results[0] is initial.results[0]
    assert merged.results[1].status is EvidenceUnitExecutionStatus.READY
    assert _eids(merged.results[1]) == ["E002"]


def test_existing_exact_view_keeps_one_eid_across_units() -> None:
    unit_a, unit_b = _units()
    shared_a = _doc(unit_a, "shared", "同一精确证据视图。", start=17)
    initial = _batch(
        _ready(unit_a, shared_a),
        _ready(unit_b, _doc(unit_b, "chunk-b", "原限制证据。")),
    )
    shared_b = _doc(unit_b, "shared", "同一精确证据视图。", start=17)
    retry = _batch(_ready(unit_b, shared_b))

    merged = merge_targeted_evidence_retry(
        initial,
        retry,
        [unit_b.unit_id],
        budget=_budget(),
    )

    assert _eids(merged.results[0]) == ["E001"]
    assert _eids(merged.results[1]) == ["E001", "E002"]
    assert len(merged.evidence_ledger) == 2
    new_tags = merged.results[1].selected_docs[0]["retrieval"]
    assert new_tags["matched_requirement_ids"] == [unit_b.unit_id]
    assert new_tags["matched_unit_ids"] == [unit_b.unit_id]


def test_same_chunk_different_span_gets_a_new_global_eid() -> None:
    unit_a, unit_b = _units()
    initial = _batch(
        _ready(unit_a, _doc(unit_a, "chunk-a", "规则。")),
        _ready(unit_b, _doc(unit_b, "shared", "前一段限制。", start=10)),
    )
    retry = _batch(_ready(unit_b, _doc(unit_b, "shared", "后一段限制。", start=80)))

    merged = merge_targeted_evidence_retry(
        initial,
        retry,
        [unit_b.unit_id],
        budget=_budget(),
    )

    assert _eids(merged.results[1]) == ["E003", "E002"]
    spans = [
        (entry["chunk_id"], entry["span_start"], entry["span_end"])
        for entry in merged.evidence_ledger
    ]
    assert spans[1][0] == spans[2][0] == "shared"
    assert spans[1][1:] != spans[2][1:]


def test_retry_scope_violation_is_an_operational_error_and_drops_retry_target() -> None:
    unit_a, unit_b = _units()
    initial = _batch(
        _ready(unit_a, _doc(unit_a, "chunk-a", "规则。")),
        _ready(unit_b, _doc(unit_b, "chunk-b", "旧限制。")),
    )
    retry = _batch(
        _ready(
            unit_b,
            _doc(unit_b, "wrong", "越界证据。", source="other.pdf"),
        )
    )

    merged = merge_targeted_evidence_retry(
        initial,
        retry,
        [unit_b.unit_id],
        budget=_budget(),
    )

    assert merged.results[0] is initial.results[0]
    assert merged.results[1].status is EvidenceUnitExecutionStatus.RETRIEVAL_ERROR
    assert merged.results[1].selected_docs == ()
    assert merged.results[1].reason_code == "targeted_retry_scope_violation"
    assert merged.results[1].scope_violation_count == 1
    assert [entry["evidence_id"] for entry in merged.evidence_ledger] == ["E001"]


def test_runner_failure_never_becomes_no_evidence() -> None:
    unit_a, unit_b = _units()
    initial = _batch(
        _ready(unit_a, _doc(unit_a, "chunk-a", "规则。")),
        _ready(unit_b, _doc(unit_b, "chunk-b", "旧限制。")),
    )

    def runner(units: tuple[EvidenceUnit, ...]) -> EvidenceUnitBatchResult:
        raise TimeoutError("retrieval timed out")

    merged = retry_evidence_units(
        initial,
        [unit_b.unit_id],
        budget=_budget(),
        runner=runner,
    )

    assert merged.results[1].status is EvidenceUnitExecutionStatus.RETRIEVAL_ERROR
    assert merged.results[1].error_class == "TimeoutError"
    assert merged.results[1].reason_code == "targeted_retry_failed"


def test_error_shaped_no_evidence_is_coerced_to_operational_error() -> None:
    unit_a, unit_b = _units()
    initial = _batch(
        _ready(unit_a, _doc(unit_a, "chunk-a", "规则。")),
        _ready(unit_b, _doc(unit_b, "chunk-b", "旧限制。")),
    )
    retry = _batch(
        EvidenceUnitExecutionResult(
            unit=unit_b,
            status=EvidenceUnitExecutionStatus.NO_EVIDENCE,
            reason_code="retrieval_error",
            error_class="ConnectionError",
        )
    )

    merged = merge_targeted_evidence_retry(
        initial,
        retry,
        [unit_b.unit_id],
        budget=_budget(),
    )

    assert merged.results[1].status is EvidenceUnitExecutionStatus.RETRIEVAL_ERROR
    assert merged.results[1].error_class == "ConnectionError"


def test_retry_budget_failure_is_not_reclassified_as_ready_old_evidence() -> None:
    unit_a, unit_b = _units()
    initial = _batch(
        _ready(unit_a, _doc(unit_a, "chunk-a", "规则。")),
        _ready(unit_b, _doc(unit_b, "chunk-b", "旧限制。")),
    )
    retry = _batch(
        EvidenceUnitExecutionResult(
            unit=unit_b,
            status=EvidenceUnitExecutionStatus.BUDGET_EXHAUSTED,
            retrieval_round=1,
            reason_code="unit_evidence_budget_exceeded",
        )
    )

    merged = merge_targeted_evidence_retry(
        initial,
        retry,
        [unit_b.unit_id],
        budget=_budget(),
    )

    assert merged.results[1].status is EvidenceUnitExecutionStatus.BUDGET_EXHAUSTED
    assert merged.results[1].selected_docs == ()
    assert merged.results[1].reason_code == "unit_evidence_budget_exceeded"
    assert [entry["evidence_id"] for entry in merged.evidence_ledger] == ["E001"]


def test_retry_budget_prefers_new_evidence_and_clips_old_docs_deterministically() -> (
    None
):
    unit_a, unit_b = _units()
    initial = _batch(
        _ready(unit_a, _doc(unit_a, "chunk-a", "规则。")),
        _ready(unit_b, _doc(unit_b, "chunk-b-old", "旧限制。")),
    )
    retry = _batch(
        _ready(
            unit_b,
            _doc(unit_b, "chunk-b-new-1", "第一个补充。"),
            _doc(unit_b, "chunk-b-new-2", "第二个补充。"),
        )
    )

    merged = merge_targeted_evidence_retry(
        initial,
        retry,
        [unit_b.unit_id],
        budget=_budget(max_total_docs=4, max_docs_per_unit=2),
    )

    assert [doc["meta"]["chunk_id"] for doc in merged.results[1].selected_docs] == [
        "chunk-b-new-1",
        "chunk-b-new-2",
    ]
    assert len(merged.ready_docs) == 3


def test_full_target_budget_replaces_old_view_and_second_verifier_grounds_new_eid() -> (
    None
):
    unit_a, unit_b = _units()
    initial = _batch(
        _ready(unit_a, _doc(unit_a, "chunk-a", "规则。")),
        _ready(unit_b, _doc(unit_b, "chunk-b-old", "旧限制。")),
    )
    retry = _batch(
        _ready(
            unit_b,
            _doc(unit_b, "chunk-b-new-1", "第一个补充。"),
            _doc(unit_b, "chunk-b-new-2", "第二个补充。"),
        )
    )

    merged = merge_targeted_evidence_retry(
        initial,
        retry,
        [unit_b.unit_id],
        budget=_budget(max_total_docs=2, max_docs_per_unit=1),
    )

    assert merged.results[1].status is EvidenceUnitExecutionStatus.READY
    assert [doc["meta"]["chunk_id"] for doc in merged.results[1].selected_docs] == [
        "chunk-b-new-1"
    ]
    assert _eids(merged.results[1]) == ["E003"]
    assert [entry["evidence_id"] for entry in merged.evidence_ledger] == [
        "E001",
        "E003",
    ]

    def verifier_client(schema, messages):
        return {
            "assessments": [
                {
                    "unit_id": unit_a.unit_id,
                    "status": "supported",
                    "evidence_ids": ["E001"],
                    "reason": "规则证据充分",
                },
                {
                    "unit_id": unit_b.unit_id,
                    "status": "supported",
                    "evidence_ids": ["E003"],
                    "reason": "重试证据直接支持限制单元",
                },
            ]
        }

    verified = verify_evidence_unit_batch(
        merged,
        structured_client=verifier_client,
    )
    assert verified.results[1].status is EvidenceClosureStatus.SUPPORTED
    assert verified.results[1].grounding_evidence_ids == ("E003",)
    assert verified.results[1].closure.evidence[0].span_start == 0
    assert verified.results[1].closure.evidence[0].span_end == len("第一个补充。")


def test_non_target_and_invalid_retry_requests_are_rejected() -> None:
    unit_a, unit_b = _units()
    initial = _batch(
        _ready(unit_a, _doc(unit_a, "chunk-a", "规则。")),
        _ready(unit_b, _doc(unit_b, "chunk-b", "限制。")),
    )
    retry = _batch(_ready(unit_b, _doc(unit_b, "chunk-b-new", "新限制。")))

    with pytest.raises(ValueError, match="non-requested"):
        merge_targeted_evidence_retry(
            initial,
            _batch(
                *retry.results,
                _ready(unit_a, _doc(unit_a, "chunk-a-new", "新规则。")),
            ),
            [unit_b.unit_id],
            budget=_budget(),
        )
    with pytest.raises(ValueError, match="duplicates"):
        merge_targeted_evidence_retry(
            initial,
            retry,
            [unit_b.unit_id, unit_b.unit_id],
            budget=_budget(),
        )


def test_missing_retry_result_becomes_an_operational_error() -> None:
    unit_a, unit_b = _units()
    initial = _batch(
        _ready(unit_a, _doc(unit_a, "chunk-a", "规则。")),
        _ready(unit_b, _doc(unit_b, "chunk-b", "限制。")),
    )

    merged = merge_targeted_evidence_retry(
        initial,
        EvidenceUnitBatchResult(results=()),
        [unit_b.unit_id],
        budget=_budget(),
    )

    assert merged.results[1].status is EvidenceUnitExecutionStatus.RETRIEVAL_ERROR
    assert merged.results[1].error_class == "MissingEvidenceUnitRetryResult"
