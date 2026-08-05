from __future__ import annotations

from types import SimpleNamespace

import pytest

import cogdoc.service.qa_evidence_unit_adapter as adapter
from cogdoc.agents.evidence_unit_verifier import (
    EvidenceUnitVerificationBatchResult,
    verify_evidence_unit_batch,
)
from cogdoc.service.qa_evidence_unit_adapter import (
    QAEvidenceUnitAdapterOutcome,
    adapt_qa_evidence_verification,
    build_qa_verification_batch,
)
from cogdoc.tools.citation_ledger import assign_evidence_ids


@pytest.fixture(autouse=True)
def _stable_settings(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "get_settings",
        lambda: SimpleNamespace(
            qa_adaptive_retrieval_max_retries=2,
            qa_evidence_verify_max_chars_per_doc=777,
            evidence_unit_verify_max_units_per_batch=5,
        ),
    )


def _requirements():
    return [
        {
            "requirement_id": "r1",
            "question": "A 的规则是什么",
            "retrieval_query": "A 规则",
            "recovery_query": "A 规则详情",
        },
        {
            "requirement_id": "r2",
            "question": "B 的限制是什么",
            "retrieval_query": "B 限制",
            "recovery_query": "B 限制详情",
        },
    ]


def _doc(chunk_id: str, text: str, *, page: int = 1):
    return {
        "text": text,
        "meta": {
            "chunk_id": chunk_id,
            "source": "handbook.pdf",
            "page": page,
            "page_start": page,
            "page_end": page,
            "chunk_index": page - 1,
        },
        "retrieval": {"bm25_score": 12.0},
    }


def _state():
    annotated, ledger = assign_evidence_ids(
        [
            _doc("chunk-a", "A 的规则是必须登记。"),
            _doc("chunk-b", "B 的限制是最多十项。", page=2),
            _doc("chunk-extra", "仅供生成使用的额外上下文。", page=3),
        ]
    )
    return {
        "query": "A 的规则和 B 的限制分别是什么",
        "evidence_requirements": _requirements(),
        "verification_docs": annotated[:2],
        "evidence_ledger": ledger,
        "retrieval_round": 1,
        "is_local": True,
    }


def _generic_verifier(assessments, *, reverse_results=False, captured=None):
    def verify(
        batch,
        *,
        is_local=False,
        max_chars_per_doc=1600,
        max_units_per_batch=8,
    ):
        if captured is not None:
            captured.update(
                {
                    "batch": batch,
                    "is_local": is_local,
                    "max_chars_per_doc": max_chars_per_doc,
                    "max_units_per_batch": max_units_per_batch,
                }
            )
        result = verify_evidence_unit_batch(
            batch,
            is_local=is_local,
            max_chars_per_doc=max_chars_per_doc,
            max_units_per_batch=max_units_per_batch,
            structured_client=lambda _schema, _messages: {
                "assessments": assessments(batch)
            },
        )
        if not reverse_results:
            return result
        return EvidenceUnitVerificationBatchResult(
            results=tuple(reversed(result.results)),
            protocol_errors=result.protocol_errors,
            error_class=result.error_class,
        )

    return verify


def test_no_requirements_is_explicitly_not_applicable():
    called = False

    def verifier(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("generic verifier must not run")

    result = adapt_qa_evidence_verification(
        {"query": "单题", "evidence_requirements": []}, verifier=verifier
    )

    assert result.outcome is QAEvidenceUnitAdapterOutcome.NOT_APPLICABLE
    assert result.reason_code == "no_evidence_requirements"
    assert result.state_update == {}
    assert result.batch is None
    assert called is False


def test_batch_reuses_shared_verification_pack_and_full_ledger_superset():
    batch = build_qa_verification_batch(_state())

    assert batch is not None
    assert [result.unit.binding.requirement_id for result in batch.results] == [
        "r1",
        "r2",
    ]
    assert all(result.unit.policy.max_retrieval_retries == 2 for result in batch.results)
    assert all(result.retrieval_round == 1 for result in batch.results)
    assert [
        doc["retrieval"]["evidence_id"] for doc in batch.results[0].selected_docs
    ] == ["E001", "E002"]
    assert [
        doc["retrieval"]["evidence_id"] for doc in batch.results[1].selected_docs
    ] == ["E001", "E002"]
    assert [entry["evidence_id"] for entry in batch.evidence_ledger] == [
        "E001",
        "E002",
        "E003",
    ]


def test_generic_results_map_to_legacy_fields_in_requirement_order():
    captured = {}

    def assessments(batch):
        first, second = batch.results
        return [
            {
                "unit_id": second.unit.unit_id,
                "status": "no_evidence",
                "evidence_ids": [],
                "reason": "没有直接说明 B 的限制",
            },
            {
                "unit_id": first.unit.unit_id,
                "status": "supported",
                "evidence_ids": ["E001"],
                "reason": "E001 直接支持 A",
            },
        ]

    result = adapt_qa_evidence_verification(
        _state(),
        verifier=_generic_verifier(
            assessments, reverse_results=True, captured=captured
        ),
    )

    assert result.outcome is QAEvidenceUnitAdapterOutcome.VERIFIED
    assert captured["is_local"] is True
    assert captured["max_chars_per_doc"] == 777
    assert captured["max_units_per_batch"] == 5
    assert [
        item["requirement_id"]
        for item in result.state_update["evidence_requirement_assessments"]
    ] == ["r1", "r2"]
    assert [
        item["verdict"]
        for item in result.state_update["evidence_requirement_assessments"]
    ] == ["supported", "missing"]
    assert result.state_update["evidence_supported"] is False
    assert result.state_update["evidence_verified_chunk_ids"] == ["chunk-a"]
    assert result.state_update["missing_evidence_requirement_ids"] == ["r2"]
    assert result.state_update["retrieval_abstained"] is True
    assert result.state_update["retrieval_abstain_reason"] == (
        "evidence_not_supported"
    )
    assert result.state_update["evidence_verifier_error"] == ""


def test_contradictory_remains_grounded_but_is_a_legacy_missing_requirement():
    def assessments(batch):
        first, second = batch.results
        return [
            {
                "unit_id": first.unit.unit_id,
                "status": "supported",
                "evidence_ids": ["E001"],
                "reason": "A 已支持",
            },
            {
                "unit_id": second.unit.unit_id,
                "status": "contradictory",
                "evidence_ids": ["E002"],
                "reason": "B 的限制存在直接冲突",
            },
        ]

    result = adapt_qa_evidence_verification(
        _state(), verifier=_generic_verifier(assessments)
    )

    second = result.state_update["evidence_requirement_assessments"][1]
    assert second["verdict"] == "contradictory"
    assert second["evidence_chunk_ids"] == ["chunk-b"]
    assert result.state_update["missing_evidence_requirement_ids"] == ["r2"]
    assert result.state_update["evidence_verified_chunk_ids"] == ["chunk-a"]


def test_empty_verification_pack_is_semantic_no_evidence_without_model_call():
    state = _state()
    state["verification_docs"] = []
    state["evidence_ledger"] = []
    model_called = False

    def verifier(
        batch,
        *,
        is_local=False,
        max_chars_per_doc=1600,
        max_units_per_batch=8,
    ):
        def model(*_args, **_kwargs):
            nonlocal model_called
            model_called = True
            raise AssertionError("no ready unit should invoke the model")

        return verify_evidence_unit_batch(batch, structured_client=model)

    result = adapt_qa_evidence_verification(state, verifier=verifier)

    assert result.outcome is QAEvidenceUnitAdapterOutcome.VERIFIED
    assert result.state_update["evidence_supported"] is False
    assert result.state_update["missing_evidence_requirement_ids"] == ["r1", "r2"]
    assert [
        item["verdict"]
        for item in result.state_update["evidence_requirement_assessments"]
    ] == ["missing", "missing"]
    assert result.state_update["evidence_verifier_error"] == ""
    assert model_called is False


@pytest.mark.parametrize("failure", ["missing_eid", "ledger_mismatch"])
def test_invalid_frozen_closure_fails_closed_before_verifier(failure):
    state = _state()
    if failure == "missing_eid":
        state["verification_docs"][0]["retrieval"].pop("evidence_id")
    else:
        state["evidence_ledger"][0]["span_end"] += 1
    called = False

    def verifier(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid closure must fail before model verification")

    result = adapt_qa_evidence_verification(state, verifier=verifier)

    assert result.outcome is QAEvidenceUnitAdapterOutcome.VERIFICATION_ERROR
    assert result.reason_code == "invalid_qa_evidence_batch"
    assert result.error_class == "CitationLedgerError"
    assert result.state_update["evidence_supported"] is False
    assert result.state_update["missing_evidence_requirement_ids"] == ["r1", "r2"]
    assert result.state_update["retrieval_abstain_reason"] == (
        "evidence_verifier_error"
    )
    assert result.state_update["evidence_verifier_error"] == "CitationLedgerError"
    assert called is False


def test_malformed_requirements_fail_closed_without_escaping_adapter():
    result = adapt_qa_evidence_verification(
        {"query": "问题", "evidence_requirements": "not-a-list"}
    )

    assert result.outcome is QAEvidenceUnitAdapterOutcome.VERIFICATION_ERROR
    assert result.reason_code == "invalid_qa_evidence_batch"
    assert result.error_class == "ValueError"
    assert result.state_update["evidence_supported"] is False
    assert result.state_update["evidence_requirement_assessments"] == []
    assert result.state_update["retrieval_abstain_reason"] == (
        "evidence_verifier_error"
    )


def test_model_and_protocol_errors_are_verification_errors_not_no_evidence():
    model_failure = adapt_qa_evidence_verification(
        _state(),
        verifier=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TimeoutError("model unavailable")
        ),
    )

    assert model_failure.outcome is QAEvidenceUnitAdapterOutcome.VERIFICATION_ERROR
    assert model_failure.error_class == "TimeoutError"
    assert model_failure.state_update["evidence_verifier_error"] == "TimeoutError"
    assert all(
        item["verdict"] == "missing"
        for item in model_failure.state_update["evidence_requirement_assessments"]
    )

    def protocol_verifier(
        batch,
        *,
        is_local=False,
        max_chars_per_doc=1600,
        max_units_per_batch=8,
    ):
        first = batch.results[0].unit
        return verify_evidence_unit_batch(
            batch,
            structured_client=lambda _schema, _messages: {
                "assessments": [
                    {
                        "unit_id": first.unit_id,
                        "status": "supported",
                        "evidence_ids": ["E001"],
                        "reason": "故意遗漏第二个 unit",
                    }
                ]
            },
        )

    protocol_failure = adapt_qa_evidence_verification(
        _state(), verifier=protocol_verifier
    )

    assert protocol_failure.outcome is QAEvidenceUnitAdapterOutcome.VERIFICATION_ERROR
    assert protocol_failure.error_class == "EvidenceUnitVerificationProtocolError"
    assert protocol_failure.state_update["evidence_verifier_error"] == (
        "EvidenceUnitVerificationProtocolError"
    )
    assert protocol_failure.state_update["missing_evidence_requirement_ids"] == [
        "r1",
        "r2",
    ]
