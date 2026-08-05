from __future__ import annotations

import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from cogdoc.agents import claim_evidence_verifier
from cogdoc.agents.claim_evidence_verifier import (
    ClaimAssessment,
    ClaimAssessmentBatch,
    ClaimEvidenceVerifierAgent,
    ClaimRepair,
    ClaimRepairAgent,
)
from cogdoc.agents.compare_generator import CompareGeneratorAgent
from cogdoc.agents.summary_generator import (
    EVIDENCE_UNIT_FAILURE_MESSAGE,
    GlobalSummaryAgent,
)
from cogdoc.graph.subgraphs import qa
from cogdoc.service.claim_audit_projection import (
    ClaimAuditProjectionSegment,
    ClaimAuditProjectionStatus,
    build_claim_audit_projection,
    load_claim_audit_projection,
)


def _doc(source: str = "a.pdf", evidence_id: str = "E001") -> dict:
    return {
        "text": "文档明确说明该方法使用向量检索。",
        "meta": {
            "chunk_id": f"chunk:{source}:1",
            "source": source,
            "source_type": "document",
            "chunk_index": 0,
            "local_chunk_index": 0,
            "page": 1,
            "page_start": 1,
            "page_end": 1,
            "origin": "test",
        },
        "retrieval": {"evidence_id": evidence_id},
    }


def _ledger(source: str = "a.pdf", evidence_id: str = "E001") -> list[dict]:
    return [
        {
            "evidence_id": evidence_id,
            "chunk_id": f"chunk:{source}:1",
            "source_type": "document",
            "source": source,
            "page": 1,
            "span_start": 0,
            "span_end": 12,
            "display_citation": f"[{source}:P1]",
        }
    ]


def _support_projected_claims(monkeypatch, captured_claims: list[dict]) -> None:
    monkeypatch.setattr(
        claim_evidence_verifier,
        "get_settings",
        lambda: SimpleNamespace(
            claim_verification_enabled=True,
            claim_verification_max_claims=32,
            claim_verification_max_claims_per_batch=8,
            claim_verification_max_docs_per_batch=8,
            claim_verification_max_chars_per_doc=1200,
        ),
    )
    monkeypatch.setattr(
        claim_evidence_verifier.Generator,
        "_get_client_for_node",
        lambda *_args, **_kwargs: object(),
    )

    def fake_invoke(_llm, schema, messages):
        assert schema is ClaimAssessmentBatch
        user_content = messages[1]["content"]
        claims_json = user_content.split("【候选声明 JSON】\n", 1)[1].split(
            "\n\n【允许证据 JSON】", 1
        )[0]
        claims = json.loads(claims_json)
        captured_claims.extend(claims)
        return ClaimAssessmentBatch(
            assessments=[
                ClaimAssessment(
                    claim_id=claim["claim_id"],
                    verdict="supported",
                    evidence_ids=list(claim["allowed_evidence_ids"][:1]),
                    reason="精确证据直接支持",
                    confidence=1.0,
                )
                for claim in claims
            ]
        )

    monkeypatch.setattr(claim_evidence_verifier, "invoke_structured", fake_invoke)


def test_mixed_summary_projects_and_audits_only_generated_section(monkeypatch):
    doc = _doc()
    results = [
        {
            "section_id": "method",
            "title": "方法",
            "content": "该方法使用向量检索[E001]。",
            "unit_id": "eu_summary_method",
            "status": "generated",
            "evidence": [{"evidence_id": "E001", "chunk_id": "chunk:a.pdf:1"}],
        },
        {
            "section_id": "metrics",
            "title": "指标",
            "content": "文档中未明确说明。",
            "unit_id": "eu_summary_metrics",
            "status": "no_evidence",
            "evidence": [],
        },
        {
            "section_id": "limits",
            "title": "限制",
            "content": EVIDENCE_UNIT_FAILURE_MESSAGE,
            "unit_id": "eu_summary_limits",
            "status": "retrieval_error",
            "evidence": [],
        },
    ]
    generated = GlobalSummaryAgent.build_final_summary(
        {
            "summary_source": "a.pdf",
            "summary_docs": [doc],
            "summary_section_results": results,
            "evidence_ledger": _ledger(),
        }
    )
    projection = load_claim_audit_projection(
        generated["claim_audit_projection"],
        answer=generated["answer"],
    )

    assert [segment.status for segment in projection.segments] == [
        ClaimAuditProjectionStatus.GENERATED,
        ClaimAuditProjectionStatus.DETERMINISTIC,
        ClaimAuditProjectionStatus.OPERATIONAL,
    ]
    assert [segment.obligation_ids for segment in projection.segments] == [
        ("eu_summary_method",),
        ("eu_summary_metrics",),
        ("eu_summary_limits",),
    ]

    captured: list[dict] = []
    _support_projected_claims(monkeypatch, captured)
    audit = ClaimEvidenceVerifierAgent.audit(
        {
            "task_type": "summary",
            "query": "总结 a.pdf",
            **generated,
        }
    )

    assert audit["claim_audit_passed"] is True
    assert [claim["text"] for claim in captured] == [
        "该方法使用向量检索[E001]。"
    ]
    assert "文档中未明确说明" not in projection.audit_text
    assert EVIDENCE_UNIT_FAILURE_MESSAGE not in projection.audit_text


def test_mixed_compare_projects_generated_no_evidence_and_operational_cells(
    monkeypatch,
):
    sources = ["a.pdf", "b.pdf", "c.pdf"]
    dimension = {"dimension_id": "method", "title": "方法", "instruction": "方法"}
    profiles = [
        {
            "source": "a.pdf",
            "cells": [
                {
                    "dimension_id": "method",
                    "source": "a.pdf",
                    "content": "A 使用向量检索[E001]。",
                    "unit_id": "eu_compare_a_method",
                    "status": "generated",
                    "evidence": [],
                }
            ],
        },
        {
            "source": "b.pdf",
            "cells": [
                {
                    "dimension_id": "method",
                    "source": "b.pdf",
                    "content": "文档中未明确说明。",
                    "unit_id": "eu_compare_b_method",
                    "status": "no_evidence",
                    "evidence": [],
                }
            ],
        },
        {
            "source": "c.pdf",
            "cells": [
                {
                    "dimension_id": "method",
                    "source": "c.pdf",
                    "content": EVIDENCE_UNIT_FAILURE_MESSAGE,
                    "unit_id": "eu_compare_c_method",
                    "status": "verification_error",
                    "evidence": [],
                }
            ],
        },
    ]
    state = {
        "compare_sources": sources,
        "compare_dimensions": [dimension],
        "compare_docs_by_source": {"a.pdf": [_doc()]},
        "document_profiles": profiles,
        "evidence_ledger": _ledger(),
        "is_local": False,
    }
    built = CompareGeneratorAgent.build_compare_answer(state)
    assert built["compare_conclusion"] == ""
    validated = CompareGeneratorAgent.validate_compare_answer({**state, **built})
    projection = load_claim_audit_projection(
        validated["claim_audit_projection"],
        answer=validated["answer"],
    )

    assert [segment.status for segment in projection.segments] == [
        ClaimAuditProjectionStatus.GENERATED,
        ClaimAuditProjectionStatus.DETERMINISTIC,
        ClaimAuditProjectionStatus.OPERATIONAL,
    ]
    assert projection.obligation_ids == (
        "eu_compare_a_method",
        "eu_compare_b_method",
        "eu_compare_c_method",
    )

    captured: list[dict] = []
    _support_projected_claims(monkeypatch, captured)
    audit = ClaimEvidenceVerifierAgent.audit(
        {
            "task_type": "compare",
            "query": "对比三篇文档",
            **state,
            **built,
            **validated,
        }
    )

    assert audit["claim_audit_passed"] is True
    assert [claim["text"] for claim in captured] == ["A 使用向量检索[E001]。"]


def test_compare_projection_includes_generated_conclusion_without_obligation(
    monkeypatch,
):
    monkeypatch.setattr(
        CompareGeneratorAgent,
        "_generate_conclusion",
        lambda *_args, **_kwargs: "A 提供了可用方法[E001]。",
    )
    sources = ["a.pdf", "b.pdf"]
    dimension = {"dimension_id": "method", "title": "方法", "instruction": "方法"}
    profiles = [
        {
            "source": "a.pdf",
            "cells": [
                {
                    "dimension_id": "method",
                    "source": "a.pdf",
                    "content": "A 使用向量检索[E001]。",
                    "unit_id": "eu_compare_a",
                    "status": "generated",
                    "evidence": [],
                }
            ],
        },
        {
            "source": "b.pdf",
            "cells": [
                {
                    "dimension_id": "method",
                    "source": "b.pdf",
                    "content": "文档中未明确说明。",
                    "unit_id": "eu_compare_b",
                    "status": "no_evidence",
                    "evidence": [],
                }
            ],
        },
    ]
    state = {
        "compare_sources": sources,
        "compare_dimensions": [dimension],
        "compare_docs_by_source": {"a.pdf": [_doc()]},
        "document_profiles": profiles,
        "evidence_ledger": _ledger(),
    }
    built = CompareGeneratorAgent.build_compare_answer(state)
    validated = CompareGeneratorAgent.validate_compare_answer({**state, **built})
    projection = load_claim_audit_projection(
        validated["claim_audit_projection"], answer=validated["answer"]
    )

    conclusion = projection.segments[-1]
    assert conclusion.segment_id == "compare:conclusion"
    assert conclusion.status is ClaimAuditProjectionStatus.GENERATED
    assert conclusion.obligation_ids == ()
    assert conclusion.content == "A 提供了可用方法[E001]。"


def test_invalid_and_repair_stale_projection_fail_closed_without_llm(monkeypatch):
    monkeypatch.setattr(
        claim_evidence_verifier,
        "get_settings",
        lambda: SimpleNamespace(claim_verification_enabled=True),
    )
    monkeypatch.setattr(
        claim_evidence_verifier.Generator,
        "_get_client_for_node",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid projection must fail before LLM")
        ),
    )
    original_answer = "原始事实[E001]。"
    projection = build_claim_audit_projection(
        original_answer,
        (ClaimAuditProjectionSegment.generated("qa:answer", original_answer),),
    ).to_state()
    invalid = json.loads(json.dumps(projection))
    invalid["segments"][0]["status"] = "skip"

    invalid_result = ClaimEvidenceVerifierAgent.audit(
        {
            "task_type": "qa",
            "answer": original_answer,
            "claim_audit_projection": invalid,
        }
    )
    stale_result = ClaimEvidenceVerifierAgent.audit(
        {
            "task_type": "qa",
            "answer": "修复后的事实[E001]。",
            "claim_repair_count": 1,
            "claim_audit_projection": projection,
        }
    )

    assert invalid_result["claim_audit_passed"] is False
    assert invalid_result["claim_audit"]["reason_code"] == (
        "claim_audit_projection_segment_invalid"
    )
    assert stale_result["claim_audit_passed"] is False
    assert stale_result["claim_audit"]["reason_code"] == (
        "claim_audit_projection_answer_mismatch"
    )


def test_claim_repair_rebuilds_projection_and_preserves_obligations(monkeypatch):
    original_answer = "原始事实[E001]。\n文档中未明确说明。"
    original_projection = build_claim_audit_projection(
        original_answer,
        (
            ClaimAuditProjectionSegment.generated(
                "qa:answer",
                "原始事实[E001]。",
                obligation_ids=("eu_qa_fact",),
            ),
            ClaimAuditProjectionSegment.deterministic(
                "qa:no_evidence",
                "文档中未明确说明。",
                source_status="no_evidence",
                obligation_ids=("eu_qa_gap",),
            ),
        ),
    )
    monkeypatch.setattr(
        claim_evidence_verifier,
        "get_settings",
        lambda: SimpleNamespace(
            claim_verification_max_docs_per_batch=8,
            claim_verification_max_chars_per_doc=1200,
        ),
    )
    monkeypatch.setattr(
        claim_evidence_verifier.Generator,
        "_get_client_for_node",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        claim_evidence_verifier,
        "invoke_structured",
        lambda *_args, **_kwargs: ClaimRepair(
            revised_answer="修复后的事实由证据支持[E001]。"
        ),
    )
    state = {
        "task_type": "qa",
        "query": "QA 问题",
        "answer": original_answer,
        "reranked_docs": [_doc()],
        "evidence_ledger": _ledger(),
        "claim_audit_projection": original_projection.to_state(),
        "claim_audit": {
            "status": "failed",
            "claims": [
                {
                    "claim_id": "c1",
                    "verdict": "unsupported",
                    "cited_evidence_ids": ["E001"],
                    "reason": "原始声明不受支持",
                }
            ],
        },
    }

    repaired = ClaimRepairAgent.repair(state)
    repaired_projection = load_claim_audit_projection(
        repaired["claim_audit_projection"],
        answer=repaired["answer"],
    )

    assert repaired_projection.obligation_ids == ("eu_qa_fact", "eu_qa_gap")
    assert len(repaired_projection.segments) == 1
    assert repaired_projection.segments[0].status is (
        ClaimAuditProjectionStatus.GENERATED
    )
    assert repaired_projection.segments[0].source_status == "repaired"

    captured: list[dict] = []
    _support_projected_claims(monkeypatch, captured)
    audit = ClaimEvidenceVerifierAgent.audit({**state, **repaired})

    assert audit["claim_audit_passed"] is True
    assert [claim["text"] for claim in captured] == [repaired["answer"]]


def test_empty_qa_generation_returns_bound_operational_error(monkeypatch):
    class _EmptyQAClient:
        def invoke(self, _messages):
            return AIMessage(content="   ")

    monkeypatch.setattr(
        qa.Generator,
        "_get_client_for_node",
        lambda *_args, **_kwargs: _EmptyQAClient(),
    )

    generated = qa.generate_node(
        {
            "query": "QA 问题",
            "reranked_docs": [_doc()],
            "chat_history": [],
            "evidence_unit_generate_ids": ["eu_qa_1"],
        }
    )
    projection = load_claim_audit_projection(
        generated["claim_audit_projection"], answer=generated["answer"]
    )

    assert generated["answer"] == qa.QA_GENERATION_FAILURE_ANSWER
    assert generated["error"] == "qa_generation_empty"
    assert projection.has_generated_content is False
    assert projection.segments[0].status is ClaimAuditProjectionStatus.OPERATIONAL
    assert projection.segments[0].obligation_ids == ("eu_qa_1",)


def test_empty_summary_section_becomes_operational_error_without_throwing():
    generated = GlobalSummaryAgent.build_final_summary(
        {
            "summary_source": "a.pdf",
            "summary_docs": [_doc()],
            "summary_section_results": [
                {
                    "section_id": "method",
                    "title": "方法",
                    "content": "   ",
                    "unit_id": "eu_summary_method",
                    "status": "generated",
                    "evidence": [],
                }
            ],
            "evidence_ledger": _ledger(),
        }
    )
    projection = load_claim_audit_projection(
        generated["claim_audit_projection"], answer=generated["answer"]
    )

    assert generated["error"] == "summary_evidence_units_incomplete"
    assert EVIDENCE_UNIT_FAILURE_MESSAGE in generated["answer"]
    assert projection.has_generated_content is False
    assert projection.segments[0].status is ClaimAuditProjectionStatus.OPERATIONAL
    assert projection.segments[0].source_status == "generation_error"
    assert projection.segments[0].obligation_ids == ("eu_summary_method",)


def test_qa_projection_binds_multiple_units_and_legacy_qa_keeps_full_audit(
    monkeypatch,
):
    doc = _doc()

    class _QAClient:
        def invoke(self, _messages):
            return AIMessage(content="QA 事实由文档支持[E001]。")

    monkeypatch.setattr(
        qa.Generator,
        "_get_client_for_node",
        lambda *_args, **_kwargs: _QAClient(),
    )
    base_state = {
        "query": "QA 问题",
        "reranked_docs": [doc],
        "evidence_ledger": _ledger(),
        "chat_history": [],
    }
    unit_output = qa.generate_node(
        {
            **base_state,
            "evidence_unit_generate_ids": ["eu_qa_1", "eu_qa_2"],
        }
    )
    unit_projection = load_claim_audit_projection(
        unit_output["claim_audit_projection"], answer=unit_output["answer"]
    )
    legacy_output = qa.generate_node(base_state)
    legacy_projection = load_claim_audit_projection(
        legacy_output["claim_audit_projection"], answer=legacy_output["answer"]
    )

    assert unit_projection.segments[0].obligation_ids == ("eu_qa_1", "eu_qa_2")
    assert legacy_projection.segments[0].obligation_ids == ()
    assert legacy_projection.audit_text == legacy_output["answer"]

    captured: list[dict] = []
    _support_projected_claims(monkeypatch, captured)
    audit = ClaimEvidenceVerifierAgent.audit(
        {"task_type": "qa", **base_state, **legacy_output}
    )

    assert audit["claim_audit_passed"] is True
    assert [claim["text"] for claim in captured] == [
        "QA 事实由文档支持[E001]。"
    ]
