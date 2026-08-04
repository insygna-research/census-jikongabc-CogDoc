from cogdoc.agents import claim_evidence_verifier
from cogdoc.agents.answer_markers import (
    NO_RELEVANT_CONTENT_ANSWER,
    NO_RELEVANT_CONTENT_MARKER,
)
from cogdoc.agents.citation_validator import CitationValidatorAgent
from cogdoc.agents.claim_evidence_verifier import (
    CLAIM_AUDIT_BLOCKED_ANSWER,
    CLAIM_AUDIT_EXEMPTION_GUIDANCE,
    CLAIM_AUDIT_EXEMPTION_UPSTREAM_ERROR,
    ClaimAssessment,
    ClaimAssessmentBatch,
    ClaimEvidenceVerifierAgent,
    ClaimRepair,
    ClaimRepairAgent,
    block_unfaithful_answer,
    extract_claim_units,
    make_claim_audit_exemption,
)
from cogdoc.config.settings import Settings


def _settings(**overrides):
    values = {
        "claim_verification_enabled": True,
        "claim_verification_max_claims": 10,
        "claim_verification_max_claims_per_batch": 8,
        "claim_verification_max_docs_per_batch": 8,
        "claim_verification_max_chars_per_doc": 1600,
        "claim_verification_max_repair_attempts": 1,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _doc(
    text: str = "报名截止日期是 8 月 30 日。",
    *,
    chunk_id: str = "chunk:guide:2",
) -> dict:
    return {
        "text": text,
        "meta": {
            "chunk_id": chunk_id,
            "source": "guide.pdf",
            "page": 2,
            "page_start": 2,
            "page_end": 2,
        },
    }


def _state(answer: str, *, doc: dict | None = None) -> dict:
    return {
        "task_type": "qa",
        "query": "报名截止日期是什么时候？",
        "answer": answer,
        "critique": "",
        "reranked_docs": [doc or _doc()],
        "is_local": False,
    }


def _assessment(
    *,
    verdict: str,
    evidence_chunk_ids: list[str],
    reason: str = "证据直接支持该声明",
) -> ClaimAssessmentBatch:
    return ClaimAssessmentBatch(
        assessments=[
            ClaimAssessment(
                claim_id="c1",
                verdict=verdict,
                evidence_chunk_ids=evidence_chunk_ids,
                reason=reason,
                confidence=0.98,
            )
        ]
    )


def _stub_verifier(monkeypatch, output: ClaimAssessmentBatch) -> None:
    monkeypatch.setattr(claim_evidence_verifier, "get_settings", _settings)
    monkeypatch.setattr(
        claim_evidence_verifier.Generator,
        "_get_client_for_node",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        claim_evidence_verifier,
        "invoke_structured",
        lambda *args, **kwargs: output,
    )


def test_claim_extraction_keeps_citation_after_sentence_punctuation_attached():
    units = extract_claim_units("报名截止日期是 8 月 30 日。[guide.pdf:P2]")

    assert len(units) == 1
    assert units[0]["text"] == "报名截止日期是 8 月 30 日。[guide.pdf:P2]"
    assert units[0]["citation_refs"] == ["document:guide.pdf:P2"]


def test_claim_extraction_audits_markdown_heading_and_fenced_code_contents():
    units = extract_claim_units(
        "# 报名截止日期是 9 月 30 日。[guide.pdf:P2]\n"
        "```text\n"
        "系统端口是 9000。[guide.pdf:P2]\n"
        "```"
    )

    assert [unit["text"] for unit in units] == [
        "报名截止日期是 9 月 30 日。[guide.pdf:P2]",
        "系统端口是 9000。[guide.pdf:P2]",
    ]
    assert all(unit["citation_refs"] == ["document:guide.pdf:P2"] for unit in units)


def test_claim_audit_downgrades_model_not_factual_for_factual_heading(monkeypatch):
    _stub_verifier(
        monkeypatch,
        _assessment(verdict="not_factual", evidence_chunk_ids=[]),
    )

    result = ClaimEvidenceVerifierAgent.audit(
        _state("# 报名截止日期是 9 月 30 日。[guide.pdf:P2]")
    )

    assert result["claim_audit_passed"] is False
    assert result["claim_audit"]["status"] == "failed"
    claim = result["claim_audit"]["claims"][0]
    assert claim["verdict"] == "insufficient"
    assert "not_factual" in claim["reason"]


def test_claim_audit_accepts_not_factual_only_for_fixed_structure(monkeypatch):
    _stub_verifier(
        monkeypatch,
        _assessment(verdict="not_factual", evidence_chunk_ids=[]),
    )

    result = ClaimEvidenceVerifierAgent.audit(_state("# 多文档对比"))

    assert result["claim_audit_passed"] is True
    assert result["claim_audit"]["status"] == "passed"
    assert result["claim_audit"]["counts"]["skipped_statements"] == 1


def test_claim_audit_accepts_supported_claim_with_cited_evidence(monkeypatch):
    _stub_verifier(
        monkeypatch,
        _assessment(verdict="supported", evidence_chunk_ids=["chunk:guide:2"]),
    )

    result = ClaimEvidenceVerifierAgent.audit(
        _state("报名截止日期是 8 月 30 日。[guide.pdf:P2]")
    )

    assert result["claim_audit_passed"] is True
    assert result["claim_verifier_error"] == ""
    audit = result["claim_audit"]
    assert audit["status"] == "passed"
    assert audit["counts"] == {
        "claim_count": 1,
        "supported": 1,
        "unsupported": 0,
        "insufficient": 0,
        "cited": 1,
        "skipped_statements": 0,
    }
    assert audit["metrics"] == {
        "claim_support_rate": 1.0,
        "citation_coverage": 1.0,
        "unsupported_claim_rate": 0.0,
    }
    assert audit["claims"][0]["supporting_chunk_ids"] == ["chunk:guide:2"]


def test_claim_audit_does_not_treat_marker_plus_facts_as_safe_abstention(monkeypatch):
    _stub_verifier(monkeypatch, ClaimAssessmentBatch(assessments=[]))

    result = ClaimEvidenceVerifierAgent.audit(
        _state(
            f"{NO_RELEVANT_CONTENT_MARKER}，但报名截止日期是 9 月 30 日。[guide.pdf:P2]"
        )
    )

    assert result["claim_audit_required"] is True
    assert result["claim_audit_passed"] is False
    assert result["claim_audit"]["status"] == "failed"


def test_claim_audit_records_repair_that_safely_abstains(monkeypatch):
    monkeypatch.setattr(claim_evidence_verifier, "get_settings", _settings)

    result = ClaimEvidenceVerifierAgent.audit(
        {
            **_state(NO_RELEVANT_CONTENT_ANSWER),
            "claim_repair_count": 1,
        }
    )

    assert result["claim_audit_passed"] is True
    assert result["claim_audit"]["status"] == "not_run"
    assert result["claim_audit"]["reason_code"] == "abstained"
    assert result["claim_audit"]["repair"] == {
        "attempted": True,
        "attempt_count": 1,
        "succeeded": True,
    }


def test_claim_audit_fails_closed_for_arbitrary_answer_without_documents(monkeypatch):
    monkeypatch.setattr(claim_evidence_verifier, "get_settings", _settings)
    state = _state("报名截止日期是 9 月 30 日。[guide.pdf:P2]")
    state["reranked_docs"] = []

    result = ClaimEvidenceVerifierAgent.audit(state)

    assert result["claim_audit_passed"] is False
    assert result["claim_audit"]["status"] == "error"
    assert result["claim_audit"]["reason_code"] == "no_evidence_documents"


def test_claim_audit_fails_closed_for_unmarked_upstream_error(monkeypatch):
    monkeypatch.setattr(claim_evidence_verifier, "get_settings", _settings)
    state = _state("报名截止日期是 9 月 30 日。[guide.pdf:P2]")
    state["error"] = "profile failed"

    result = ClaimEvidenceVerifierAgent.audit(state)

    assert result["claim_audit_passed"] is False
    assert result["claim_audit"]["status"] == "error"
    assert result["claim_audit"]["reason_code"] == "upstream_error"


def test_claim_audit_allows_answer_bound_deterministic_guidance(monkeypatch):
    monkeypatch.setattr(claim_evidence_verifier, "get_settings", _settings)
    answer = "请在摘要问题中明确指定要总结的文件名。"
    state = {
        "task_type": "summary",
        "answer": answer,
        "summary_docs": [],
        "claim_audit_exemption": make_claim_audit_exemption(
            answer,
            CLAIM_AUDIT_EXEMPTION_GUIDANCE,
        ),
    }

    result = ClaimEvidenceVerifierAgent.audit(state)

    assert result["claim_audit_passed"] is True
    assert result["claim_audit"]["status"] == "not_run"
    assert result["claim_audit"]["reason_code"] == CLAIM_AUDIT_EXEMPTION_GUIDANCE


def test_claim_audit_rejects_exemption_when_bound_answer_does_not_match(monkeypatch):
    monkeypatch.setattr(claim_evidence_verifier, "get_settings", _settings)
    guidance = "请在摘要问题中明确指定要总结的文件名。"
    state = {
        "task_type": "summary",
        "answer": f"{guidance} 报名截止日期是 9 月 30 日。",
        "summary_docs": [],
        "claim_audit_exemption": make_claim_audit_exemption(
            guidance,
            CLAIM_AUDIT_EXEMPTION_GUIDANCE,
        ),
    }

    result = ClaimEvidenceVerifierAgent.audit(state)

    assert result["claim_audit_passed"] is False
    assert result["claim_audit"]["status"] == "error"
    assert result["claim_audit"]["reason_code"] == "no_evidence_documents"


def test_claim_audit_upstream_error_exemption_requires_error_and_exact_answer(
    monkeypatch,
):
    monkeypatch.setattr(claim_evidence_verifier, "get_settings", _settings)
    answer = "模型生成对比画像失败。建议稍后重试。"
    marker = make_claim_audit_exemption(
        answer,
        CLAIM_AUDIT_EXEMPTION_UPSTREAM_ERROR,
    )

    without_error = ClaimEvidenceVerifierAgent.audit(
        {
            "task_type": "compare",
            "answer": answer,
            "compare_docs_by_source": {},
            "claim_audit_exemption": marker,
        }
    )
    with_error = ClaimEvidenceVerifierAgent.audit(
        {
            "task_type": "compare",
            "answer": answer,
            "error": "TimeoutError",
            "compare_docs_by_source": {},
            "claim_audit_exemption": marker,
        }
    )

    assert without_error["claim_audit"]["status"] == "error"
    assert without_error["claim_audit"]["reason_code"] == "no_evidence_documents"
    assert with_error["claim_audit"]["status"] == "not_run"
    assert (
        with_error["claim_audit"]["reason_code"] == CLAIM_AUDIT_EXEMPTION_UPSTREAM_ERROR
    )


def test_claim_audit_rejects_semantically_unsupported_claim_with_legal_citation(
    monkeypatch,
):
    answer = "报名截止日期是 9 月 30 日。[guide.pdf:P2]"
    doc = _doc("报名截止日期是 8 月 30 日。")
    assert CitationValidatorAgent.validate_citations(answer, [doc])["is_valid"] is True
    _stub_verifier(
        monkeypatch,
        _assessment(
            verdict="unsupported",
            evidence_chunk_ids=["chunk:guide:2"],
            reason="答案日期与证据冲突",
        ),
    )

    result = ClaimEvidenceVerifierAgent.audit(_state(answer, doc=doc))

    assert result["claim_audit_passed"] is False
    audit = result["claim_audit"]
    assert audit["status"] == "failed"
    assert audit["reason_code"] == "claims_not_supported"
    assert audit["counts"]["unsupported"] == 1
    assert audit["metrics"]["citation_coverage"] == 1.0
    assert audit["metrics"]["claim_support_rate"] == 0.0
    assert audit["claims"][0]["reason"] == "答案日期与证据冲突"


def test_claim_audit_rejects_model_fabricated_chunk_id(monkeypatch):
    _stub_verifier(
        monkeypatch,
        _assessment(verdict="supported", evidence_chunk_ids=["fabricated-chunk"]),
    )

    result = ClaimEvidenceVerifierAgent.audit(
        _state("报名截止日期是 8 月 30 日。[guide.pdf:P2]")
    )

    assert result["claim_audit_passed"] is False
    claim = result["claim_audit"]["claims"][0]
    assert claim["verdict"] == "insufficient"
    assert claim["supporting_chunk_ids"] == []
    assert claim["cited_chunk_ids"] == ["chunk:guide:2"]
    assert "有效的引用证据标识" in claim["reason"]


def test_claim_audit_rejects_uncited_fact_even_if_model_claims_support(monkeypatch):
    _stub_verifier(
        monkeypatch,
        _assessment(verdict="supported", evidence_chunk_ids=["chunk:guide:2"]),
    )

    result = ClaimEvidenceVerifierAgent.audit(_state("报名截止日期是 8 月 30 日。"))

    assert result["claim_audit_passed"] is False
    audit = result["claim_audit"]
    claim = audit["claims"][0]
    assert claim["verdict"] == "unsupported"
    assert claim["citation_refs"] == []
    assert claim["cited_chunk_ids"] == []
    assert claim["supporting_chunk_ids"] == []
    assert claim["reason"] == "事实声明没有显式引用"
    assert audit["metrics"]["citation_coverage"] == 0.0


def test_claim_audit_verifier_exception_fails_closed(monkeypatch):
    monkeypatch.setattr(claim_evidence_verifier, "get_settings", _settings)
    monkeypatch.setattr(
        claim_evidence_verifier.Generator,
        "_get_client_for_node",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    result = ClaimEvidenceVerifierAgent.audit(
        _state("报名截止日期是 8 月 30 日。[guide.pdf:P2]")
    )

    assert result["claim_audit_required"] is True
    assert result["claim_audit_passed"] is False
    assert result["claim_verifier_error"] == "RuntimeError"
    assert result["claim_audit"]["status"] == "error"
    assert result["claim_audit"]["reason_code"] == "RuntimeError"

    blocked = block_unfaithful_answer({**_state("候选答案"), **result})
    assert blocked["answer"] == CLAIM_AUDIT_BLOCKED_ANSWER
    assert blocked["claim_audit_passed"] is False
    assert blocked["sources"] == []
    assert blocked["evidence"] == []
    assert "RuntimeError" in blocked["critique"]


def test_claim_audit_does_not_bypass_failed_physical_citation_gate(monkeypatch):
    monkeypatch.setattr(claim_evidence_verifier, "get_settings", _settings)

    result = ClaimEvidenceVerifierAgent.audit(
        {
            **_state("报名截止日期是 8 月 30 日。[missing.pdf:P9]"),
            "critique": "引用来源不存在",
        }
    )

    assert result["claim_audit_required"] is True
    assert result["claim_audit_passed"] is False
    assert result["claim_audit"]["status"] == "error"
    assert result["claim_audit"]["reason_code"] == "physical_citation_rejected"


def test_claim_audit_fails_closed_when_answer_exceeds_max_claims(monkeypatch):
    monkeypatch.setattr(
        claim_evidence_verifier,
        "get_settings",
        lambda: _settings(claim_verification_max_claims=2),
    )
    monkeypatch.setattr(
        claim_evidence_verifier.Generator,
        "_get_client_for_node",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("overflow must fail before calling the verifier")
        ),
    )
    answer = "\n".join(
        [
            "报名截止日期是 8 月 30 日。[guide.pdf:P2]",
            "报名材料包括身份证。[guide.pdf:P2]",
            "报名需要线上提交。[guide.pdf:P2]",
        ]
    )

    result = ClaimEvidenceVerifierAgent.audit(_state(answer))

    assert result["claim_audit_passed"] is False
    assert result["claim_verifier_error"] == ""
    audit = result["claim_audit"]
    assert audit["status"] == "failed"
    assert audit["reason_code"] == "max_claims_exceeded"
    assert audit["counts"]["insufficient"] == 1
    assert audit["claims"][0]["claim_id"] == "overflow"


def test_claim_repair_returns_revised_answer_and_increments_attempt(monkeypatch):
    captured = {}
    monkeypatch.setattr(claim_evidence_verifier, "get_settings", _settings)
    monkeypatch.setattr(
        claim_evidence_verifier.Generator,
        "_get_client_for_node",
        lambda *args, **kwargs: object(),
    )

    def fake_invoke(llm, schema, messages):
        captured["schema"] = schema
        captured["messages"] = messages
        return ClaimRepair(revised_answer="报名截止日期是 8 月 30 日。[guide.pdf:P2]")

    monkeypatch.setattr(claim_evidence_verifier, "invoke_structured", fake_invoke)
    state = _state("报名截止日期是 9 月 30 日。[guide.pdf:P2]")
    state["claim_audit"] = {
        "status": "failed",
        "claims": [
            {
                "claim_id": "c1",
                "text": state["answer"],
                "verdict": "unsupported",
                "cited_chunk_ids": ["chunk:guide:2"],
                "reason": "日期冲突",
            }
        ],
    }

    result = ClaimRepairAgent.repair(state)

    assert result["answer"] == "报名截止日期是 8 月 30 日。[guide.pdf:P2]"
    assert result["messages"][0].content == result["answer"]
    assert result["claim_repair_count"] == 1
    assert result["claim_repair_error"] == ""
    assert captured["schema"] is ClaimRepair
    assert "日期冲突" in captured["messages"][1]["content"]
    assert "报名截止日期是 8 月 30 日" in captured["messages"][1]["content"]


def test_claim_repair_error_is_returned_for_fail_closed_routing(monkeypatch):
    monkeypatch.setattr(claim_evidence_verifier, "get_settings", _settings)
    monkeypatch.setattr(
        claim_evidence_verifier.Generator,
        "_get_client_for_node",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        claim_evidence_verifier,
        "invoke_structured",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("offline")),
    )
    state = _state("报名截止日期是 9 月 30 日。[guide.pdf:P2]")
    state["claim_audit"] = {
        "status": "failed",
        "claims": [
            {
                "claim_id": "c1",
                "verdict": "unsupported",
                "cited_chunk_ids": ["chunk:guide:2"],
                "reason": "日期冲突",
            }
        ],
    }

    result = ClaimRepairAgent.repair(state)

    assert result == {
        "claim_repair_count": 1,
        "claim_repair_error": "TimeoutError",
    }
