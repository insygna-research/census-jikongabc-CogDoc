import pytest

from cogdoc.agents import evidence_verifier
from cogdoc.agents.evidence_verifier import (
    EvidenceVerification,
    EvidenceVerifierAgent,
    requires_evidence_verification,
    select_verification_docs,
    should_verify_evidence,
)
from cogdoc.config.settings import Settings
from cogdoc.graph.subgraphs import qa


def _settings(**overrides):
    return Settings(_env_file=None, **overrides)


def _doc(chunk_id: str, source: str, text: str = "evidence") -> dict:
    return {
        "text": text,
        "meta": {
            "chunk_id": chunk_id,
            "source": source,
            "page": 1,
            "page_start": 1,
            "page_end": 1,
        },
    }


# 精确事实问题进入校验，普通概念解释不增加额外模型调用。
def test_fact_query_detection_is_selective():
    assert requires_evidence_verification("比赛时长分别是多少") is True
    assert requires_evidence_verification("报名费报销比例是多少") is True
    assert requires_evidence_verification("What is the deadline?") is True
    assert requires_evidence_verification("介绍一下这个比赛") is False


# 校验候选优先覆盖不同来源，再按原始排名补足。
def test_select_verification_docs_diversifies_sources():
    docs = [
        _doc("a1", "a.pdf"),
        _doc("a2", "a.pdf"),
        _doc("b1", "b.pdf"),
        _doc("c1", "c.pdf"),
    ]

    selected = select_verification_docs(docs, 3)

    assert [doc["meta"]["chunk_id"] for doc in selected] == ["a1", "b1", "c1"]


# 第一阶段放行和阈值附近的事实问题都进入二阶段，明显低分仍直接拒答。
def test_should_verify_supported_and_borderline_fact_queries():
    settings = _settings(qa_evidence_verify_borderline_min_score=0.75)

    assert should_verify_evidence(
        {
            "query": "比赛时长是多少",
            "retrieval_first_stage_supported": True,
        },
        settings,
    )
    assert should_verify_evidence(
        {
            "query": "比赛时长是多少",
            "retrieval_first_stage_supported": False,
            "retrieval_abstained": True,
            "retrieval_abstain_reason": "below_threshold",
            "retrieval_confidence": 0.9,
        },
        settings,
    )
    assert not should_verify_evidence(
        {
            "query": "比赛时长是多少",
            "retrieval_first_stage_supported": False,
            "retrieval_abstained": True,
            "retrieval_abstain_reason": "below_threshold",
            "retrieval_confidence": 0.5,
        },
        settings,
    )


# 结构化结论只有引用闭集中的真实 chunk_id 才能放行。
def test_verifier_accepts_supported_evidence_with_valid_chunk_id(monkeypatch):
    monkeypatch.setattr(evidence_verifier, "get_settings", lambda: _settings())
    monkeypatch.setattr(
        evidence_verifier.Generator, "_get_client", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        evidence_verifier,
        "invoke_structured",
        lambda *_args, **_kwargs: EvidenceVerification(
            supported=True,
            evidence_chunk_ids=["a1"],
            reason="证据明确给出时长",
        ),
    )

    output = EvidenceVerifierAgent.verify(
        {
            "query": "比赛时长是多少",
            "retrieval_first_stage_supported": False,
            "retrieval_abstained": True,
            "verification_docs": [_doc("a1", "a.pdf", "比赛持续 5 小时")],
        }
    )

    assert output["evidence_supported"] is True
    assert output["retrieval_abstained"] is False
    assert output["evidence_verified_chunk_ids"] == ["a1"]


# 模型声称支持但编造证据标识时必须拒答。
def test_verifier_rejects_fabricated_chunk_id(monkeypatch):
    monkeypatch.setattr(evidence_verifier, "get_settings", lambda: _settings())
    monkeypatch.setattr(
        evidence_verifier.Generator, "_get_client", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        evidence_verifier,
        "invoke_structured",
        lambda *_args, **_kwargs: EvidenceVerification(
            supported=True,
            evidence_chunk_ids=["fabricated"],
            reason="声称有证据",
        ),
    )

    output = EvidenceVerifierAgent.verify(
        {
            "query": "比赛时长是多少",
            "retrieval_first_stage_supported": True,
            "verification_docs": [_doc("a1", "a.pdf")],
        }
    )

    assert output["evidence_supported"] is False
    assert output["retrieval_abstain_reason"] == "evidence_not_supported"
    assert output["evidence_verified_chunk_ids"] == []


# 校验器使用了 Top 3 之外的来源时，该 chunk 必须进入后续生成和引用上下文。
def test_evidence_verify_node_merges_verified_docs_into_generation(monkeypatch):
    top_doc = _doc("a1", "a.pdf")
    cross_source_doc = _doc("b1", "b.pdf")
    monkeypatch.setattr(
        qa.EvidenceVerifierAgent,
        "verify",
        lambda _state: {
            "evidence_verification_required": True,
            "evidence_supported": True,
            "evidence_verification_reason": "两篇证据均完整",
            "evidence_verified_chunk_ids": ["a1", "b1"],
            "retrieval_abstained": False,
            "retrieval_abstain_reason": "evidence_supported",
        },
    )
    monkeypatch.setattr(qa, "log_event", lambda *args, **kwargs: None)

    output = qa.evidence_verify_node(
        {
            "reranked_docs": [top_doc],
            "verification_docs": [top_doc, cross_source_doc],
        }
    )

    assert [doc["meta"]["chunk_id"] for doc in output["reranked_docs"]] == [
        "a1",
        "b1",
    ]


# 校验器异常时不改变第一阶段结论：原本放行则放行，原本拒答则继续拒答。
@pytest.mark.parametrize("first_stage_supported", [True, False])
def test_verifier_error_preserves_first_stage_decision(
    monkeypatch, first_stage_supported
):
    monkeypatch.setattr(evidence_verifier, "get_settings", lambda: _settings())
    monkeypatch.setattr(
        evidence_verifier.Generator,
        "_get_client",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    output = EvidenceVerifierAgent.verify(
        {
            "query": "比赛时长是多少",
            "retrieval_first_stage_supported": first_stage_supported,
            "retrieval_abstained": not first_stage_supported,
            "retrieval_abstain_reason": "below_threshold",
            "verification_docs": [_doc("a1", "a.pdf")],
        }
    )

    assert output["evidence_supported"] is first_stage_supported
    assert output["retrieval_abstained"] is (not first_stage_supported)
    assert output["evidence_verifier_error"] == "RuntimeError"
