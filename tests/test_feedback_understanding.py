from cogdoc.agents.feedback_understanding import analyze_feedback


# 验证明确纠错会建议生成知识草稿。
def test_analyze_feedback_recommends_pending_knowledge_for_correction():
    analysis = analyze_feedback(
        {
            "trace_id": "t1",
            "feedback": "correction",
            "kb_id": "kb",
            "query": "问题",
            "correction_text": "正确说法",
            "citations": [{"chunk_id": "c1", "source": "a.pdf"}],
        }
    )

    assert analysis["feedback_type"] == "correction"
    assert analysis["recommended_action"] == "create_pending_knowledge"
    assert analysis["extracted_claim"] == "正确说法"
    assert analysis["confidence"] >= 0.8
    assert analysis["target"]["chunk_ids"] == ["c1"]


# 验证坏检索反馈会建议调权。
def test_analyze_feedback_recommends_retrieval_adjustment():
    analysis = analyze_feedback(
        {
            "trace_id": "t2",
            "feedback": "thumbs_down",
            "kb_id": "kb",
            "query": "问题",
            "feedback_text": "引用不相关",
            "evidence": [{"chunk_id": "c2", "source": "b.pdf"}],
        }
    )

    assert analysis["feedback_type"] == "bad_retrieval"
    assert analysis["recommended_action"] == "adjust_retrieval"
    assert analysis["weight_delta"] < 0
    assert analysis["target"]["sources"] == ["b.pdf"]


# 验证缺少目标的普通反馈只记录。
def test_analyze_feedback_records_low_context_feedback_only():
    analysis = analyze_feedback({"trace_id": "t3", "feedback": "thumbs_down"})

    assert analysis["recommended_action"] == "record_only"
    assert analysis["needs_review"] is True


def _ledger_entry(chunk_id, citation, answer, *, evidence_id="E001"):
    start = answer.index(citation)
    return {
        "evidence_id": evidence_id,
        "chunk_id": chunk_id,
        "source_type": "document",
        "source": "a.pdf",
        "page": 1,
        "span_start": 0,
        "span_end": 20,
        "occurrences": [
            {
                "index": 0,
                "answer_start": start,
                "answer_end": start + len(citation),
            }
        ],
    }


def _internal_evidence_entry(
    chunk_id,
    *,
    evidence_id="E001",
    source="a.pdf",
    page=1,
    span_start=0,
    span_end=20,
):
    return {
        "evidence_id": evidence_id,
        "chunk_id": chunk_id,
        "source_type": "document",
        "source": source,
        "page": page,
        "page_start": page,
        "page_end": page,
        "span_start": span_start,
        "span_end": span_end,
        "display_citation": f"[{source}:P{page}]",
    }


# 非空精确账本只归因实际引用的分块，不扩散到同轮其他候选证据。
def test_analyze_feedback_prefers_cited_ledger_chunks():
    citation = "[a.pdf:P1]"
    answer = f"结论{citation}。"
    analysis = analyze_feedback(
        {
            "trace_id": "t-ledger",
            "feedback": "thumbs_down",
            "feedback_type": "bad_retrieval",
            "kb_id": "kb",
            "query": "问题",
            "answer": answer,
            "citations": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
            "evidence": [
                {
                    "chunk_id": "c1",
                    "source_type": "document",
                    "source": "a.pdf",
                    "page": 1,
                },
                {
                    "chunk_id": "c2",
                    "source_type": "document",
                    "source": "a.pdf",
                    "page": 1,
                },
            ],
            "citation_ledger": [_ledger_entry("c2", citation, answer)],
        }
    )

    assert analysis["target"]["chunk_ids"] == ["c2"]
    assert analysis["recommended_action"] == "adjust_retrieval"


# repair 可引用全局 registry 中未进入公开 evidence 摘要的证据；
# 反馈归因必须使用 internal evidence_ledger 的精确视图。
def test_analyze_feedback_uses_internal_registry_when_public_evidence_omits_citation():
    citation = "[b.pdf:P2]"
    answer = f"结论{citation}。"
    ledger = _ledger_entry("c2", citation, answer, evidence_id="E002")
    ledger.update(
        {
            "source": "b.pdf",
            "page": 2,
            "page_start": 2,
            "page_end": 2,
            "span_start": 40,
            "span_end": 60,
        }
    )
    analysis = analyze_feedback(
        {
            "trace_id": "t-global-registry",
            "feedback": "thumbs_down",
            "feedback_type": "bad_retrieval",
            "kb_id": "kb",
            "query": "问题",
            "answer": answer,
            # 面向用户的证据摘要不含修复时引用的 c2。
            "evidence": [
                {
                    "chunk_id": "c1",
                    "source_type": "document",
                    "source": "a.pdf",
                    "page": 1,
                }
            ],
            "evidence_ledger": [
                _internal_evidence_entry("c1"),
                _internal_evidence_entry(
                    "c2",
                    evidence_id="E002",
                    source="b.pdf",
                    page=2,
                    span_start=40,
                    span_end=60,
                ),
            ],
            "citation_ledger": [ledger],
        }
    )

    assert analysis["target"]["chunk_ids"] == ["c2"]
    assert analysis["target"]["sources"] == ["b.pdf"]
    assert analysis["recommended_action"] == "adjust_retrieval"


# 伪造 chunk 即使与合法证据混在同一载荷中也不能越过证据闭集。
def test_analyze_feedback_fails_closed_for_forged_ledger_chunk():
    citation = "[a.pdf:P1]"
    answer = f"结论{citation}。"
    analysis = analyze_feedback(
        {
            "trace_id": "t-forged",
            "feedback": "thumbs_down",
            "feedback_type": "bad_retrieval",
            "kb_id": "kb",
            "query": "问题",
            "answer": answer,
            "evidence": [
                {
                    "chunk_id": "c1",
                    "source_type": "document",
                    "source": "a.pdf",
                    "page": 1,
                }
            ],
            "citation_ledger": [_ledger_entry("forged", citation, answer)],
        }
    )

    assert analysis["target"]["chunk_ids"] == []
    assert analysis["recommended_action"] == "record_only"


# 空 ledger 是旧 trace 的兼容形态，仍可在同一载荷的可信引用内回退。
def test_analyze_feedback_empty_ledger_uses_legacy_targets():
    analysis = analyze_feedback(
        {
            "trace_id": "t-empty-ledger",
            "feedback": "thumbs_down",
            "feedback_type": "bad_retrieval",
            "kb_id": "kb",
            "query": "问题",
            "citations": [{"chunk_id": "c1", "source": "a.pdf"}],
            "evidence": [{"chunk_id": "c2", "source": "b.pdf"}],
            "citation_ledger": [],
        }
    )

    assert analysis["target"]["chunk_ids"] == ["c1", "c2"]


def test_analyze_feedback_non_list_ledger_fails_closed_without_legacy_fallback():
    analysis = analyze_feedback(
        {
            "trace_id": "t-malformed-ledger",
            "feedback": "thumbs_down",
            "feedback_type": "bad_retrieval",
            "kb_id": "kb",
            "query": "问题",
            "citations": [{"chunk_id": "c1", "source": "a.pdf"}],
            "evidence": [{"chunk_id": "c1", "source": "a.pdf"}],
            "citation_ledger": {"evidence_id": "E001"},
        }
    )

    assert analysis["target"]["chunk_ids"] == []
    assert analysis["recommended_action"] == "record_only"


def test_analyze_feedback_malformed_nonempty_ledger_does_not_fallback_to_registry():
    citation = "[a.pdf:P1]"
    answer = f"结论{citation}。"
    malformed = _ledger_entry("c1", citation, answer, evidence_id="E000")
    analysis = analyze_feedback(
        {
            "trace_id": "t-malformed-list-ledger",
            "feedback": "thumbs_down",
            "feedback_type": "bad_retrieval",
            "kb_id": "kb",
            "query": "问题",
            "answer": answer,
            "citations": [{"chunk_id": "legacy", "source": "legacy.pdf"}],
            "evidence": [{"chunk_id": "c1", "source": "a.pdf", "page": 1}],
            "evidence_ledger": [_internal_evidence_entry("c1")],
            "citation_ledger": [malformed],
        }
    )

    assert analysis["target"]["chunk_ids"] == []
    assert analysis["recommended_action"] == "record_only"


def test_analyze_feedback_requires_one_to_one_physical_citation_coverage():
    citation = "[a.pdf:P1]"
    answer = f"结论{citation}和补充{citation}。"
    analysis = analyze_feedback(
        {
            "trace_id": "t-partial-ledger",
            "feedback": "thumbs_down",
            "feedback_type": "bad_retrieval",
            "kb_id": "kb",
            "query": "问题",
            "answer": answer,
            "citations": [{"chunk_id": "c1", "source": "a.pdf"}],
            "evidence": [
                {
                    "chunk_id": "c1",
                    "source_type": "document",
                    "source": "a.pdf",
                    "page": 1,
                }
            ],
            # 只声明第一个实体引用，不得对部分通过的表做归因。
            "citation_ledger": [_ledger_entry("c1", citation, answer)],
        }
    )

    assert analysis["target"]["chunk_ids"] == []


def test_analyze_feedback_binds_retrieval_visible_span_offsets():
    citation = "[a.pdf:P1]"
    answer = f"结论{citation}。"
    analysis = analyze_feedback(
        {
            "trace_id": "t-span-mismatch",
            "feedback": "thumbs_down",
            "feedback_type": "bad_retrieval",
            "kb_id": "kb",
            "query": "问题",
            "answer": answer,
            "evidence": [
                {
                    "evidence_id": "E001",
                    "chunk_id": "c1",
                    "source_type": "document",
                    "source": "a.pdf",
                    "page": 1,
                    "retrieval": {
                        "evidence_id": "E001",
                        "evidence_text_start": 1,
                        "evidence_text_end": 20,
                    },
                }
            ],
            "citation_ledger": [_ledger_entry("c1", citation, answer)],
        }
    )

    assert analysis["target"]["chunk_ids"] == []
