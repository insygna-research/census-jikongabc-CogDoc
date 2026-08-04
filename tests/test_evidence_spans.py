import copy

import pytest

from cogdoc.tools.evidence_rendering import render_evidence_block
from cogdoc.tools.retriever.evidence_pack import (
    EvidencePackCandidate,
    build_evidence_pack,
)
from cogdoc.tools.retriever.evidence_spans import (
    REASON_FALLBACK_NO_MATCH,
    REASON_FALLBACK_NO_TERMS,
    REASON_LONG_SENTENCE_WINDOW,
    REASON_QUERY_SPAN,
    REASON_WITHIN_BUDGET,
    EvidenceSpanSelector,
    select_evidence_span,
    select_evidence_spans,
)


def _doc(
    text: str,
    *,
    chunk_id: str = "chunk-1",
    retrieval: dict | None = None,
) -> dict:
    doc = {
        "text": text,
        "meta": {
            "chunk_id": chunk_id,
            "source": "paper.pdf",
            "page": 1,
            "page_start": 1,
            "page_end": 1,
        },
    }
    if retrieval is not None:
        doc["retrieval"] = retrieval
    return doc


def _retrieval(doc: dict) -> dict:
    return doc["retrieval"]


def test_short_text_is_deep_copied_without_compression():
    original = _doc(
        "Short retrieval fact.",
        retrieval={"matched_requirement_ids": ["r1"], "nested": {"rank": 1}},
    )
    before = copy.deepcopy(original)

    batch = select_evidence_spans(
        [original],
        query="retrieval",
        evidence_requirements=[{"requirement_id": "r1", "question": "retrieval"}],
        max_chars_per_doc=100,
    )

    selected = batch.docs[0]
    assert selected["text"] == original["text"]
    assert selected is not original
    assert selected["meta"] is not original["meta"]
    assert selected["retrieval"] is not original["retrieval"]
    assert _retrieval(selected)["evidence_span_reason"] == REASON_WITHIN_BUDGET
    assert _retrieval(selected)["evidence_span_selected"] is False
    assert batch.input_count == batch.output_count == 1
    assert batch.compressed_count == batch.fallback_count == 0
    assert batch.input_chars == batch.selected_chars == len(original["text"])
    assert batch.reason_counts == {REASON_WITHIN_BUDGET: 1}
    selected["meta"]["source"] = "changed.pdf"
    selected["retrieval"]["nested"]["rank"] = 2
    assert original == before


def test_context_fact_is_removed_from_model_visible_span_snapshot():
    original = _doc("The selected retrieval fact.")
    original["meta"].update(
        {
            "context": "PRIVATE FACT OUTSIDE THE VERIFIED BODY",
            "section_path": "Methods > Retrieval",
        }
    )

    selected = select_evidence_span(
        original,
        query="retrieval",
        max_chars_per_doc=100,
    )
    rendered = render_evidence_block(selected)

    assert "context" not in selected["meta"]
    assert "PRIVATE FACT" not in rendered
    assert "Methods > Retrieval" in rendered
    assert original["meta"]["context"] == "PRIVATE FACT OUTSIDE THE VERIFIED BODY"


def test_english_selection_preserves_exact_sentence_and_offsets():
    text = "Intro sentence. The retrieval system uses hybrid search. Closing note."
    expected = "The retrieval system uses hybrid search."
    expected_start = text.index(expected)

    selected = select_evidence_span(
        _doc(text),
        query="hybrid retrieval",
        max_chars_per_doc=len(expected),
        context_sentences=0,
    )

    assert selected["text"] == expected
    assert _retrieval(selected)["evidence_span_start"] == expected_start
    assert _retrieval(selected)["evidence_span_end"] == expected_start + len(expected)
    assert _retrieval(selected)["evidence_span_input_start"] == 0
    assert _retrieval(selected)["evidence_span_input_end"] == len(text)
    assert _retrieval(selected)["evidence_text_start"] == expected_start
    assert _retrieval(selected)["evidence_text_end"] == expected_start + len(expected)
    assert _retrieval(selected)["evidence_trimmed_overlap_chars"] == 0
    assert _retrieval(selected)["evidence_span_reason"] == REASON_QUERY_SPAN
    assert "retriev" in _retrieval(selected)["evidence_span_matched_terms"]


def test_chinese_requirement_selects_verbatim_sentence():
    text = "背景信息没有答案。系统采用向量检索提升召回率。最后讨论部署成本。"
    expected = "系统采用向量检索提升召回率。"

    selected = select_evidence_span(
        _doc(text, retrieval={"matched_requirement_ids": ["recall"]}),
        query="系统质量",
        evidence_requirements=[
            {"requirement_id": "recall", "question": "向量检索的召回率"}
        ],
        max_chars_per_doc=len(expected),
        context_sentences=0,
    )

    assert selected["text"] == expected
    assert _retrieval(selected)["matched_requirement_ids"] == ["recall"]
    assert _retrieval(selected)["evidence_span_matched_requirement_ids"] == ["recall"]


def test_common_requirement_words_do_not_claim_distant_single_letter_entities():
    first = "A 的日期是 8 月 1 日。"
    second = "B 的日期是 9 月 2 日。"
    text = f"{first}{'背景说明。' * 30}{second}"

    selected = select_evidence_span(
        _doc(text, retrieval={"matched_requirement_ids": ["r1", "r2"]}),
        query="A 和 B 的日期分别是什么？",
        evidence_requirements=[
            {"requirement_id": "r1", "question": "A 的日期"},
            {"requirement_id": "r2", "question": "B 的日期"},
        ],
        max_chars_per_doc=len(first),
        context_sentences=0,
    )

    assert selected["text"] == first
    assert _retrieval(selected)["matched_requirement_ids"] == ["r1"]
    assert _retrieval(selected)["evidence_span_matched_requirement_ids"] == ["r1"]


def test_identical_requirement_terms_are_not_used_as_false_attribution():
    selected = select_evidence_span(
        _doc(
            f"{'背景。' * 30}日期是 8 月 1 日。",
            retrieval={"matched_requirement_ids": ["r1", "r2"]},
        ),
        query="日期是什么？",
        evidence_requirements=[
            {"requirement_id": "r1", "question": "日期是什么？"},
            {"requirement_id": "r2", "question": "日期是什么？"},
        ],
        max_chars_per_doc=20,
        context_sentences=0,
    )

    assert _retrieval(selected)["matched_requirement_ids"] == []


def test_numeric_identifier_is_a_stable_selection_term():
    text = "Old release was 2024-01-01. Current release is 2026-08-04. Appendix."
    expected = "Current release is 2026-08-04."

    selected = select_evidence_span(
        _doc(text),
        query="2026-08-04",
        max_chars_per_doc=len(expected),
        context_sentences=0,
    )

    assert selected["text"] == expected
    assert "2026-08-04" in _retrieval(selected)["evidence_span_matched_terms"]


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("", REASON_FALLBACK_NO_TERMS),
        ("unrelated-token", REASON_FALLBACK_NO_MATCH),
        ("art", REASON_FALLBACK_NO_MATCH),
    ],
)
def test_long_text_without_reliable_match_fails_open(query, reason):
    text = "This partial document discusses retrieval only. " * 8

    batch = select_evidence_spans(
        [_doc(text)], query=query, max_chars_per_doc=30, context_sentences=0
    )

    assert batch.docs[0]["text"] == text
    assert _retrieval(batch.docs[0])["evidence_span_reason"] == reason
    assert _retrieval(batch.docs[0])["evidence_span_selected"] is False
    assert batch.compressed_count == 0
    assert batch.fallback_count == 1
    assert batch.selected_chars == len(text)


def test_context_sentences_are_kept_on_both_sides_when_budget_allows():
    text = "Outside first. Previous context. Retrieval target. Following context. Outside last."
    expected = "Previous context. Retrieval target. Following context."

    selected = select_evidence_span(
        _doc(text),
        query="retrieval",
        max_chars_per_doc=len(expected),
        context_sentences=1,
    )

    assert selected["text"] == expected


def test_single_long_sentence_uses_exact_window_around_matching_word():
    text = f"{'a' * 80} retrieval {'b' * 80}"

    selected = select_evidence_span(
        _doc(text),
        query="retrieval",
        max_chars_per_doc=40,
        context_sentences=1,
    )

    retrieval = _retrieval(selected)
    assert len(selected["text"]) == 40
    assert "retrieval" in selected["text"]
    assert (
        selected["text"]
        == text[retrieval["evidence_span_start"] : retrieval["evidence_span_end"]]
    )
    assert retrieval["evidence_span_reason"] == REASON_LONG_SENTENCE_WINDOW


def test_selection_is_deterministic_for_equal_score_and_duplicate_identity():
    text = "Retrieval first. Padding padding. Retrieval second."
    docs = [
        _doc(text, chunk_id="same"),
        _doc(text, chunk_id="same"),
        _doc(text, chunk_id=""),
    ]
    selector = EvidenceSpanSelector(
        query="retrieval",
        max_chars_per_doc=len("Retrieval first."),
        context_sentences=0,
    )

    first = selector.select_many(docs)
    second = selector.select_many(docs)

    assert len(first.docs) == 3
    assert [doc["text"] for doc in first.docs] == ["Retrieval first."] * 3
    assert first == second


def test_existing_visible_range_composes_ultimate_source_offsets():
    text = "Noise sentence. Retrieval answer. Tail."
    expected = "Retrieval answer."
    local_start = text.index(expected)
    doc = _doc(
        text,
        retrieval={
            "evidence_text_start": 100,
            "evidence_text_end": 100 + len(text),
            "evidence_trimmed_overlap_chars": 7,
        },
    )

    selected = select_evidence_span(
        doc,
        query="retrieval",
        max_chars_per_doc=len(expected),
        context_sentences=0,
    )

    retrieval = _retrieval(selected)
    assert selected["text"] == expected
    assert retrieval["evidence_span_input_start"] == 100
    assert retrieval["evidence_span_input_end"] == 100 + len(text)
    assert retrieval["evidence_span_start"] == 100 + local_start
    assert retrieval["evidence_span_end"] == 100 + local_start + len(expected)
    assert retrieval["evidence_trimmed_overlap_chars"] == 0
    assert _retrieval(doc)["evidence_trimmed_overlap_chars"] == 7


def test_mismatched_stored_end_is_repaired_with_an_observable_warning(caplog):
    text = "Retrieval answer."
    doc = _doc(
        text,
        retrieval={
            "evidence_text_start": 50,
            "evidence_text_end": 50 + len(text) + 1,
        },
    )

    with caplog.at_level("WARNING", logger="cogdoc.tools.retriever.evidence_spans"):
        selected = select_evidence_span(
            doc,
            query="retrieval",
            max_chars_per_doc=100,
        )

    assert selected["_evidence_span_source_end"] == 50 + len(text)
    assert _retrieval(selected)["evidence_span_input_end"] == 50 + len(text)
    assert "evidence_span_source_end_mismatch" in caplog.text
    assert "chunk_id=chunk-1" in caplog.text


def test_old_pack_private_source_is_promoted_but_never_exposed_to_repack():
    full_text = "Noise sentence. Retrieval answer. Tail sentence."
    expected = "Retrieval answer."
    doc = _doc("visible-only")
    doc.update(
        {
            "_evidence_source_text": full_text,
            "_evidence_source_start": 50,
            "_evidence_source_end": 50 + len(full_text),
            "_evidence_source_overlap_chars": 4,
        }
    )

    selected = select_evidence_span(
        doc,
        query="retrieval",
        max_chars_per_doc=len(expected),
        context_sentences=0,
    )

    assert selected["text"] == expected
    assert not any(key.startswith("_evidence_source_") for key in selected)
    assert selected["_evidence_span_source_text"] == full_text
    assert _retrieval(selected)["evidence_span_start"] == 50 + full_text.index(expected)
    assert doc["text"] == "visible-only"
    assert doc["_evidence_source_text"] == full_text


def test_adaptive_span_pack_round_trip_can_reselect_without_repack_bypass():
    full_text = (
        "Retrieval evidence is in this sentence. "
        "Neutral middle sentence. "
        "Budget evidence is in the final sentence."
    )
    raw = _doc(full_text)
    first_selector = EvidenceSpanSelector(
        query="retrieval", max_chars_per_doc=45, context_sentences=0
    )
    first_span = first_selector.select(raw)
    first_pack = build_evidence_pack(
        [EvidencePackCandidate(first_span)], max_docs=1, max_chars=45
    )
    assert first_pack.kept_docs[0]["text"] == "Retrieval evidence is in this sentence."

    second_selector = EvidenceSpanSelector(
        query="budget", max_chars_per_doc=45, context_sentences=0
    )
    second_span = second_selector.select(first_pack.kept_docs[0])
    assert second_span["text"] == "Budget evidence is in the final sentence."
    assert not any(key.startswith("_evidence_source_") for key in second_span)

    second_pack = build_evidence_pack(
        [EvidencePackCandidate(second_span)], max_docs=1, max_chars=45
    )
    assert second_pack.kept_docs[0]["text"] == second_span["text"]
    assert len(second_pack.kept_docs[0]["text"]) <= 45
    assert second_pack.kept_docs[0]["text"] != full_text


def test_compressed_span_drops_requirement_ids_supported_only_outside_span():
    date_sentence = "The filing deadline date is 2026-08-04."
    budget_sentence = "The approved budget limit is 9000 credits."
    text = f"{date_sentence} {'Neutral material. ' * 8}{budget_sentence}"
    requirements = [
        {
            "requirement_id": "r1",
            "question": "What rule gives the deadline date?",
        },
        {
            "requirement_id": "r2",
            "question": "What rule gives the budget limit?",
        },
    ]
    original = _doc(text, retrieval={"matched_requirement_ids": ["r1", "r2"]})

    selected = select_evidence_span(
        original,
        query="deadline date",
        evidence_requirements=requirements,
        matched_requirement_ids=["r1", "r2"],
        max_chars_per_doc=len(date_sentence),
        context_sentences=0,
    )

    retrieval = _retrieval(selected)
    assert selected["text"] == date_sentence
    assert retrieval["matched_requirement_ids"] == ["r1"]
    assert retrieval["evidence_span_matched_requirement_ids"] == ["r1"]
    assert _retrieval(original)["matched_requirement_ids"] == ["r1", "r2"]


def test_full_text_fallback_preserves_existing_requirement_attribution():
    text = "No lexical overlap is available here. " * 5
    original = _doc(text, retrieval={"matched_requirement_ids": ["r1", "r2"]})

    selected = select_evidence_span(
        original,
        query="missing",
        evidence_requirements=[
            {"requirement_id": "r1", "question": "alpha"},
            {"requirement_id": "r2", "question": "beta"},
        ],
        max_chars_per_doc=20,
    )

    assert selected["text"] == text
    assert _retrieval(selected)["matched_requirement_ids"] == ["r1", "r2"]
    assert _retrieval(selected)["evidence_span_matched_requirement_ids"] == [
        "r1",
        "r2",
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_chars_per_doc": 0}, "max_chars_per_doc"),
        ({"max_chars_per_doc": 10, "context_sentences": -1}, "context_sentences"),
    ],
)
def test_invalid_limits_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        select_evidence_spans([_doc("text")], query="query", **kwargs)
