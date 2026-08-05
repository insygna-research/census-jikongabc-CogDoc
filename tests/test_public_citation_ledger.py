import pytest

from cogdoc.tools.public_citation_ledger import validate_public_citation_ledger


def _entry(answer: str, *, evidence_id: str = "E001", source: str = "a.pdf") -> dict:
    citation = f"[{source}:P1]"
    start = answer.index(citation)
    return {
        "evidence_id": evidence_id,
        "chunk_id": "c1",
        "source_type": "document",
        "source": source,
        "page": 1,
        "page_start": 1,
        "page_end": 1,
        "span_start": 4,
        "span_end": 20,
        "occurrences": [
            {
                "index": 0,
                "answer_start": start,
                "answer_end": start + len(citation),
            }
        ],
    }


def _evidence(evidence_id: str = "E001") -> list[dict]:
    return [
        {
            "evidence_id": evidence_id,
            "chunk_id": "c1",
            "source_type": "document",
            "source": "a.pdf",
            "page": 1,
            "page_start": 1,
            "page_end": 1,
            "retrieval": {
                "evidence_id": evidence_id,
                "evidence_text_start": 4,
                "evidence_text_end": 20,
            },
        }
    ]


def test_public_ledger_validates_full_table_and_unicode_offsets():
    answer = "😀结论[a.pdf:P1]。"

    result = validate_public_citation_ledger(
        answer,
        [_entry(answer)],
        evidence=_evidence(),
        require_evidence=True,
    )

    assert result.is_valid is True
    assert result.occurrence_mapped == result.occurrence_total == 1
    # Python/API contract uses Unicode code points: the emoji occupies one offset.
    assert result.entries[0]["occurrences"][0]["answer_start"] == 3


def test_public_ledger_accepts_canonical_four_digit_evidence_id():
    answer = "结论[a.pdf:P1]。"

    result = validate_public_citation_ledger(
        answer,
        [_entry(answer, evidence_id="E1000")],
        evidence=_evidence("E1000"),
        require_evidence=True,
    )

    assert result.is_valid is True


@pytest.mark.parametrize("evidence_id", ["E000", "E01000", "e001", "E01"])
def test_public_ledger_rejects_out_of_contract_evidence_ids(evidence_id):
    answer = "结论[a.pdf:P1]。"
    entry = _entry(answer, evidence_id=evidence_id)

    result = validate_public_citation_ledger(answer, [entry])

    assert result.is_valid is False
    assert result.reason == "invalid_or_duplicate_evidence_id"


def test_public_ledger_rejects_missing_visible_citation_and_duplicate_index():
    answer = "甲[a.pdf:P1]，乙[a.pdf:P1]。"
    entry = _entry(answer)
    entry["occurrences"].append(dict(entry["occurrences"][0]))

    result = validate_public_citation_ledger(answer, [entry])

    assert result.is_valid is False
    assert result.occurrence_total == 2
    assert result.reason in {
        "occurrence_indices_not_unique_contiguous",
        "occurrences_overlap",
    }


def test_public_ledger_rejects_bad_span_and_evidence_binding():
    answer = "结论[a.pdf:P1]。"
    entry = _entry(answer)
    entry["span_end"] = entry["span_start"]

    result = validate_public_citation_ledger(
        answer,
        [entry],
        evidence=_evidence(),
        require_evidence=True,
    )

    assert result.is_valid is False
    assert result.reason == "invalid_span"


def test_public_ledger_rejects_wrong_answer_slice_as_one_table():
    answer = "结论[a.pdf:P1]。"
    entry = _entry(answer)
    entry["occurrences"][0]["answer_start"] += 1

    result = validate_public_citation_ledger(answer, [entry])

    assert result.is_valid is False
    assert result.occurrence_mapped == 0
    assert result.occurrence_total == 1


@pytest.mark.parametrize(
    "internal",
    [
        "[E1000]",
        "[E-002]",
        "[E001,E002]",
        "[Evidence ID 002]",
        "[EvidenceID002]",
        "[E-I-D-002]",
        "[E001：P1]",
        "［Ｅ００１：Ｐ１］",
        "[E001:P1",
        "[[E001]]",
        "[prefix [E001]]",
    ],
)
def test_public_ledger_rejects_internal_eid_variants(internal):
    citation = "[a.pdf:P1]"
    answer = f"结论{citation}{internal}"
    entry = _entry(answer)

    result = validate_public_citation_ledger(answer, [entry])

    assert result.is_valid is False
    assert result.reason == "internal_evidence_reference_exposed"


def test_public_ledger_does_not_misclassify_e_digit_filename():
    answer = "结论[e1.pdf:P1]"
    entry = _entry(answer, source="e1.pdf")

    result = validate_public_citation_ledger(answer, [entry])

    assert result.is_valid is True


def test_public_ledger_exempts_exact_closed_eid_shaped_filename_citation():
    answer = "结论[E001:P1]"
    entry = _entry(answer, source="E001")

    result = validate_public_citation_ledger(answer, [entry])

    assert result.is_valid is True


@pytest.mark.parametrize(
    "natural_bracket",
    [
        "[Evidence 2024]",
        "[Type E2]",
        "[Type E200]",
        "[E=2]",
        "[Example 1]",
    ],
)
def test_public_ledger_does_not_misclassify_natural_bracket_text(natural_bracket):
    answer = f"结论[a.pdf:P1]，分类为 {natural_bracket}。"

    result = validate_public_citation_ledger(
        answer,
        [_entry(answer)],
        evidence=_evidence(),
        require_evidence=True,
    )

    assert result.is_valid is True
