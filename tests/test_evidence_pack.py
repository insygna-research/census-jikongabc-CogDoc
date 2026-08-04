from dataclasses import FrozenInstanceError

import pytest

from cogdoc.tools.retriever.evidence_pack import (
    DROP_DUPLICATE_CHUNK_ID,
    DROP_MAX_CHARS,
    DROP_MAX_DOCS,
    EvidencePackBudgetExceeded,
    EvidencePackCandidate,
    build_evidence_pack,
    build_evidence_pack_from_sources,
    evidence_candidates_from_sources,
)
from cogdoc.tools.evidence_rendering import evidence_block_char_count


def _doc(
    chunk_id: str,
    text: str,
    *,
    source: str = "paper.pdf",
    parent: str = "parent-1",
    child_order: int | None = None,
    context_anchor_chunk_id: str = "",
    context: str = "",
    section_path: str = "",
) -> dict:
    meta = {
        "chunk_id": chunk_id,
        "source": source,
        "page": 1,
        "page_start": 1,
        "page_end": 1,
        "parent_chunk_id": parent,
    }
    if child_order is not None:
        meta["child_index_in_parent"] = child_order
    if context:
        meta["context"] = context
    if section_path:
        meta["section_path"] = section_path
    doc = {"text": text, "meta": meta}
    if context_anchor_chunk_id:
        doc["retrieval"] = {
            "context_anchor_chunk_id": context_anchor_chunk_id,
        }
    return doc


def _ids(pack) -> list[str]:
    return [item.ref.chunk_id for item in pack.kept]


def test_hard_anchor_and_pinned_docs_survive_both_global_budgets():
    pack = build_evidence_pack(
        [
            EvidencePackCandidate(
                _doc("anchor", "a" * 20),
                priority=0,
                provenance="anchor",
                hard_required=True,
                anchor_rank=0,
            ),
            EvidencePackCandidate(
                _doc("pinned", "p" * 10),
                priority=10,
                provenance="verified",
                hard_required=True,
            ),
            EvidencePackCandidate(_doc("normal", "n"), priority=20),
        ],
        max_docs=1,
        max_chars=5,
    )

    assert _ids(pack) == ["anchor", "pinned"]
    assert pack.estimated_chars == 30
    assert pack.input_count == 3
    assert pack.input_estimated_chars == 31
    assert pack.overlap_removed_chars == 0
    assert pack.over_budget_hard_constraints is True
    with pytest.raises(EvidencePackBudgetExceeded):
        pack.require_within_budget()
    assert [(item.ref.chunk_id, item.reason) for item in pack.dropped] == [
        ("normal", DROP_MAX_DOCS)
    ]


def test_normal_selection_is_stable_by_anchor_rank_distance_then_input_order():
    pack = build_evidence_pack(
        [
            EvidencePackCandidate(_doc("rank-1", "1"), anchor_rank=1, distance=0),
            EvidencePackCandidate(_doc("far", "f"), anchor_rank=0, distance=2),
            EvidencePackCandidate(_doc("near-first", "a"), anchor_rank=0, distance=1),
            EvidencePackCandidate(_doc("near-second", "b"), anchor_rank=0, distance=1),
        ],
        max_docs=4,
        max_chars=100,
    )

    assert _ids(pack) == ["near-first", "near-second", "far", "rank-1"]


def test_char_budget_can_skip_large_candidate_and_keep_later_small_candidate():
    pack = build_evidence_pack(
        [
            EvidencePackCandidate(_doc("first", "aaaa")),
            EvidencePackCandidate(_doc("too-large", "bbbbbb")),
            EvidencePackCandidate(_doc("fits", "cc")),
        ],
        max_docs=3,
        max_chars=6,
    )

    assert _ids(pack) == ["first", "fits"]
    assert pack.estimated_chars == 6
    assert [(item.ref.chunk_id, item.reason) for item in pack.dropped] == [
        ("too-large", DROP_MAX_CHARS)
    ]


def test_doc_budget_reports_every_later_candidate():
    pack = build_evidence_pack(
        [
            EvidencePackCandidate(_doc("first", "a")),
            EvidencePackCandidate(_doc("second", "b")),
            EvidencePackCandidate(_doc("third", "c")),
        ],
        max_docs=1,
        max_chars=100,
    )

    assert _ids(pack) == ["first"]
    assert [(item.ref.chunk_id, item.reason) for item in pack.dropped] == [
        ("second", DROP_MAX_DOCS),
        ("third", DROP_MAX_DOCS),
    ]


def test_chunk_id_dedup_merges_provenance_and_uses_strongest_snapshot():
    expanded = _doc("same", "expanded")
    anchor = _doc("same", "anchor")
    pack = build_evidence_pack(
        [
            EvidencePackCandidate(
                expanded, priority=20, provenance="expanded", anchor_rank=0
            ),
            EvidencePackCandidate(
                anchor,
                priority=0,
                provenance="anchor",
                hard_required=True,
                anchor_rank=0,
            ),
        ],
        max_docs=1,
        max_chars=100,
    )

    assert _ids(pack) == ["same"]
    assert pack.kept[0].doc["text"] == "anchor"
    assert pack.kept[0].provenance == ("expanded", "anchor")
    assert pack.kept[0].hard_required is True
    assert [(item.ref.input_order, item.reason) for item in pack.dropped] == [
        (0, DROP_DUPLICATE_CHUNK_ID)
    ]


def test_missing_chunk_ids_are_never_deduplicated_by_content_or_metadata():
    first = _doc("", "identical")
    second = _doc("", "identical")

    pack = build_evidence_pack(
        [EvidencePackCandidate(first), EvidencePackCandidate(second)],
        max_docs=2,
        max_chars=100,
    )

    assert len(pack.kept) == 2
    assert not pack.dropped
    assert [item.ref.input_order for item in pack.kept] == [0, 1]


def test_adjacent_parent_children_use_exact_overlap_only_for_display_budget():
    overlap = "0123456789abcdef"
    first = _doc("c0", f"left-{overlap}", child_order=0)
    second = _doc("c1", f"{overlap}-right", child_order=1)

    pack = build_evidence_pack(
        [EvidencePackCandidate(first), EvidencePackCandidate(second)],
        max_docs=2,
        max_chars=27,
    )

    assert _ids(pack) == ["c0", "c1"]
    assert [item.display_text for item in pack.kept] == [
        f"left-{overlap}",
        "-right",
    ]
    assert pack.estimated_chars == 27
    assert pack.input_estimated_chars == 27
    assert pack.overlap_removed_chars == len(overlap)
    assert first["text"] == f"left-{overlap}"
    assert second["text"] == f"{overlap}-right"
    assert pack.kept[1].doc["text"] == "-right"
    assert pack.kept[1].evidence_text_start == len(overlap)
    assert pack.kept[1].evidence_text_end == len(second["text"])
    assert pack.kept[1].doc["retrieval"] == {
        "evidence_text_start": len(overlap),
        "evidence_text_end": len(second["text"]),
        "evidence_trimmed_overlap_chars": len(overlap),
    }
    assert pack.kept_docs[1]["text"] == "-right"


def test_repacking_adaptive_carryover_composes_original_text_offsets():
    overlap = "0123456789abcdef"
    predecessor = _doc("c0", f"left-{overlap}", child_order=0)
    carryover = _doc("c1", f"{overlap}-right", child_order=1)
    carryover["retrieval"] = {
        "search_channel": "adaptive_carryover",
        "evidence_text_start": 10,
        "evidence_text_end": 32,
        "evidence_trimmed_overlap_chars": 10,
    }

    pack = build_evidence_pack(
        [EvidencePackCandidate(predecessor), EvidencePackCandidate(carryover)],
        max_docs=2,
        max_chars=100,
    )

    repacked = pack.kept[1]
    assert repacked.doc["text"] == "-right"
    assert repacked.evidence_text_start == 26
    assert repacked.evidence_text_end == 32
    assert repacked.evidence_trimmed_overlap_chars == 26
    assert repacked.doc["retrieval"] == {
        "search_channel": "adaptive_carryover",
        "evidence_text_start": 26,
        "evidence_text_end": 32,
        "evidence_trimmed_overlap_chars": 26,
    }
    assert carryover["text"] == f"{overlap}-right"
    assert carryover["retrieval"]["evidence_text_start"] == 10


def test_real_materialized_carryover_restores_available_prefix_without_predecessor():
    overlap = "0123456789abcdef"
    predecessor = _doc("c0", f"left-{overlap}", child_order=0)
    carryover = _doc("c1", f"{overlap}-right", child_order=1)
    first_round = build_evidence_pack(
        [EvidencePackCandidate(predecessor), EvidencePackCandidate(carryover)],
        max_docs=2,
        max_chars=100,
    )
    packed_carryover = next(
        item.doc for item in first_round.kept if item.ref.chunk_id == "c1"
    )
    assert packed_carryover["text"] == "-right"

    second_round = build_evidence_pack(
        [EvidencePackCandidate(packed_carryover, hard_required=True)],
        max_docs=1,
        max_chars=100,
    )

    restored = second_round.kept[0]
    assert restored.doc["text"] == f"{overlap}-right"
    assert restored.evidence_text_start == 0
    assert restored.evidence_text_end == len(carryover["text"])
    assert restored.evidence_trimmed_overlap_chars == 0
    assert carryover["text"] == f"{overlap}-right"


@pytest.mark.parametrize(
    ("second_overrides", "overlap"),
    [
        ({"parent": "parent-2"}, "0123456789abcdef"),
        ({"source": "other.pdf"}, "0123456789abcdef"),
        ({"child_order": 2}, "0123456789abcdef"),
        ({}, "short-overlap"),
    ],
)
def test_display_overlap_is_not_trimmed_without_every_safe_signal(
    second_overrides, overlap
):
    first = _doc("c0", f"left-{overlap}", child_order=0)
    options = {
        "source": "paper.pdf",
        "parent": "parent-1",
        "child_order": 1,
        **second_overrides,
    }
    second = _doc("c1", f"{overlap}-right", **options)
    raw_chars = len(first["text"]) + len(second["text"])

    pack = build_evidence_pack(
        [EvidencePackCandidate(first), EvidencePackCandidate(second)],
        max_docs=2,
        max_chars=raw_chars,
    )

    assert [item.display_text for item in pack.kept] == [
        first["text"],
        second["text"],
    ]
    assert pack.estimated_chars == raw_chars


def test_ambiguous_child_order_disables_display_overlap_for_whole_parent():
    overlap = "0123456789abcdef"
    docs = [
        _doc("a", f"left-{overlap}", child_order=0),
        _doc("b", f"{overlap}-right", child_order=1),
        _doc("duplicate-order", "other", child_order=1),
    ]

    pack = build_evidence_pack(
        [EvidencePackCandidate(doc) for doc in docs],
        max_docs=3,
        max_chars=100,
    )

    assert [item.display_text for item in pack.kept] == [doc["text"] for doc in docs]


def test_locator_context_is_included_in_global_char_estimate():
    doc = _doc(
        "contextual",
        "body",
        context="locator",
        section_path="Methods > Limits",
    )
    overhead = len("locator") + len("定位上下文：\n\n正文：\n")
    overhead += len("Methods > Limits") + len("章节路径：\n\n")

    dropped = build_evidence_pack(
        [EvidencePackCandidate(doc)],
        max_docs=1,
        max_chars=len("body") + overhead - 1,
    )
    kept = build_evidence_pack(
        [EvidencePackCandidate(doc)],
        max_docs=1,
        max_chars=len("body") + overhead,
    )

    assert [(item.ref.chunk_id, item.reason) for item in dropped.dropped] == [
        ("contextual", DROP_MAX_CHARS)
    ]
    assert kept.estimated_chars == len("body") + overhead


def test_custom_rendered_block_cost_enforces_tag_and_identity_overhead():
    doc = _doc("chunk-with-a-long-stable-identity", "body", source="long-name.pdf")
    rendered_chars = evidence_block_char_count(doc, doc["text"])

    dropped = build_evidence_pack(
        [EvidencePackCandidate(doc)],
        max_docs=1,
        max_chars=rendered_chars - 1,
        document_char_cost=evidence_block_char_count,
    )
    kept = build_evidence_pack(
        [EvidencePackCandidate(doc)],
        max_docs=1,
        max_chars=rendered_chars,
        document_char_cost=evidence_block_char_count,
    )

    assert not dropped.kept
    assert dropped.dropped[0].reason == DROP_MAX_CHARS
    assert kept.estimated_chars == rendered_chars


def test_source_adapter_marks_anchors_and_pinned_as_hard_and_derives_proximity():
    anchor = _doc("anchor", "a", child_order=2)
    near = _doc(
        "near",
        "n",
        child_order=1,
        context_anchor_chunk_id="anchor",
    )
    far = _doc(
        "far",
        "f",
        child_order=0,
        context_anchor_chunk_id="anchor",
    )
    pinned = _doc("verified", "v", source="other.pdf", child_order=4)

    candidates = evidence_candidates_from_sources(
        anchors=[anchor],
        expanded_docs=[far, near, anchor],
        verification_candidates=[pinned],
        pinned_chunk_ids={"verified"},
    )
    pack = build_evidence_pack(candidates, max_docs=3, max_chars=100)

    assert _ids(pack) == ["near", "anchor", "verified"]
    assert [item.hard_required for item in pack.kept] == [False, True, True]
    assert pack.kept[0].anchor_rank == 0
    assert pack.kept[0].distance == 1
    assert [(item.ref.chunk_id, item.reason) for item in pack.dropped] == [
        ("far", DROP_MAX_DOCS),
        ("anchor", DROP_DUPLICATE_CHUNK_ID),
    ]


def test_source_adapter_merges_stage_provenance_for_same_chunk():
    anchor = _doc("anchor", "a", child_order=1)
    shared = _doc(
        "shared",
        "s",
        child_order=0,
        context_anchor_chunk_id="anchor",
    )

    pack = build_evidence_pack_from_sources(
        anchors=[anchor],
        expanded_docs=[shared],
        verification_candidates=[shared],
        pinned_chunk_ids={"shared"},
        max_docs=1,
        max_chars=1,
    )

    assert _ids(pack) == ["shared", "anchor"]
    assert pack.kept[0].provenance == ("expanded", "verification")
    assert pack.over_budget_hard_constraints is True
    assert [(item.ref.chunk_id, item.reason) for item in pack.dropped] == [
        ("shared", DROP_DUPLICATE_CHUNK_ID)
    ]


def test_deduped_requirement_attribution_is_materialized_for_verifier_selection():
    anchor = _doc("shared", "anchor")
    attributed = _doc("shared", "candidate")
    attributed["retrieval"] = {"matched_requirement_ids": ["r2"]}

    pack = build_evidence_pack_from_sources(
        anchors=[anchor],
        verification_candidates=[attributed],
        requirement_ids=["r2"],
        max_docs=1,
        max_chars=100,
    )

    assert pack.kept[0].matched_requirement_ids == ("r2",)
    assert pack.kept_docs[0]["retrieval"]["matched_requirement_ids"] == ["r2"]


def test_document_transform_runs_once_after_identity_and_requirement_merge():
    anchor = _doc("shared", "anchor")
    attributed = _doc("shared", "candidate")
    attributed["retrieval"] = {"matched_requirement_ids": ["r2"]}
    calls = []

    def transform(doc, matched_requirement_ids):
        calls.append((doc["meta"]["chunk_id"], matched_requirement_ids))
        return {**doc, "text": "selected-span", "meta": dict(doc["meta"])}

    pack = build_evidence_pack_from_sources(
        anchors=[anchor],
        verification_candidates=[attributed],
        max_docs=1,
        max_chars=100,
        document_transform=transform,
    )

    assert calls == [("shared", ("r2",))]
    assert pack.kept_docs[0]["text"] == "selected-span"
    assert anchor["text"] == "anchor"


def test_document_transform_can_downgrade_effective_requirement_coverage():
    doc = _doc("shared", "r1 evidence then distant r2 evidence")
    doc["retrieval"] = {"matched_requirement_ids": ["r1", "r2"]}

    def transform(canonical_doc, _matched_requirement_ids):
        transformed = {
            **canonical_doc,
            "text": "r1 evidence",
            "meta": dict(canonical_doc["meta"]),
            "retrieval": dict(canonical_doc["retrieval"]),
        }
        transformed["retrieval"]["matched_requirement_ids"] = ["r1"]
        return transformed

    pack = build_evidence_pack(
        [EvidencePackCandidate(doc, matched_requirement_ids=("r1", "r2"))],
        requirement_ids=["r1", "r2"],
        max_docs=1,
        max_chars=100,
        document_transform=transform,
    )

    assert pack.kept[0].matched_requirement_ids == ("r1",)
    assert pack.kept_docs[0]["retrieval"]["matched_requirement_ids"] == ["r1"]


def test_membership_priority_and_natural_parent_presentation_are_separate():
    anchor = _doc("c1", "middle", child_order=1)
    left = _doc("c0", "left", child_order=0, context_anchor_chunk_id="c1")
    right = _doc("c2", "right", child_order=2, context_anchor_chunk_id="c1")

    pack = build_evidence_pack_from_sources(
        anchors=[anchor],
        expanded_docs=[left, anchor, right],
        max_docs=3,
        max_chars=100,
    )

    assert _ids(pack) == ["c0", "c1", "c2"]


def test_multiple_anchors_in_one_parent_follow_single_expanded_stage_order():
    docs = [
        _doc("c0", "zero", child_order=0),
        _doc("c1", "one", child_order=1),
        _doc("c2", "two", child_order=2),
        _doc("c3", "three", child_order=3),
    ]

    pack = build_evidence_pack_from_sources(
        anchors=[docs[3], docs[1]],
        expanded_docs=docs,
        max_docs=4,
        max_chars=100,
    )

    assert _ids(pack) == ["c0", "c1", "c2", "c3"]


def test_requirement_coverage_precedes_nearer_generic_context():
    anchor = _doc("anchor", "a", child_order=2)
    generic = _doc("generic", "g", child_order=1, context_anchor_chunk_id="anchor")
    requirement = _doc(
        "requirement", "r", child_order=0, context_anchor_chunk_id="anchor"
    )
    requirement["retrieval"]["matched_requirement_ids"] = ["r2"]

    pack = build_evidence_pack_from_sources(
        anchors=[anchor],
        expanded_docs=[generic, requirement],
        requirement_ids=["r1", "r2"],
        max_docs=2,
        max_chars=100,
    )

    assert set(_ids(pack)) == {"anchor", "requirement"}
    requirement_item = next(
        item for item in pack.kept if item.ref.chunk_id == "requirement"
    )
    assert requirement_item.matched_requirement_ids == ("r2",)
    assert [(item.ref.chunk_id, item.reason) for item in pack.dropped] == [
        ("generic", DROP_MAX_DOCS)
    ]


def test_tight_doc_budget_prefers_candidate_covering_more_requirements():
    pack = build_evidence_pack(
        [
            EvidencePackCandidate(
                _doc("anchor", "a"),
                priority=0,
                provenance="anchor",
                hard_required=True,
                anchor_rank=0,
            ),
            EvidencePackCandidate(_doc("single", "s"), matched_requirement_ids=("r1",)),
            EvidencePackCandidate(
                _doc("multi", "m"), matched_requirement_ids=("r1", "r2")
            ),
        ],
        requirement_ids=["r1", "r2"],
        max_docs=2,
        max_chars=100,
    )

    assert set(_ids(pack)) == {"anchor", "multi"}
    assert [(item.ref.chunk_id, item.reason) for item in pack.dropped] == [
        ("single", DROP_MAX_DOCS)
    ]


def test_tight_char_budget_prefers_candidate_covering_more_requirements():
    pack = build_evidence_pack(
        [
            EvidencePackCandidate(
                _doc("anchor", "a"),
                priority=0,
                provenance="anchor",
                hard_required=True,
                anchor_rank=0,
            ),
            EvidencePackCandidate(
                _doc("single", "s" * 4), matched_requirement_ids=("r1",)
            ),
            EvidencePackCandidate(
                _doc("multi", "m" * 4), matched_requirement_ids=("r1", "r2")
            ),
        ],
        requirement_ids=["r1", "r2"],
        max_docs=3,
        max_chars=5,
    )

    assert set(_ids(pack)) == {"anchor", "multi"}
    assert [(item.ref.chunk_id, item.reason) for item in pack.dropped] == [
        ("single", DROP_MAX_CHARS)
    ]


def test_result_is_deterministic_frozen_and_does_not_alias_input():
    source = _doc("a", "original")
    candidates = [EvidencePackCandidate(source, provenance="anchor")]

    first = build_evidence_pack(candidates, max_docs=1, max_chars=100)
    second = build_evidence_pack(candidates, max_docs=1, max_chars=100)

    assert first == second
    assert isinstance(first.kept, tuple)
    assert isinstance(first.kept_docs, tuple)
    source["text"] = "mutated later"
    assert first.kept[0].doc["text"] == "original"
    assert first.require_within_budget() is first
    with pytest.raises(FrozenInstanceError):
        first.estimated_chars = 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_docs": -1, "max_chars": 1}, "max_docs"),
        ({"max_docs": 1, "max_chars": -1}, "max_chars"),
        (
            {"max_docs": 1, "max_chars": 1, "min_overlap_chars": 0},
            "min_overlap_chars",
        ),
    ],
)
def test_invalid_global_budgets_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_evidence_pack([], **kwargs)


def test_invalid_candidate_ordering_hints_are_rejected():
    with pytest.raises(ValueError, match="anchor_rank"):
        build_evidence_pack(
            [EvidencePackCandidate(_doc("a", "a"), anchor_rank=-1)],
            max_docs=1,
            max_chars=1,
        )
