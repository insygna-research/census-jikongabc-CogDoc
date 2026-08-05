import pytest

from cogdoc.tools.eval.retrieval_eval_drafts import (
    DatasetPartition,
    DraftStatus,
    EvidenceUnitDraft,
    EvidenceUnitTask,
    RetrievalIndexSnapshot,
    apply_review_annotations,
    approval_errors,
    approve_draft,
    build_retrieval_eval_draft,
    create_pending_draft,
    detect_stale_reasons,
    export_retrieval_eval_case,
)


PROVENANCE = {
    "index_generation": "generation-7",
    "index_build_version": "hybrid-v2",
    "chunk_identity_version": "chunk-v5",
    "source_versions": [{"source": "a.pdf", "sha256": "sha-a"}],
}


def _qa_trace():
    return {
        "trace_id": "trace-1",
        "task_type": "qa",
        "input": {"doc_id": "kb-1", "query": "报名条件是什么？"},
        "config": PROVENANCE,
        "output": {
            "evidence_requirements": [
                {
                    "requirement_id": "r1",
                    "question": "年龄条件是什么？",
                    "retrieval_query": "报名 年龄 条件",
                    "recovery_query": "参赛者 年龄限制",
                }
            ],
            # Runtime observations must never become reviewer-authored labels.
            "retrieved_docs": [
                {
                    "meta": {
                        "chunk_id": "observed",
                        "source": "a.pdf",
                        "source_sha256": "sha-a",
                    }
                }
            ],
        },
    }


def test_builder_creates_stable_unlabelled_qa_proposal():
    first = build_retrieval_eval_draft(
        {"feedback_id": "feedback-1", "kb_id": "kb-1"},
        _qa_trace(),
        now="2026-01-01T00:00:00+00:00",
    )
    second_trace = _qa_trace()
    second_trace["input"]["query"] = "  报名条件是什么？  "
    second = build_retrieval_eval_draft(
        {"feedback_id": "feedback-2", "kb_id": "kb-1"},
        second_trace,
        now="2026-01-02T00:00:00+00:00",
    )

    assert first.status is DraftStatus.PENDING
    assert first.draft_id == second.draft_id
    assert first.units[0].task_kind is EvidenceUnitTask.QA_REQUIREMENT
    assert first.units[0].expected_status is None
    assert first.units[0].acceptable_evidence == []
    assert first.units[0].hard_negative_chunks == []
    assert first.hard_negative_chunks == []
    assert first.index_generation == "generation-7"


def test_builder_dedupes_per_index_snapshot_not_forever():
    baseline = build_retrieval_eval_draft({}, _qa_trace())
    same_snapshot = build_retrieval_eval_draft({}, _qa_trace())
    generation_trace = _qa_trace()
    generation_trace["config"] = {
        **PROVENANCE,
        "index_generation": "generation-8",
    }
    identity_trace = _qa_trace()
    identity_trace["config"] = {
        **PROVENANCE,
        "chunk_identity_version": "chunk-v6",
    }
    source_trace = _qa_trace()
    source_trace["config"] = {
        **PROVENANCE,
        "source_versions": [{"source": "a.pdf", "sha256": "sha-a-new"}],
    }
    generation = build_retrieval_eval_draft({}, generation_trace)
    identity = build_retrieval_eval_draft({}, identity_trace)
    source = build_retrieval_eval_draft({}, source_trace)

    assert baseline.draft_id == same_snapshot.draft_id
    assert (
        len(
            {baseline.draft_id, generation.draft_id, identity.draft_id, source.draft_id}
        )
        == 4
    )


@pytest.mark.parametrize(
    ("task_type", "output", "expected_kinds", "expected_ids"),
    [
        (
            "summary",
            {
                "summary_source": "a.pdf",
                "summary_section_plans": [
                    {
                        "section_id": "method",
                        "title": "方法",
                        "instruction": "提炼实施流程",
                    }
                ],
            },
            [EvidenceUnitTask.SUMMARY_SECTION],
            ["method"],
        ),
        (
            "compare",
            {
                "compare_sources": ["a.pdf", "b.pdf"],
                "compare_dimensions": [
                    {
                        "dimension_id": "limits",
                        "title": "限制",
                        "instruction": "提炼限制",
                    }
                ],
            },
            [
                EvidenceUnitTask.COMPARE_SOURCE_DIMENSION,
                EvidenceUnitTask.COMPARE_SOURCE_DIMENSION,
            ],
            ["a.pdf::limits", "b.pdf::limits"],
        ),
    ],
)
def test_builder_covers_summary_and_compare_units(
    task_type, output, expected_kinds, expected_ids
):
    draft = build_retrieval_eval_draft(
        {},
        {
            "task_type": task_type,
            "input": {"doc_id": "kb-1", "query": "请处理文档"},
            "output": output,
            "config": PROVENANCE,
        },
    )

    assert [unit.task_kind for unit in draft.units] == expected_kinds
    assert [unit.unit_id for unit in draft.units] == expected_ids
    assert all(not unit.acceptable_evidence for unit in draft.units)


def test_review_annotations_are_required_before_approval_and_export():
    pending = build_retrieval_eval_draft({}, _qa_trace())
    with pytest.raises(ValueError, match="acceptable_evidence_required"):
        approve_draft(pending, reviewer="reviewer")
    with pytest.raises(ValueError, match="review units must be a list"):
        apply_review_annotations(pending, {"units": {"unit_id": "r1"}})

    annotated = apply_review_annotations(
        pending,
        {
            "units": [
                {
                    "unit_id": "r1",
                    "acceptable_evidence": [
                        {
                            "chunk_id": "chunk-1",
                            "parent_chunk_id": "parent-1",
                            "source": "a.pdf",
                            "source_sha256": "sha-a",
                            "start": 10,
                            "end": 30,
                        }
                    ],
                }
            ],
            "hard_negative_chunks": [
                {
                    "chunk_id": "chunk-bad",
                    "source": "a.pdf",
                    "source_sha256": "sha-a",
                }
            ],
        },
    )
    approved = approve_draft(
        annotated, reviewer="reviewer", now="2026-01-03T00:00:00+00:00"
    )
    exported = export_retrieval_eval_case(approved)

    assert approved.status is DraftStatus.APPROVED
    assert approved.no_answer is False
    assert approved.units[0].expected_status == "supported"
    assert [item.chunk_id for item in approved.units[0].hard_negative_chunks] == [
        "chunk-bad"
    ]
    assert exported["expected_sources"] == ["a.pdf"]
    assert exported["expected_unit_statuses"] == {"r1": "supported"}
    assert exported["gold_requirements"][0]["acceptable_chunk_ids"] == ["chunk-1"]
    assert exported["gold_requirements"][0]["acceptable_spans"] == [
        {"chunk_id": "chunk-1", "start": 10, "end": 30}
    ]
    assert exported["hard_negative_chunk_ids"] == ["chunk-bad"]
    assert exported["hard_negative_chunk_ids_by_unit"] == {"r1": ["chunk-bad"]}
    assert exported["annotation_provenance"]["reviewed_by"] == "reviewer"


def test_reviewer_can_complete_provenance_without_changing_draft_identity():
    pending = create_pending_draft(
        kb_id="kb-1",
        query="问题",
        units=[
            EvidenceUnitDraft(
                unit_id="r1",
                task_kind=EvidenceUnitTask.QA_REQUIREMENT,
                label="问题",
                retrieval_query="问题",
                recovery_query="替代问题",
            )
        ],
    )
    annotated = apply_review_annotations(
        pending,
        {
            **PROVENANCE,
            "units": [
                {
                    "unit_id": "r1",
                    "acceptable_evidence": [
                        {
                            "chunk_id": "chunk-1",
                            "source": "a.pdf",
                            "source_sha256": "sha-a",
                        }
                    ],
                }
            ],
        },
    )

    assert annotated.draft_id == pending.draft_id
    assert annotated.dedupe_key == pending.dedupe_key
    assert annotated.identity_snapshot == pending.identity_snapshot
    assert approve_draft(annotated, reviewer="reviewer").status is DraftStatus.APPROVED


def test_no_answer_export_has_no_gold_and_partition_changes_identity():
    unit = EvidenceUnitDraft(
        unit_id="r1",
        task_kind=EvidenceUnitTask.QA_REQUIREMENT,
        label="未收录问题",
        retrieval_query="未收录问题",
        recovery_query="替代表达",
    )
    common = {
        "kb_id": "kb-1",
        "query": "未收录问题",
        "units": [unit],
        "no_answer": True,
        **PROVENANCE,
    }
    training = create_pending_draft(**common)
    release = create_pending_draft(
        **common, dataset_partition=DatasetPartition.RELEASE_GATE
    )
    approved = approve_draft(training, reviewer="reviewer")

    assert training.draft_id != release.draft_id
    assert approved.no_answer is True
    assert approved.units[0].expected_status == "no_evidence"
    exported = export_retrieval_eval_case(approved)
    assert exported["expected_sources"] == []
    assert exported["gold_requirements"] == []
    assert exported["expected_unit_statuses"] == {"r1": "no_evidence"}
    assert exported["layer"] == "no-answer"


@pytest.mark.parametrize(
    "task_kind",
    [
        EvidenceUnitTask.SUMMARY_SECTION,
        EvidenceUnitTask.COMPARE_SOURCE_DIMENSION,
    ],
)
def test_mixed_unit_statuses_keep_hard_negatives_local(task_kind):
    pending = create_pending_draft(
        kb_id="kb-1",
        query="提炼方法和限制",
        no_answer=True,
        units=[
            EvidenceUnitDraft(
                unit_id="method",
                task_kind=task_kind,
                label="方法",
                retrieval_query="方法",
                recovery_query="实施流程",
                source="a.pdf",
                dimension_id="method",
            ),
            EvidenceUnitDraft(
                unit_id="limits",
                task_kind=task_kind,
                label="限制",
                retrieval_query="限制",
                recovery_query="风险边界",
                source="a.pdf",
                dimension_id="limits",
            ),
        ],
        **PROVENANCE,
    )
    annotated = apply_review_annotations(
        pending,
        {
            "units": [
                {
                    "unit_id": "method",
                    "expected_status": "supported",
                    "acceptable_evidence": [
                        {
                            "chunk_id": "shared-chunk",
                            "source": "a.pdf",
                            "source_sha256": "sha-a",
                        }
                    ],
                },
                {
                    "unit_id": "limits",
                    "expected_status": "no_evidence",
                    "hard_negative_chunks": [
                        {
                            "chunk_id": "shared-chunk",
                            "source": "a.pdf",
                            "source_sha256": "sha-a",
                        }
                    ],
                },
            ]
        },
    )

    assert approval_errors(annotated) == []
    approved = approve_draft(annotated, reviewer="reviewer")
    exported = export_retrieval_eval_case(approved)

    assert approved.no_answer is False
    assert [unit.expected_status for unit in approved.units] == [
        "supported",
        "no_evidence",
    ]
    assert exported["expected_unit_statuses"] == {
        "method": "supported",
        "limits": "no_evidence",
    }
    assert [row["requirement_id"] for row in exported["gold_requirements"]] == [
        "method"
    ]
    assert exported["hard_negative_chunk_ids_by_unit"] == {
        "method": [],
        "limits": ["shared-chunk"],
    }
    # A negative for one unit may be positive for another; the legacy global
    # field must not flatten that relationship into a contradiction.
    assert exported["hard_negative_chunk_ids"] == []


def test_unit_local_positive_negative_overlap_is_rejected():
    pending = build_retrieval_eval_draft({}, _qa_trace())
    annotated = apply_review_annotations(
        pending,
        {
            "units": [
                {
                    "unit_id": "r1",
                    "expected_status": "supported",
                    "acceptable_evidence": [
                        {
                            "chunk_id": "same",
                            "source": "a.pdf",
                            "source_sha256": "sha-a",
                        }
                    ],
                    "hard_negative_chunks": [
                        {
                            "chunk_id": "same",
                            "source": "a.pdf",
                            "source_sha256": "sha-a",
                        }
                    ],
                }
            ]
        },
    )

    assert "acceptable_hard_negative_overlap:r1:same" in approval_errors(annotated)


def test_unit_local_negative_is_not_flattened_into_legacy_global_field():
    pending = create_pending_draft(
        kb_id="kb-1",
        query="总结",
        units=[
            EvidenceUnitDraft(
                unit_id="method",
                task_kind=EvidenceUnitTask.SUMMARY_SECTION,
                label="方法",
                retrieval_query="方法",
                recovery_query="实施流程",
                dimension_id="method",
            ),
            EvidenceUnitDraft(
                unit_id="limits",
                task_kind=EvidenceUnitTask.SUMMARY_SECTION,
                label="限制",
                retrieval_query="限制",
                recovery_query="注意事项",
                dimension_id="limits",
            ),
        ],
        **PROVENANCE,
    )
    annotated = apply_review_annotations(
        pending,
        {
            "units": [
                {
                    "unit_id": "method",
                    "expected_status": "no_evidence",
                    "hard_negative_chunks": [
                        {
                            "chunk_id": "method-only-negative",
                            "source": "a.pdf",
                            "source_sha256": "sha-a",
                        }
                    ],
                },
                {"unit_id": "limits", "expected_status": "no_evidence"},
            ]
        },
    )

    exported = export_retrieval_eval_case(
        approve_draft(annotated, reviewer="reviewer")
    )

    assert exported["hard_negative_chunk_ids_by_unit"] == {
        "method": ["method-only-negative"],
        "limits": [],
    }
    assert exported["hard_negative_chunk_ids"] == []


def test_approval_rejects_source_mismatch_and_hard_negative_overlap():
    pending = build_retrieval_eval_draft({}, _qa_trace())
    annotated = apply_review_annotations(
        pending,
        {
            "units": [
                {
                    "unit_id": "r1",
                    "acceptable_evidence": [
                        {
                            "chunk_id": "same",
                            "source": "a.pdf",
                            "source_sha256": "wrong-sha",
                        }
                    ],
                }
            ],
            "hard_negative_chunks": [
                {
                    "chunk_id": "same",
                    "source": "a.pdf",
                    "source_sha256": "sha-a",
                }
            ],
        },
    )

    errors = approval_errors(annotated)
    assert "source_version_mismatch:r1:a.pdf" in errors
    assert "acceptable_hard_negative_overlap:same" in errors


def test_stale_detection_covers_generation_identity_and_source_changes():
    draft = build_retrieval_eval_draft({}, _qa_trace())
    current = RetrievalIndexSnapshot(
        kb_id="kb-1",
        index_generation="generation-8",
        index_build_version="hybrid-v2",
        chunk_identity_version="chunk-v6",
        source_versions=[
            {"source": "a.pdf", "sha256": "sha-a-new"},
            {"source": "b.pdf", "sha256": "sha-b"},
        ],
    )

    assert detect_stale_reasons(draft, current) == (
        "index_generation_changed",
        "chunk_identity_version_changed",
        "source_added:b.pdf",
        "source_sha256_changed:a.pdf",
    )
