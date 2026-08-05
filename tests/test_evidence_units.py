import re

import pytest

from cogdoc.service.evidence_units import (
    EvidenceClosureStatus,
    EvidenceSourceScope,
    EvidenceSourceType,
    EvidenceTaskBinding,
    EvidenceUnit,
    EvidenceUnitBudget,
    EvidenceUnitClosure,
    EvidenceUnitKind,
    EvidenceUnitPolicy,
    EvidenceView,
    build_compare_evidence_units,
    build_qa_evidence_units,
    build_summary_evidence_units,
    validate_evidence_unit_closures,
)


UNIT_ID = re.compile(r"^eu_[0-9a-f]{24}$")
GROUP_ID = re.compile(r"^eg_[0-9a-f]{24}$")


def _summary_units():
    return build_summary_evidence_units(
        "请总结 handbook.pdf 的规则与限制",
        "handbook.pdf",
        [
            {
                "section_id": "rules",
                "title": "规则",
                "instruction": "提炼报名条件",
            },
            {
                "section_id": "limits",
                "title": "限制",
                "instruction": "提炼明确限制",
            },
        ],
    )


def _budget(**overrides):
    values = {
        "max_total_docs": 8,
        "max_total_chars": 8_000,
        "max_docs_per_unit": 4,
        "max_chars_per_unit": 4_000,
    }
    values.update(overrides)
    return EvidenceUnitBudget(**values)


def _supported(unit: EvidenceUnit, *, source: str, chars: int = 100):
    return EvidenceUnitClosure(
        unit_id=unit.unit_id,
        status=EvidenceClosureStatus.SUPPORTED,
        evidence=(
            EvidenceView(
                chunk_id=f"chunk:{source}:1",
                source=source,
                estimated_chars=chars,
            ),
        ),
        grounding_chunk_ids=(f"chunk:{source}:1",),
    )


def test_qa_builder_preserves_requirement_binding_and_uses_one_atomic_group():
    units = build_qa_evidence_units(
        "报名条件是什么？",
        [
            {
                "requirement_id": "r-age",
                "question": "年龄限制是什么？",
                "retrieval_query": "报名 年龄 限制",
                "recovery_query": "参赛者 年龄 条件",
            },
            {
                "requirement_id": "r-fee",
                "question": "费用是多少？",
                "retrieval_query": "报名 费用",
                "recovery_query": "参赛 价格",
            },
        ],
    )

    assert [unit.binding.requirement_id for unit in units] == ["r-age", "r-fee"]
    assert [unit.retrieval_query for unit in units] == ["报名 年龄 限制", "报名 费用"]
    assert all(unit.scope.allowed_sources == () for unit in units)
    assert all(unit.scope.allow_derived_knowledge for unit in units)
    assert len({unit.policy.admission_group for unit in units}) == 1
    assert all(UNIT_ID.fullmatch(unit.unit_id) for unit in units)
    assert all(GROUP_ID.fullmatch(unit.policy.admission_group) for unit in units)


def test_summary_builder_keeps_source_only_in_scope_and_binding():
    first = _summary_units()
    second = _summary_units()

    assert first == second
    assert [unit.binding.section_id for unit in first] == ["rules", "limits"]
    assert all(unit.binding.source == "handbook.pdf" for unit in first)
    assert all(unit.scope.allowed_sources == ("handbook.pdf",) for unit in first)
    assert all(not unit.scope.allow_derived_knowledge for unit in first)
    assert all("handbook.pdf" not in unit.retrieval_query.lower() for unit in first)
    assert all("handbook.pdf" not in unit.recovery_query.lower() for unit in first)
    assert all("handbook.pdf" not in unit.unit_id for unit in first)


def test_compare_builder_is_dimension_major_and_source_names_are_not_queries():
    units = build_compare_evidence_units(
        "请对比 A.PDF 与 b.pdf 的方法和限制",
        ["A.PDF", "b.pdf"],
        [
            {
                "dimension_id": "method",
                "title": "方法",
                "instruction": "概括技术路线",
            },
            {
                "dimension_id": "limits",
                "title": "限制",
                "instruction": "概括明确限制",
            },
        ],
    )

    assert [(unit.binding.dimension_id, unit.binding.source) for unit in units] == [
        ("method", "A.PDF"),
        ("method", "b.pdf"),
        ("limits", "A.PDF"),
        ("limits", "b.pdf"),
    ]
    assert units[0].policy.admission_group == units[1].policy.admission_group
    assert units[2].policy.admission_group == units[3].policy.admission_group
    assert units[0].policy.admission_group != units[2].policy.admission_group
    assert units[0].retrieval_query == units[1].retrieval_query
    assert units[2].retrieval_query == units[3].retrieval_query
    for unit in units:
        assert unit.scope.allowed_sources == (unit.binding.source,)
        assert "a.pdf" not in unit.retrieval_query.casefold()
        assert "b.pdf" not in unit.retrieval_query.casefold()


def test_custom_summary_and_compare_queries_still_strip_source_names():
    summary = build_summary_evidence_units(
        "总结 a.pdf",
        "a.pdf",
        [
            {
                "section_id": "s1",
                "title": "方法",
                "instruction": "提炼方法",
                "retrieval_query": "从 a.pdf 查找方法",
                "recovery_query": "a.pdf 技术路线",
            }
        ],
    )[0]
    compare = build_compare_evidence_units(
        "比较 a.pdf b.pdf",
        ["a.pdf", "b.pdf"],
        [
            {
                "dimension_id": "d1",
                "title": "方法",
                "instruction": "提炼方法",
                "retrieval_query": "a.pdf 与 b.pdf 的技术路线",
                "recovery_query": "b.pdf 方法",
            }
        ],
    )

    assert "a.pdf" not in summary.retrieval_query.casefold()
    assert "a.pdf" not in summary.recovery_query.casefold()
    assert all("a.pdf" not in unit.retrieval_query.casefold() for unit in compare)
    assert all("b.pdf" not in unit.retrieval_query.casefold() for unit in compare)


@pytest.mark.parametrize(
    ("builder", "match"),
    [
        (
            lambda: build_qa_evidence_units(
                "q",
                [
                    {"requirement_id": "r1", "question": "one"},
                    {"requirement_id": "r1", "question": "two"},
                ],
            ),
            "requirement_id",
        ),
        (
            lambda: build_summary_evidence_units(
                "q",
                "a.pdf",
                [
                    {"section_id": "s", "title": "one"},
                    {"section_id": "s", "title": "two"},
                ],
            ),
            "section_id",
        ),
        (
            lambda: build_compare_evidence_units(
                "q",
                ["a.pdf", "A.PDF"],
                [{"dimension_id": "d", "title": "one"}],
            ),
            "duplicates",
        ),
    ],
)
def test_builders_reject_ambiguous_duplicate_coordinates(builder, match):
    with pytest.raises(ValueError, match=match):
        builder()


def test_source_scope_requires_explicit_related_source_for_derived_knowledge():
    scope = EvidenceSourceScope(
        allowed_sources=("a.pdf",), allow_derived_knowledge=True
    )
    document_only = EvidenceSourceScope(
        allowed_sources=("a.pdf",), allow_derived_knowledge=False
    )

    assert scope.contains(source="a.pdf")
    assert not scope.contains(source="b.pdf")
    assert scope.contains(
        source="knowledge:k1",
        source_type=EvidenceSourceType.DERIVED_KNOWLEDGE,
        related_source="a.pdf",
    )
    assert not scope.contains(
        source="knowledge:k1",
        source_type=EvidenceSourceType.DERIVED_KNOWLEDGE,
        related_source="b.pdf",
    )
    assert not document_only.contains(
        source="knowledge:k1",
        source_type=EvidenceSourceType.DERIVED_KNOWLEDGE,
        related_source="a.pdf",
    )


def test_task_binding_rejects_cross_task_coordinates():
    with pytest.raises(ValueError, match="non-QA"):
        EvidenceTaskBinding(
            task_kind=EvidenceUnitKind.QA_REQUIREMENT,
            requirement_id="r1",
            source="a.pdf",
        )


def test_unit_contract_requires_opaque_ids_and_exact_source_scope():
    unit = _summary_units()[0]
    with pytest.raises(ValueError, match="opaque"):
        EvidenceUnit(
            unit_id="a.pdf::rules",
            binding=unit.binding,
            label=unit.label,
            instruction=unit.instruction,
            retrieval_query=unit.retrieval_query,
            recovery_query=unit.recovery_query,
            scope=unit.scope,
            policy=unit.policy,
        )
    with pytest.raises(ValueError, match="exact one-source"):
        EvidenceUnit(
            unit_id=unit.unit_id,
            binding=unit.binding,
            label=unit.label,
            instruction=unit.instruction,
            retrieval_query=unit.retrieval_query,
            recovery_query=unit.recovery_query,
            scope=EvidenceSourceScope(allowed_sources=("other.pdf",)),
            policy=unit.policy,
        )


def test_budget_rejects_invalid_caps_and_unfair_required_reservation():
    with pytest.raises(ValueError, match="max_docs_per_unit"):
        _budget(max_total_docs=2, max_docs_per_unit=3)

    units = _summary_units()
    with pytest.raises(ValueError, match="reserve every required unit"):
        _budget(max_total_docs=1, max_docs_per_unit=1).validate_plan_capacity(units)


def test_budget_can_reserve_plan_capacity_without_raising_per_unit_caps():
    units = _summary_units()
    baseline = _budget(
        max_total_docs=1,
        max_total_chars=1,
        max_docs_per_unit=1,
        max_chars_per_unit=1,
    )

    reserved = baseline.reserve_plan_capacity(units)

    assert baseline.max_total_docs == 1
    assert baseline.max_total_chars == 1
    assert reserved.max_total_docs == 2
    assert reserved.max_total_chars == 2
    assert reserved.max_docs_per_unit == baseline.max_docs_per_unit
    assert reserved.max_chars_per_unit == baseline.max_chars_per_unit
    reserved.validate_plan_capacity(units)


def test_supported_closure_requires_grounding_subset_and_unique_views():
    unit = _summary_units()[0]
    with pytest.raises(ValueError, match="contained"):
        EvidenceUnitClosure(
            unit_id=unit.unit_id,
            status=EvidenceClosureStatus.SUPPORTED,
            evidence=(
                EvidenceView(
                    chunk_id="chunk:a:1",
                    source="handbook.pdf",
                    estimated_chars=20,
                ),
            ),
            grounding_chunk_ids=("fabricated",),
        )
    with pytest.raises(ValueError, match="duplicate evidence views"):
        EvidenceUnitClosure(
            unit_id=unit.unit_id,
            status=EvidenceClosureStatus.SUPPORTED,
            evidence=(
                EvidenceView("chunk:a:1", "handbook.pdf", 20),
                EvidenceView("chunk:a:1", "handbook.pdf", 20),
            ),
            grounding_chunk_ids=("chunk:a:1",),
        )


def test_distinct_spans_of_one_chunk_are_distinct_evidence_views():
    unit = _summary_units()[0]
    closure = EvidenceUnitClosure(
        unit_id=unit.unit_id,
        status=EvidenceClosureStatus.SUPPORTED,
        evidence=(
            EvidenceView("chunk:a:1", "handbook.pdf", 20, span_start=0, span_end=20),
            EvidenceView("chunk:a:1", "handbook.pdf", 20, span_start=30, span_end=50),
        ),
        grounding_chunk_ids=("chunk:a:1",),
    )

    assert closure.estimated_chars == 40


@pytest.mark.parametrize(
    "status",
    [
        EvidenceClosureStatus.NO_EVIDENCE,
        EvidenceClosureStatus.RETRIEVAL_ERROR,
        EvidenceClosureStatus.VERIFICATION_ERROR,
        EvidenceClosureStatus.BUDGET_EXHAUSTED,
    ],
)
def test_non_grounded_closure_statuses_fail_closed(status):
    unit = _summary_units()[0]
    closure = EvidenceUnitClosure(
        unit_id=unit.unit_id,
        status=status,
        reason_code="not_supported"
        if status is EvidenceClosureStatus.NO_EVIDENCE
        else "error",
    )
    closure.validate_for(unit, _budget())

    with pytest.raises(ValueError, match="fail closed"):
        EvidenceUnitClosure(
            unit_id=unit.unit_id,
            status=status,
            evidence=(EvidenceView("chunk:a:1", "handbook.pdf", 20),),
            reason_code="error",
        )


def test_closure_validation_enforces_source_and_per_unit_budget():
    unit = _summary_units()[0]
    wrong_source = _supported(unit, source="other.pdf")
    with pytest.raises(ValueError, match="source scope"):
        wrong_source.validate_for(unit, _budget())

    too_large = _supported(unit, source="handbook.pdf", chars=4_001)
    with pytest.raises(ValueError, match="character budget"):
        too_large.validate_for(unit, _budget())


def test_batch_closure_validation_requires_exact_results_and_global_budget():
    units = _summary_units()
    first = _supported(units[0], source="handbook.pdf", chars=100)
    second = _supported(units[1], source="handbook.pdf", chars=100)
    validate_evidence_unit_closures(units, [first, second], _budget())

    with pytest.raises(ValueError, match="exactly one"):
        validate_evidence_unit_closures(units, [first], _budget())
    with pytest.raises(ValueError, match="global character budget"):
        validate_evidence_unit_closures(
            units,
            [
                _supported(units[0], source="handbook.pdf", chars=100),
                _supported(units[1], source="handbook.pdf", chars=100),
            ],
            _budget(max_total_chars=150, max_chars_per_unit=100),
        )


def test_policy_rejects_semantic_admission_group_names():
    with pytest.raises(ValueError, match="opaque"):
        EvidenceUnitPolicy(admission_group="method::a.pdf")
