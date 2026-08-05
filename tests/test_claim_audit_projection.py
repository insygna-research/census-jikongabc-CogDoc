from __future__ import annotations

from copy import deepcopy

import pytest

from cogdoc.service.claim_audit_projection import (
    CLAIM_AUDIT_PROJECTION_VERSION,
    ClaimAuditProjectionError,
    ClaimAuditProjectionSegment,
    ClaimAuditProjectionStatus,
    build_claim_audit_projection,
    load_claim_audit_projection,
)


def test_projection_audits_only_generated_segments_in_structured_order():
    answer = """# 结构化输出
## 方法
方法使用向量检索[E001]。
## 指标
文档中未明确说明。
## 限制
本单元证据处理未完成，请重试。
## 结论
该方法支持混合检索[E002]。"""
    segments = (
        ClaimAuditProjectionSegment.generated("method", "方法使用向量检索[E001]。"),
        ClaimAuditProjectionSegment.deterministic(
            "metrics",
            "文档中未明确说明。",
            source_status="no_evidence",
        ),
        ClaimAuditProjectionSegment.operational(
            "limits",
            "本单元证据处理未完成，请重试。",
            source_status="retrieval_error",
        ),
        ClaimAuditProjectionSegment.generated(
            "conclusion", "该方法支持混合检索[E002]。"
        ),
    )

    projection = build_claim_audit_projection(answer, segments)

    assert projection.audit_text == (
        "方法使用向量检索[E001]。\n\n该方法支持混合检索[E002]。"
    )
    assert [segment.segment_id for segment in projection.generated_segments] == [
        "method",
        "conclusion",
    ]
    assert projection.has_generated_content
    assert projection.metrics == {
        "segment_count": 4,
        "generated_count": 2,
        "deterministic_count": 1,
        "operational_count": 1,
        "obligation_count": 0,
    }


def test_projection_round_trips_through_strict_state_contract():
    answer = "生成事实[E001]。\n文档中未明确说明。"
    projection = build_claim_audit_projection(
        answer,
        (
            ClaimAuditProjectionSegment.generated(
                "u1",
                "生成事实[E001]。",
                obligation_ids=("eu_requirement_1", "eu_requirement_2"),
            ),
            ClaimAuditProjectionSegment.deterministic(
                "u2",
                "文档中未明确说明。",
                source_status="future_task_no_evidence",
            ),
        ),
    )

    restored = load_claim_audit_projection(projection.to_state(), answer=answer)

    assert restored == projection
    assert restored.version == CLAIM_AUDIT_PROJECTION_VERSION
    assert restored.segments[1].source_status == "future_task_no_evidence"
    assert restored.obligation_ids == (
        "eu_requirement_1",
        "eu_requirement_2",
    )
    assert restored.to_state() == projection.to_state()


def test_deterministic_and_operational_text_never_becomes_a_claim():
    answer = "确定性状态里看起来像事实：上限为 99。\n操作失败：服务不可用。"
    projection = build_claim_audit_projection(
        answer,
        (
            ClaimAuditProjectionSegment.deterministic(
                "deterministic",
                "确定性状态里看起来像事实：上限为 99。",
                source_status="no_evidence",
            ),
            ClaimAuditProjectionSegment.operational(
                "operational",
                "操作失败：服务不可用。",
                source_status="future_terminal_state",
            ),
        ),
    )

    assert projection.audit_text == ""
    assert not projection.has_generated_content
    assert projection.metrics["deterministic_count"] == 1
    assert projection.metrics["operational_count"] == 1


def test_projection_is_bound_to_the_exact_final_answer():
    projection = build_claim_audit_projection(
        "原始事实[E001]。",
        (ClaimAuditProjectionSegment.generated("u1", "原始事实[E001]。"),),
    )

    with pytest.raises(
        ClaimAuditProjectionError,
        match="claim_audit_projection_answer_mismatch",
    ) as exc_info:
        load_claim_audit_projection(
            projection.to_state(),
            answer="答案已被后续节点改写[E001]。",
        )

    assert exc_info.value.reason_code == "claim_audit_projection_answer_mismatch"


def test_projection_rejects_segment_missing_from_digest_bound_answer():
    answer = "第一条事实[E001]。"
    state = build_claim_audit_projection(
        answer,
        (ClaimAuditProjectionSegment.generated("u1", answer),),
    ).to_state()
    state["segments"][0]["content"] = "投影中伪造的事实[E001]。"

    with pytest.raises(
        ClaimAuditProjectionError,
        match="claim_audit_projection_segment_missing",
    ) as exc_info:
        load_claim_audit_projection(state, answer=answer)

    assert exc_info.value.reason_code == "claim_audit_projection_segment_missing"


def test_projection_rejects_segments_outside_final_render_order():
    answer = "第一条事实[E001]。\n第二条事实[E002]。"
    projection = build_claim_audit_projection(
        answer,
        (
            ClaimAuditProjectionSegment.generated("first", "第一条事实[E001]。"),
            ClaimAuditProjectionSegment.generated("second", "第二条事实[E002]。"),
        ),
    )
    state = projection.to_state()
    state["segments"] = list(reversed(state["segments"]))

    with pytest.raises(
        ClaimAuditProjectionError,
        match="claim_audit_projection_segment_order_invalid",
    ) as exc_info:
        load_claim_audit_projection(state, answer=answer)

    assert exc_info.value.reason_code == "claim_audit_projection_segment_order_invalid"


def test_projection_rejects_duplicate_segment_ids():
    answer = "第一条。\n第二条。"

    with pytest.raises(
        ClaimAuditProjectionError,
        match="claim_audit_projection_duplicate_segment_id",
    ):
        build_claim_audit_projection(
            answer,
            (
                ClaimAuditProjectionSegment.generated("same", "第一条。"),
                ClaimAuditProjectionSegment.generated("same", "第二条。"),
            ),
        )


def test_projection_rejects_duplicate_obligation_across_segments():
    answer = "第一条。\n第二条。"

    with pytest.raises(
        ClaimAuditProjectionError,
        match="claim_audit_projection_duplicate_obligation_id",
    ):
        build_claim_audit_projection(
            answer,
            (
                ClaimAuditProjectionSegment.generated(
                    "first",
                    "第一条。",
                    obligation_ids=("eu_shared",),
                ),
                ClaimAuditProjectionSegment.generated(
                    "second",
                    "第二条。",
                    obligation_ids=("eu_shared",),
                ),
            ),
        )


@pytest.mark.parametrize("invalid", [[""], ["eu_1", "eu_1"], "eu_1"])
def test_projection_state_requires_unique_nonempty_obligation_id_list(invalid):
    answer = "第一条。"
    state = build_claim_audit_projection(
        answer,
        (ClaimAuditProjectionSegment.generated("first", answer),),
    ).to_state()
    state["segments"][0]["obligation_ids"] = invalid

    with pytest.raises(ClaimAuditProjectionError):
        load_claim_audit_projection(state, answer=answer)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: state.update({"unexpected": True}),
        lambda state: state["segments"][0].update({"unexpected": True}),
        lambda state: state["segments"][0].update({"status": "silently_skip"}),
        lambda state: state.update({"version": "v2"}),
    ],
)
def test_projection_state_fails_closed_for_unknown_contract_values(mutation):
    answer = "有依据的事实[E001]。"
    original = build_claim_audit_projection(
        answer,
        (ClaimAuditProjectionSegment.generated("u1", answer),),
    ).to_state()
    state = deepcopy(original)
    mutation(state)

    with pytest.raises(ClaimAuditProjectionError):
        load_claim_audit_projection(state, answer=answer)


def test_projection_requires_typed_status_in_memory():
    with pytest.raises(
        ClaimAuditProjectionError,
        match="claim_audit_projection_segment_invalid",
    ):
        ClaimAuditProjectionSegment(
            segment_id="u1",
            content="事实。",
            status="generated",  # type: ignore[arg-type]
        )


def test_repeated_segment_content_must_have_a_distinct_answer_occurrence():
    repeated = "文档中未明确说明。"
    projection = build_claim_audit_projection(
        f"{repeated}\n{repeated}",
        (
            ClaimAuditProjectionSegment.deterministic(
                "first", repeated, source_status="no_evidence"
            ),
            ClaimAuditProjectionSegment.deterministic(
                "second", repeated, source_status="no_evidence"
            ),
        ),
    )

    assert projection.metrics["deterministic_count"] == 2
    state = projection.to_state()
    third = dict(state["segments"][1])
    third["segment_id"] = "third"
    state["segments"].append(third)
    with pytest.raises(
        ClaimAuditProjectionError,
        match="claim_audit_projection_segment_order_invalid",
    ):
        load_claim_audit_projection(state, answer=f"{repeated}\n{repeated}")


def test_projection_accepts_typed_instance_and_outer_answer_whitespace():
    projection = build_claim_audit_projection(
        "  事实[E001]。\n  ",
        (ClaimAuditProjectionSegment.generated("u1", "事实[E001]。"),),
    )

    restored = load_claim_audit_projection(
        projection,
        answer="\n事实[E001]。\n",
    )

    assert restored is projection
    assert restored.generated_segments[0].status is (
        ClaimAuditProjectionStatus.GENERATED
    )
