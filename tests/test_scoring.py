import json

from cogdoc.tools.eval.scoring import (
    LLMJudge,
    aggregate_case,
    aggregate_run,
    evaluate_trial,
)


def test_deterministic_quality_and_fatal_gate_are_independent():
    report = evaluate_trial(
        {
            "trial_id": "t1",
            "execution_status": "SUCCESS",
            "agent_output": "答案 [a.pdf:P1] sk-secret-value-12345678901234567890",
            "expected": "答案 [a.pdf:P1] sk-secret-value-12345678901234567890",
        },
        [
            {"type": "exact_match", "role": "QUALITY", "requires": ["agent_output", "expected"]},
            {
                "type": "safety_assertion",
                "role": "GATE",
                "requires": ["agent_output"],
                "gate_policy": {"level": "FATAL", "required": True},
                "config": {"forbidden_patterns": [r"sk-[A-Za-z0-9-]{20,}"]},
            },
        ],
    )

    assert report["quality_score"] == 1.0
    assert report["gate_decision"] == "FATAL"
    assert report["decision"] == "FAIL"


def test_trace_incomplete_can_score_but_never_passes():
    report = evaluate_trial(
        {
            "trial_id": "t2",
            "execution_status": "TRACE_INCOMPLETE",
            "agent_output": "ok",
            "expected": "ok",
        },
        [{"type": "exact_match", "requires": ["agent_output", "expected"]}],
    )
    assert report["quality_score"] == 1.0
    assert report["decision"] == "NEEDS_REVIEW"


def test_missing_required_evidence_is_not_observable():
    report = evaluate_trial(
        {"trial_id": "t3", "execution_status": "SUCCESS", "agent_output": "ok"},
        [{"type": "ragas_metric", "requires": ["agent_output", "retrieved_context"], "config": {"metric": "faithfulness"}}],
    )
    assert report["quality_score"] is None
    assert report["decision"] == "NEEDS_REVIEW"
    assert report["evaluators"][0]["status"] == "NOT_OBSERVABLE"


def test_case_counts_execution_failures_and_mutually_exclusive_buckets():
    case = aggregate_case(
        [
            {"execution_status": "SUCCESS", "decision": "PASS", "quality_score": 1.0},
            {"execution_status": "TIMEOUT", "decision": "FAIL", "quality_score": None},
            {"execution_status": "TRACE_INCOMPLETE", "decision": "NEEDS_REVIEW", "quality_score": 0.5},
        ],
        min_trials=3,
        min_success_rate=0.8,
    )
    assert case["n_total"] == 3
    assert case["n_completed"] == 2
    assert case["n_passed"] == 1
    assert case["execution_completion_rate"] == 2 / 3
    assert case["observed_pass_rate"] == 1 / 3
    assert case["stability_status"] == "UNSTABLE"


def test_run_decision_and_stable_case_rate():
    report = aggregate_run(
        [
            {"quality_score": 0.9, "stability_status": "STABLE", "execution_completion_rate": 1.0},
            {"quality_score": None, "stability_status": "INSUFFICIENT", "execution_completion_rate": 0.0},
        ]
    )
    assert report["stable_case_rate"] == 0.5
    assert report["decision"] == "NEEDS_REVIEW"


def test_llm_judge_uses_common_output_schema(monkeypatch):
    expected = {
        "overall_score": 4,
        "dimension_scores": {"correctness": 4},
        "pass": True,
        "confidence": 0.9,
        "rationale": "有证据支持",
        "concerns": [],
        "evidence": [{"dimension": "correctness", "source": "agent_output", "quote": "ok"}],
        "recommended_action": "PASS",
    }

    monkeypatch.setattr(LLMJudge, "_client", lambda self: object())
    monkeypatch.setattr(
        "cogdoc.tools.eval.scoring.invoke_structured",
        lambda client, schema, messages: schema.model_validate(expected),
    )
    judge = LLMJudge()
    result = judge.evaluate(
        {"case_input": "问题", "agent_output": "ok", "expected": "ok"},
        type("Spec", (), {"config": {"dimensions": ["correctness"]}, "type": "llm_judge"})(),
    )
    assert result["score"] == 0.75
    assert result["status"] == "PASS"
    assert result["evidence"]
