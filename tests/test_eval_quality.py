import json
import sys

import pytest
import scripts.eval_quality as eval_quality
from cogdoc.tools.eval.quality_metrics import audit_coverage, compare_baseline, run_eval


# 构造测试用文档。
def _doc(source: str = "a.pdf", page: int = 1) -> dict:
    return {
        "text": "报名要求。",
        "meta": {
            "chunk_id": f"chunk:{source}:{page}",
            "source": source,
            "page": page,
            "page_start": page,
            "page_end": page,
            "chunk_index": 0,
            "local_chunk_index": 0,
            "source_sha256": "sha",
            "origin": "file",
        },
    }


# 验证质量评测报告包含路由、引用和忠实性指标。
def test_quality_eval_reports_router_citation_and_faithfulness():
    report = run_eval(
        [
            {
                "case_type": "router",
                "layer": "easy",
                "query": "请总结 a.pdf",
                "expected_task_type": "summary",
            },
            {
                "case_type": "citation",
                "layer": "easy",
                "answer": "文档说明报名要求。[a.pdf:P1]",
                "expected_valid": True,
                "docs": [_doc()],
            },
            {
                "case_type": "faithfulness",
                "layer": "hard",
                "answer": "人工标注为不忠实的答案。",
                "is_faithful": False,
            },
        ]
    )

    assert report["aggregate"]["router_rule_accuracy"] == 1.0
    assert report["aggregate"]["citation_accuracy"] == 1.0
    assert report["aggregate"]["faithfulness_manual_support_rate"] == 0.0
    assert report["baseline_gated_metrics"] == [
        "router_rule_accuracy",
        "citation_accuracy",
    ]
    assert report["by_layer"]["easy"]["count"] == 2
    assert report["by_layer"]["hard"]["count"] == 1


# 验证声明指标按 claim 数微平均，且不信任运行时附带的汇总值。
def test_quality_eval_reports_micro_averaged_claim_audit_diagnostics():
    report = run_eval(
        [
            {
                "case_type": "faithfulness",
                "layer": "hard",
                "is_faithful": False,
                "claim_audit": {
                    "status": "failed",
                    "claims": [
                        {
                            "claim_id": "c1",
                            "verdict": "supported",
                            "cited_chunk_ids": ["chunk-1"],
                        },
                        {
                            "claim_id": "c2",
                            "verdict": "supported",
                            "cited_chunk_ids": ["chunk-2"],
                        },
                        {
                            "claim_id": "c3",
                            "verdict": "unsupported",
                            "cited_chunk_ids": [],
                        },
                    ],
                    # 故意写错，评测必须从 claims 重算。
                    "counts": {"claim_count": 99, "supported": 99},
                    "metrics": {"claim_support_rate": 1.0},
                    "repair": {"attempted": True, "succeeded": False},
                    "verifier": {"duration_ms": 100},
                },
            },
            {
                "case_type": "faithfulness",
                "layer": "hard",
                "is_faithful": True,
                "trace": {
                    "output": {
                        "claim_audit": {
                            "status": "repaired",
                            "claims": [
                                {
                                    "claim_id": "c1",
                                    "verdict": "supported",
                                    "cited_chunk_ids": ["chunk-4"],
                                }
                            ],
                            "repair": {"attempted": True, "succeeded": True},
                            "verifier": {"duration_ms": 300},
                        }
                    }
                },
            },
            {
                "case_type": "faithfulness",
                "layer": "hard",
                "is_faithful": True,
            },
        ]
    )

    aggregate = report["aggregate"]
    assert aggregate["claim_audit_observable_rate"] == 2 / 3
    assert aggregate["claim_support_rate"] == 3 / 4
    assert aggregate["citation_coverage"] == 3 / 4
    assert aggregate["unsupported_claim_rate"] == 1 / 4
    assert aggregate["insufficient_claim_rate"] == 0.0
    assert aggregate["repair_success_rate"] == 1 / 2
    assert aggregate["claim_verifier_mean_duration_ms"] == 200.0
    assert "claim_support_rate" not in report["baseline_gated_metrics"]
    assert report["by_layer"]["hard"]["claim_support_rate"] == 3 / 4


# 验证可观测但没有事实声明时比率保持为空。
def test_quality_eval_keeps_empty_claim_rates_unset():
    report = run_eval(
        [
            {
                "case_type": "faithfulness",
                "layer": "easy",
                "claim_audit": {
                    "status": "passed",
                    "claims": [
                        {
                            "claim_id": "c1",
                            "verdict": "not_factual",
                            "cited_chunk_ids": [],
                        }
                    ],
                },
            }
        ]
    )

    assert report["aggregate"]["claim_audit_observable_rate"] == 1.0
    assert report["aggregate"]["claim_support_rate"] is None
    assert report["aggregate"]["citation_coverage"] is None
    assert report["aggregate"]["repair_success_rate"] is None
    assert report["aggregate"]["faithfulness_manual_support_rate"] is None


# 验证禁用门禁写入的 not_run 占位不冒充可观测审计。
def test_quality_eval_treats_not_run_claim_audit_as_unobservable():
    report = run_eval(
        [
            {
                "case_type": "faithfulness",
                "layer": "easy",
                "claim_audit": {"status": "not_run", "claims": []},
            }
        ]
    )

    assert report["aggregate"]["claim_audit_observable_rate"] == 0.0
    assert report["rows"][0]["metrics"]["claim_audit_observable"] == 0.0


# 验证未知状态与非有限耗时不污染质量汇总。
def test_quality_eval_rejects_unknown_status_and_nonfinite_duration():
    report = run_eval(
        [
            {
                "case_type": "faithfulness",
                "claim_audit": {"status": "mystery", "claims": []},
            },
            {
                "case_type": "faithfulness",
                "claim_audit": {
                    "status": "passed",
                    "claims": [],
                    "verifier": {"duration_ms": float("nan")},
                },
            },
        ]
    )

    assert report["aggregate"]["claim_audit_observable_rate"] == 0.0
    assert report["aggregate"]["claim_verifier_mean_duration_ms"] is None


# 验证质量评测能识别非法引用。
def test_quality_eval_catches_invalid_citation():
    report = run_eval(
        [
            {
                "case_type": "citation",
                "layer": "hard",
                "answer": "文档说明报名要求。[missing.pdf:P1]",
                "expected_valid": False,
                "docs": [_doc()],
            }
        ]
    )

    row = report["rows"][0]
    assert row["actual"] is False
    assert row["metrics"]["citation_accuracy"] == 1.0


# 验证质量基线对比在回退时返回非零。
def test_quality_baseline_compare_returns_nonzero_on_regression(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "aggregate": {
                    "router_rule_accuracy": 1.0,
                    "citation_accuracy": 1.0,
                    "faithfulness_manual_support_rate": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )
    report = {
        "aggregate": {
            "router_rule_accuracy": 0.5,
            "citation_accuracy": 1.0,
            "faithfulness_manual_support_rate": 0.0,
        }
    }

    assert compare_baseline(report, baseline) == 1


# 验证质量基线忽略人工忠实性台账回退。
def test_quality_baseline_ignores_manual_faithfulness_regression(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"aggregate": {"faithfulness_manual_support_rate": 1.0}}),
        encoding="utf-8",
    )
    report = {
        "aggregate": {"faithfulness_manual_support_rate": 0.0},
        "baseline_gated_metrics": ["router_rule_accuracy", "citation_accuracy"],
    }

    assert compare_baseline(report, baseline) == 0


# 验证质量基线把门禁指标缺失视为回退。
def test_quality_baseline_treats_missing_gated_metric_as_regression(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"aggregate": {"router_rule_accuracy": 1.0}}),
        encoding="utf-8",
    )
    report = {
        "aggregate": {"router_rule_accuracy": None},
        "baseline_gated_metrics": ["router_rule_accuracy"],
    }

    assert compare_baseline(report, baseline) == 1


# 验证质量覆盖审计能报告缺失维度。
def test_quality_coverage_audit_reports_missing_dimensions():
    coverage = audit_coverage(
        [
            {"case_type": "router", "layer": "easy"},
            {"case_type": "citation", "layer": "hard"},
        ]
    )

    assert coverage["missing_case_types"] == ["faithfulness"]
    assert coverage["missing_layers"] == [
        "no-answer",
        "summary",
        "compare",
        "multi-turn",
        "feedback",
    ]
    assert coverage["is_coverage_complete"] is False


# 写入覆盖完整的质量评测集。
def _write_complete_quality_eval(path):
    rows = [
        {"case_type": "router", "layer": "easy"},
        {"case_type": "citation", "layer": "hard"},
        {"case_type": "faithfulness", "layer": "no-answer"},
        {"case_type": "faithfulness", "layer": "summary"},
        {"case_type": "router", "layer": "compare"},
        {"case_type": "router", "layer": "multi-turn"},
        {"case_type": "faithfulness", "layer": "feedback"},
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


# 验证质量覆盖快速模式跳过完整评测。
def test_quality_cli_coverage_only_skips_eval(tmp_path, monkeypatch, capsys):
    eval_set = tmp_path / "quality.jsonl"
    _write_complete_quality_eval(eval_set)

    # 阻止覆盖快速模式误入完整评测。
    def fail_run_eval(_items):
        raise AssertionError("run_eval should not be called")

    monkeypatch.setattr(eval_quality, "run_eval", fail_run_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_quality.py", "--eval-set", str(eval_set), "--coverage-only"],
    )

    assert eval_quality.main() == 0
    out = capsys.readouterr().out
    assert "覆盖完整" in out
    assert "质量评测" not in out


# 验证质量覆盖快速模式拒绝写报告参数。
def test_quality_cli_coverage_only_rejects_json(tmp_path, monkeypatch):
    eval_set = tmp_path / "quality.jsonl"
    _write_complete_quality_eval(eval_set)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_quality.py",
            "--eval-set",
            str(eval_set),
            "--coverage-only",
            "--json",
            str(tmp_path / "report.json"),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        eval_quality.main()
    assert exc.value.code == 2


# 验证质量覆盖快速模式拒绝重复覆盖参数。
def test_quality_cli_coverage_only_rejects_check_coverage(tmp_path, monkeypatch):
    eval_set = tmp_path / "quality.jsonl"
    _write_complete_quality_eval(eval_set)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_quality.py",
            "--eval-set",
            str(eval_set),
            "--coverage-only",
            "--check-coverage",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        eval_quality.main()
    assert exc.value.code == 2
