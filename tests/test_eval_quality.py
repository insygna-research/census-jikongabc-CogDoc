import json
from cogdoc.tools.eval.quality_metrics import compare_baseline, run_eval


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
